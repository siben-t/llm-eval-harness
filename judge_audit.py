"""Audit an LLM-as-a-judge for the failure modes that make its scores untrustworthy.

Four things, each with a number and a confidence interval, each with a
named source for the definition:

  position bias      does the verdict change when the two responses swap
                     places? Measured by showing every pair in both orders.
                     (Shi et al., "A Systematic Study of Position Bias in
                     LLM-as-a-Judge", IJCNLP 2025)
  self-preference    does the judge favour outputs from its own model family,
                     over and above ordinary consistency? beta = PIR - null-PIR
                     on equal-quality pairs. (arXiv:2604.22891)
  directional bias   on correct/incorrect verdicts: FPR - FNR. Positive = lenient,
                     negative = strict. Plus capability-dependent leniency: does
                     FPR rise with the examinee's true accuracy?
                     (arXiv:2609.12002, Sept 2026)
  drift              do FPR / FNR move across batches over time? Reported as
                     first-vs-last with a two-proportion z-test.

And one repair: Dawid-Skene EM over several judges estimates each judge's
error rates WITHOUT ground truth and produces a weighted vote. When truth is
available the audit reports how close the label-free estimate came, which
is the test of whether you could run this on a live system.

Nothing here calls a model. Verdicts are data you already have.

Input formats (JSONL, one row per verdict):

  pairwise:  pair_id, judge, order (1|2), first_model, second_model,
             first_quality, second_quality, verdict ("first"|"second"|"tie")
             -- each pair_id must appear with order 1 AND order 2 per judge
  binary:    item_id, judge, examinee, verdict (0|1), truth (0|1, optional),
             batch (optional, for drift)

CLI:
  python judge_audit.py --pairwise data/judge_pairwise.jsonl \
                        --binary data/judge_binary.jsonl \
                        --own judge_alpha=model_alpha --own judge_beta=model_beta \
                        --json audit.json --fail-position-bias 0.10 --fail-beta 0.15 --fail-drift 0.10
Exit 0 if every judge clears the gates; 1 if any judge fails one; 2 on bad input.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

EXIT_OK, EXIT_FAIL, EXIT_BAD_INPUT = 0, 1, 2
Z_95 = 1.959964


# --------------------------------------------------------------------------
# small statistics helpers
# --------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple:
    """Wilson score interval for a proportion. Well-behaved at 0 and 1,
    unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: np.ndarray, stat, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap of `stat` over rows of `values`."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(values[i]) for i in idx])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple:
    """z statistic and two-sided p-value for p1 != p2."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p2 - p1) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, p)


# --------------------------------------------------------------------------
# loading and validation
# --------------------------------------------------------------------------

def load_jsonl(path: str) -> pd.DataFrame:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    return pd.DataFrame(rows)


PAIRWISE_COLS = ["pair_id", "judge", "order", "first_model", "second_model",
                 "verdict"]
BINARY_COLS = ["item_id", "judge", "examinee", "verdict"]


def validate_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PAIRWISE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"pairwise data missing columns: {missing}")
    bad = set(df["verdict"]) - {"first", "second", "tie"}
    if bad:
        raise ValueError(f"pairwise verdict must be first/second/tie, got {sorted(bad)}")
    counts = df.groupby(["judge", "pair_id"])["order"].nunique()
    incomplete = counts[counts != 2]
    if len(incomplete):
        raise ValueError(
            f"{len(incomplete)} (judge, pair) combinations are not shown in "
            f"both orders; position bias needs both. First few: "
            f"{incomplete.index[:3].tolist()}")
    return df


def validate_binary(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BINARY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"binary data missing columns: {missing}")
    if not set(df["verdict"].unique()) <= {0, 1}:
        raise ValueError("binary verdict must be 0 or 1")
    if "truth" in df.columns and not set(df["truth"].dropna().unique()) <= {0, 1}:
        raise ValueError("binary truth must be 0 or 1")
    return df


# --------------------------------------------------------------------------
# pairwise: position bias and self-preference
# --------------------------------------------------------------------------

def _winner_identity(row) -> str:
    """Which RESPONSE (by model id) the judge picked, independent of order."""
    if row["verdict"] == "tie":
        return "tie"
    return row["first_model"] if row["verdict"] == "first" else row["second_model"]


def _pairs_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (judge, pair): the picked model in each order."""
    d = df.copy()
    d["picked"] = d.apply(_winner_identity, axis=1)
    o1 = d[d["order"] == 1].set_index(["judge", "pair_id"])
    o2 = d[d["order"] == 2].set_index(["judge", "pair_id"])
    wide = pd.DataFrame({
        "picked_o1": o1["picked"], "picked_o2": o2["picked"],
        "verdict_o1": o1["verdict"], "verdict_o2": o2["verdict"],
        # the two responses, named by the models that produced them
        "model_x": o1["first_model"], "model_y": o1["second_model"],
        "quality_x": o1.get("first_quality"), "quality_y": o1.get("second_quality"),
    }).reset_index()
    wide["consistent"] = wide["picked_o1"] == wide["picked_o2"]
    return wide


def position_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Per judge:
      first_pick_rate  P(verdict == first) over all rows. Every pair is shown
                       both ways, so an order-blind judge sits at exactly 0.5
                       regardless of the quality gap. Deviation = position bias.
      flip_rate        share of pairs whose picked response CHANGED with order
      primacy          among flipped pairs, share where the judge took the
                       first slot both times (1.0 = pure primacy, 0.0 = recency)
    """
    wide = _pairs_wide(df)
    out = []
    for judge, g in wide.groupby("judge"):
        rows = df[df["judge"] == judge]
        decided = rows[rows["verdict"] != "tie"]
        k_first = int((decided["verdict"] == "first").sum())
        n = len(decided)
        flipped = g[~g["consistent"]]
        n_flip = len(flipped)
        primacy_k = int(((flipped["verdict_o1"] == "first") &
                         (flipped["verdict_o2"] == "first")).sum())
        lo, hi = wilson_ci(k_first, n)
        flo, fhi = wilson_ci(n_flip, len(g))
        out.append({
            "judge": judge,
            "n_pairs": int(len(g)),
            "first_pick_rate": k_first / n if n else float("nan"),
            "first_pick_ci_low": lo, "first_pick_ci_high": hi,
            "position_bias": (k_first / n - 0.5) if n else float("nan"),
            "flip_rate": n_flip / len(g),
            "flip_ci_low": flo, "flip_ci_high": fhi,
            "primacy_share_of_flips": primacy_k / n_flip if n_flip else float("nan"),
        })
    return pd.DataFrame(out).set_index("judge")


def self_preference(df: pd.DataFrame, own: dict, epsilon: float = 0.15,
                    seed: int = 0) -> pd.DataFrame:
    """beta = PIR - null-PIR, per judge, following arXiv:2604.22891.

    PIR      on equal-quality pairs (|q_x - q_y| <= epsilon) where one response
             is from the judge's own model: share where the judge picked its own
             response in BOTH orders.
    null-PIR on equal-quality third-party pairs (neither response is its own):
             the rate at which the judge consistently prefers an arbitrary one
             of the two -- P(consistent) / 2. This is the base rate of
             "consistently preferring one of two equal responses" and absorbs
             position bias and acuity, so the difference isolates self-preference.

    Judges with no `own` mapping, or too few qualifying pairs, come back NaN.
    """
    if "first_quality" not in df.columns:
        raise ValueError("self-preference needs first_quality / second_quality")
    wide = _pairs_wide(df)
    wide = wide[(wide["quality_x"] - wide["quality_y"]).abs() <= epsilon]
    out = []
    for judge, g in wide.groupby("judge"):
        mine = own.get(judge)
        if mine is None:
            out.append({"judge": judge, "own_model": None, "n_own_pairs": 0,
                        "n_null_pairs": int(len(g)), "PIR": float("nan"),
                        "null_PIR": float("nan"), "beta": float("nan"),
                        "beta_ci_low": float("nan"), "beta_ci_high": float("nan")})
            continue
        involves = (g["model_x"] == mine) | (g["model_y"] == mine)
        own_pairs = g[involves]
        null_pairs = g[~involves]
        own_hits = ((own_pairs["picked_o1"] == mine) &
                    (own_pairs["picked_o2"] == mine)).to_numpy().astype(float)
        # null: a third-party pair has two responses; exactly one of them can be
        # the consistent pick. Averaging over both as targets gives
        # P(consistent)/2 per pair -- same expectation as a random target,
        # half the variance.
        null_hits = null_pairs["consistent"].to_numpy().astype(float) / 2.0
        pir = own_hits.mean() if len(own_hits) else float("nan")
        null = null_hits.mean() if len(null_hits) else float("nan")
        # bootstrap the difference by resampling each group independently
        if len(own_hits) and len(null_hits):
            b = np.random.default_rng(seed + 1)
            diffs = []
            for _ in range(2000):
                a = own_hits[b.integers(0, len(own_hits), len(own_hits))].mean()
                c = null_hits[b.integers(0, len(null_hits), len(null_hits))].mean()
                diffs.append(a - c)
            lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
        else:
            lo = hi = float("nan")
        out.append({"judge": judge, "own_model": mine,
                    "n_own_pairs": int(len(own_pairs)),
                    "n_null_pairs": int(len(null_pairs)),
                    "PIR": pir, "null_PIR": null, "beta": pir - null,
                    "beta_ci_low": lo, "beta_ci_high": hi})
    return pd.DataFrame(out).set_index("judge")


# --------------------------------------------------------------------------
# binary: directional bias, capability-dependent leniency, drift
# --------------------------------------------------------------------------

def _rates(g: pd.DataFrame) -> dict:
    neg = g[g["truth"] == 0]
    pos = g[g["truth"] == 1]
    fp, tn = int((neg["verdict"] == 1).sum()), int((neg["verdict"] == 0).sum())
    fn, tp = int((pos["verdict"] == 0).sum()), int((pos["verdict"] == 1).sum())
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    fnr = fn / (fn + tp) if fn + tp else float("nan")
    return {"n": int(len(g)), "fp": fp, "tn": tn, "fn": fn, "tp": tp,
            "FPR": fpr, "FNR": fnr, "directional_bias": fpr - fnr}


def directional_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Per judge: FPR, FNR, and FPR - FNR (positive = lenient). Requires truth."""
    if "truth" not in df.columns:
        raise ValueError("directional bias needs a truth column")
    out = []
    for judge, g in df.groupby("judge"):
        r = _rates(g)
        r["judge"] = judge
        r["FPR_ci_low"], r["FPR_ci_high"] = wilson_ci(r["fp"], r["fp"] + r["tn"])
        r["FNR_ci_low"], r["FNR_ci_high"] = wilson_ci(r["fn"], r["fn"] + r["tp"])
        out.append(r)
    return pd.DataFrame(out).set_index("judge")


def capability_leniency(df: pd.DataFrame) -> pd.DataFrame:
    """Does the judge get more lenient as the examinee gets stronger?

    Per judge: FPR by examinee, the Pearson r between examinee accuracy and
    FPR (the statistic arXiv:2609.12002 reports, r >= 0.83 there), and --
    because r over a handful of examinees is unstable -- the FPR gap between
    the strongest and weakest examinee with a two-proportion p-value. A row
    "ALL" pools every judge, which is the best-powered version of the test.
    """
    if "truth" not in df.columns:
        raise ValueError("capability leniency needs a truth column")
    acc = df.groupby("examinee")["truth"].mean().rename("examinee_accuracy")
    strongest, weakest = acc.idxmax(), acc.idxmin()

    def one(g: pd.DataFrame, label: str) -> dict:
        per = g.groupby("examinee").apply(lambda x: _rates(x), include_groups=False)
        fpr = per.apply(lambda r: r["FPR"]).rename("FPR")
        table = pd.concat([acc, fpr], axis=1).dropna()
        r = float(np.corrcoef(table["examinee_accuracy"], table["FPR"])[0, 1]) \
            if len(table) >= 3 else float("nan")
        s, w = per[strongest], per[weakest]
        _, p = two_proportion_z(w["fp"], w["fp"] + w["tn"], s["fp"], s["fp"] + s["tn"])
        return {"judge": label, "r_accuracy_vs_FPR": r,
                "FPR_strongest": s["FPR"], "FPR_weakest": w["FPR"],
                "strong_minus_weak": s["FPR"] - w["FPR"], "gap_p_value": p,
                "FPR_by_examinee": table["FPR"].round(4).to_dict()}

    out = [one(g, j) for j, g in df.groupby("judge")]
    out.append(one(df, "ALL"))
    return pd.DataFrame(out).set_index("judge")


def drift(df: pd.DataFrame) -> pd.DataFrame:
    """FPR and FNR per batch; first-vs-last two-proportion test per judge."""
    if "truth" not in df.columns or "batch" not in df.columns:
        raise ValueError("drift needs truth and batch columns")
    out = []
    batches = sorted(df["batch"].unique())
    for judge, g in df.groupby("judge"):
        per = {b: _rates(g[g["batch"] == b]) for b in batches}
        f, l = per[batches[0]], per[batches[-1]]
        z_fpr, p_fpr = two_proportion_z(f["fp"], f["fp"] + f["tn"],
                                        l["fp"], l["fp"] + l["tn"])
        z_fnr, p_fnr = two_proportion_z(f["fn"], f["fn"] + f["tp"],
                                        l["fn"], l["fn"] + l["tp"])
        out.append({
            "judge": judge,
            "FPR_by_batch": {b: round(per[b]["FPR"], 4) for b in batches},
            "FNR_by_batch": {b: round(per[b]["FNR"], 4) for b in batches},
            "FPR_first": f["FPR"], "FPR_last": l["FPR"],
            "FPR_change": l["FPR"] - f["FPR"], "FPR_p_value": p_fpr,
            "FNR_change": l["FNR"] - f["FNR"], "FNR_p_value": p_fnr,
        })
    return pd.DataFrame(out).set_index("judge")


# --------------------------------------------------------------------------
# repair: label-free error-rate estimation and weighted voting
# --------------------------------------------------------------------------

def dawid_skene(df: pd.DataFrame, n_iter: int = 100, tol: float = 1e-6) -> tuple:
    """Binary Dawid-Skene EM. Estimates each judge's FPR and FNR and the
    posterior P(truth = 1) per item using only the verdicts.

    Returns (rates: DataFrame indexed by judge, posterior: Series by item_id).
    """
    piv = df.pivot_table(index="item_id", columns="judge", values="verdict",
                         aggfunc="first")
    V = piv.to_numpy(dtype=float)              # items x judges, NaN if absent
    present = ~np.isnan(V)
    mu = np.nanmean(V, axis=1)                 # init: majority-vote fraction
    mu = np.clip(mu, 0.01, 0.99)
    eps = 1e-3
    for _ in range(n_iter):
        pi = float(mu.mean())
        w1 = mu[:, None] * present
        w0 = (1 - mu)[:, None] * present
        fnr = np.clip((w1 * (V == 0)).sum(0) / np.maximum(w1.sum(0), 1e-12), eps, 1 - eps)
        fpr = np.clip((w0 * (V == 1)).sum(0) / np.maximum(w0.sum(0), 1e-12), eps, 1 - eps)
        # E-step in log space
        log1 = np.log(pi) + np.where(present, np.where(V == 1, np.log(1 - fnr), np.log(fnr)), 0).sum(1)
        log0 = np.log(1 - pi) + np.where(present, np.where(V == 1, np.log(fpr), np.log(1 - fpr)), 0).sum(1)
        new_mu = 1 / (1 + np.exp(log0 - log1))
        if np.max(np.abs(new_mu - mu)) < tol:
            mu = new_mu
            break
        mu = new_mu
    rates = pd.DataFrame({"FPR_est": fpr, "FNR_est": fnr}, index=piv.columns)
    posterior = pd.Series(mu, index=piv.index, name="p_correct")
    return rates, posterior


def ensemble_accuracy(df: pd.DataFrame, posterior: pd.Series) -> dict:
    """Accuracy of plain majority vote vs the Dawid-Skene weighted vote,
    against truth. Only meaningful when truth exists."""
    if "truth" not in df.columns:
        raise ValueError("ensemble accuracy needs truth")
    truth = df.groupby("item_id")["truth"].first()
    maj = (df.groupby("item_id")["verdict"].mean() > 0.5).astype(int)
    ds = (posterior.reindex(truth.index) > 0.5).astype(int)
    per_judge = {j: float((g.set_index("item_id")["verdict"]
                           .reindex(truth.index) == truth).mean())
                 for j, g in df.groupby("judge")}
    return {"majority_vote": float((maj.reindex(truth.index) == truth).mean()),
            "dawid_skene_vote": float((ds == truth).mean()),
            "best_single_judge": max(per_judge.values()),
            "per_judge": per_judge}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_own(values) -> dict:
    own = {}
    for v in values or []:
        if "=" not in v:
            raise ValueError(f"--own expects judge=model, got {v!r}")
        j, m = v.split("=", 1)
        own[j.strip()] = m.strip()
    return own


def _json_safe(obj):
    if isinstance(obj, pd.DataFrame):
        return {str(k): _json_safe(v) for k, v in obj.to_dict("index").items()}
    if isinstance(obj, pd.Series):
        return {str(k): _json_safe(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return None if (isinstance(obj, float) and math.isnan(obj)) or \
            (isinstance(obj, np.floating) and np.isnan(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="judge-audit",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--pairwise", metavar="JSONL")
    p.add_argument("--binary", metavar="JSONL")
    p.add_argument("--own", action="append", metavar="JUDGE=MODEL",
                   help="which examinee model a judge shares a family with")
    p.add_argument("--epsilon", type=float, default=0.15,
                   help="quality gap that still counts as equal (self-preference)")
    p.add_argument("--fail-position-bias", type=float, metavar="BIAS",
                   help="exit 1 if any judge's |first_pick_rate - 0.5| exceeds this")
    p.add_argument("--fail-flip-rate", type=float, metavar="RATE",
                   help="exit 1 if any judge flips more than this share of pairs "
                        "(noise gate; position bias is the directional one)")
    p.add_argument("--fail-beta", type=float, metavar="BETA",
                   help="exit 1 if any judge's self-preference beta exceeds this")
    p.add_argument("--fail-drift", type=float, metavar="DELTA",
                   help="exit 1 if any judge's FPR moved more than this first-to-last "
                        "AND the move is significant at p < 0.05")
    p.add_argument("--json", metavar="PATH")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.pairwise and not args.binary:
        print("error: give --pairwise and/or --binary", file=sys.stderr)
        return EXIT_BAD_INPUT
    results, failures = {}, []
    say = (lambda *a, **k: None) if args.quiet else print

    try:
        own = _parse_own(args.own)
        if args.pairwise:
            if not os.path.exists(args.pairwise):
                raise ValueError(f"no such file: {args.pairwise}")
            pw = validate_pairwise(load_jsonl(args.pairwise))
            pos = position_bias(pw)
            results["position_bias"] = pos
            say("=" * 66)
            say("JUDGE AUDIT")
            say("=" * 66)
            say("\nPosition bias  (first_pick_rate: 0.5 = order-blind)")
            say(pos[["n_pairs", "first_pick_rate", "position_bias", "flip_rate",
                     "primacy_share_of_flips"]].round(3).to_string())
            if own and "first_quality" in pw.columns:
                sp = self_preference(pw, own, epsilon=args.epsilon)
                results["self_preference"] = sp
                say(f"\nSelf-preference  (beta = PIR - null-PIR on pairs within "
                    f"{args.epsilon} quality)")
                say(sp[["own_model", "n_own_pairs", "n_null_pairs", "PIR",
                        "null_PIR", "beta", "beta_ci_low", "beta_ci_high"]]
                    .round(3).to_string())
            if args.fail_position_bias is not None:
                for j, r in pos.iterrows():
                    if abs(r["position_bias"]) > args.fail_position_bias:
                        failures.append(f"{j}: position bias {r['position_bias']:+.3f} "
                                        f"exceeds {args.fail_position_bias}")
            if args.fail_flip_rate is not None:
                for j, r in pos.iterrows():
                    if r["flip_rate"] > args.fail_flip_rate:
                        failures.append(f"{j}: flip rate {r['flip_rate']:.3f} > "
                                        f"{args.fail_flip_rate}")
            if args.fail_beta is not None and "self_preference" in results:
                for j, r in results["self_preference"].iterrows():
                    if r["beta"] == r["beta"] and r["beta"] > args.fail_beta:
                        failures.append(f"{j}: beta {r['beta']:.3f} > {args.fail_beta}")

        if args.binary:
            if not os.path.exists(args.binary):
                raise ValueError(f"no such file: {args.binary}")
            bn = validate_binary(load_jsonl(args.binary))
            rates, posterior = dawid_skene(bn)
            results["label_free_rates"] = rates
            if "truth" in bn.columns:
                db = directional_bias(bn)
                results["directional_bias"] = db
                say("\nDirectional bias  (FPR - FNR: + lenient, - strict)")
                say(db[["n", "FPR", "FPR_ci_low", "FPR_ci_high", "FNR",
                        "directional_bias"]].round(3).to_string())
                cl = capability_leniency(bn)
                results["capability_leniency"] = cl
                say("\nCapability-dependent leniency  (FPR on strongest examinee minus weakest)")
                for j, r in cl.iterrows():
                    say(f"  {j:<14} gap = {r['strong_minus_weak']:+.3f} (p={r['gap_p_value']:.4f})"
                        f"   r = {r['r_accuracy_vs_FPR']:+.2f}   by examinee: {r['FPR_by_examinee']}")
                if "batch" in bn.columns:
                    dr = drift(bn)
                    results["drift"] = dr
                    say("\nDrift  (first batch -> last batch)")
                    for j, r in dr.iterrows():
                        say(f"  {j:<14} FPR {r['FPR_first']:.3f} -> {r['FPR_last']:.3f} "
                            f"(change {r['FPR_change']:+.3f}, p={r['FPR_p_value']:.4f})")
                    if args.fail_drift is not None:
                        # both the size AND the significance: a gate that fires
                        # on noise teaches people to ignore it
                        for j, r in dr.iterrows():
                            if abs(r["FPR_change"]) > args.fail_drift and r["FPR_p_value"] < 0.05:
                                failures.append(f"{j}: FPR drifted {r['FPR_change']:+.3f} "
                                                f"(p={r['FPR_p_value']:.4f})")
                comp = pd.concat([db[["FPR", "FNR"]], rates], axis=1)
                comp["FPR_abs_err"] = (comp["FPR"] - comp["FPR_est"]).abs()
                comp["FNR_abs_err"] = (comp["FNR"] - comp["FNR_est"]).abs()
                results["label_free_vs_truth"] = comp
                say("\nLabel-free error rates (Dawid-Skene) vs truth")
                say(comp.round(3).to_string())
                ens = ensemble_accuracy(bn, posterior)
                results["ensemble"] = ens
                say(f"\nEnsemble accuracy: majority {ens['majority_vote']:.3f}, "
                    f"Dawid-Skene weighted {ens['dawid_skene_vote']:.3f}, "
                    f"best single judge {ens['best_single_judge']:.3f}")
            else:
                say("\nLabel-free error-rate estimates (no truth column given)")
                say(rates.round(3).to_string())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    status = EXIT_OK
    if any(f is not None for f in (args.fail_position_bias, args.fail_flip_rate,
                                   args.fail_beta, args.fail_drift)):
        results["gates"] = {"failures": failures, "passed": not failures}
        status = EXIT_FAIL if failures else EXIT_OK
        say(f"\n[{'FAIL' if failures else 'PASS'}] gates")
        for f in failures:
            say(f"       {f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(_json_safe(results), f, indent=2)
        say(f"Wrote {args.json}")
    return status


if __name__ == "__main__":
    sys.exit(main())
