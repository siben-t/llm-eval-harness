"""judge_audit.py is checked against PLANTED biases, not eyeballed.

make_judge_data.py plants: judge_alpha has primacy bias, self-preference and
FPR drift from batch 3; judge_beta is the clean control; judge_gamma has mild
recency bias and nothing else; every judge is more lenient toward stronger
examinees. Each test asserts the audit recovers the planted fact and does
NOT flag the control.
"""

import hashlib
import json
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import make_judge_data as gen                                  # noqa: E402
from judge_audit import (                                      # noqa: E402
    EXIT_BAD_INPUT, EXIT_FAIL, EXIT_OK,
    capability_leniency, dawid_skene, directional_bias, drift,
    ensemble_accuracy, position_bias, self_preference,
    validate_binary, validate_pairwise, wilson_ci,
)

OWN = gen.OWN_MODEL


@pytest.fixture(scope="module")
def generated() -> tuple:
    # mirror make_judge_data.main(): ONE rng, pairwise drawn before binary,
    # so the fixtures are byte-for-byte the shipped data files
    rng = random.Random(gen.SEED)
    pw = pd.DataFrame(gen.make_pairwise(rng))
    bn = pd.DataFrame(gen.make_binary(rng))
    return validate_pairwise(pw), validate_binary(bn)


@pytest.fixture(scope="module")
def pairwise(generated) -> pd.DataFrame:
    return generated[0]


@pytest.fixture(scope="module")
def binary(generated) -> pd.DataFrame:
    return generated[1]


def test_fixtures_match_the_shipped_files(pairwise, binary):
    shipped = pd.read_json(os.path.join(ROOT, "data", "judge_binary.jsonl"), lines=True)
    assert len(shipped) == len(binary)
    assert (shipped["verdict"].to_numpy() == binary["verdict"].to_numpy()).all()


# ---- generator ------------------------------------------------------------

def test_generator_is_deterministic():
    a = json.dumps(gen.make_pairwise(random.Random(3)))
    b = json.dumps(gen.make_pairwise(random.Random(3)))
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()


def test_every_pair_is_shown_in_both_orders(pairwise):
    counts = pairwise.groupby(["judge", "pair_id"])["order"].nunique()
    assert (counts == 2).all()


# ---- position bias --------------------------------------------------------

def test_deterministic_quality_judge_sits_at_exactly_half():
    """A judge that always picks the better response, regardless of order,
    must show first_pick_rate == 0.5 exactly: each pair contributes one
    'first' and one 'second'."""
    rows = []
    for i in range(50):
        qa, qb = 0.9, 0.1
        rows.append(dict(pair_id=i, judge="j", order=1, first_model="A", second_model="B",
                         first_quality=qa, second_quality=qb, verdict="first"))
        rows.append(dict(pair_id=i, judge="j", order=2, first_model="B", second_model="A",
                         first_quality=qb, second_quality=qa, verdict="second"))
    pos = position_bias(validate_pairwise(pd.DataFrame(rows)))
    assert pos.loc["j", "first_pick_rate"] == 0.5
    assert pos.loc["j", "flip_rate"] == 0.0


def test_pure_primacy_judge_is_fully_flagged():
    rows = []
    for i in range(40):
        for order, (f, s) in enumerate([("A", "B"), ("B", "A")], start=1):
            rows.append(dict(pair_id=i, judge="j", order=order, first_model=f,
                             second_model=s, first_quality=0.5, second_quality=0.5,
                             verdict="first"))
    pos = position_bias(validate_pairwise(pd.DataFrame(rows)))
    assert pos.loc["j", "first_pick_rate"] == 1.0
    assert pos.loc["j", "flip_rate"] == 1.0
    assert pos.loc["j", "primacy_share_of_flips"] == 1.0


def test_planted_position_biases_are_recovered(pairwise):
    pos = position_bias(pairwise)
    assert pos.loc["judge_alpha", "position_bias"] > 0.10         # primacy planted
    assert pos.loc["judge_alpha", "primacy_share_of_flips"] > 0.8
    assert abs(pos.loc["judge_beta", "position_bias"]) < 0.05     # control
    assert pos.loc["judge_gamma", "position_bias"] < -0.03        # recency planted
    # the interval for the control must contain 0.5
    assert pos.loc["judge_beta", "first_pick_ci_low"] <= 0.5 <= pos.loc["judge_beta", "first_pick_ci_high"]


# ---- self-preference ------------------------------------------------------

def test_self_preference_flags_only_the_planted_judge(pairwise):
    sp = self_preference(pairwise, OWN)
    assert sp.loc["judge_alpha", "beta"] > 0.15
    assert sp.loc["judge_alpha", "beta_ci_low"] > 0.0             # interval excludes zero
    for control in ("judge_beta", "judge_gamma"):
        assert abs(sp.loc[control, "beta"]) < 0.10
        assert sp.loc[control, "beta_ci_low"] <= 0.0 <= sp.loc[control, "beta_ci_high"]


def test_self_preference_needs_quality_columns(pairwise):
    with pytest.raises(ValueError):
        self_preference(pairwise.drop(columns=["first_quality", "second_quality"]), OWN)


def test_unmapped_judge_gets_nan_not_a_crash(pairwise):
    sp = self_preference(pairwise, {"judge_alpha": "model_alpha"})
    assert np.isnan(sp.loc["judge_beta", "beta"])
    assert sp.loc["judge_alpha", "beta"] > 0.15


# ---- directional bias, leniency, drift -----------------------------------

def test_directional_bias_matches_planted_rates(binary):
    db = directional_bias(binary)
    for j in gen.JUDGES:
        # planted FPR > FNR for every judge, so all read lenient
        assert db.loc[j, "directional_bias"] > 0
        # Wilson interval must contain the point estimate
        assert db.loc[j, "FPR_ci_low"] <= db.loc[j, "FPR"] <= db.loc[j, "FPR_ci_high"]
    # leniency (extra FPR) plus drift make alpha the most lenient
    assert db["directional_bias"].idxmax() == "judge_alpha"


def test_capability_dependent_leniency_is_detected(binary):
    cl = capability_leniency(binary)
    # pooled across judges: strongest examinee gets a higher FPR than the weakest
    assert cl.loc["ALL", "strong_minus_weak"] > 0.05
    assert cl.loc["ALL", "gap_p_value"] < 0.01
    for j in gen.JUDGES:
        assert cl.loc[j, "strong_minus_weak"] > 0
    assert cl.loc["ALL", "r_accuracy_vs_FPR"] > 0.5


def test_drift_is_found_only_where_planted(binary):
    dr = drift(binary)
    d = gen.PLANT["drift"]
    assert dr.loc[d["judge"], "FPR_change"] > d["fpr_increase"] * 0.5
    assert dr.loc[d["judge"], "FPR_p_value"] < 0.01
    assert dr.loc["judge_beta", "FPR_p_value"] > 0.05


# ---- Dawid-Skene ---------------------------------------------------------

def test_dawid_skene_recovers_error_rates_without_truth(binary):
    rates, posterior = dawid_skene(binary.drop(columns=["truth"]))
    truth_rates = directional_bias(binary)
    for j in gen.JUDGES:
        assert abs(rates.loc[j, "FPR_est"] - truth_rates.loc[j, "FPR"]) < 0.05
        assert abs(rates.loc[j, "FNR_est"] - truth_rates.loc[j, "FNR"]) < 0.05
    assert posterior.between(0, 1).all()


def test_dawid_skene_tolerates_missing_verdicts(binary):
    thinned = binary.sample(frac=0.8, random_state=0)
    rates, posterior = dawid_skene(thinned.drop(columns=["truth"]))
    truth_rates = directional_bias(binary)
    for j in gen.JUDGES:
        assert abs(rates.loc[j, "FPR_est"] - truth_rates.loc[j, "FPR"]) < 0.08


def test_weighted_vote_is_at_least_as_good_as_the_best_judge(binary):
    _, posterior = dawid_skene(binary)
    ens = ensemble_accuracy(binary, posterior)
    assert ens["dawid_skene_vote"] >= ens["best_single_judge"]
    assert ens["dawid_skene_vote"] >= ens["majority_vote"] - 0.01


# ---- validation ----------------------------------------------------------

def test_single_order_data_is_rejected(pairwise):
    with pytest.raises(ValueError, match="both orders"):
        validate_pairwise(pairwise[pairwise["order"] == 1])


def test_bad_verdict_value_is_rejected(pairwise):
    bad = pairwise.copy()
    bad.loc[bad.index[0], "verdict"] = "left"
    with pytest.raises(ValueError, match="first/second/tie"):
        validate_pairwise(bad)


def test_wilson_interval_behaves_at_the_edges():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = wilson_ci(20, 20)
    assert 0.8 < lo < 1 and hi == 1.0
    assert all(np.isnan(v) for v in wilson_ci(0, 0))


# ---- CLI -----------------------------------------------------------------

def run(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "judge_audit.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_cli_gates_fail_on_the_planted_judge(tmp_path):
    out = tmp_path / "audit.json"
    r = run("--pairwise", "data/judge_pairwise.jsonl", "--binary", "data/judge_binary.jsonl",
            "--own", "judge_alpha=model_alpha", "--own", "judge_beta=model_beta",
            "--fail-position-bias", "0.10", "--fail-beta", "0.15", "--fail-drift", "0.10",
            "--json", str(out), "--quiet")
    assert r.returncode == EXIT_FAIL, r.stderr
    d = json.loads(out.read_text())
    failing = {f.split(":")[0] for f in d["gates"]["failures"]}
    assert failing == {"judge_alpha"}
    for key in ("position_bias", "self_preference", "directional_bias",
                "capability_leniency", "drift", "label_free_vs_truth", "ensemble"):
        assert key in d


def test_cli_passes_when_gates_are_loose():
    r = run("--pairwise", "data/judge_pairwise.jsonl", "--fail-position-bias", "0.5", "--quiet")
    assert r.returncode == EXIT_OK, r.stderr


def test_cli_bad_input_exits_two(tmp_path):
    assert run("--pairwise", "nope.jsonl").returncode == EXIT_BAD_INPUT
    assert run().returncode == EXIT_BAD_INPUT
    assert run("--pairwise", "data/judge_pairwise.jsonl", "--own", "broken").returncode == EXIT_BAD_INPUT
