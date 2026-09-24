"""Synthetic pairwise and binary judge verdicts with PLANTED biases.

The point of planting them: judge_audit.py has to recover numbers that are
known in advance, so the audit is checked against the truth rather than
eyeballed. tests/test_judge_audit.py asserts the recovery.

Two files come out:

  data/judge_pairwise.jsonl  -- three judges compare pairs of responses,
                                every pair shown in BOTH orders
  data/judge_binary.jsonl    -- three judges mark single answers correct /
                                incorrect, across five batches over time

Planted effects (see PLANT below):
  judge_alpha : strong primacy bias, prefers its own model's outputs, and its
                false-positive rate drifts upward from batch 3 onward
  judge_beta  : close to unbiased -- the control
  judge_gamma : mild recency bias, no self-preference
  all judges  : more lenient toward stronger examinees (capability-dependent
                leniency, the effect reported in arXiv:2609.12002)

No model is called. Everything is a coin flip with a known bias.

On sample size: with 600 prompts the control judges' self-preference beta
lands within about +/-0.08 of zero; at 240 prompts it swung +/-0.2 across
seeds, which is the practical lesson -- equal-quality pairs are a small
subset, and beta needs a few hundred comparisons before its interval
excludes zero for a judge that really has the bias.
"""

import json
import math
import os
import random

SEED = 1
N_PROMPTS = 600           # pairwise comparisons; each becomes 2 rows
N_ITEMS_PER_BATCH = 320   # binary verdicts per batch
N_BATCHES = 5

JUDGES = ["judge_alpha", "judge_beta", "judge_gamma"]
# which examinee model each judge shares a family with (self-preference target)
OWN_MODEL = {"judge_alpha": "model_alpha", "judge_beta": "model_beta",
             "judge_gamma": "model_gamma"}
MODELS = ["model_alpha", "model_beta", "model_gamma", "model_delta"]

PLANT = {
    # log-odds added toward picking the FIRST-presented response
    "position": {"judge_alpha": 0.9, "judge_beta": 0.05, "judge_gamma": -0.5},
    # log-odds added toward picking the judge's OWN model when present
    "self_pref": {"judge_alpha": 1.2, "judge_beta": 0.0, "judge_gamma": 0.0},
    # how sharply a judge tracks true quality differences (bigger = better)
    "acuity": {"judge_alpha": 4.0, "judge_beta": 5.0, "judge_gamma": 4.5},
    # binary task: examinee true accuracy
    "examinee_accuracy": {"model_alpha": 0.85, "model_beta": 0.70,
                          "model_gamma": 0.60, "model_delta": 0.50},
    # binary task: base FPR / FNR per judge
    "fpr": {"judge_alpha": 0.10, "judge_beta": 0.08, "judge_gamma": 0.12},
    "fnr": {"judge_alpha": 0.06, "judge_beta": 0.07, "judge_gamma": 0.10},
    # capability-dependent leniency: extra FPR per unit of examinee accuracy
    # above 0.5 (strong examinees get the benefit of the doubt)
    "leniency_slope": 0.35,
    # drift: judge_alpha's FPR rises by this much from batch 3 onward
    "drift": {"judge": "judge_alpha", "from_batch": 3, "fpr_increase": 0.18},
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_pairwise(rng: random.Random) -> list:
    rows = []
    for p in range(N_PROMPTS):
        m_a, m_b = rng.sample(MODELS, 2)
        q_a, q_b = rng.random(), rng.random()
        pair_id = f"pair_{p:04d}"
        for judge in JUDGES:
            own = OWN_MODEL[judge]
            for order in (1, 2):
                first, second = ((m_a, q_a), (m_b, q_b)) if order == 1 \
                    else ((m_b, q_b), (m_a, q_a))
                logit = PLANT["acuity"][judge] * (first[1] - second[1])
                logit += PLANT["position"][judge]
                if first[0] == own:
                    logit += PLANT["self_pref"][judge]
                if second[0] == own:
                    logit -= PLANT["self_pref"][judge]
                pick_first = rng.random() < sigmoid(logit)
                rows.append({
                    "pair_id": pair_id, "judge": judge, "order": order,
                    "first_model": first[0], "second_model": second[0],
                    "first_quality": round(first[1], 4),
                    "second_quality": round(second[1], 4),
                    "verdict": "first" if pick_first else "second",
                })
    return rows


def make_binary(rng: random.Random) -> list:
    rows = []
    acc = PLANT["examinee_accuracy"]
    for batch in range(1, N_BATCHES + 1):
        for i in range(N_ITEMS_PER_BATCH):
            examinee = rng.choice(MODELS)
            truth = 1 if rng.random() < acc[examinee] else 0
            item_id = f"b{batch}_item_{i:03d}"
            for judge in JUDGES:
                fpr = PLANT["fpr"][judge] + PLANT["leniency_slope"] * (acc[examinee] - 0.5)
                fnr = PLANT["fnr"][judge]
                d = PLANT["drift"]
                if judge == d["judge"] and batch >= d["from_batch"]:
                    fpr += d["fpr_increase"]
                if truth == 1:
                    verdict = 0 if rng.random() < fnr else 1
                else:
                    verdict = 1 if rng.random() < fpr else 0
                rows.append({"item_id": item_id, "batch": batch,
                             "examinee": examinee, "judge": judge,
                             "verdict": verdict, "truth": truth})
    return rows


def write_jsonl(path: str, rows: list) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    os.makedirs("data", exist_ok=True)
    pw = make_pairwise(rng)
    bn = make_binary(rng)
    write_jsonl("data/judge_pairwise.jsonl", pw)
    write_jsonl("data/judge_binary.jsonl", bn)
    print(f"wrote {len(pw)} pairwise rows and {len(bn)} binary rows")


if __name__ == "__main__":
    main()
