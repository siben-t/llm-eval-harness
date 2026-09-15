"""Harness core: score a run of model responses against a rubric.

The rubric lives in rubric.json as data, not code -- dimensions map to
check functions by name, with a weight per dimension. Scores come out
in long format (task_id, category, dimension, score), the same tidy
shape used in annotation-qa-toolkit, so the same aggregation habits
apply.
"""

import json

import pandas as pd

from checks import CHECKS


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rubric(path: str = "rubric.json") -> dict:
    with open(path) as f:
        rubric = json.load(f)
    unknown = set(rubric["dimensions"]) - set(CHECKS)
    if unknown:
        raise ValueError(f"rubric names unknown checks: {sorted(unknown)}")
    return rubric


def score_run(tasks: list, responses: list, rubric: dict,
              run_name: str) -> pd.DataFrame:
    """Score one model run. Returns long-format DataFrame with columns:
    run, task_id, category, dimension, score. Dimensions that don't
    apply to a task are simply absent (None scores are dropped)."""
    by_id = {r["task_id"]: r["response"] for r in responses}
    missing = [t["task_id"] for t in tasks if t["task_id"] not in by_id]
    if missing:
        raise ValueError(f"run '{run_name}' missing responses for: {missing}")

    rows = []
    for task in tasks:
        response = by_id[task["task_id"]]
        for dim in rubric["dimensions"]:
            score = CHECKS[dim](task, response)
            if score is None:
                continue
            rows.append({
                "run": run_name,
                "task_id": task["task_id"],
                "category": task["category"],
                "dimension": dim,
                "score": float(score),
            })
    return pd.DataFrame(rows)


def overall_scores(df: pd.DataFrame, rubric: dict) -> pd.Series:
    """Weighted overall score per task: weighted mean over the
    dimensions that applied, re-normalizing weights to those present."""
    weights = {d: spec["weight"] for d, spec in rubric["dimensions"].items()}

    def weighted(group: pd.DataFrame) -> float:
        w = group["dimension"].map(weights)
        return float((group["score"] * w).sum() / w.sum())

    return df.groupby("task_id").apply(weighted, include_groups=False)
