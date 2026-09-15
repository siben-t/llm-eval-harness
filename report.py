"""Reporting layer: aggregate scores, compare runs, validate the judge.

The three questions an eval report must answer, in order:
1. How good is each model, per dimension and per category?
2. Where exactly does the weaker model lose -- uniformly, or in a
   pattern? (Patterned losses have specific fixes; uniform losses
   usually mean a capability gap.)
3. Can we trust the judge? Automated checks are only as good as their
   agreement with careful human labels on an audited subset.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness import overall_scores


def dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score per (run, dimension), wide format for reading."""
    return (df.groupby(["dimension", "run"])["score"].mean()
              .unstack("run").round(3))


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score per (run, category), wide format."""
    return (df.groupby(["category", "run"])["score"].mean()
              .unstack("run").round(3))


def compare_runs(df: pd.DataFrame, rubric: dict,
                 run_a: str, run_b: str) -> pd.DataFrame:
    """Per-task overall scores side by side, plus win/tie/loss."""
    a = overall_scores(df[df["run"] == run_a], rubric).rename(run_a)
    b = overall_scores(df[df["run"] == run_b], rubric).rename(run_b)
    out = pd.concat([a, b], axis=1)
    out["delta"] = (out[run_a] - out[run_b]).round(3)
    return out.sort_values("delta", ascending=False)


def win_rate(comparison: pd.DataFrame, run_a: str, run_b: str) -> dict:
    wins = int((comparison["delta"] > 1e-9).sum())
    losses = int((comparison["delta"] < -1e-9).sum())
    ties = len(comparison) - wins - losses
    return {"wins": wins, "ties": ties, "losses": losses,
            "n": len(comparison)}


def review_queue(df: pd.DataFrame, rubric: dict, run: str,
                 threshold: float = 0.7) -> pd.DataFrame:
    """Lowest-scoring tasks for a run -- the human-review queue."""
    overall = overall_scores(df[df["run"] == run], rubric)
    low = overall[overall < threshold].sort_values()
    return low.rename("overall_score").reset_index()


def validate_judge(df: pd.DataFrame, human_path: str, run: str) -> pd.DataFrame:
    """Agreement between the programmatic judge and hand labels on an
    audited subset. Reports, per dimension: n audited, mean absolute
    difference, and agreement rate within 0.25."""
    human = pd.read_csv(human_path)
    judge = df[df["run"] == run][["task_id", "dimension", "score"]]
    merged = human.merge(judge, on=["task_id", "dimension"],
                         suffixes=("_human", "_judge"))
    merged["abs_diff"] = (merged["human_score"] - merged["score"]).abs()
    merged["agree"] = merged["abs_diff"] <= 0.25
    summary = (merged.groupby("dimension")
                     .agg(n=("agree", "size"),
                          mean_abs_diff=("abs_diff", "mean"),
                          agreement=("agree", "mean"))
                     .round(3))
    return summary, merged


def plot_dimension_comparison(df: pd.DataFrame, path: str) -> None:
    summary = dimension_summary(df)
    ax = summary.plot.bar(figsize=(7, 4), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean score")
    ax.set_title("Mean score by dimension")
    ax.legend(title="run")
    ax.grid(axis="y", alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_category_heatmap(df: pd.DataFrame, run: str, path: str) -> None:
    pivot = (df[df["run"] == run]
             .groupby(["category", "dimension"])["score"].mean()
             .unstack("dimension"))
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8)
    ax.set_title(f"{run}: mean score by category x dimension")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
