"""End-to-end demo: score two model runs, compare, validate the judge.

Run:  python make_example_data.py   (once, to create data/)
      python demo.py
"""

import os

from harness import load_jsonl, load_rubric, score_run
import pandas as pd

from report import (
    category_summary,
    compare_runs,
    dimension_summary,
    plot_category_heatmap,
    plot_dimension_comparison,
    review_queue,
    validate_judge,
    win_rate,
)


def main() -> None:
    os.makedirs("output", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    tasks = load_jsonl("data/tasks.jsonl")
    rubric = load_rubric("rubric.json")

    frames = []
    for run in ("model_a", "model_b"):
        responses = load_jsonl(f"data/responses_{run}.jsonl")
        frames.append(score_run(tasks, responses, rubric, run))
    df = pd.concat(frames, ignore_index=True)

    print("=" * 62)
    print(f"EVAL REPORT  |  rubric: {rubric['name']}  |  "
          f"{len(tasks)} tasks, 2 runs")
    print("=" * 62)

    print("\nMean score by dimension:")
    print(dimension_summary(df).to_string())

    print("\nMean score by category:")
    print(category_summary(df).to_string())

    comparison = compare_runs(df, rubric, "model_a", "model_b")
    wr = win_rate(comparison, "model_a", "model_b")
    print(f"\nHead-to-head (weighted overall, model_a vs model_b): "
          f"{wr['wins']} wins / {wr['ties']} ties / {wr['losses']} losses "
          f"of {wr['n']}")
    print("\nLargest per-task gaps:")
    print(comparison.head(5).to_string())

    queue = review_queue(df, rubric, "model_b", threshold=0.7)
    queue.to_csv("output/review_queue_model_b.csv", index=False)
    print(f"\nWrote {len(queue)} low-scoring model_b tasks to "
          "output/review_queue_model_b.csv")

    print("\nJudge validation against hand labels (model_b subset):")
    summary, merged = validate_judge(df, "data/human_labels.csv", "model_b")
    print(summary.to_string())
    disagreements = merged[~merged["agree"]]
    if len(disagreements):
        print("\nJudge-vs-human disagreements (each one is a lesson "
              "about the judge):")
        print(disagreements[["task_id", "dimension", "human_score",
                             "score"]].to_string(index=False))

    plot_dimension_comparison(df, "figures/dimension_comparison.png")
    plot_category_heatmap(df, "model_b", "figures/category_heatmap.png")
    print("\nWrote figures/dimension_comparison.png and "
          "figures/category_heatmap.png")

    print("\n" + "-" * 62)
    print("READING THE SIGNAL")
    print("-" * 62)
    print(
        "model_b's gap is not uniform: accuracy is nearly level with\n"
        "model_a, while the losses concentrate in format (broken JSON on\n"
        "extraction) and safety (over-refusal on benign-but-sensitive\n"
        "questions, plus one unsafe compliance). Patterned losses have\n"
        "specific fixes -- output templating for the format failures,\n"
        "refusal calibration for the safety ones -- where a uniform gap\n"
        "would have suggested a capability problem no prompt fix solves."
    )


if __name__ == "__main__":
    main()
