"""Command-line interface: score your own runs without editing the code.

Point it at your tasks and responses. Nothing here calls a model API — you
generate responses however you like, save them as JSONL, and this scores them.
That makes a run reproducible and free to repeat.

    python cli.py --tasks data/tasks.jsonl --responses data/responses_model_a.jsonl
    python cli.py --tasks t.jsonl --responses a.jsonl --responses b.jsonl --compare
    python cli.py --tasks t.jsonl --responses a.jsonl --human-labels labels.csv
    python cli.py --tasks t.jsonl --responses a.jsonl --json scores.json --fail-under 0.8

Response files are named by their filename stem, so `responses_model_a.jsonl`
becomes the run `responses_model_a` unless you pass `--name`.

Exit codes:
    0  scored, and any --fail-under threshold was met
    1  scored, but the overall weighted score fell below the floor
    2  bad input -- missing file, malformed JSONL, rubric naming unknown checks
"""

import argparse
import json
import os
import sys

import pandas as pd

from harness import load_jsonl, load_rubric, overall_scores, score_run
from report import (
    category_summary,
    compare_runs,
    dimension_summary,
    review_queue,
    validate_judge,
    win_rate,
)

EXIT_OK, EXIT_BELOW_THRESHOLD, EXIT_BAD_INPUT = 0, 1, 2


def bad_input(message: str):
    """Exit 2 on stderr. Kept distinct from 1 so a missing file and a
    low-scoring run are different signals to a build system."""
    print(message, file=sys.stderr)
    return SystemExit(EXIT_BAD_INPUT)


def read_jsonl(path: str, what: str) -> list:
    if not os.path.exists(path):
        raise bad_input(f"error: no such {what} file: {path}")
    try:
        rows = load_jsonl(path)
    except json.JSONDecodeError as exc:
        raise bad_input(f"error: {path} is not valid JSONL — {exc}")
    if not rows:
        raise bad_input(f"error: {path} is empty")
    return rows


def run_name(path: str, override: str = None) -> str:
    return override or os.path.splitext(os.path.basename(path))[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="llm-eval",
        description="Rubric-driven offline scoring for saved LLM responses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tasks", required=True, help="JSONL of task definitions")
    p.add_argument("--responses", required=True, action="append",
                   metavar="PATH",
                   help="JSONL of responses; repeat the flag for a second run")
    p.add_argument("--name", action="append", metavar="NAME",
                   help="name for each responses file, in the same order")
    p.add_argument("--rubric", default="rubric.json")
    p.add_argument("--compare", action="store_true",
                   help="head-to-head between the first two runs")
    p.add_argument("--human-labels", metavar="CSV",
                   help="hand labels to validate the judge against")
    p.add_argument("--queue", metavar="PATH", help="write the review queue as CSV")
    p.add_argument("--queue-threshold", type=float, default=0.7)
    p.add_argument("--fail-under", type=float, metavar="SCORE",
                   help="exit 1 if a run's mean weighted score is below this")
    p.add_argument("--json", metavar="PATH", help="write all results as JSON")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.name and len(args.name) != len(args.responses):
        raise bad_input(
            f"error: {len(args.name)} --name values for "
            f"{len(args.responses)} --responses files"
        )
    if args.compare and len(args.responses) < 2:
        raise bad_input("error: --compare needs two --responses files")

    tasks = read_jsonl(args.tasks, "tasks")
    if not os.path.exists(args.rubric):
        raise bad_input(f"error: no such rubric: {args.rubric}")
    try:
        rubric = load_rubric(args.rubric)
    except ValueError as exc:
        # rubric names a check that doesn't exist -- fail loudly rather than
        # silently scoring three dimensions and calling it four
        raise bad_input(f"error: {exc}")

    names, frames = [], []
    for i, path in enumerate(args.responses):
        name = run_name(path, args.name[i] if args.name else None)
        responses = read_jsonl(path, "responses")
        try:
            frames.append(score_run(tasks, responses, rubric, name))
        except ValueError as exc:
            raise bad_input(f"error: {exc}")
        names.append(name)

    df = pd.concat(frames, ignore_index=True)
    results = {
        "rubric": rubric["name"],
        "n_tasks": len(tasks),
        "runs": names,
        "dimension_means": dimension_summary(df).round(4).to_dict(),
        "category_means": category_summary(df).round(4).to_dict(),
        "overall": {},
    }
    for name in names:
        scores = overall_scores(df[df["run"] == name], rubric)
        results["overall"][name] = round(float(scores.mean()), 4)

    if not args.quiet:
        print("=" * 62)
        print(f"EVAL REPORT  |  rubric: {rubric['name']}  |  "
              f"{len(tasks)} tasks, {len(names)} run(s)")
        print("=" * 62)
        print("\nMean score by dimension:")
        print(dimension_summary(df).round(3).to_string())
        print("\nWeighted overall:")
        for name, value in results["overall"].items():
            print(f"  {name:<28} {value:.3f}")

    if args.compare:
        comparison = compare_runs(df, rubric, names[0], names[1])
        wr = win_rate(comparison, names[0], names[1])
        results["head_to_head"] = {k: int(v) if k != "n" else int(v)
                                   for k, v in wr.items()}
        if not args.quiet:
            print(f"\nHead-to-head {names[0]} vs {names[1]}: "
                  f"{wr['wins']}W / {wr['ties']}T / {wr['losses']}L of {wr['n']}")
            print("\nLargest per-task gaps:")
            print(comparison.head(5).to_string())
            print("\nNote: no significance test. With a few dozen tasks a gap of")
            print("a few points is noise. Treat this as a direction, not a verdict.")

    if args.human_labels:
        if not os.path.exists(args.human_labels):
            raise bad_input(f"error: no such labels file: {args.human_labels}")
        target = names[-1]
        summary, merged = validate_judge(df, args.human_labels, target)
        results["judge_validation"] = {
            "run": target,
            "per_dimension": summary.to_dict(),
            "n_disagreements": int((~merged["agree"]).sum()),
        }
        if not args.quiet:
            print(f"\nJudge validation against hand labels ({target}):")
            print(summary.to_string())
            bad = merged[~merged["agree"]]
            if len(bad):
                print("\nDisagreements — each one is a lesson about the judge:")
                print(bad[["task_id", "dimension", "human_score", "score"]]
                      .to_string(index=False))
            else:
                print("No disagreements on the audited subset.")

    if args.queue:
        queue = review_queue(df, rubric, names[-1], threshold=args.queue_threshold)
        queue.to_csv(args.queue, index=False)
        results["review_queue_size"] = int(len(queue))
        if not args.quiet:
            print(f"\nWrote {len(queue)} low-scoring tasks to {args.queue}")

    status = EXIT_OK
    if args.fail_under is not None:
        failing = {n: v for n, v in results["overall"].items()
                   if v < args.fail_under}
        results["fail_under"] = {"threshold": args.fail_under,
                                 "failing_runs": failing,
                                 "passed": not failing}
        status = EXIT_BELOW_THRESHOLD if failing else EXIT_OK
        if not args.quiet:
            verdict = "FAIL" if failing else "PASS"
            print(f"\n[{verdict}] floor {args.fail_under:.2f}")
            for name, value in failing.items():
                print(f"       {name}: {value:.3f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"Wrote {args.json}")

    return status


if __name__ == "__main__":
    sys.exit(main())
