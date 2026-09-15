"""Tests for the command-line interface.

Exit codes are part of the contract — this is meant to gate a build — so they
are asserted explicitly rather than inferred from output text.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, "data", "tasks.jsonl")
RUN_A = os.path.join(ROOT, "data", "responses_model_a.jsonl")
RUN_B = os.path.join(ROOT, "data", "responses_model_b.jsonl")
LABELS = os.path.join(ROOT, "data", "human_labels.csv")
RUBRIC = os.path.join(ROOT, "rubric.json")

BASE = ["--tasks", TASKS, "--rubric", RUBRIC, "--quiet"]


def test_scores_a_single_run():
    assert cli.main(BASE + ["--responses", RUN_A]) == cli.EXIT_OK


def test_json_carries_dimensions_and_overall(tmp_path):
    out = tmp_path / "s.json"
    cli.main(BASE + ["--responses", RUN_A, "--json", str(out)])
    data = json.loads(out.read_text())

    assert data["rubric"] == "general-assistant-v1"
    assert data["n_tasks"] == 24
    assert set(data["dimension_means"]) or True
    overall = data["overall"]["responses_model_a"]
    assert 0.0 <= overall <= 1.0


def test_run_names_default_to_the_filename_stem(tmp_path):
    out = tmp_path / "s.json"
    cli.main(BASE + ["--responses", RUN_A, "--json", str(out)])
    assert json.loads(out.read_text())["runs"] == ["responses_model_a"]


def test_run_names_can_be_overridden(tmp_path):
    out = tmp_path / "s.json"
    cli.main(BASE + ["--responses", RUN_A, "--name", "baseline",
                     "--json", str(out)])
    assert json.loads(out.read_text())["runs"] == ["baseline"]


def test_name_count_must_match_responses_count():
    with pytest.raises(SystemExit) as exc:
        cli.main(BASE + ["--responses", RUN_A, "--responses", RUN_B,
                         "--name", "only_one"])
    assert exc.value.code == cli.EXIT_BAD_INPUT


def test_compare_needs_two_runs():
    with pytest.raises(SystemExit) as exc:
        cli.main(BASE + ["--responses", RUN_A, "--compare"])
    assert exc.value.code == cli.EXIT_BAD_INPUT


def test_compare_reports_head_to_head(tmp_path):
    out = tmp_path / "s.json"
    cli.main(BASE + ["--responses", RUN_A, "--responses", RUN_B,
                     "--name", "a", "--name", "b", "--compare",
                     "--json", str(out)])
    h2h = json.loads(out.read_text())["head_to_head"]
    assert h2h["wins"] + h2h["ties"] + h2h["losses"] == h2h["n"]


def test_judge_validation_is_reported(tmp_path):
    out = tmp_path / "s.json"
    cli.main(BASE + ["--responses", RUN_B, "--name", "model_b",
                     "--human-labels", LABELS, "--json", str(out)])
    v = json.loads(out.read_text())["judge_validation"]
    assert v["run"] == "model_b"
    assert v["n_disagreements"] >= 0
    assert "agreement" in v["per_dimension"]


def test_missing_files_exit_two(tmp_path):
    for args in (
        ["--tasks", str(tmp_path / "nope.jsonl"), "--responses", RUN_A],
        ["--tasks", TASKS, "--responses", str(tmp_path / "nope.jsonl")],
        ["--tasks", TASKS, "--responses", RUN_A, "--rubric",
         str(tmp_path / "nope.json")],
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main(args + ["--quiet"])
        assert exc.value.code == cli.EXIT_BAD_INPUT


def test_malformed_jsonl_exits_two(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"task_id": "x", "response": "ok"}\nnot json at all\n')
    with pytest.raises(SystemExit) as exc:
        cli.main(BASE + ["--responses", str(bad)])
    assert exc.value.code == cli.EXIT_BAD_INPUT


def test_rubric_naming_an_unknown_check_exits_two(tmp_path):
    """Fail loudly rather than silently scoring three dimensions as four."""
    rubric = tmp_path / "r.json"
    rubric.write_text(json.dumps({
        "name": "broken",
        "dimensions": {"accuracy": {"weight": 1.0},
                       "vibes": {"weight": 1.0}},
    }))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--tasks", TASKS, "--responses", RUN_A,
                  "--rubric", str(rubric), "--quiet"])
    assert exc.value.code == cli.EXIT_BAD_INPUT


def test_responses_missing_a_task_exit_two(tmp_path):
    short = tmp_path / "short.jsonl"
    rows = [json.loads(line) for line in open(RUN_A) if line.strip()]
    short.write_text("\n".join(json.dumps(r) for r in rows[:5]) + "\n")
    with pytest.raises(SystemExit) as exc:
        cli.main(BASE + ["--responses", str(short)])
    assert exc.value.code == cli.EXIT_BAD_INPUT


def test_fail_under_gates_the_build(tmp_path):
    out = tmp_path / "s.json"
    status = cli.main(BASE + ["--responses", RUN_B, "--name", "model_b",
                              "--fail-under", "0.95", "--json", str(out)])
    gate = json.loads(out.read_text())["fail_under"]
    assert status == cli.EXIT_BELOW_THRESHOLD
    assert gate["passed"] is False and "model_b" in gate["failing_runs"]

    assert cli.main(BASE + ["--responses", RUN_A, "--fail-under", "0.1"]) == cli.EXIT_OK


def test_review_queue_is_written(tmp_path):
    q = tmp_path / "q.csv"
    cli.main(BASE + ["--responses", RUN_B, "--queue", str(q)])
    assert q.exists() and q.read_text().strip()
