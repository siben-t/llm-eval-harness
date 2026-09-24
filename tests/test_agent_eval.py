"""agent_eval.py is checked against PLANTED faults, not eyeballed.

make_agent_data.py records every fault as it plants it. Each test asserts that
the evaluator flags exactly the planted trajectories, with no misses and no
false alarms, and that it leaves the control agent (agent_ref) alone,
retries included. The estimators and statistics are checked by hand.
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import make_agent_data as gen                                  # noqa: E402
from agent_eval import (                                       # noqa: E402
    EXIT_BAD_INPUT, EXIT_FAIL, EXIT_OK, GRADER_CHECKS,
    cohen_kappa, fit_step_reliability, grader_audit, human_agreement,
    length_profile, load_human_labels, main, pass_at_k, pass_hat_k,
    poisson_binomial_cdf, reliability, score_all, score_trajectory,
    validate_args, validate_tasks, validate_trajectories,
)

DATA = os.path.join(ROOT, "data")
ARGS = ["--tools", os.path.join(DATA, "agent_tools.json"),
        "--tasks", os.path.join(DATA, "agent_tasks.jsonl"),
        "--trajectories", os.path.join(DATA, "agent_trajectories.jsonl")]
LABELS = os.path.join(DATA, "agent_human_labels.csv")


def planted(tags: list, kind: str) -> set:
    return {t.split(":", 1)[1] for t in tags if t.startswith(kind + ":")}


@pytest.fixture(scope="module")
def generated():
    return gen.generate()


@pytest.fixture(scope="module")
def tasks():
    return validate_tasks(gen.TASKS, gen.TOOLS)


@pytest.fixture(scope="module")
def scored(generated, tasks):
    rows, _ = generated
    df = score_all(tasks, gen.TOOLS, rows)
    df["planted"] = [r["planted"] for r in rows]
    return df


def test_shipped_files_match_the_generator(generated):
    rows, human = generated
    with open(os.path.join(DATA, "agent_trajectories.jsonl")) as f:
        shipped = [json.loads(line) for line in f]
    assert len(shipped) == len(rows) == 4 * len(gen.TASKS) * gen.TRIALS
    for s, r in zip(shipped, rows):
        assert s == {k: v for k, v in r.items() if k != "planted"}
    labels = pd.read_csv(LABELS)
    assert labels["human_success"].tolist() == [h["human_success"] for h in human]


# ---------------------------------------------------------------- estimators

def test_pass_at_k_and_pass_hat_k_by_hand():
    # 5 trials, 2 successes, k = 2: 3 of the 10 possible pairs are all-fail, 1 is all-pass
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)
    assert pass_hat_k(5, 2, 2) == pytest.approx(0.1)
    for n, c in ((6, 0), (6, 4), (6, 6)):
        assert pass_at_k(n, c, 1) == pytest.approx(c / n) == pytest.approx(pass_hat_k(n, c, 1))
    assert pass_at_k(6, 0, 3) == 0.0 and pass_hat_k(6, 0, 3) == 0.0
    assert pass_at_k(6, 6, 3) == 1.0 and pass_hat_k(6, 6, 3) == 1.0
    assert pass_at_k(6, 4, 3) == 1.0        # only 2 failures: any 3 draws include a success
    # Anthropic's example: 75% per trial gives pass^3 of about 42%; the estimator converges to p^k
    assert pass_hat_k(400, 300, 3) == pytest.approx(0.75 ** 3, abs=0.002)
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 4)
    with pytest.raises(ValueError):
        pass_hat_k(3, 1, 0)


def test_reliability_needs_k_trials_per_task(scored):
    rel = reliability(scored, [1, 6])
    assert rel.loc["agent_ref", "pass^6"] == 1.0
    assert rel.loc["agent_replay", "pass@6"] == 0.0
    with pytest.raises(ValueError, match="k=7"):
        reliability(scored, [7])


def test_kappa_by_hand():
    assert cohen_kappa(20, 5, 10, 15) == pytest.approx(0.4)      # the textbook 2x2 example
    assert cohen_kappa(10, 0, 0, 10) == 1.0
    assert cohen_kappa(10, 0, 0, 0) != cohen_kappa(10, 0, 0, 0)  # one class only: undefined


def test_step_reliability_fit_and_poisson_binomial_by_hand():
    # (2q + 2q^2) / 4 = 0.75  ->  q = (sqrt(7) - 1) / 2
    q = fit_step_reliability([1, 1, 2, 2], [1, 1, 1, 0])
    assert q == pytest.approx((7 ** 0.5 - 1) / 2, abs=1e-9)
    assert fit_step_reliability([1, 2], [0, 0]) == 0.0
    assert fit_step_reliability([1, 2], [1, 1]) == 1.0
    assert poisson_binomial_cdf([0.5] * 4, 1) == pytest.approx(5 / 16)
    assert poisson_binomial_cdf([1.0] * 3, 2) == pytest.approx(0.0)


# ---------------------------------------------------------------- one trajectory, by hand

FLIGHT = {"task_id": "t", "required_tools": ["search_flights", "book_flight"],
          "required_order": [["search_flights", "book_flight"]],
          "forbidden_tools": ["refund_payment"], "optional_tools": [],
          "max_steps": 6, "expected_final_contains": "confirmation"}
S = {"origin": "IAD", "destination": "SFO", "date": "2026-10-14"}
B = {"flight_id": "FL1234", "passengers": 1}


def call(tool, args, ok=True):
    return [{"type": "tool_call", "tool": tool, "args": dict(args)},
            {"type": "tool_result", "tool": tool, "ok": ok}]


def answer(text="Confirmation: CX12345"):
    return [{"type": "final", "content": text}]


def score(steps, task=FLIGHT, **kw):
    return score_trajectory(task, gen.TOOLS, steps, **kw)


@pytest.mark.parametrize("tool,args,expected", [
    ("book_flight", {"flight_id": "FL1234", "passengers": 2}, []),
    ("book_flight", {"flight_id": "FL1234"}, ["missing required argument 'passengers'"]),
    ("book_flight", {"flight_id": "FL1234", "passengers": "one"}, ["passengers: expected int, got str"]),
    ("book_flight", {"flight_id": "FL1234", "passengers": True}, ["passengers: expected int, got bool"]),
    ("book_flight", {"flight_id": "FL1234", "passengers": 0}, ["passengers: 0 below minimum 1"]),
    ("book_flight", {"flight_id": "FL1234", "passengers": 10}, ["passengers: 10 above maximum 9"]),
    ("search_flights", dict(S, origin="iad"), ["origin: 'iad' does not match ^[A-Z]{3}$"]),
    ("charge_card", {"amount": 412, "currency": "USD"}, []),      # an int is a valid float
    ("charge_card", {"amount": 412.5, "currency": "dollars"},
     ["currency: 'dollars' not one of ['USD', 'EUR']"]),
    ("get_weather", {"city": "Chicago", "unexpected_field": True},
     ["unexpected argument 'unexpected_field'"]),
    ("teleport", {}, ["unknown tool 'teleport'"]),
])
def test_argument_validation(tool, args, expected):
    assert validate_args(tool, args, gen.TOOLS) == expected


def test_a_clean_run_passes_the_reference():
    m = score(call("search_flights", S) + call("book_flight", B) + answer())
    assert m["reference_success"] and m["step_efficiency"] == 1.0
    assert m["tool_recall"] == m["tool_precision"] == m["arg_validity"] == 1.0


def test_order_needs_a_successful_prerequisite():
    backwards = score(call("book_flight", B) + call("search_flights", S) + answer())
    assert backwards["order_violations"] == ["search_flights->book_flight"]
    assert not backwards["reference_success"]
    failed_first = score(call("search_flights", S, ok=False) + call("book_flight", B) + answer())
    assert failed_first["order_violations"] == ["search_flights->book_flight"]
    retried = score(call("search_flights", S, ok=False) + call("search_flights", S)
                    + call("book_flight", B) + answer())
    assert retried["order_violations"] == [] and retried["reference_success"]


def test_a_retry_after_failure_is_not_redundant_but_a_repeat_after_success_is():
    retry = score(call("search_flights", S, ok=False) + call("search_flights", S)
                  + call("book_flight", B) + answer())
    assert retry["redundant_calls"] == 0
    assert retry["step_efficiency"] == pytest.approx(2 / 3)       # the retry still costs a step
    repeat = score(call("search_flights", S) + call("search_flights", S)
                   + call("book_flight", B) + answer())
    assert repeat["redundant_calls"] == 1 and repeat["reference_success"]


def test_loops_and_budget():
    three = call("search_flights", S, ok=False) * 3
    assert score(three + answer(""))["loop_detected"]
    assert not score(three[:4] + answer(""))["loop_detected"]
    assert score(three[:4] + answer(""), loop_threshold=2)["loop_detected"]
    tight = dict(FLIGHT, max_steps=2)
    assert score(three + answer(""), task=tight)["over_budget"]
    assert not score(call("search_flights", S) + call("book_flight", B) + answer(), task=tight)["over_budget"]


def test_empty_and_wrong_answers_fail_even_when_every_tool_worked():
    tools_ok = call("search_flights", S) + call("book_flight", B)
    empty = score(tools_ok + answer("   "))
    assert empty["final_empty"] and not empty["reference_success"]
    vague = score(tools_ok + answer("Done."))
    assert not vague["final_empty"] and not vague["final_matches"] and not vague["reference_success"]
    none = score(tools_ok)                                         # no final step at all
    assert none["final_empty"] and not none["reference_success"]


def test_confusion_is_a_sibling_called_instead_of_a_needed_tool():
    m = score(call("search_flights_legacy", {"route": "IAD-SFO"}) + call("book_flight", B) + answer())
    assert m["confusions"] == ["search_flights_legacy->search_flights"]
    assert m["missing_required"] == ["search_flights"] and not m["reference_success"]
    assert m["tool_precision"] == 0.5
    stray = score(call("get_user", {"user_id": "U04213"}) + call("search_flights", S)
                  + call("book_flight", B) + answer())
    assert stray["confusions"] == [] and stray["tool_precision"] == pytest.approx(2 / 3)
    charge = {"task_id": "c", "required_tools": ["charge_card"], "forbidden_tools": ["refund_payment"]}
    wrong_payment = score(call("refund_payment", {"payment_id": "P-1"}) + answer("ok"), task=charge)
    assert wrong_payment["confusions"] == [] and wrong_payment["forbidden_calls"] == 1


def test_task_and_trajectory_schema_errors_are_caught(tasks):
    with pytest.raises(ValueError, match="required AND forbidden"):
        validate_tasks([{"task_id": "x", "required_tools": ["refund_payment"],
                         "forbidden_tools": ["refund_payment"]}], gen.TOOLS)
    with pytest.raises(ValueError, match="not in the tool schema"):
        validate_tasks([{"task_id": "x", "required_tools": ["teleport"]}], gen.TOOLS)
    with pytest.raises(ValueError, match="required_order"):
        validate_tasks([{"task_id": "x", "required_tools": ["get_user"],
                         "required_order": [["get_user", "book_flight"]]}], gen.TOOLS)
    one = {"agent": "a", "task_id": "weather_01", "trial": 1, "steps": answer("forecast")}
    with pytest.raises(ValueError, match="every trajectory or on none"):
        validate_trajectories([dict(one, graded_success=True), dict(one, trial=2)], tasks)
    with pytest.raises(ValueError, match="duplicate"):
        validate_trajectories([one, dict(one)], tasks)


# ---------------------------------------------------------------- planted faults, recovered exactly

def test_every_planted_confusion_is_found_and_nothing_else(scored):
    assert scored["planted"].apply(lambda p: bool(planted(p, "confusion"))).sum() > 0
    for found, tags in zip(scored["confusions"], scored["planted"]):
        assert set(found) == planted(tags, "confusion")


def test_every_planted_bad_argument_is_found_and_nothing_else(scored):
    assert scored["planted"].apply(lambda p: bool(planted(p, "bad_args"))).sum() > 0
    for errors, tags in zip(scored["arg_errors"], scored["planted"]):
        assert {e.split(":")[0] for e in errors} == planted(tags, "bad_args")


def test_loops_failed_tools_and_empty_answers_match_the_plant(scored):
    for _, r in scored.iterrows():
        tags = r["planted"]
        assert r["loop_detected"] == bool(planted(tags, "loop"))
        assert set(r["unrecovered_errors"]) == planted(tags, "bad_args") | planted(tags, "loop")
        assert r["final_empty"] == ("empty_final" in tags)


def test_order_violations_come_only_from_broken_prerequisites(scored, tasks):
    for _, r in scored.iterrows():
        tags = r["planted"]
        broken = planted(tags, "bad_args") | {c.split("->")[1] for c in planted(tags, "confusion")}
        pairs = tasks[r["task_id"]]["required_order"]
        # the confused agent always goes on to call b; the others either run clean or stop
        expected = {f"{a}->{b}" for a, b in pairs if a in broken} if r["agent"] == "agent_confused" else set()
        assert set(r["order_violations"]) == expected


def test_replays_and_forbidden_calls_are_exactly_the_replay_agent(scored):
    is_replay = scored["planted"].apply(lambda p: "replay" in p)
    assert (scored["replayed"] == is_replay).all()
    assert ((scored["forbidden_calls"] > 0) == is_replay).all()


def test_the_reference_verdict_is_exactly_the_planted_truth(scored):
    assert (scored["reference_success"] == scored["planted"].apply(gen.is_clean)).all()


def test_the_control_agent_is_left_alone_retries_included(scored):
    ref = scored[scored["agent"] == "agent_ref"]
    retried = ref["planted"].apply(lambda p: any(t.startswith("retry:") for t in p))
    assert retried.sum() >= 3                               # so the test means something
    assert ref["reference_success"].all()
    for col in ("confusions", "arg_errors", "order_violations", "unrecovered_errors", "missing_required"):
        assert ref[col].apply(len).sum() == 0
    assert (ref["redundant_calls"] == 0).all() and (ref["forbidden_calls"] == 0).all()
    assert not ref[["loop_detected", "over_budget", "final_empty", "replayed"]].to_numpy().any()
    assert (ref.loc[retried, "step_efficiency"] < 1).all()  # a retry is not a fault, but it costs a step


def test_length_profile_flags_only_the_planted_bottleneck(scored):
    lp = length_profile(scored)
    assert set(lp.index[lp["bottleneck"]]) == {"agent_long_fail"}
    assert lp.loc["agent_long_fail", "q"] == 1.0
    # agent_confused also drops sharply from short to mid, but compounding explains it
    assert lp.loc["agent_confused", "short_to_mid_drop"] > 0.25
    assert abs(lp.loc["agent_confused", "mid_shortfall"]) < 0.1


# ---------------------------------------------------------------- the grader audit

def test_grader_audit_recovers_each_planted_bug(scored):
    audit = grader_audit(scored)
    checks, tags = audit["checks"], scored["planted"]
    # bug 1: every empty answer was passed, and only agent_long_fail gives them
    empties = tags.apply(lambda p: "empty_final" in p).sum()
    assert checks["passed_empty_answer"]["count"] == empties > 0
    assert set(checks["passed_empty_answer"]["by_agent"]) == {"agent_long_fail"}
    # bug 2: tool calls are never read, so the replayed transcript with its forbidden
    # refund passes on every task whose answer should be a confirmation (9 tasks x 6)
    assert checks["passed_forbidden_call"]["count"] == checks["passed_replayed_transcript"]["count"] == 54
    assert set(checks["passed_missing_tool"]["by_agent"]) == {"agent_confused", "agent_long_fail", "agent_replay"}
    # bug 3: correct answers phrased "confirmation number" are rejected, and nothing else is
    natural_valid = tags.apply(lambda p: gen.is_clean(p) and "natural_phrasing" in p)
    rejected = ~scored["graded_success"] & scored["reference_success"]
    assert natural_valid.sum() > 0 and (rejected == natural_valid).all()
    assert checks["rejected_valid"]["count"] == natural_valid.sum()


def test_every_false_pass_is_explained_by_a_named_check(scored):
    graded = scored["graded_success"]
    named = pd.Series(False, index=scored.index)
    for _, _, rule in GRADER_CHECKS:
        named |= graded & rule(scored)
    assert (named == (graded & ~scored["reference_success"])).all()


def test_grader_headline_numbers(scored):
    audit = grader_audit(scored)
    assert audit["kappa"] < 0.2                     # barely better than chance
    assert audit["false_pass_rate"] > 0.7
    by = audit["by_agent"]
    assert by.loc["agent_replay", "graded_pass"] == 0.75
    assert by.loc["agent_replay", "reference_pass"] == 0.0
    assert by.loc["agent_ref", "inflation"] < 0     # the grader under-reports the good agent


def test_human_review_surfaces_exactly_the_two_slips(scored, generated):
    _, human = generated
    out = human_agreement(scored, load_human_labels(LABELS))
    slips = {(h["agent"], h["task_id"], str(h["trial"])) for h in human if h["slip"]}
    found = {(d["agent"], d["task_id"], d["trial"]) for d in out["reference_disagreements"]}
    assert found == slips and len(slips) == gen.HUMAN_SLIPS
    assert out["reference"]["agreement"] == pytest.approx(58 / 60)
    assert out["grader"]["kappa"] < out["reference"]["kappa"]


# ---------------------------------------------------------------- CLI

def test_cli_writes_the_full_report(tmp_path):
    out = tmp_path / "eval.json"
    assert main(ARGS + ["--human-labels", LABELS, "--json", str(out), "--quiet"]) == EXIT_OK
    data = json.loads(out.read_text())
    assert {"agents", "faults", "reliability", "reliability_by_grader", "length_profile",
            "grader_audit", "human_agreement"} <= set(data)
    assert data["grader_audit"]["checks"]["passed_empty_answer"]["count"] > 0
    assert data["length_profile"]["agent_long_fail"]["bottleneck"] is True
    assert data["reliability"]["agent_ref"]["pass^3"] == 1.0


@pytest.mark.parametrize("gate", [["--fail-kappa", "0.6"], ["--fail-false-pass", "0.2"],
                                  ["--fail-forbidden"], ["--fail-pass-hat-k", "0.9"]])
def test_cli_gates_fail_on_the_planted_faults(gate):
    assert main(ARGS + ["--quiet"] + gate) == EXIT_FAIL


def test_cli_gates_pass_on_the_control_alone(tmp_path):
    ref_only = tmp_path / "ref.jsonl"
    with open(os.path.join(DATA, "agent_trajectories.jsonl")) as src, open(ref_only, "w") as dst:
        dst.writelines(line for line in src if '"agent": "agent_ref"' in line)
    args = ARGS[:4] + ["--trajectories", str(ref_only), "--quiet", "--k", "1", "3", "6"]
    assert main(args + ["--fail-forbidden", "--fail-pass-hat-k", "0.99"]) == EXIT_OK


def test_cli_bad_input_is_exit_2(tmp_path):
    assert main(ARGS + ["--quiet", "--k", "7"]) == EXIT_BAD_INPUT          # only 6 trials per task
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"agent": "a", "task_id": "nope", "trial": 1, "steps": []}) + "\n")
    assert main(ARGS[:4] + ["--trajectories", str(bad), "--quiet"]) == EXIT_BAD_INPUT
    assert main(ARGS[:4] + ["--trajectories", str(tmp_path / "missing.jsonl"), "--quiet"]) == EXIT_BAD_INPUT
    labels = tmp_path / "labels.csv"
    labels.write_text("agent,task_id,trial,human_success\nagent_ref,weather_01,1,maybe\n")
    assert main(ARGS + ["--quiet", "--human-labels", str(labels)]) == EXIT_BAD_INPUT
    orphan = tmp_path / "orphan.csv"
    orphan.write_text("agent,task_id,trial,human_success\nagent_ref,weather_01,99,1\n")
    assert main(ARGS + ["--quiet", "--human-labels", str(orphan)]) == EXIT_BAD_INPUT


def test_cli_exit_code_from_a_real_process():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "agent_eval.py")] + ARGS
                       + ["--quiet", "--fail-forbidden"], capture_output=True, text=True)
    assert r.returncode == EXIT_FAIL, r.stderr
