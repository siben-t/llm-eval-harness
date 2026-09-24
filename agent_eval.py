r"""Evaluate tool-using agents from their trajectories, and audit the grader
that scored them.

Most agent evals report one number: the share of runs a grader passed. That
number is only as good as the grader, and published audits keep finding
graders that pass empty answers, never look at the actions taken, or can be
satisfied by a transcript that ignores the task. This module answers two
questions from trajectories you already have:

  1. How good is the agent?  Process, not just outcome: tool recall and
     precision, argument validity against the tool schema, dependency order,
     forbidden calls, loops, redundant calls, similar-tool confusion, and
     errors it never recovered from. Reliability is reported as pass@k AND
     pass^k with unbiased estimators over n trials. pass@k is "succeeds at
     least once in k tries"; pass^k is "succeeds on all k", which is the one
     a user of a deployed agent actually feels.

  2. Can the grader be trusted?  Every trajectory is re-scored by a strict
     reference: every required tool succeeded, in dependency order, nothing
     forbidden was called, and the final answer is non-empty and says what
     the task needs. The grader's verdicts are cross-tabulated against the
     reference and each kind of disagreement is named and counted: empty
     answers passed, missing tools passed, forbidden calls passed, failed
     tools passed, the same transcript replayed across different tasks
     passed, valid answers rejected. With a human-labelled sample, both the
     grader and the reference are checked against people, and every
     reference-vs-human disagreement is listed for adjudication instead of
     being silently resolved.

Plus a length profile. Success falls as tasks get longer, and much of that
is plain compounding: an agent that gets each step right with probability q
finishes an L-step task with probability about q^L. The profile fits q on
short tasks, predicts mid-length success from it, and flags only the
shortfall that compounding does not explain. That separates the
short-to-mid bottleneck TRAJECT-Bench describes from ordinary per-step error.

Nothing here calls a model.

Sources
  He et al., TRAJECT-Bench, ICLR 2026 (arXiv:2510.04550): tool selection,
    argument correctness, dependency/order; similar-tool confusion and
    parameter-blind selection; the short-to-mid length bottleneck
  Anthropic, "Demystifying evals for AI agents", 9 Jan 2026: pass@k, pass^k
  Zhu et al., "Establishing Best Practices for Building Rigorous Agentic
    Benchmarks" (arXiv:2507.02825): tau-bench counted empty responses as
    successful

Inputs
  --tools         JSON   {tool: {"family": str, "args": {name: spec}}}
                         spec: type (str|int|float|bool|list|dict), required,
                               pattern (regex, full match), enum, min, max
  --tasks         JSONL  task_id, required_tools, required_order ([[a, b], ...]:
                         a successful call to a before the first call to b),
                         forbidden_tools, optional_tools, max_steps,
                         expected_final_contains
  --trajectories  JSONL  agent, task_id, trial, steps, graded_success (optional)
                         steps: {"type": "tool_call", "tool": ..., "args": {...}}
                                {"type": "tool_result", "tool": ..., "ok": bool}
                                {"type": "final", "content": "..."}
  --human-labels  CSV    agent, task_id, trial, human_success (0|1), optional

CLI
  python agent_eval.py --tools data/agent_tools.json --tasks data/agent_tasks.jsonl
                       --trajectories data/agent_trajectories.jsonl
                       --human-labels data/agent_human_labels.csv --k 1 3
                       --json agent_eval.json --fail-kappa 0.6 --fail-forbidden
Exit 0 if every gate passes; 1 if one fails; 2 on bad input.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

EXIT_OK, EXIT_FAIL, EXIT_BAD_INPUT = 0, 1, 2
ARG_TYPES = {"str": (str,), "int": (int,), "float": (int, float), "bool": (bool,),
             "list": (list,), "dict": (dict,)}
STEP_TYPES = ("tool_call", "tool_result", "final")
NAN = float("nan")


# --------------------------------------------------------------------------
# loading and validation: bad input is exit 2, never a silent skip
# --------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        raise ValueError(f"no such file: {path}")
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {n}: not JSON ({exc.msg})") from None
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def load_tools(path: str) -> dict:
    if not os.path.exists(path):
        raise ValueError(f"no such file: {path}")
    with open(path) as f:
        try:
            tools = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not JSON ({exc.msg})") from None
    return validate_tools(tools)


def validate_tools(tools: dict) -> dict:
    if not isinstance(tools, dict) or not tools:
        raise ValueError("tool schema must be a non-empty JSON object")
    for name, spec in tools.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("args") or {}, dict):
            raise ValueError(f"tool {name!r}: expected an object with an 'args' object")
        for arg, rule in (spec.get("args") or {}).items():
            if not isinstance(rule, dict):
                raise ValueError(f"tool {name!r} argument {arg!r}: expected an object")
            if rule.get("type") is not None and rule["type"] not in ARG_TYPES:
                raise ValueError(f"tool {name!r} argument {arg!r}: unknown type {rule['type']!r}")
            if "pattern" in rule:
                try:
                    re.compile(rule["pattern"])
                except re.error as exc:
                    raise ValueError(f"tool {name!r} argument {arg!r}: bad pattern ({exc})") from None
    return tools


def validate_tasks(rows: list, tools: dict) -> dict:
    tasks = {}
    for n, t in enumerate(rows, 1):
        if not isinstance(t, dict):
            raise ValueError(f"task row {n}: expected an object")
        tid = t.get("task_id")
        if not tid:
            raise ValueError(f"task row {n}: missing task_id")
        if tid in tasks:
            raise ValueError(f"duplicate task_id {tid!r}")
        if not isinstance(t.get("required_tools"), list):
            raise ValueError(f"task {tid}: required_tools must be a list")
        task = {
            "task_id": tid,
            "required_tools": list(dict.fromkeys(t["required_tools"])),
            "required_order": [list(p) for p in t.get("required_order") or []],
            "forbidden_tools": list(t.get("forbidden_tools") or []),
            "optional_tools": list(t.get("optional_tools") or []),
            "max_steps": t.get("max_steps"),
            "expected_final_contains": t.get("expected_final_contains"),
        }
        for field in ("required_tools", "forbidden_tools", "optional_tools"):
            unknown = [x for x in task[field] if x not in tools]
            if unknown:
                raise ValueError(f"task {tid}: {field} not in the tool schema: {unknown}")
        for pair in task["required_order"]:
            if len(pair) != 2 or not set(pair) <= set(task["required_tools"]):
                raise ValueError(f"task {tid}: required_order {pair} must name two required tools")
        if task["max_steps"] is not None and (isinstance(task["max_steps"], bool)
                                              or not isinstance(task["max_steps"], int)):
            raise ValueError(f"task {tid}: max_steps must be an integer")
        if task["expected_final_contains"] is not None and not isinstance(task["expected_final_contains"], str):
            raise ValueError(f"task {tid}: expected_final_contains must be a string")
        clash = set(task["required_tools"]) & set(task["forbidden_tools"])
        if clash:
            raise ValueError(f"task {tid}: required AND forbidden: {sorted(clash)}")
        tasks[tid] = task
    return tasks


def validate_trajectories(rows: list, tasks: dict) -> bool:
    """Checks every row; returns whether grader verdicts are present."""
    seen, graded = set(), None
    for n, r in enumerate(rows, 1):
        if not isinstance(r, dict):
            raise ValueError(f"trajectory row {n}: expected an object")
        for key in ("agent", "task_id", "trial", "steps"):
            if key not in r:
                raise ValueError(f"trajectory row {n}: missing {key!r}")
        if r["task_id"] not in tasks:
            raise ValueError(f"trajectory row {n}: unknown task_id {r['task_id']!r}")
        ident = (str(r["agent"]), str(r["task_id"]), str(r["trial"]))
        if ident in seen:
            raise ValueError(f"trajectory row {n}: duplicate agent/task/trial {ident}")
        seen.add(ident)
        if not isinstance(r["steps"], list):
            raise ValueError(f"trajectory row {n}: steps must be a list")
        for s in r["steps"]:
            if not isinstance(s, dict) or s.get("type") not in STEP_TYPES:
                raise ValueError(f"trajectory row {n}: bad step {s!r}")
            if s["type"] == "tool_call" and not isinstance(s.get("tool"), str):
                raise ValueError(f"trajectory row {n}: tool_call without a tool name")
        has = "graded_success" in r
        if graded is None:
            graded = has
        elif has != graded:
            raise ValueError("graded_success must be on every trajectory or on none")
        if has and r["graded_success"] not in (True, False, 0, 1):
            raise ValueError(f"trajectory row {n}: graded_success must be true/false")
    return bool(graded)


# --------------------------------------------------------------------------
# one trajectory
# --------------------------------------------------------------------------

def validate_args(tool: str, args, tools: dict) -> list:
    """Problems with one call's arguments against the schema; [] means valid."""
    if tool not in tools:
        return [f"unknown tool {tool!r}"]
    if not isinstance(args, dict):
        return ["arguments are not an object"]
    spec = tools[tool].get("args") or {}
    problems = []
    for name, rule in spec.items():
        if name not in args:
            if rule.get("required"):
                problems.append(f"missing required argument {name!r}")
            continue
        value = args[name]
        typ = rule.get("type")
        if typ is not None:
            numeric_bool = typ in ("int", "float") and isinstance(value, bool)
            if numeric_bool or not isinstance(value, ARG_TYPES[typ]):
                problems.append(f"{name}: expected {typ}, got {type(value).__name__}")
                continue
        if "enum" in rule and value not in rule["enum"]:
            problems.append(f"{name}: {value!r} not one of {rule['enum']}")
        if "pattern" in rule and isinstance(value, str) and not re.fullmatch(rule["pattern"], value):
            problems.append(f"{name}: {value!r} does not match {rule['pattern']}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "min" in rule and value < rule["min"]:
                problems.append(f"{name}: {value} below minimum {rule['min']}")
            if "max" in rule and value > rule["max"]:
                problems.append(f"{name}: {value} above maximum {rule['max']}")
    for name in args:
        if name not in spec:
            problems.append(f"unexpected argument {name!r}")
    return problems


def tool_calls(steps: list) -> list:
    """Calls in order, each paired with the first result for that tool that follows it."""
    calls = []
    for s in steps:
        if s["type"] == "tool_call":
            args = s.get("args")
            calls.append({"tool": s["tool"], "args": {} if args is None else args, "ok": None})
        elif s["type"] == "tool_result":
            for c in reversed(calls):
                if c["tool"] == s.get("tool") and c["ok"] is None:
                    c["ok"] = bool(s.get("ok"))
                    break
    return calls


def final_answer(steps: list) -> str:
    finals = [s for s in steps if s["type"] == "final"]
    if not finals:
        return ""
    content = finals[-1].get("content")
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, sort_keys=True)


def _call_key(call: dict) -> tuple:
    return call["tool"], json.dumps(call["args"], sort_keys=True, default=str)


def transcript_signature(steps: list) -> str:
    """Everything the agent itself emitted: each call with its arguments, then the answer."""
    calls = [[c["tool"], c["args"]] for c in tool_calls(steps)]
    return json.dumps([calls, final_answer(steps).strip()], sort_keys=True, default=str)


def score_trajectory(task: dict, tools: dict, steps: list, loop_threshold: int = 3) -> dict:
    """Process metrics for one trajectory, and the strict reference verdict."""
    calls = tool_calls(steps)
    required = list(dict.fromkeys(task.get("required_tools", [])))
    optional = set(task.get("optional_tools", []))
    forbidden = set(task.get("forbidden_tools", []))
    relevant = set(required) | optional
    called = [c["tool"] for c in calls]
    distinct = set(called)
    succeeded = {c["tool"] for c in calls if c["ok"]}

    missing = [t for t in required if t not in distinct]
    unrecovered = [t for t in required if t in distinct and t not in succeeded]

    arg_errors, valid_calls = [], 0
    for c in calls:
        problems = validate_args(c["tool"], c["args"], tools)
        arg_errors.extend(f"{c['tool']}: {p}" for p in problems)
        valid_calls += not problems

    order_violations = []
    for a, b in task.get("required_order", []):
        first_b = next((i for i, c in enumerate(calls) if c["tool"] == b), None)
        if first_b is not None and not any(c["tool"] == a and c["ok"] for c in calls[:first_b]):
            order_violations.append(f"{a}->{b}")

    counts = Counter(_call_key(c) for c in calls)
    redundant, done = 0, set()
    for c in calls:
        key = _call_key(c)
        redundant += key in done            # the same call again after it already worked
        if c["ok"]:
            done.add(key)

    confusions = []                         # a sibling tool called instead of a needed one
    for t in called:
        if t in relevant or t in forbidden or t not in tools:
            continue
        family = tools[t].get("family")
        for r in missing:
            pair = f"{t}->{r}"
            if family and tools[r].get("family") == family and pair not in confusions:
                confusions.append(pair)

    max_steps = task.get("max_steps")
    # one successful call per required tool is the work; everything else (retries,
    # wrong tools, repeats, forbidden calls) is overhead
    useful = sum(t in succeeded for t in required)
    efficiency = useful / len(calls) if calls else (1.0 if not required else NAN)

    final = final_answer(steps)
    final_empty = final.strip() == ""
    want = task.get("expected_final_contains")
    final_matches = not final_empty and (not want or want.lower() in final.lower())
    forbidden_calls = sum(t in forbidden for t in called)

    return {
        "n_calls": len(calls),
        "tool_recall": len(set(required) & distinct) / len(required) if required else 1.0,
        "tool_precision": len(distinct & relevant) / len(distinct) if distinct else 1.0,
        "arg_validity": valid_calls / len(calls) if calls else 1.0,
        "arg_errors": arg_errors,
        "order_violations": order_violations,
        "forbidden_calls": forbidden_calls,
        "redundant_calls": redundant,
        "loop_detected": bool(counts) and max(counts.values()) >= loop_threshold,
        "over_budget": max_steps is not None and len(calls) > max_steps,
        "step_efficiency": efficiency,
        "confusions": confusions,
        "missing_required": missing,
        "unrecovered_errors": unrecovered,
        "final_empty": final_empty,
        "final_matches": final_matches,
        "reference_success": (not missing and not unrecovered and not order_violations
                              and forbidden_calls == 0 and final_matches),
    }


def reference_reasons(m) -> list:
    """Why the reference failed a trajectory, in words; [] if it passed."""
    why = []
    if len(m["missing_required"]):
        why.append("never called " + ", ".join(m["missing_required"]))
    if len(m["unrecovered_errors"]):
        why.append("no successful call to " + ", ".join(m["unrecovered_errors"]))
    if len(m["order_violations"]):
        why.append("order " + ", ".join(m["order_violations"]))
    if m["forbidden_calls"]:
        why.append(f"{m['forbidden_calls']} forbidden call(s)")
    if m["final_empty"]:
        why.append("empty answer")
    elif not m["final_matches"]:
        why.append("answer missing the expected content")
    return why


# --------------------------------------------------------------------------
# all trajectories
# --------------------------------------------------------------------------

def bucket_of(n_required: int) -> str:
    return "short" if n_required <= 2 else "mid" if n_required <= 4 else "long"


def score_all(tasks: dict, tools: dict, rows: list, loop_threshold: int = 3) -> pd.DataFrame:
    out = []
    for r in rows:
        task = tasks[r["task_id"]]
        rec = {"agent": str(r["agent"]), "task_id": str(r["task_id"]), "trial": str(r["trial"]),
               "n_required": len(task["required_tools"]),
               "bucket": bucket_of(len(task["required_tools"]))}
        rec.update(score_trajectory(task, tools, r["steps"], loop_threshold))
        rec["signature"] = transcript_signature(r["steps"])
        if "graded_success" in r:
            rec["graded_success"] = bool(r["graded_success"])
        out.append(rec)
    df = pd.DataFrame(out)
    df["replayed"] = find_replays(df)
    return df


def find_replays(df: pd.DataFrame) -> list:
    """True where an agent produced the identical transcript (every call with its
    arguments, and the same non-empty answer) on two or more different tasks.
    An empty answer is its own finding, so it is not counted as a replay."""
    tasks_by_sig = defaultdict(set)
    for agent, tid, sig, empty in zip(df["agent"], df["task_id"], df["signature"], df["final_empty"]):
        if not empty:
            tasks_by_sig[(agent, sig)].add(tid)
    return [not empty and len(tasks_by_sig[(agent, sig)]) >= 2
            for agent, sig, empty in zip(df["agent"], df["signature"], df["final_empty"])]


def _flags(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "agent": df["agent"],
        "order_violation": df["order_violations"].apply(len) > 0,
        "forbidden": df["forbidden_calls"] > 0,
        "arg_error": df["arg_errors"].apply(len) > 0,
        "confusion": df["confusions"].apply(len) > 0,
        "unrecovered": df["unrecovered_errors"].apply(len) > 0,
        "missing_tool": df["missing_required"].apply(len) > 0,
        "loop": df["loop_detected"],
        "redundant": df["redundant_calls"] > 0,
        "over_budget": df["over_budget"],
        "empty_answer": df["final_empty"],
        "replayed": df["replayed"],
    })


def agent_summary(df: pd.DataFrame) -> tuple:
    """(rates, fault counts) per agent. Counts are trajectories affected."""
    g = df.groupby("agent")
    rates = pd.DataFrame({
        "n": g.size(),
        "success": g["reference_success"].mean(),
        "tool_recall": g["tool_recall"].mean(),
        "tool_precision": g["tool_precision"].mean(),
        "arg_validity": g["arg_validity"].mean(),
        "step_efficiency": g["step_efficiency"].mean(),
    })
    if "graded_success" in df.columns:
        rates.insert(2, "graded_pass", g["graded_success"].mean())
    faults = _flags(df).groupby("agent").sum()
    return rates, faults


# --------------------------------------------------------------------------
# reliability: pass@k and pass^k, unbiased over n trials
# --------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k trials drawn without replacement from n succeeds),
    given c of the n succeeded: 1 - C(n-c, k) / C(n, k)."""
    if not 1 <= k <= n or not 0 <= c <= n:
        raise ValueError(f"need 1 <= k <= n and 0 <= c <= n (n={n}, c={c}, k={k})")
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """P(all k trials drawn without replacement from n succeed): C(c, k) / C(n, k)."""
    if not 1 <= k <= n or not 0 <= c <= n:
        raise ValueError(f"need 1 <= k <= n and 0 <= c <= n (n={n}, c={c}, k={k})")
    return math.comb(c, k) / math.comb(n, k)


def reliability(df: pd.DataFrame, ks: list, column: str = "reference_success") -> pd.DataFrame:
    """Per agent: the mean over tasks of pass@k and pass^k for each k."""
    out = []
    for agent, g in df.groupby("agent"):
        per_task = g.groupby("task_id")[column].agg(["size", "sum"])
        rec = {"agent": agent, "tasks": len(per_task), "min_trials": int(per_task["size"].min())}
        for k in ks:
            if k > rec["min_trials"]:
                short = per_task.index[per_task["size"] < k][0]
                raise ValueError(f"k={k} needs at least {k} trials per task; {agent} has "
                                 f"{int(per_task.loc[short, 'size'])} on {short}")
            rec[f"pass@{k}"] = sum(pass_at_k(int(n), int(c), k) for n, c in per_task.values) / len(per_task)
            rec[f"pass^{k}"] = sum(pass_hat_k(int(n), int(c), k) for n, c in per_task.values) / len(per_task)
        out.append(rec)
    return pd.DataFrame(out).set_index("agent")


# --------------------------------------------------------------------------
# length profile: bottleneck beyond compounding
# --------------------------------------------------------------------------

def fit_step_reliability(lengths: list, successes: list) -> float:
    """The per-step success q for which mean(q ** L) equals the observed success rate."""
    lengths = [int(x) for x in lengths]
    if not lengths:
        return NAN
    target = sum(bool(s) for s in successes) / len(lengths)
    if target <= 0:
        return 0.0
    if target >= 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(mid ** L for L in lengths) / len(lengths) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def poisson_binomial_cdf(ps: list, x: int) -> float:
    """P(X <= x), X the number of successes among independent trials with probabilities ps."""
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for j, v in enumerate(dist):
            nxt[j] += v * (1 - p)
            nxt[j + 1] += v * p
        dist = nxt
    return min(1.0, sum(dist[: x + 1]))


def length_profile(df: pd.DataFrame, threshold: float = 0.25, alpha: float = 0.05) -> pd.DataFrame:
    """Success by length bucket (short <= 2 required tools, mid 3-4, long >= 5), the
    per-step reliability q fitted on short tasks, the mid/long success that q predicts,
    and a bottleneck flag: mid success falls short of the prediction by more than
    `threshold` AND the shortfall is significant (one-sided, Poisson-binomial)."""
    out = []
    for agent, g in df.groupby("agent"):
        rec = {"agent": agent}
        for name in ("short", "mid", "long"):
            b = g[g["bucket"] == name]
            rec[name] = b["reference_success"].mean() if len(b) else NAN
            rec[f"n_{name}"] = len(b)
        fit_on = g[(g["bucket"] == "short") & (g["n_required"] >= 1)]
        q = fit_step_reliability(fit_on["n_required"], fit_on["reference_success"])
        rec["q"] = q
        for name in ("mid", "long"):
            b = g[g["bucket"] == name]
            if len(b) and q == q:
                ps = [q ** int(L) for L in b["n_required"]]
                rec[f"{name}_predicted"] = sum(ps) / len(ps)
                rec[f"{name}_shortfall"] = rec[f"{name}_predicted"] - rec[name]
                rec[f"{name}_p"] = poisson_binomial_cdf(ps, int(b["reference_success"].sum()))
            else:
                rec[f"{name}_predicted"] = rec[f"{name}_shortfall"] = rec[f"{name}_p"] = NAN
        rec["short_to_mid_drop"] = rec["short"] - rec["mid"]
        rec["bottleneck"] = bool(rec["mid_shortfall"] > threshold and rec["mid_p"] < alpha)
        out.append(rec)
    return pd.DataFrame(out).set_index("agent")


# --------------------------------------------------------------------------
# grader audit
# --------------------------------------------------------------------------

def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's kappa for two binary raters from a 2x2 table (rater A = rows)."""
    n = tp + fp + fn + tn
    if n == 0:
        return NAN
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return NAN if pe == 1 else (po - pe) / (1 - pe)


def _agreement(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.astype(bool), b.astype(bool)
    tp, fp = int((a & b).sum()), int((a & ~b).sum())
    fn, tn = int((~a & b).sum()), int((~a & ~b).sum())
    n = tp + fp + fn + tn
    return {"n": n, "agreement": (tp + tn) / n if n else NAN, "kappa": cohen_kappa(tp, fp, fn, tn),
            "a_pass_b_pass": tp, "a_pass_b_fail": fp, "a_fail_b_pass": fn, "a_fail_b_fail": tn}


GRADER_CHECKS = (
    ("passed_empty_answer", "passed with an empty final answer",
     lambda d: d["final_empty"]),
    ("passed_missing_tool", "passed without calling a required tool",
     lambda d: d["missing_required"].apply(len) > 0),
    ("passed_failed_tool", "passed although a required tool never succeeded",
     lambda d: d["unrecovered_errors"].apply(len) > 0),
    ("passed_order_violation", "passed with a step run before its prerequisite succeeded",
     lambda d: d["order_violations"].apply(len) > 0),
    ("passed_forbidden_call", "passed although a forbidden tool was called",
     lambda d: d["forbidden_calls"] > 0),
    ("passed_wrong_answer", "passed with an answer missing the expected content",
     lambda d: ~d["final_empty"] & ~d["final_matches"]),
    ("passed_replayed_transcript", "passed a transcript the agent also produced for other tasks",
     lambda d: d["replayed"]),
)


def _ids(mask: pd.Series, df: pd.DataFrame, limit: int = 3) -> list:
    hit = df[mask]
    return [f"{a}/{t}/{r}" for a, t, r in zip(hit["agent"], hit["task_id"], hit["trial"])][:limit]


def grader_audit(df: pd.DataFrame) -> dict:
    """Grader verdicts against the strict reference: agreement, kappa, false-pass
    and false-fail rates, and each kind of false pass named and counted."""
    graded, ref = df["graded_success"].astype(bool), df["reference_success"].astype(bool)
    agree = _agreement(graded, ref)
    fp, tn = agree["a_pass_b_fail"], agree["a_fail_b_fail"]
    fn, tp = agree["a_fail_b_pass"], agree["a_pass_b_pass"]
    checks = {}
    for name, description, rule in GRADER_CHECKS:
        mask = graded & rule(df)
        checks[name] = {"description": description, "count": int(mask.sum()),
                        "by_agent": {a: int(v) for a, v in mask.groupby(df["agent"]).sum().items() if v},
                        "examples": _ids(mask, df)}
    rejected = ~graded & ref
    checks["rejected_valid"] = {"description": "failed although the reference passes it",
                                "count": int(rejected.sum()),
                                "by_agent": {a: int(v) for a, v in rejected.groupby(df["agent"]).sum().items() if v},
                                "examples": _ids(rejected, df)}
    by_agent = pd.DataFrame({"graded_pass": graded.groupby(df["agent"]).mean(),
                             "reference_pass": ref.groupby(df["agent"]).mean()})
    by_agent["inflation"] = by_agent["graded_pass"] - by_agent["reference_pass"]
    return {"n": agree["n"], "agreement": agree["agreement"], "kappa": agree["kappa"],
            "false_pass_rate": fp / (fp + tn) if fp + tn else NAN,
            "false_fail_rate": fn / (fn + tp) if fn + tp else NAN,
            "graded_pass_reference_pass": tp, "graded_pass_reference_fail": fp,
            "graded_fail_reference_pass": fn, "graded_fail_reference_fail": tn,
            "checks": checks, "by_agent": by_agent}


def load_human_labels(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise ValueError(f"no such file: {path}")
    lab = pd.read_csv(path, dtype=str)
    need = {"agent", "task_id", "trial", "human_success"}
    if not need <= set(lab.columns):
        raise ValueError(f"{path}: needs columns {sorted(need)}")
    lab["human_success"] = lab["human_success"].str.strip().str.lower()
    bad = ~lab["human_success"].isin(["0", "1", "true", "false"])
    if bad.any():
        raise ValueError(f"{path}: human_success must be 0/1 (row {int(bad.idxmax()) + 2})")
    lab["human_success"] = lab["human_success"].isin(["1", "true"])
    return lab


def human_agreement(df: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """The reference and (if present) the grader, each against a human-labelled sample.
    Reference-vs-human disagreements are listed for someone to adjudicate."""
    merged = labels.merge(df, on=["agent", "task_id", "trial"], how="left", indicator=True)
    orphans = merged[merged["_merge"] != "both"]
    if len(orphans):
        r = orphans.iloc[0]
        raise ValueError(f"{len(orphans)} human label(s) match no trajectory, e.g. "
                         f"{r['agent']}/{r['task_id']}/{r['trial']}")
    out = {"n": len(merged),
           "reference": _agreement(merged["reference_success"], merged["human_success"])}
    if "graded_success" in merged.columns:
        out["grader"] = _agreement(merged["graded_success"], merged["human_success"])
    diff = merged[merged["reference_success"].astype(bool) != merged["human_success"]]
    out["reference_disagreements"] = [
        {"agent": r["agent"], "task_id": r["task_id"], "trial": r["trial"],
         "reference": bool(r["reference_success"]), "human": bool(r["human_success"]),
         "reference_reasons": reference_reasons(r)}
        for _, r in diff.iterrows()]
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _json_safe(obj):
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict(orient="index"))
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agent-eval", description=__doc__.split("\n\n")[0])
    p.add_argument("--tools", required=True, metavar="JSON")
    p.add_argument("--tasks", required=True, metavar="JSONL")
    p.add_argument("--trajectories", required=True, metavar="JSONL")
    p.add_argument("--human-labels", metavar="CSV")
    p.add_argument("--k", type=int, nargs="+", default=[1, 3],
                   help="k values for pass@k and pass^k (each needs k trials per task)")
    p.add_argument("--loop-threshold", type=int, default=3,
                   help="the same call with the same arguments this many times is a loop")
    p.add_argument("--bottleneck-threshold", type=float, default=0.25,
                   help="mid-length shortfall beyond compounding that counts as a bottleneck")
    p.add_argument("--fail-kappa", type=float, metavar="K",
                   help="exit 1 if the grader's kappa against the reference is below K")
    p.add_argument("--fail-false-pass", type=float, metavar="RATE",
                   help="exit 1 if the grader passes more than RATE of the trajectories the reference fails")
    p.add_argument("--fail-pass-hat-k", type=float, metavar="P",
                   help="exit 1 if any agent's pass^k at the largest k is below P")
    p.add_argument("--fail-forbidden", action="store_true",
                   help="exit 1 if any trajectory calls a forbidden tool")
    p.add_argument("--json", metavar="PATH")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    say = (lambda *a, **k: None) if args.quiet else print

    try:
        if min(args.k) < 1:
            raise ValueError("--k values must be >= 1")
        ks = sorted(set(args.k))
        tools = load_tools(args.tools)
        tasks = validate_tasks(load_jsonl(args.tasks), tools)
        rows = load_jsonl(args.trajectories)
        graded = validate_trajectories(rows, tasks)
        df = score_all(tasks, tools, rows, args.loop_threshold)
        rel = reliability(df, ks)
        rel_graded = reliability(df, ks, "graded_success") if graded else None
        labels = load_human_labels(args.human_labels) if args.human_labels else None
        humans = human_agreement(df, labels) if labels is not None else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    rates, faults = agent_summary(df)
    lp = length_profile(df, threshold=args.bottleneck_threshold)
    results = {"n_trajectories": len(df), "n_tasks": df["task_id"].nunique(),
               "agents": rates, "faults": faults, "reliability": rel, "length_profile": lp}

    say("=" * 70)
    say(f"AGENT EVALUATION   {df['agent'].nunique()} agents, {results['n_tasks']} tasks, "
        f"{len(df)} trajectories")
    say("=" * 70)
    say("\nOutcome and process (means per trajectory; success = strict reference)")
    say(rates.round(3).to_string())
    say("\nFaults (trajectories affected)")
    say(faults.to_string())
    hide = ["tasks", "min_trials"] + (["pass^1"] if 1 in ks else [])   # pass^1 == pass@1
    say("\nReliability (reference; unbiased over the trials per task)")
    say(rel.drop(columns=hide).round(3).to_string())
    if rel_graded is not None:
        results["reliability_by_grader"] = rel_graded
        say("\n  ...and as the grader under audit would report it")
        say(rel_graded.drop(columns=hide).round(3).to_string())
    say("\nLength profile (success by required-tool count; q = per-step reliability "
        "fitted on short tasks)")
    say(lp[["short", "mid", "long", "short_to_mid_drop", "q", "mid_predicted",
            "mid_shortfall", "mid_p", "bottleneck"]].round(3).to_string())

    if graded:
        audit = grader_audit(df)
        results["grader_audit"] = audit
        say("\n" + "=" * 70)
        say("GRADER AUDIT   graded_success against the strict reference")
        say("=" * 70)
        say(f"agreement {audit['agreement']:.3f}   kappa {audit['kappa']:.3f}   "
            f"false-pass rate {audit['false_pass_rate']:.3f}   "
            f"false-fail rate {audit['false_fail_rate']:.3f}")
        say(f"graded pass & reference fail: {audit['graded_pass_reference_fail']}   "
            f"graded fail & reference pass: {audit['graded_fail_reference_pass']}")
        for name, c in audit["checks"].items():
            if c["count"]:
                agents = ", ".join(f"{a} {v}" for a, v in c["by_agent"].items())
                say(f"  {name:<28}{c['count']:>5}   {agents}")
        say("\nPass rate by agent: grader vs reference")
        say(audit["by_agent"].round(3).to_string())

    if humans is not None:
        results["human_agreement"] = humans
        say(f"\nHuman sample (n={humans['n']})")
        if "grader" in humans:
            h = humans["grader"]
            say(f"  grader    agrees {h['agreement']:.3f}   kappa {h['kappa']:.3f}")
        h = humans["reference"]
        say(f"  reference agrees {h['agreement']:.3f}   kappa {h['kappa']:.3f}")
        if humans["reference_disagreements"]:
            say("  reference vs human, for adjudication:")
            for d in humans["reference_disagreements"]:
                why = "; ".join(d["reference_reasons"]) or "reference found nothing wrong"
                say(f"    {d['agent']}/{d['task_id']}/{d['trial']}: reference "
                    f"{'pass' if d['reference'] else 'fail'}, human "
                    f"{'pass' if d['human'] else 'fail'} ({why})")

    failures = []
    if args.fail_kappa is not None and graded:
        if not audit["kappa"] >= args.fail_kappa:
            failures.append(f"grader kappa {audit['kappa']:.3f} < {args.fail_kappa}")
    if args.fail_false_pass is not None and graded:
        if audit["false_pass_rate"] > args.fail_false_pass:
            failures.append(f"grader false-pass rate {audit['false_pass_rate']:.3f} > "
                            f"{args.fail_false_pass}")
    if args.fail_pass_hat_k is not None:
        col = f"pass^{ks[-1]}"
        for agent, v in rel[col].items():
            if v < args.fail_pass_hat_k:
                failures.append(f"{agent}: {col} {v:.3f} < {args.fail_pass_hat_k}")
    if args.fail_forbidden:
        for agent, v in faults["forbidden"].items():
            if v:
                failures.append(f"{agent}: {int(v)} trajectories call a forbidden tool")
    gates = [args.fail_kappa is not None and graded, args.fail_false_pass is not None and graded,
             args.fail_pass_hat_k is not None, args.fail_forbidden]
    status = EXIT_OK
    if any(gates):
        results["gates"] = {"failures": failures, "passed": not failures}
        status = EXIT_FAIL if failures else EXIT_OK
        say(f"\n[{'FAIL' if failures else 'PASS'}] gates")
        for f in failures:
            say(f"       {f}")
    if (args.fail_kappa is not None or args.fail_false_pass is not None) and not graded:
        say("\nnote: grader gates skipped -- the trajectories carry no graded_success")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(_json_safe(results), f, indent=2)
        say(f"\nWrote {args.json}")
    return status


if __name__ == "__main__":
    sys.exit(main())
