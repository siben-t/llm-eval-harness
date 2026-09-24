"""Synthetic agent trajectories with PLANTED faults, for agent_eval.py.

Four agents run twelve tool-use tasks, six trials each: 288 trajectories.
Every trajectory carries the verdict of a deliberately buggy grader, so the
audit has real grader failures to find. Every fault is recorded as it is
planted, so tests/test_agent_eval.py can assert that the evaluator finds
exactly the planted faults, no more and no fewer, and leaves the clean agent
alone.

Agents
  agent_ref        the control. Now and then a tool times out and it retries,
                   which must NOT be flagged
  agent_confused   similar-tool confusion (the deprecated flight search, the
                   wrong user lookup) and parameter-blind arguments (lowercase
                   airport codes, dates in the wrong format, a string where
                   the schema wants an integer): TRAJECT-Bench's two named
                   failure modes
  agent_long_fail  fine on short tasks, collapses on mid and long ones: it
                   loops on a failing call, then either returns nothing or
                   claims success anyway
  agent_replay     a degenerate policy that emits the same canned transcript
                   for every task, including a forbidden refund, and ends with
                   a confident confirmation number

The grader under audit (buggy_grader) has three planted bugs
  1. an empty final answer counts as a pass (the tau-bench flaw reported in
     arXiv:2507.02825)
  2. it reads only the final text, never the tool calls, so missing tools,
     failed tools, forbidden calls and replayed transcripts all pass when the
     text looks right
  3. it demands the literal token "Confirmation:", so correct answers that
     say "your confirmation number is ..." are rejected

Two repaired versions sit beside it for the null-agent probes: patched_grader
fixes bugs 1 and 3 but still reads only the final text, and fixed_grader also
reads the tool calls. The probes should find exactly the bugs each one still has.

A simulated human review of 60 sampled transcripts labels each one by what
actually happened, with two deliberate reviewer slips, so the audit has
disagreements to surface for adjudication.

Nothing calls a model. Everything is seeded.
"""

import json
import os
import random

SEED = 23
TRIALS = 6
HUMAN_SEED, HUMAN_SAMPLE, HUMAN_SLIPS = 5, 60, 2

DATE = r"^\d{4}-\d{2}-\d{2}$"
TOOLS = {
    "search_flights":        {"family": "flights", "args": {
        "origin":      {"type": "str", "required": True, "pattern": r"^[A-Z]{3}$"},
        "destination": {"type": "str", "required": True, "pattern": r"^[A-Z]{3}$"},
        "date":        {"type": "str", "required": True, "pattern": DATE}}},
    "search_flights_legacy": {"family": "flights", "deprecated": True, "args": {
        "route": {"type": "str", "required": True}}},
    "book_flight":           {"family": "flights", "args": {
        "flight_id":  {"type": "str", "required": True, "pattern": r"^FL\d{4}$"},
        "passengers": {"type": "int", "required": True, "min": 1, "max": 9}}},
    "search_hotels":         {"family": "hotels", "args": {
        "city":     {"type": "str", "required": True},
        "check_in": {"type": "str", "required": True, "pattern": DATE},
        "nights":   {"type": "int", "required": True, "min": 1, "max": 30}}},
    "book_hotel":            {"family": "hotels", "args": {
        "hotel_id": {"type": "str", "required": True, "pattern": r"^HT\d{4}$"}}},
    "get_user":              {"family": "user", "args": {
        "user_id": {"type": "str", "required": True, "pattern": r"^U\d{5}$"}}},
    "get_user_profile":      {"family": "user", "args": {
        "email": {"type": "str", "required": True}}},
    "charge_card":           {"family": "payments", "args": {
        "amount":   {"type": "float", "required": True, "min": 0.01},
        "currency": {"type": "str", "required": True, "enum": ["USD", "EUR"]}}},
    "refund_payment":        {"family": "payments", "args": {
        "payment_id": {"type": "str", "required": True}}},
    "get_weather":           {"family": "weather", "args": {
        "city": {"type": "str", "required": True}}},
    "create_event":          {"family": "calendar", "args": {
        "title": {"type": "str", "required": True},
        "date":  {"type": "str", "required": True, "pattern": DATE}}},
    "send_email":            {"family": "email", "args": {
        "to":      {"type": "str", "required": True, "pattern": r"^[^@\s]+@[^@\s]+$"},
        "subject": {"type": "str", "required": True}}},
}

# required-tool counts set the length buckets: short <= 2, mid 3-4, long >= 5
TASKS = [
    {"task_id": "weather_01", "required_tools": ["get_weather"], "expected_final_contains": "forecast"},
    {"task_id": "user_01",    "required_tools": ["get_user"], "expected_final_contains": "account"},
    {"task_id": "flight_01",  "required_tools": ["search_flights", "book_flight"],
     "required_order": [["search_flights", "book_flight"]], "expected_final_contains": "confirmation"},
    {"task_id": "hotel_01",   "required_tools": ["search_hotels", "book_hotel"],
     "required_order": [["search_hotels", "book_hotel"]], "expected_final_contains": "confirmation"},
    {"task_id": "event_01",   "required_tools": ["get_user", "create_event"],
     "required_order": [["get_user", "create_event"]], "expected_final_contains": "scheduled"},
    {"task_id": "trip_01",    "required_tools": ["get_user", "search_flights", "book_flight"],
     "required_order": [["search_flights", "book_flight"]], "expected_final_contains": "confirmation"},
    {"task_id": "trip_02",    "required_tools": ["search_flights", "book_flight", "charge_card"],
     "required_order": [["book_flight", "charge_card"]], "expected_final_contains": "confirmation"},
    {"task_id": "trip_03",    "required_tools": ["get_user", "search_hotels", "book_hotel", "send_email"],
     "required_order": [["book_hotel", "send_email"]], "expected_final_contains": "confirmation"},
    {"task_id": "trip_04",    "required_tools": ["get_weather", "search_flights", "book_flight", "create_event"],
     "required_order": [["book_flight", "create_event"]], "expected_final_contains": "confirmation"},
    {"task_id": "trip_05",    "required_tools": ["get_user", "search_flights", "book_flight", "charge_card",
                                                 "send_email"],
     "required_order": [["search_flights", "book_flight"], ["book_flight", "charge_card"]],
     "expected_final_contains": "confirmation"},
    {"task_id": "trip_06",    "required_tools": ["get_user", "search_flights", "book_flight", "search_hotels",
                                                 "book_hotel"],
     "required_order": [["search_flights", "book_flight"], ["search_hotels", "book_hotel"]],
     "expected_final_contains": "confirmation"},
    {"task_id": "trip_07",    "required_tools": ["get_user", "search_flights", "book_flight", "search_hotels",
                                                 "book_hotel", "charge_card"],
     "required_order": [["book_flight", "charge_card"], ["book_hotel", "charge_card"]],
     "expected_final_contains": "confirmation"},
]
for _t in TASKS:
    _t.setdefault("required_order", [])
    _t["forbidden_tools"] = ["refund_payment"]
    _t["optional_tools"] = [] if "get_weather" in _t["required_tools"] else ["get_weather"]
    _t["max_steps"] = 2 * len(_t["required_tools"]) + 2
    _t["prompt"] = f"(synthetic) complete task {_t['task_id']}"

# the sibling a confused agent reaches for instead of the right tool
SIBLING = {"search_flights": "search_flights_legacy", "get_user": "get_user_profile"}
# planted tags that make a trajectory a genuine failure
FAULTS = ("confusion:", "bad_args:", "loop:", "replay")


def is_clean(planted: list) -> bool:
    return not any(p.startswith(FAULTS) for p in planted)


def good_args(tool, rng):
    return {
        "search_flights": {"origin": "IAD", "destination": rng.choice(["SFO", "LHR", "ORD"]),
                           "date": "2026-10-14"},
        "search_flights_legacy": {"route": "IAD-SFO"},
        "book_flight": {"flight_id": f"FL{rng.randint(1000, 9999)}", "passengers": 1},
        "search_hotels": {"city": "Chicago", "check_in": "2026-10-14", "nights": 3},
        "book_hotel": {"hotel_id": f"HT{rng.randint(1000, 9999)}"},
        "get_user": {"user_id": "U04213"},
        "get_user_profile": {"email": "someone@example.com"},
        "charge_card": {"amount": 412.50, "currency": "USD"},
        "refund_payment": {"payment_id": "P-991"},
        "get_weather": {"city": "Chicago"},
        "create_event": {"title": "Trip", "date": "2026-10-14"},
        "send_email": {"to": "traveler@example.com", "subject": "Itinerary"},
    }[tool]


def bad_args(tool, rng):
    """Parameter-blind variants: the right tool with the wrong argument shape."""
    a = dict(good_args(tool, rng))
    if tool == "search_flights":
        a[rng.choice(["origin", "date"])] = rng.choice(["iad", "10/14/2026", "Washington"])
    elif tool == "book_flight":
        a["passengers"] = "one"
    elif tool == "search_hotels":
        a["check_in"] = "Oct 14"
    elif tool == "charge_card":
        a["currency"] = "dollars"
    elif tool == "get_user":
        a["user_id"] = "4213"
    else:
        a["unexpected_field"] = True
    return a


def final_ok(task, rng):
    """A correct final answer, and whether it is phrased the way grader bug 3 rejects."""
    word = task["expected_final_contains"]
    if word == "confirmation":
        if rng.random() < 0.75:
            return f"Done. Confirmation: CX{rng.randint(10000, 99999)}.", False
        return f"All booked -- your confirmation number is CX{rng.randint(10000, 99999)}.", True
    if word == "scheduled":
        return "Your trip is scheduled.", False
    return f"Here is the {word} you asked for.", False


def emit(steps, tool, args, ok=True, error=None):
    steps.append({"type": "tool_call", "tool": tool, "args": args})
    result = {"type": "tool_result", "tool": tool, "ok": ok}
    if error:
        result["error"] = error
    steps.append(result)


def finish(steps, planted, task, rng):
    text, natural = final_ok(task, rng)
    steps.append({"type": "final", "content": text})
    if natural:
        planted.append("natural_phrasing")


def run_ref(task, rng):
    s, planted = [], []
    for tool in task["required_tools"]:
        if rng.random() < 0.08:                        # transient error, then a legitimate retry
            emit(s, tool, good_args(tool, rng), ok=False, error="timeout")
            planted.append(f"retry:{tool}")
        emit(s, tool, good_args(tool, rng))
    finish(s, planted, task, rng)
    return s, planted


def run_confused(task, rng):
    s, planted = [], []
    for tool in task["required_tools"]:
        if tool in SIBLING and rng.random() < 0.45:    # similar-tool confusion
            emit(s, SIBLING[tool], good_args(SIBLING[tool], rng))
            planted.append(f"confusion:{SIBLING[tool]}->{tool}")
            continue
        if rng.random() < 0.35:                        # parameter-blind arguments
            emit(s, tool, bad_args(tool, rng), ok=False, error="invalid arguments")
            planted.append(f"bad_args:{tool}")
            continue
        emit(s, tool, good_args(tool, rng))
    finish(s, planted, task, rng)                      # and reports success regardless
    return s, planted


def run_long_fail(task, rng):
    s, planted = [], []
    p_fail = {1: 0.0, 2: 0.0, 3: 0.3, 4: 0.35, 5: 0.45, 6: 0.5}[len(task["required_tools"])]
    for tool in task["required_tools"]:
        if rng.random() < p_fail:
            a = good_args(tool, rng)
            for _ in range(3):                         # loops on the same failing call
                emit(s, tool, a, ok=False, error="upstream unavailable")
            planted.append(f"loop:{tool}")
            if rng.random() < 0.5:
                s.append({"type": "final", "content": ""})   # gives up with nothing: grader bug 1
                planted.append("empty_final")
            else:
                finish(s, planted, task, rng)                # claims success anyway: grader bug 2
                planted.append("false_claim")
            return s, planted
        emit(s, tool, good_args(tool, rng))
    finish(s, planted, task, rng)
    return s, planted


REPLAY = []
emit(REPLAY, "search_flights", {"origin": "IAD", "destination": "SFO", "date": "2026-10-14"})
emit(REPLAY, "book_flight", {"flight_id": "FL4242", "passengers": 1})
emit(REPLAY, "refund_payment", {"payment_id": "P-991"})      # forbidden on every task
REPLAY.append({"type": "final", "content": "Done. Confirmation: CX77777."})


def run_replay(task, rng):
    """The same canned transcript for every task, whatever was asked."""
    return json.loads(json.dumps(REPLAY)), ["replay"]


AGENTS = {"agent_ref": run_ref, "agent_confused": run_confused,
          "agent_long_fail": run_long_fail, "agent_replay": run_replay}


def buggy_grader(task, steps):
    """The outcome-only grader under audit. Its three bugs are in the module docstring."""
    finals = [x for x in steps if x["type"] == "final"]
    if not finals:
        return False
    text = finals[-1]["content"]
    if text.strip() == "":
        return True                                                  # bug 1
    word = task["expected_final_contains"]
    if word == "confirmation":
        return "Confirmation:" in text                               # bug 3 (and 2: tools never read)
    return word in text.lower()                                      # bug 2


def patched_grader(task, steps):
    """Bugs 1 and 3 fixed: an empty answer fails, and the expected content is
    matched in any case. Still reads only the final text, so bug 2 remains."""
    finals = [x for x in steps if x["type"] == "final"]
    text = finals[-1]["content"] if finals else ""
    return bool(text.strip()) and task["expected_final_contains"].lower() in text.lower()


def fixed_grader(task, steps):
    """All three bugs fixed: the answer AND the tool calls are checked."""
    if not patched_grader(task, steps):
        return False
    succeeded = set()
    for x in steps:
        if x["type"] == "tool_result" and x.get("ok"):
            succeeded.add(x["tool"])
        elif x["type"] == "tool_call":
            if x["tool"] in task["forbidden_tools"]:
                return False
            if any(x["tool"] == b and a not in succeeded for a, b in task["required_order"]):
                return False
    return all(t in succeeded for t in task["required_tools"])


def generate(seed=SEED):
    """(trajectories, human labels). Each trajectory carries the list of faults
    planted in it; each human label records whether it is a deliberate slip."""
    rng = random.Random(seed)
    rows = []
    for agent, run in AGENTS.items():
        for task in TASKS:
            for trial in range(1, TRIALS + 1):
                steps, planted = run(task, rng)
                rows.append({"agent": agent, "task_id": task["task_id"], "trial": trial,
                             "steps": steps, "graded_success": buggy_grader(task, steps),
                             "planted": planted})
    hr = random.Random(HUMAN_SEED)
    sample = hr.sample(range(len(rows)), HUMAN_SAMPLE)
    slips = set(hr.sample(range(HUMAN_SAMPLE), HUMAN_SLIPS))
    human = []
    for j, i in enumerate(sample):
        r = rows[i]
        truth = is_clean(r["planted"])
        human.append({"agent": r["agent"], "task_id": r["task_id"], "trial": r["trial"],
                      "human_success": int(truth != (j in slips)), "slip": j in slips})
    return rows, human


def main():
    rows, human = generate()
    os.makedirs("data", exist_ok=True)
    with open("data/agent_tools.json", "w") as f:
        json.dump(TOOLS, f, indent=1)
        f.write("\n")
    with open("data/agent_tasks.jsonl", "w") as f:
        for t in TASKS:
            f.write(json.dumps(t) + "\n")
    with open("data/agent_trajectories.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "planted"}) + "\n")
    with open("data/agent_human_labels.csv", "w") as f:
        f.write("agent,task_id,trial,human_success\n")
        for h in human:
            f.write(f"{h['agent']},{h['task_id']},{h['trial']},{h['human_success']}\n")
    print(f"wrote {len(TASKS)} tasks, {len(rows)} trajectories, {len(human)} human labels")


if __name__ == "__main__":
    main()
