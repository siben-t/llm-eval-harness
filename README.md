# llm-eval-harness

A rubric-driven evaluation harness for LLM outputs: deterministic
scoring checks, A/B model comparison, a human-review queue, and —
the part most eval scripts skip — **validation of the judge itself
against hand labels**.

Runs fully offline on saved responses (JSONL in → scored report out).
No API keys required.

Since 2026-09-18 it also ships a **judge audit** (`judge_audit.py`): position
bias, self-preference, directional bias with capability-dependent leniency,
and drift — each with a confidence interval and a source for the definition —
plus a label-free repair via Dawid–Skene. See *Auditing an LLM judge* below.

Since 2026-09-24 it also evaluates **agent trajectories** (`agent_eval.py`):
process metrics for tool-using agents, pass@k and pass^k, a length profile
that separates a real bottleneck from compounding error, an audit of the
grader that scored the runs, and null-agent probes that test a grader before
any agent runs. See *Evaluating agents, and the grader that scored them* below.

## Quickstart

```bash
pip install -r requirements.txt
python make_example_data.py    # writes 24 tasks + 2 simulated runs
python demo.py                 # full report + figures/ + output/
python -m pytest tests/ -v     # hand-computable tests for every check
```

Or run it against your own saved responses, no code changes:

```bash
python cli.py --tasks your_tasks.jsonl --responses your_run.jsonl --json scores.json
```

## Using it on your own data

Nothing here calls a model API. You generate responses however you like, save them as JSONL
with `task_id` and `response`, and this scores them — which is what makes a run reproducible
and free to repeat.

```bash
# two runs, head to head
python cli.py --tasks t.jsonl \
              --responses baseline.jsonl --name baseline \
              --responses candidate.jsonl --name candidate \
              --compare

# check the judge against your own labels before trusting its scores
python cli.py --tasks t.jsonl --responses run.jsonl --human-labels labels.csv

# gate a build
python cli.py --tasks t.jsonl --responses run.jsonl --fail-under 0.80
```

Exit codes: `0` passed, `1` a run fell below `--fail-under`, `2` bad input. Bad input is
deliberately not `1` — a malformed JSONL file and a genuinely low-scoring model should not
look the same to a build system.

A rubric that names a check which doesn't exist is an error, not a warning. Scoring three
dimensions while believing you scored four is worse than failing.

`--compare` prints a head-to-head but **no significance test**, and says so in the output. On
a few dozen tasks a gap of a few points is noise.

## What it measures — and what it honestly can't

The rubric (`rubric.json`) defines four dimensions, each scored by a
pure function in `checks.py`:

| Dimension | Check | Example |
|---|---|---|
| `accuracy` | numeric match vs reference, with tolerance; exact-text match | final answer within 0.1% |
| `format` | structural requirements | response contains parseable JSON |
| `instruction` | required keywords, forbidden terms, length caps | "do not mention pricing" |
| `safety` | refusal behavior **in both directions** | refuses a phishing request; does *not* refuse a benign first-aid question |

Reasoning soundness is deliberately absent: it cannot be graded
deterministically. That boundary is where human raters or model-based
judges enter — and where any judge must itself be audited, which is
what `validate_judge` does against `data/human_labels.csv`.

Measuring safety in both directions matters: under-refusal is the
failure everyone tests for; **over-refusal** is the one that quietly
destroys helpfulness and rarely gets measured.

## Auditing an LLM judge

Deterministic checks can grade numbers and formats. Anything subtler gets
graded by a model, and a model judge has failure modes of its own. The
audit measures four of them from verdicts you already have — it never calls
a model — and reports each one with an interval, because a bias estimate
without one is a guess.

| Failure mode | What is measured | Definition follows |
|---|---|---|
| **Position bias** | every pair shown in both orders; an order-blind judge picks "first" exactly 50% of the time. Deviation from 0.5, flip rate, and primacy vs recency | Shi et al., *A Systematic Study of Position Bias in LLM-as-a-Judge*, IJCNLP 2025 |
| **Self-preference** | β = PIR − null-PIR on equal-quality pairs: how much more often the judge consistently picks its own model's output than it consistently picks an arbitrary one of two equal third-party outputs | arXiv:2604.22891 |
| **Directional bias** | FPR − FNR on correct/incorrect verdicts (+ lenient, − strict), and *capability-dependent leniency*: does FPR rise with the examinee's true accuracy? | arXiv:2609.12002 (Sept 2026) |
| **Drift** | FPR and FNR per batch over time; first-vs-last two-proportion test | — |

And one repair: **Dawid–Skene EM** across several judges estimates each
judge's FPR and FNR *without ground truth* and produces a posterior-weighted
vote. When truth is available the audit reports how close the label-free
estimate came, which is the test of whether you could run it on a live system.

```bash
python make_judge_data.py          # synthetic verdicts with PLANTED biases
python judge_audit.py --pairwise data/judge_pairwise.jsonl \
                      --binary   data/judge_binary.jsonl \
                      --own judge_alpha=model_alpha --own judge_beta=model_beta \
                      --own judge_gamma=model_gamma \
                      --fail-position-bias 0.10 --fail-beta 0.15 --fail-drift 0.10 \
                      --json audit.json
```

The demo plants three faults in `judge_alpha` — primacy bias, preference for
its own model, and an FPR that jumps from batch 3 onward — leaves
`judge_beta` clean as the control, and gives `judge_gamma` mild recency bias
only. The gates fail on exactly `judge_alpha`, on exactly the three planted
faults:

```
Position bias           first_pick  bias    flip   primacy
  judge_alpha   n=600     0.661   +0.161   0.408   0.894    <- planted
  judge_beta    n=600     0.508   +0.008   0.337   0.525
  judge_gamma   n=600     0.415   -0.085   0.347   0.255    <- planted (recency)

Self-preference (β, 95% CI)
  judge_alpha   +0.288  [ 0.172,  0.406]                    <- planted
  judge_beta    +0.001  [-0.084,  0.098]
  judge_gamma   +0.010  [-0.090,  0.111]

Drift, FPR first batch -> last
  judge_alpha   0.095 -> 0.284   +0.189   p = 0.0004        <- planted
  judge_beta    0.114 -> 0.138   +0.023   p = 0.61
  judge_gamma   0.124 -> 0.183   +0.060   p = 0.23

Capability-dependent leniency, pooled: FPR on the strongest examinee is
+0.127 above the weakest (p < 0.0001). Every judge shows it — it was
planted in all three, as the paper found in the wild.

Label-free error rates (Dawid–Skene) vs truth: max abs error 0.008.
```

Three things worth knowing before pointing this at a real judge:

- **Position bias needs both orders.** The audit refuses data where a pair
  was only shown one way, because there is no honest way to separate order
  effects from quality effects otherwise.
- **The drift gate requires significance, not just size.** A gate that fires
  on noise trains people to ignore it. `--fail-drift 0.10` fails only when the
  change exceeds 0.10 *and* p < 0.05; `judge_gamma` above moved +0.060 at
  p = 0.23 and correctly passes.
- **Leniency is the subtlest of the four.** With four examinees the Pearson r
  the paper reports is unstable, so the audit also gives the strongest-minus-
  weakest FPR gap with a p-value, and pools across judges for the headline.
  Across eight random seeds of the generator the pooled gap was positive in
  all eight and significant in five: the effect is real and needs data.

`tests/test_judge_audit.py` asserts every planted number is recovered and
the control is left alone — 21 tests, including a deterministic judge that
must sit at exactly 0.5 and a pure-primacy judge that must read 1.0.

## Evaluating agents, and the grader that scored them

An agent's score is only as good as the grader that produced it, and published
audits keep finding graders that don't hold up. TAU-bench counted empty
responses as successful (Zhu et al., arXiv:2507.02825). BenchJack found 219
flaws across ten agent benchmarks and reached near-perfect scores without
solving a single task (arXiv:2605.12673, May 2026). A stricter harness cut one
frontier model's SWE-bench Pro score from 87.1% to 73.0% (Cursor, June 2026).
So `agent_eval.py` answers two questions from saved trajectories. It never
calls a model.

**1. How good is the agent: process, not just outcome.** Every trajectory is
scored on:

| Metric | What counts |
|---|---|
| tool recall / precision | required tools that were called; share of the distinct tools called that the task needed |
| argument validity | every call checked against the tool schema: required, type, pattern, enum, range, unknown arguments |
| dependency order | a *successful* call to `a` before the first call to `b`. Booking before the search worked is a violation |
| forbidden calls | any call to a tool the task forbids. One is enough to fail |
| similar-tool confusion | a sibling from the same tool family called while the needed tool never was |
| unrecovered errors | a required tool that failed and never succeeded. A retry that works is fine |
| loops, redundancy, budget | the same call 3+ times; the same call again after it already worked; more calls than `max_steps` |
| step efficiency | one successful call per required tool is the work; everything else is overhead |

These follow TRAJECT-Bench (He et al., ICLR 2026), which names similar-tool
confusion and parameter-blind selection as the dominant tool-use failures.

Reliability is reported as **pass@k and pass^k**, with unbiased estimators
over n trials per task. pass@k = 1 − C(n−c, k)/C(n, k) asks whether the agent
succeeds at least once in k tries. pass^k = C(c, k)/C(n, k) asks whether it
succeeds every time, which is the number a user of a deployed agent feels
(Anthropic, *Demystifying evals for AI agents*, January 2026).

**2. Can the grader be trusted.** Every trajectory is re-scored by a strict
reference: every required tool succeeded, in dependency order, nothing
forbidden was called, and the final answer is non-empty and says what the task
needs. The grader's verdicts are cross-tabulated against it. Each kind of
false pass is named and counted: empty answer, missing tool, failed tool,
order violation, forbidden call, wrong answer, replayed transcript. Together
the named kinds account for every false pass, and a test enforces that.
Valid answers the grader rejected are counted too. With a human-labelled
sample, the grader *and the reference* are both checked against people.

```bash
python make_agent_data.py      # 4 agents x 12 tasks x 6 trials, planted faults, a buggy grader
python agent_eval.py --tools data/agent_tools.json --tasks data/agent_tasks.jsonl \
                     --trajectories data/agent_trajectories.jsonl \
                     --human-labels data/agent_human_labels.csv \
                     --k 1 3 --fail-kappa 0.6 --fail-forbidden --json agent_eval.json
```

The demo plants faults in three agents and leaves `agent_ref` clean as the
control. The control retries after occasional timeouts, and that must not be
flagged. `agent_confused` calls deprecated sibling tools and passes malformed
arguments. `agent_long_fail` loops on a failing call in longer tasks, then
returns nothing or claims success. `agent_replay` emits one canned transcript
for every task, forbidden refund included. The grader under audit has three
planted bugs: empty answers pass; it reads only the final text, never the tool
calls; and it demands the literal token "Confirmation:". Excerpt:

```
                      reference            as the grader reports it
Reliability          pass@1   pass^3          pass@1   pass^3
  agent_ref           1.000    1.000           0.903    0.758
  agent_confused      0.167    0.004           0.792    0.554
  agent_long_fail     0.514    0.417           0.861    0.662
  agent_replay        0.000    0.000           0.750    0.750    <- never completes a task

Grader audit: agreement 0.497, kappa 0.088, false-pass rate 0.784
  passed_empty_answer          18   long_fail 18                      <- bug 1
  passed_missing_tool         106   confused 28, long_fail 30, replay 48
  passed_failed_tool           64   confused 33, long_fail 31
  passed_order_violation       22   confused 22
  passed_forbidden_call        54   replay 54                         <- bug 2
  passed_replayed_transcript   54   replay 54
  rejected_valid               14   ref 7, long_fail 6, confused 1    <- bug 3

Length profile       short    mid    drop      q   predicted mid   shortfall        p
  agent_confused     0.333  0.042   0.292  0.483           0.084       0.042     0.39
  agent_long_fail    1.000  0.167   0.833  1.000           1.000       0.833   <0.001   <- bottleneck

Human sample (n=60): grader kappa 0.146, reference kappa 0.933
  2 reference-vs-human disagreements listed for adjudication
```

With the gates in the command above, the run exits `1`, as it should: the
grader's kappa is 0.088 against a floor of 0.6, and `agent_replay` calls a
forbidden tool.

Four things the numbers show:

- **The grader would have shipped the replay agent.** It rates `agent_replay`
  at 75% pass@1 and a perfectly consistent 0.750 pass^3. The agent never
  completes a task and issues a forbidden refund on every run. A grader that
  reads only the final message cannot see what the agent did.
- **pass^k is the release number.** `agent_long_fail` looks passable at
  pass@3 = 0.675 and falls to pass^3 = 0.417. `agent_confused` goes from 0.404
  to 0.004.
- **A length drop is not automatically a bottleneck.** `agent_confused` loses
  29 points from short to mid-length tasks, but an agent that gets 48% of steps
  right predicts that loss (shortfall 0.04, p = 0.39). `agent_long_fail` gets
  every short-task step right and still collapses (shortfall 0.83). That is the
  bottleneck, and it is the only one flagged.
- **The reference is checked too.** It agrees with the human sample on 58 of
  60. Both disagreements are printed with the reference's reasons so a person
  can adjudicate them; they are not silently resolved in either direction.
  They are the two planted reviewer slips.

### Probing the grader before any agent runs

The audit above needs trajectories a grader has already scored. A cheaper
check comes first. Hand `agent_eval.py` the grader itself, as a Python
callable, and it builds transcripts the grader must fail and runs the grader
on every one, for every task. Each probe is a valid transcript with exactly
one defect:

| Probe | The one defect |
|---|---|
| `empty_answer` | every tool worked; the final answer is empty |
| `no_tools` | a confident answer, and not one tool call |
| `missing_tool` | the last required tool is never called |
| `failed_tool` | the last required tool returns an error and is never retried |
| `forbidden_call` | a forbidden tool is called along the way |
| `wrong_order` | a step runs before its prerequisite |
| `wrong_answer` | every tool worked; the answer lacks the expected content |
| `other_task` | another task's transcript, under this task's answer |

The control is the same transcript without the defect, and the grader must
pass it. A grader that rejects everything catches every probe and is still
broken. Probe arguments are copied from calls that real agents made
successfully, so each probe's defect is its only defect; a test checks every
probe against the reference to make sure.

```bash
python agent_eval.py --tools data/agent_tools.json --tasks data/agent_tasks.jsonl \
                     --trajectories data/agent_trajectories.jsonl \
                     --grader make_agent_data:buggy_grader \
                     --grader make_agent_data:patched_grader \
                     --grader make_agent_data:fixed_grader --fail-probes
```

```
Share of tasks where the grader PASSED (probes should be 0, the control 1)
                control  empty  no_tools  missing  failed  forbidden  order  wrong_answer  other_task
buggy_grader       1.0    1.0      1.0      1.0     1.0       1.0    1.0          0.0         1.0   BROKEN
patched_grader     1.0    0.0      1.0      1.0     1.0       1.0    1.0          0.0         1.0   BROKEN
fixed_grader       1.0    0.0      0.0      0.0     0.0       0.0    0.0          0.0         0.0   holds
```

`patched_grader` fixes the empty-answer and literal-token bugs but still reads
only the final text. The probes show exactly that, in about a second, without
running an agent. This is the check that catches the TAU-bench flaw: Zhu et
al. report that a trivial agent returning empty responses scored 38% there,
because on intentionally impossible tasks an unchanged environment counted
as success.

Limits, stated plainly:

- The reference needs a task spec: required tools, order, forbidden tools,
  expected content. Writing that spec is the real work of an agent eval. For
  open-ended tasks, grade with a rubric judge and audit the judge with
  `judge_audit.py`.
- Argument checks are schema-level. A well-formed date that is the wrong date
  passes; semantic checks belong in the task spec.
- Loop detection is exact-match. An agent that legitimately polls a status
  endpoint needs a higher `--loop-threshold`.
- Replay detection compares whole transcripts across tasks, so genuinely
  duplicate tasks will trip it. De-duplicate the task set first (`rl-env-qa`
  does that).

- Probes need arguments for every required tool. They come from the tool
  schema's `example_args` or from a successful call in the trajectories. A
  task with neither is skipped and listed, never probed with guessed arguments.

Exit codes match the rest of the harness: `0` all gates pass, `1` a gate
fails, `2` bad input. `tests/test_agent_eval.py` has 50 tests. They check the
estimators by hand (n = 5, c = 2, k = 2 gives pass@2 = 0.7 and pass^2 = 0.1)
and every argument rule. They also check that every planted fault and grader
bug is recovered exactly, with no misses and no false alarms, and that the
control is left alone. The probe tests check that each probe carries exactly
one defect, and that the probes find exactly the bugs each of the three
graders still has.

## What the demo shows

Two simulated model runs over 24 tasks in six categories. The weaker
model's gap is not uniform — accuracy is nearly level, while losses
concentrate in format (broken JSON on extraction tasks) and safety
(over-refusing benign-but-sensitive questions). Patterned losses have
specific fixes; a uniform gap would have meant a capability problem.

The judge validation step reports 9/10 agreement with hand labels and
surfaces the one disagreement on purpose: a response describing a
"light**weight** bike" trips the substring-based forbidden-term check
for "weight", though a human reads it as compliant. The repo ships
with that false positive documented rather than hidden, because the
audit layer existing to catch exactly this *is the point*. (Fixing it
with word-boundary matching is the first item under Extending.)

## Files

| File | Purpose |
|---|---|
| `checks.py` | the check library — pure, unit-tested scoring functions |
| `harness.py` | tasks + responses + rubric → tidy long-format scores |
| `report.py` | aggregates, A/B comparison, review queue, judge validation, plots |
| `rubric.json` | dimensions and weights, as data not code |
| `make_example_data.py` | deterministic generator for the example tasks/runs |
| `demo.py` | end-to-end run |
| `judge_audit.py` | position bias, self-preference, directional bias, drift, Dawid–Skene — library and CLI |
| `make_judge_data.py` | synthetic judge verdicts with planted, recoverable biases |
| `agent_eval.py` | agent trajectory metrics, pass@k / pass^k, length profile, grader audit — library and CLI |
| `make_agent_data.py` | synthetic agent trajectories with planted faults, graded by a planted-buggy grader; a patched and a fixed grader for the probes |
| `tests/test_checks.py` | hand-computable cases for every check |
| `tests/test_judge_audit.py` | every planted bias is recovered; the control is left alone |
| `tests/test_agent_eval.py` | every planted agent fault and grader bug is recovered exactly; the control is left alone |

## Data formats

`data/tasks.jsonl` — one task per line:

```json
{"task_id": "math_01", "category": "math", "prompt": "...",
 "reference": 3.0, "answer_type": "numeric", "tolerance": 1e-3}
```

Optional fields per task: `format`, `must_include`, `must_exclude`,
`max_words`, `should_refuse`. Checks that don't apply return None and
are excluded from that task's weighted overall score.

`data/responses_<run>.jsonl` — `{"task_id": ..., "response": ...}`.

`data/human_labels.csv` — the audited subset: `task_id, dimension,
human_score`.

Agent evaluation (`agent_eval.py`):

```jsonc
// data/agent_tools.json: one schema per tool; "family" groups siblings that get confused
{"book_flight": {"family": "flights", "args": {
    "flight_id": {"type": "str", "required": true, "pattern": "^FL\\d{4}$"},
    "passengers": {"type": "int", "required": true, "min": 1, "max": 9}}}}

// data/agent_tasks.jsonl: one task per line
{"task_id": "flight_01", "required_tools": ["search_flights", "book_flight"],
 "required_order": [["search_flights", "book_flight"]], "forbidden_tools": ["refund_payment"],
 "optional_tools": ["get_weather"], "max_steps": 6, "expected_final_contains": "confirmation"}

// data/agent_trajectories.jsonl: one run per line; graded_success is the grader under audit
{"agent": "agent_ref", "task_id": "flight_01", "trial": 1, "graded_success": true, "steps": [
  {"type": "tool_call", "tool": "search_flights", "args": {"origin": "IAD", "destination": "SFO", "date": "2026-10-14"}},
  {"type": "tool_result", "tool": "search_flights", "ok": true},
  {"type": "final", "content": "Done. Confirmation: CX12345."}]}
```

`data/agent_human_labels.csv` — optional sample: `agent, task_id, trial,
human_success`.

## Extending

- word-boundary matching for `must_exclude` (fixes the documented
  false positive — then re-run `validate_judge` to confirm 10/10)
- LLM-as-judge plug-in for the reasoning dimension, validated the same
  way against the human-label subset
- bootstrap confidence intervals on the head-to-head win rate
- regression gating: fail CI when a dimension mean drops vs baseline
- per-category score thresholds for release decisions
- agent eval: semantic argument checks from the task spec (the right date,
  not just a well-formed one)

## License

MIT
