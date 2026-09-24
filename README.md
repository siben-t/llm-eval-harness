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
| `tests/test_checks.py` | hand-computable cases for every check |
| `tests/test_judge_audit.py` | every planted bias is recovered; the control is left alone |

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

## Extending

- word-boundary matching for `must_exclude` (fixes the documented
  false positive — then re-run `validate_judge` to confirm 10/10)
- LLM-as-judge plug-in for the reasoning dimension, validated the same
  way against the human-label subset
- bootstrap confidence intervals on the head-to-head win rate
- regression gating: fail CI when a dimension mean drops vs baseline
- per-category score thresholds for release decisions

## License

MIT
