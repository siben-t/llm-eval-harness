# llm-eval-harness

A rubric-driven evaluation harness for LLM outputs: deterministic
scoring checks, A/B model comparison, a human-review queue, and —
the part most eval scripts skip — **validation of the judge itself
against hand labels**.

Runs fully offline on saved responses (JSONL in → scored report out).
No API keys required.

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
| `tests/test_checks.py` | hand-computable cases for every check |

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
