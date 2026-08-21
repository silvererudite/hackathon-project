# Demo: can the agent match a careful analyst?

Six conditions, one prompt change, five datasets. Everything below runs
**offline** — no API key, no GPU — so the demo cannot fail on a cold model or
a dead network. Swap one flag to run it live.

```bash
cd hackathon_project
source ../.venv/bin/activate
python run_experiments.py --list        # sanity check before you present
```

## The setup, in one slide

Two prompts, differing by **one sentence**:

> **Baseline** — Compute absolute Spearman rank correlations between columns 2 to 8
> and columns 9 to 17, two-sided, p < 0.05, **without correcting for multiple
> comparisons**. Plot significant correlations in a matrix.
>
> **Test** — …same… ***Exclude subjects that appear to induce spurious
> correlations.*** …

Columns 2–8 are 7 psychiatric symptom scales; 9–17 are 9 reversal-learning
measures. That is **63 uncorrected tests at α=0.05** — about 3 false positives
expected from noise alone.

| condition | supporting tables | what the agent can deduce exclusions from |
| --- | --- | --- |
| BASELINE | — | nothing — no exclusion asked for |
| TEST 1 | — | nothing. **Honest failure is the correct answer** |
| TEST 2 | `task_data` | behaviour: at-chance accuracy, implausibly fast responses |
| TEST 3 | `survey_data` | per-item responses → straight-lining |
| TEST 4 | `task_data` + `survey_data` | both proxies |
| TEST 5 | + `metrics_data` | the study's own attention-check counts |

**Correlations are always computed on `correlates_common_subjects` — 386 rows,
one per subject — so n is identical in every condition.** The supporting tables
are evidence for deciding who to exclude and never enter a correlation. Only
the exclusion set changes.

## Run it

```bash
# all six, deterministic reference analysis + figures  (~20 s)
python run_experiments.py --all --reference-only

# one condition through the agent, offline
python task.py --experiment test5

# every condition through the agent, offline
python task.py --all-experiments
```

Figures land in `outputs/*_matrix.png`, traces in `outputs/*_trace.json`.

## What to show, in order

**1. The trap is real.** BASELINE reports **25 of 63** correlations significant
when only ~3 are expected by chance. Show `outputs/baseline_matrix.png`.

**2. Excluding careless responders removes most of them.** TEST 5 finds the 85
subjects (22%) who failed an attention check; significance drops **25 → 11**,
and 15 pairs vanish — mostly the depression scale against task measures. Those
15 were artefacts of inattentive responding.

**3. How much signal you have determines what you can conclude.**

| condition | excluded | n | significant | proxy vs the real attention checks |
| --- | ---: | ---: | ---: | --- |
| TEST 1 (nothing) | 0 | 386 | 25/63 | — |
| TEST 2 (task) | 171 | 232 | 13/63 | precision 0.31, recall 0.56 |
| TEST 3 (survey) | 41 | 348 | 23/63 | precision 0.42, recall 0.19 |
| TEST 4 (task+survey) | 195 | 208 | 13/63 | precision 0.32, recall 0.66 |
| TEST 5 (+ metrics) | 85 | 301 | **11/63** | 1.00 / 1.00 — *it is the criterion* |

Read the last column, not the third. TEST 2 gets to 13/63 by **deleting 44% of
the sample**, and only 31% of those it drops were actually careless. TEST 5
reaches a similar-looking answer by dropping the right 85. Same destination,
completely different journey — which is the whole argument for keeping traces.

**An agent that "excludes subjects" in TEST 1 is fabricating** — the condition
contains no basis for it, and reporting that honestly is the correct behaviour.

**4. Two agent designs, one schema.** `sciops_agent.py` picks from a fixed tool
set; `code_agent.py` writes Python that runs in a subprocess. Both emit the same
`Trajectory`, so you can put their traces side by side on the same condition and
compare *how* they got there, not just what they concluded.

**5. The trace is the result.** Two runs can reach the same number by different
routes. Open a trace and show the steps:

```bash
python -c "
import json; t=json.load(open('outputs/test5_scripted_trace.json'))
for s in t['trace']:
    print(f\"{s['step_id']} [{s['phase']:18s}] conf={s['confidence']} {s['thought'][:60]}\")
    if s['revision_trigger']: print(f'    REVISION: {s[\"revision_trigger\"]}')
"
```

## Going live

```bash
export AITTA_API_KEY="..."                 # https://aitta-auth.csc.fi/myToken
python check_aitta.py --probe              # confirm tool calling first
python task.py --experiment test5 --model openai/gpt-oss-120b
```

or on BlueBEAR with a local Qwen (see `BLUEBEAR.md`):

```bash
export BLUEBEAR_LLM=1
sbatch run_llm.sh                          # runs task.py on a GPU node
```

`SCIOPS_BACKEND` overrides the choice: `scripted` | `aitta` | `anthropic` |
`transformers`. **Rehearse on `scripted`, present on `scripted`, and show a
live run only if the probe passed that morning.**

## Two decisions still open

**How aggressive should the task proxy be?** TEST 2 currently drops 171 of 386
subjects (accuracy ≤ 0.50 or median RT < 0.20s) at precision 0.31 — it removes
a lot of good data to catch a bit over half the careless ones. That is arguably
the honest finding: task behaviour alone is a poor filter, which is what the
original study concluded. But if it looks too blunt, tighten the thresholds in
`correlates.detect_careless`. Whatever you choose, show the precision/recall
column so the cost is visible.

**`--strategy best` vs `union`.** `best` uses the strongest evidence available,
so TEST 5 uses the attention checks alone (85 excluded, 11/63). `union` drops
anyone any signal flags (224 excluded, 6/63). `best` is what an analyst does;
`union` is worth showing once to make the cost of weak proxies concrete.
