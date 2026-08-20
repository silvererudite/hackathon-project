# Agentic Data Selection — SCI OPS

Hackathon project: an agent that decides *which participants* should support a
scientific conclusion, quantifies uncertainty in that decision, and tests
whether the conclusion survives alternative defensible selections.

Data: [SCI OPS](https://github.com/nivlab/sciops) — Zorowitz, Solis, Niv &
Bennett (2023), *Inattentive responding can induce spurious associations
between task behaviour and symptom measures*, **Nature Human Behaviour** 7,
1667–1681. [Open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC11170515/)

## Run it

```bash
source ../.venv/bin/activate
jupyter lab Agentic_Data_Selection_Quick_Demo.ipynb
```

Everything runs offline. For the **live** agent loop in section 6:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Without a key, `run_agent(live=False)` replays the highest-rated human
trajectory so the notebook still produces the full analysis.

## Layout

```
01_Original/data/       386 participants, reversal-learning task   (36,540 trials)
02_Replication/data/    393 participants, two-step task            (80,400 trials)
sciops_agent.py         analysis tools + reactive loop + trajectory scoring
llm_backends.py         aitta / anthropic / scripted; one shared tool spec
trace_schema.py         process-trace Pydantic models
trace_ui.py             ipywidgets: ask, rate, browse
check_aitta.py          connectivity + tool-calling check
expert_trajectories.json  seed exemplars for the system prompt
traces/                 trajectories you rate in the notebook
Agentic_Data_Selection_Quick_Demo.ipynb
```

Data files are downloaded from the SCI OPS repo and are not committed; re-fetch with:

```bash
for S in 01_Original 02_Replication; do
  for F in metrics scores metadata surveys items data; do
    curl -sfL "https://raw.githubusercontent.com/nivlab/sciops/main/$S/data/$F.csv" \
      -o "$S/data/$F.csv"
  done
done
```

## What the analysis shows

`gad7`–accuracy is **selection-sensitive** — *r* moves 0.150 across policies:

| policy | r | note |
|---|---|---|
| all data | −0.122 | significant |
| oracle clean | +0.006 | vanishes; uses the true label, evaluation reference only |
| agent hard | +0.028 | learned policy, **never sees the label** |
| agent weighted | −0.069 | |

The learned policy tracks the oracle without access to it. Selector quality:
AUROC 0.755, but **31% of participants land in the ambiguous band**, so
inclusion is not confidently decidable from task and timing features alone —
which is why the agent spends budget on extra attention checks rather than
concluding.

Policy transfer across two *different* cognitive tasks: **AUROC 0.728**
(in-sample 0.755), using the 6 features shared by both samples.

## The two samples are less comparable than they look

|  | Original | Replication |
|---|---|---|
| symptoms | `7u 7d gad7 pswq bis bas shaps` | `mania depression anxiety artistic greed` |
| behaviour | `accuracy`, `variability`, `wsls` | `choice_var`, `side_var`, `wsls` |
| shared quality features | `isd reliability mahalanobis survey_rt task_rt wsls` | same |

Only `wsls` is common to both tasks. A selector using the full 8-feature list
**cannot be applied to the replication sample at all** — the columns do not
exist. Cross-sample claims must use `TRANSFER_FEATURES`, and construct matches
(`gad7` → `anxiety`) are judgements to state explicitly, not identities.
