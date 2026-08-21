# Agentic Data Selection — SCI OPS

Hackathon project: an agent that decides *which participants* should support a
scientific conclusion, quantifies uncertainty in that decision, and tests
whether the conclusion survives alternative defensible selections.

Data: [SCI OPS](https://github.com/nivlab/sciops) — Zorowitz, Solis, Niv &
Bennett (2023), *Inattentive responding can induce spurious associations
between task behaviour and symptom measures*, **Nature Human Behaviour** 7,
1667–1681. [Open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC11170515/)

## Start the project

Install the environment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`config.py` may contain `OPENAI_API_KEY`; the file is ignored by Git.

### Interactive agent trace studio

```bash
source .venv/bin/activate
python agent_web.py
```

Open <http://127.0.0.1:7860>. Enter a research question to stream the agent's
tool-free ReAct exploration and inspect every rationale, generated Python block,
raw execution observation, error recovery, generated figure, saved PNG, and
conclusion. Completed traces are saved under `traces/`; each run's figures are
saved under `figures/<trajectory-id>/` and displayed inline in the web trace.

### CLI

Run the default correlation-matrix analysis:

```bash
source .venv/bin/activate
python -u agent_cli.py --backend openai --max-steps 8
```

Or provide a custom prompt and trace filename:

```bash
python -u agent_cli.py "Compute absolute Spearman rank correlations between columns 2 to 8 and columns 9 to 17 and run a two-sided, p < 0.05 significance test on the correlations without correcting for multiple comparisons. Plot significant correlations." \
  --backend openai \
  --max-steps 3 \
  --output traces/my-run.json
```

The CLI prints the saved JSON and PNG paths when the run completes. Generated
code calls the preloaded `plot_corr_matrix(...)` function with a unique
`figure_path`, so figures from different runs do not overwrite each other.
For comparison plots, generated code can pass `reference_pval=...`; the PNG and
web matrix mark cells that were significant only in the reference with `x`.

### Notebook

```bash
source .venv/bin/activate
jupyter lab Agentic_Data_Selection_Quick_Demo.ipynb
```

Everything runs offline. For the **live** agent loop in section 6:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Without a key, `run_agent(live=False)` replays the highest-rated human
trajectory so the notebook still produces the full analysis.

The OpenAI backend accepts `OPENAI_API_KEY` from the environment or from the
local, git-ignored `config.py` file.

## Layout

```
correlates_common_subjects.csv  386 subjects, survey/task summary measures
code_agent.py           tool-free generated-code ReAct loop + trace recording
code_runner.py          restricted local executor for model-generated Python
llm_backends.py         JSON generation backends; no function/tool schemas
sciops_agent.py         compatibility import for the existing notebook
trace_schema.py         process-trace Pydantic models
trace_ui.py             ipywidgets: ask, rate, browse
check_aitta.py          connectivity + JSON/code-generation check
expert_trajectories.json  seed exemplars for the system prompt
traces/                 trajectories you rate in the notebook
Agentic_Data_Selection_Quick_Demo.ipynb  optional notebook interface
```

The runtime loads only `correlates_common_subjects.csv`. Column 1 is the subject
identifier, columns 2–8 are survey summaries, and columns 9–17 are task
summaries. Raw task/survey datasets can be added later as separate experimental
conditions without changing the baseline correlation target.
