"""Tool-free ReAct agent that explores SCI OPS by generating Python code."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from pydantic import ValidationError

import llm_backends as backends
from trace_schema import (
    Action, Outcome, SelectionSensitivity, StepError, TraceStep, Trajectory,
    TrajectoryMetadata, Verification, now_iso,
)

HERE = Path(__file__).resolve().parent
PHASES = {"inspect", "quality_model", "policy_comparison", "budget_request",
          "revision", "conclusion"}
DEFAULT_TASK = (
    "Compute absolute Spearman rank correlations between columns 2 to 8 and columns 9 to 17 and run a two-sided p<0.05 significance test without correcting for multiple comparisons. Plot significant correlations."
)


def _dataset_description() -> str:
    import pandas as pd
    correlates = pd.read_csv(HERE / "correlates_common_subjects.csv", nrows=0)
    return "- `correlates` (correlates_common_subjects.csv): " + ", ".join(correlates.columns)


SYSTEM_HEADER = """You are a code-generating ReAct research agent analysing the SCI OPS dataset.
The dataset contains subject identifiers, survey summary measures, and task summary measures.
Answer the requested analysis, quantify uncertainty, and state limitations.

You do not use API tool-calling or function schemas. At each ReAct turn, either generate Python
code for the restricted executor or return the final structured conclusion. The executor
preloads these objects, so NEVER import anything:

  correlates                     correlates_common_subjects.csv DataFrame
  np, pd, stats                  NumPy, pandas, scipy.stats
  pearsonr, spearmanr
  plot_corr_matrix(corr, pval, out_path, xticklabels, yticklabels, title,
                   alpha=0.05, vmax=0.40, figsize=(8, 7), reference_pval=None)
  figure_path                   unique PNG path reserved for this code step

Available datasets and columns:
{datasets}

For every correlation-matrix or heatmap request:
- Compute the correlation and p-value matrices in the requested row and column order.
- Call `plot_corr_matrix(...)` with `out_path=figure_path`. Do not create another plot or invent
  another output path.
- Keep the plotting function's defaults, including `vmax=0.40` and `figsize=(8, 7)`, unless the
  user explicitly requests different visual settings.
- When an actual reference p-value matrix is available, pass it as `reference_pval=...`. Do not
  invent reference values. Cells significant in the reference but not the current matrix will
  be marked with `x` in the PNG.
- Assign a compact JSON-serializable summary to `result` containing `subjects_analyzed`,
  `screening_rule` (or "none"), `significant_correlations`, `alpha`, `correction`, and
  `limitations`. Each significant-correlation item must identify its row, column, r, and p.
  Include `lost_from_reference` when applicable.
- Do not return complete correlation or p-value matrices unless the user explicitly requests
  their numeric values.

When the task mentions numbered columns, count them using the CSV's one-based column positions;
column 1 is `subject`, columns 2–8 are survey summaries, and columns 9–17 are task summaries.

For an execution turn, respond with ONLY this JSON shape:
{{
  "action": "execute",
  "phase": "inspect|quality_model|policy_comparison|budget_request|revision",
  "thought": "a concise, auditable rationale for this computation",
  "confidence": 0.0,
  "revision_trigger": null,
  "code": "Python code that assigns the compact JSON-serializable answer to result"
}}

For the final turn, respond with ONLY this JSON shape:
{{
  "action": "final",
  "phase": "conclusion",
  "thought": "why the evidence is sufficient to stop",
  "confidence": 0.0,
  "outcome": {{
    "success": true,
    "final_claim": "conclusion grounded in observed numbers",
    "confidence": 0.0,
    "selection_sensitivity": {{
      "verdict": "STABLE|SELECTION_SENSITIVE|UNDETERMINED",
      "r_spread": null,
      "policies_compared": [],
      "flips_significance": null,
      "flips_sign": null
    }},
    "verification": {{"method": "pending", "result": "pending", "score": null}},
    "failure_type": null,
    "recovery_attempted": false,
    "recovery_successful": null,
    "limitations": [],
    "resolving_measurement": null
  }}
}}

Code rules:
- Do not import, access files/network/environment, or call private/dunder attributes.
- The execution session retains variables from successful prior code blocks. Reuse them when
  useful, but make revisions explicit when an earlier definition was wrong.
- Assign the useful, compact observation to `result`; avoid full participant tables.
- If screening is requested, use only defensible, outcome-independent criteria available in
  `correlates`, report the retained subject count and exact rule, and state that summary data
  cannot establish whether a participant was careless.
- Never select or exclude subjects because doing so creates, removes, or weakens a desired
  correlation or p-value; that is circular analysis.
- On code failure, diagnose the observation and generate corrected code in a revision turn.
- `r_spread` is one number (maximum r minus minimum r), never a range or list.
- `verification.method` must be oracle_match, llm_judge, human, or pending.
- `verification.result` must be correct, incorrect, partial, or pending.
- `failure_type` must be null, code_error, reasoning_error, or hallucination.
"""


def build_system_prompt(*_, **__) -> str:
    return SYSTEM_HEADER.format(datasets=_dataset_description())


def _rated_from_disk() -> list[dict]:
    traces = HERE / "traces"
    out = []
    for path in sorted(traces.glob("*.json"), reverse=True) if traces.exists() else []:
        try:
            item = json.loads(path.read_text())
        except Exception:
            continue
        if item.get("human_rating"):
            out.append(item)
    return out


def execute_code(code: str, history: list | None = None,
                 figure_path: str | None = None, timeout: int = 35) -> dict:
    code = _normalise_generated_code(code)
    try:
        process = subprocess.run(
            [sys.executable, str(HERE / "code_runner.py")],
            input=json.dumps({"code": code, "history": history or [],
                              "figure_path": figure_path}),
            text=True, capture_output=True,
            cwd=HERE, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_type": "TimeoutError",
                "error": f"Generated code exceeded the {timeout}s execution limit."}
    if process.returncode:
        return {"ok": False, "error_type": "RunnerError",
                "error": process.stderr[-2000:] or f"runner exited {process.returncode}"}
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error_type": "ProtocolError",
                "error": f"Runner returned invalid JSON: {process.stdout[-1000:]}"}


def _normalise_generated_code(code: str) -> str:
    """Remove harmless model formatting without changing Python block structure."""
    code = str(code or "")
    fenced = code.strip()
    if fenced.startswith("```") and fenced.endswith("```"):
        lines = fenced.splitlines()
        if len(lines) >= 2:
            lines = lines[1:-1]
            code = "\n".join(lines)
    code = textwrap.dedent(code).strip()
    lines = code.splitlines()
    for _ in range(8):
        try:
            ast.parse("\n".join(lines), mode="exec")
            break
        except IndentationError as exc:
            index = (exc.lineno or 0) - 1
            if "unexpected indent" not in str(exc) or not 0 <= index < len(lines):
                break
            indentation = len(lines[index]) - len(lines[index].lstrip())
            if not 0 < indentation < 4:
                break
            lines[index] = lines[index].lstrip()
    return "\n".join(lines)


def _record_execution(trajectory: Trajectory, action: dict, verbose: bool,
                      history: list | None = None) -> dict:
    phase = action.get("phase") if action.get("phase") in PHASES else "revision"
    thought = str(action.get("thought") or "No rationale supplied.")
    code = _normalise_generated_code(action.get("code") or "")
    action["code"] = code
    confidence = action.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        confidence = None
    if verbose:
        print(f"\n--- code step {len(trajectory.trace) + 1} · {phase} ---")
        print(thought)
        print("\n" + code)
    started = time.time()
    history = history if history is not None else []
    figure_path = str(
        HERE / "figures" / trajectory.trajectory_id
        / f"step-{len(trajectory.trace) + 1:02d}-correlation-matrix.png")
    observation = execute_code(code, history, figure_path)
    elapsed = round(time.time() - started, 3)
    failed = not observation.get("ok", False)
    trajectory.trace.append(TraceStep(
        step_id=len(trajectory.trace), timestamp=now_iso(), phase=phase,
        thought=thought,
        action=Action(type="code_execution", input=code,
                      output=json.dumps(observation, ensure_ascii=False)),
        observation=json.dumps(observation, ensure_ascii=False),
        error=StepError(
            occurred=failed,
            type=observation.get("error_type") if failed else None,
            message=observation.get("error") if failed else None,
        ),
        revision_trigger=action.get("revision_trigger") or None,
        confidence=confidence, wall_time=elapsed,
    ))
    if verbose:
        print("\nObservation:")
        print(json.dumps(observation, ensure_ascii=False, indent=2)[:5000])
    if not failed:
        history.append({"code": code, "figure_path": figure_path})
    return observation


def _fallback_outcome() -> Outcome:
    return Outcome(
        success=False,
        final_claim="The run ended without enough validated evidence for a conclusion.",
        confidence=0.0,
        selection_sensitivity=SelectionSensitivity(verdict="UNDETERMINED"),
        limitations=["The code-generating agent did not return a valid final outcome."],
        resolving_measurement="Run additional validated analysis steps.",
    )


SCRIPTED_ACTIONS = [
    {
        "action": "execute", "phase": "inspect", "confidence": 0.95,
        "thought": "Compute the requested 7 by 9 Spearman matrix using all subjects and render it with the provided plotting function.",
        "code": (
            "row_labels = list(correlates.columns[1:8])\n"
            "column_labels = list(correlates.columns[8:17])\n"
            "corr = np.zeros((len(row_labels), len(column_labels)))\n"
            "pval = np.ones_like(corr)\n"
            "for i, row_name in enumerate(row_labels):\n"
            "    for j, column_name in enumerate(column_labels):\n"
            "        pair = correlates[[row_name, column_name]].dropna()\n"
            "        corr[i, j], pval[i, j] = spearmanr(pair[row_name], pair[column_name])\n"
            "plot_corr_matrix(corr, pval, out_path=figure_path, xticklabels=column_labels, yticklabels=row_labels, title='Significant Spearman correlations')\n"
            "significant_pairs = []\n"
            "for i, row_name in enumerate(row_labels):\n"
            "    for j, column_name in enumerate(column_labels):\n"
            "        if pval[i, j] < 0.05:\n"
            "            significant_pairs.append({'row': row_name, 'column': column_name, 'r': float(corr[i, j]), 'p': float(pval[i, j])})\n"
            "result = {'subjects_analyzed': int(len(correlates)), 'screening_rule': 'none', 'significant_correlations': significant_pairs, 'alpha': 0.05, 'correction': 'none', 'limitations': ['No participant screening was applied.']}"
        ),
    },
]


def _scripted_outcome() -> Outcome:
    return Outcome(
        success=True,
        final_claim="Computed the requested unfiltered Spearman correlation matrix using all subjects.",
        confidence=0.9,
        selection_sensitivity=SelectionSensitivity(
            verdict="UNDETERMINED",
            policies_compared=["all_subjects"],
        ),
        verification=Verification(method="pending", result="pending"),
        limitations=["No participant-screening policy was evaluated."],
        resolving_measurement="Provide a prespecified, outcome-independent screening rule for a screened comparison.",
    )


def run_agent(
    task: str = DEFAULT_TASK, *, live: bool | None = None, max_steps: int = 12,
    model: str | None = None, verbose: bool = True,
    trajectory_id: str | None = None, backend: str = "auto",
) -> Trajectory:
    chosen = "scripted" if live is False else backends.resolve(backend)
    model = model or backends.model_for(chosen)
    trajectory = Trajectory(
        trajectory_id=trajectory_id or f"sciops_{now_iso().replace(':', '')}",
        prompt=task, model=model,
        metadata=TrajectoryMetadata(
            model_version=f"{chosen}:{model}", collection_timestamp=now_iso()),
    )
    started = time.time()
    if verbose:
        print(f"[backend: {chosen}  model: {model}  mode: generated-code ReAct]")

    if chosen == "scripted":
        history: list = []
        for action in SCRIPTED_ACTIONS[:max_steps]:
            _record_execution(trajectory, action, verbose, history)
        trajectory.outcome = _scripted_outcome()
    else:
        history = []
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content":
             f"Research task:\n{task}\n\nBegin with the most informative computation."},
        ]
        for _ in range(max_steps):
            try:
                action = backends.chat_json(chosen, messages, model=model)
            except Exception as exc:
                action = {
                    "action": "execute", "phase": "revision", "confidence": 0.0,
                    "thought": "Record a model response failure.",
                    "code": "result = {'model_error': 'response unavailable'}",
                    "revision_trigger": f"{type(exc).__name__}: {exc}",
                }
                _record_execution(trajectory, action, verbose, history)
                trajectory.trace[-1].error = StepError(
                    occurred=True, type="model_error",
                    message=f"{type(exc).__name__}: {exc}")
                break
            messages.append({"role": "assistant", "content": json.dumps(action)})
            if action.get("action") == "final":
                try:
                    trajectory.outcome = Outcome.model_validate(action.get("outcome"))
                except ValidationError as exc:
                    if verbose:
                        print(f"\nInvalid final outcome: {exc}")
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your final outcome failed schema validation. Correct only the "
                            "types/enums while preserving evidence, then return action=final again.\n"
                            + str(exc)[:5000]
                        ),
                    })
                    continue
                break
            if action.get("action") != "execute":
                action = {
                    "action": "execute", "phase": "revision", "confidence": 0.0,
                    "thought": "Repair an invalid action envelope.",
                    "code": "result = {'error': 'model returned neither execute nor final'}",
                    "revision_trigger": "invalid action envelope",
                }
            observation = _record_execution(trajectory, action, verbose, history)
            messages.append({
                "role": "user",
                "content": "Execution observation:\n"
                + json.dumps(observation, ensure_ascii=False)[:12000]
                + "\n\nChoose another independent computation, revise failed code, or return final JSON.",
            })
        if trajectory.outcome is None:
            trajectory.metadata.max_steps_reached = len(trajectory.trace) >= max_steps
            messages.append({"role": "user", "content":
                             "Stop computing. Return action=final with a schema-valid outcome now."})
            for _ in range(2):
                try:
                    final = backends.chat_json(
                        chosen, messages, model=model, temperature=0.0)
                    trajectory.outcome = Outcome.model_validate(final.get("outcome"))
                    break
                except Exception as exc:
                    messages.extend([
                        {"role": "assistant", "content": json.dumps(final) if 'final' in locals() else "{}"},
                        {"role": "user", "content":
                         "Repair the final JSON to this schema. Error:\n"
                         + str(exc)[:3000] + "\nSchema:\n"
                         + json.dumps(Outcome.model_json_schema())[:7000]},
                    ])
            if trajectory.outcome is None:
                trajectory.outcome = _fallback_outcome()

    trajectory.recompute_metadata()
    trajectory.metadata.wall_time_seconds = round(time.time() - started, 2)
    if verbose and trajectory.outcome:
        print("\n" + "=" * 68)
        print(trajectory.outcome.final_claim)
    return trajectory
