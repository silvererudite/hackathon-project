"""
Interactive notebook UI. Nothing here requires editing a JSON file by hand.

Two panels:

  ask()     -- type the research question, pick settings, hit Run. The agent
               executes and the full trajectory, with its process trace, is
               kept in memory.

  rate(t)   -- read the trace step by step, score it 1-5 with a slider, write
               a critique, hit Save. The rated trajectory is written to
               traces/ and immediately becomes part of the system prompt for
               the next run.

That second loop is the point. The agent's guidance is human ratings of real
trajectories, and a person supplies them through a form rather than by editing
a config file. Rate a few runs and the prompt changes under you.
"""
from __future__ import annotations

import json
from pathlib import Path

import ipywidgets as widgets
from IPython.display import HTML, clear_output, display

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
TRACES.mkdir(exist_ok=True)

_LAST: dict = {}


# ---------------------------------------------------------------- persistence

def save_trajectory(traj) -> Path:
    path = TRACES / f"{traj.trajectory_id}.json"
    path.write_text(traj.model_dump_json(indent=1))
    return path


def load_rated(min_rating: int = 1) -> list:
    """Every human-rated trajectory on disk, newest first."""
    from trace_schema import Trajectory
    out = []
    for p in sorted(TRACES.glob("*.json"), reverse=True):
        try:
            t = Trajectory.model_validate_json(p.read_text())
        except Exception:
            continue
        if t.human_rating and t.human_rating.rating >= min_rating:
            out.append(t)
    return out


# ---------------------------------------------------------------- rendering

def _esc(x) -> str:
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def trace_html(traj) -> str:
    rows = []
    for s in traj.trace:
        bad = s.error.occurred
        rev = (f'<div style="color:#b45309;font-size:11px;margin-top:3px">'
               f'REVISION: {_esc(s.revision_trigger)}</div>') if s.revision_trigger else ""
        err = (f'<div style="color:#dc2626;font-size:11px">ERROR '
               f'({_esc(s.error.type)}): {_esc(s.error.message)}</div>') if bad else ""
        conf = f"{s.confidence:.2f}" if s.confidence is not None else "—"
        rows.append(f"""
        <div style="border-left:3px solid {'#dc2626' if bad else '#2563eb'};
                    padding:6px 10px;margin:6px 0;background:rgba(127,127,127,.06)">
          <div style="font-size:11px;opacity:.75">
            step {s.step_id} &middot; <b>{_esc(s.phase)}</b> &middot;
            {_esc(s.action.tool or s.action.type)} &middot; conf {conf} &middot;
            {s.wall_time or 0:.2f}s
          </div>
          <div style="margin:4px 0;font-style:italic">{_esc(s.thought)}</div>
          <div style="font-size:11px;font-family:monospace;opacity:.8">
            {_esc((s.observation or '')[:260])}</div>
          {err}{rev}
        </div>""")
    m = traj.metadata
    o = traj.outcome
    head = (f"<b>{_esc(traj.trajectory_id)}</b><br>"
            f"<span style='font-size:12px;opacity:.8'>{m.total_steps} steps &middot; "
            f"{m.total_tool_calls} tool calls &middot; {m.total_failures} errors &middot; "
            f"{m.total_revisions} revisions &middot; {m.wall_time_seconds}s &middot; "
            f"model {_esc(m.model_version)}</span>")
    tail = ""
    if o:
        lim = "".join(f"<li>{_esc(x)}</li>" for x in o.limitations)
        tail = f"""
        <div style="border:1px solid rgba(127,127,127,.4);padding:8px;margin-top:8px">
          <b>{_esc(o.selection_sensitivity.verdict)}</b>
          (r spread {o.selection_sensitivity.r_spread}) &middot;
          confidence {o.confidence:.2f}
          <div style="margin:6px 0">{_esc(o.final_claim)}</div>
          {'<b>Could not determine:</b><ul>' + lim + '</ul>' if lim else ''}
          {'<b>Would resolve it:</b> ' + _esc(o.resolving_measurement) if o.resolving_measurement else ''}
        </div>"""
    return head + "".join(rows) + tail


# ---------------------------------------------------------------- ask panel

DEFAULT_Q = ("Is the association between anxiety (gad7) and task accuracy real, "
             "or an artefact of including careless responders? "
             "Then check whether your inclusion policy transfers to the replication sample.")

PRESETS = {
    "gad7 -> accuracy (the known spurious one)": DEFAULT_Q,
    "7u -> accuracy (survives filtering)":
        "Does the 7u-accuracy association survive quality control? Compare all policies.",
    "bas -> accuracy (negative control)":
        "Test bas against accuracy as a negative control. Report honestly if there is nothing there.",
    "Screen every symptom scale":
        "Screen all symptom scales against accuracy. Which associations are selection-sensitive "
        "and which are stable? Be explicit about multiplicity.",
    "(write my own)": "",
}


def ask(on_done=None):
    """Question panel. Returns the container; the Trajectory lands in ui.last()."""
    import sciops_agent as agent

    preset = widgets.Dropdown(options=list(PRESETS), value=list(PRESETS)[0],
                              description="preset:", layout=widgets.Layout(width="99%"),
                              style={"description_width": "70px"})
    question = widgets.Textarea(value=DEFAULT_Q, placeholder="Ask a research question…",
                                layout=widgets.Layout(width="99%", height="90px"),
                                description="question:", style={"description_width": "70px"})
    import llm_backends as B
    auto = B.resolve()
    backend = widgets.Dropdown(
        options=[(f"auto  (-> {auto})", "auto"),
                 ("aitta  (CSC, needs AITTA_API_KEY)", "aitta"),
                 ("anthropic  (needs ANTHROPIC_API_KEY)", "anthropic"),
                 ("scripted  (no key)", "scripted")],
        value="auto", description="backend:", style={"description_width": "70px"},
        layout=widgets.Layout(width="46%"))
    steps = widgets.IntSlider(value=12, min=4, max=25, description="max steps:",
                              style={"description_width": "70px"})
    run = widgets.Button(description="Run agent", button_style="primary", icon="play")
    out = widgets.Output()

    def _sync(change):
        if PRESETS[change["new"]]:
            question.value = PRESETS[change["new"]]
    preset.observe(_sync, names="value")

    def _run(_):
        run.disabled, run.description = True, "Running…"
        with out:
            clear_output()
            try:
                traj = agent.run_agent(task=question.value.strip() or DEFAULT_Q,
                                       backend=backend.value, max_steps=steps.value)
                _LAST["trajectory"] = traj
                display(HTML(trace_html(traj)))
                print("\nRate this run with:  ui.rate()")
                if on_done:
                    on_done(traj)
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")
                if backend.value != "scripted":
                    print("\nTip: pick backend 'scripted' to run without any API key.")
            finally:
                run.disabled, run.description = False, "Run agent"

    run.on_click(_run)
    box = widgets.VBox([widgets.HTML("<h4 style='margin:2px'>Ask the agent</h4>"),
                        preset, question, widgets.HBox([backend, steps]), run, out])
    display(box)
    return box


def last():
    return _LAST.get("trajectory")


# ---------------------------------------------------------------- rate panel

def rate(traj=None):
    """Rating panel: read the trace, score it, save. No JSON editing."""
    from trace_schema import HumanRating, now_iso
    traj = traj or last()
    if traj is None:
        print("No trajectory yet — run ui.ask() first.")
        return

    stars = widgets.IntSlider(value=3, min=1, max=5, description="rating:",
                              style={"description_width": "80px"})
    verdict = widgets.HTML()
    rater = widgets.Text(value="", placeholder="your name / role",
                         description="rater:", style={"description_width": "80px"},
                         layout=widgets.Layout(width="60%"))
    label = widgets.Text(value="", placeholder="short name for this trajectory",
                         description="label:", style={"description_width": "80px"},
                         layout=widgets.Layout(width="90%"))
    critique = widgets.Textarea(
        value="", placeholder="What is the rating FOR? Be specific — this text is "
                              "shown to the agent verbatim on later runs.",
        description="critique:", style={"description_width": "80px"},
        layout=widgets.Layout(width="99%", height="90px"))
    save = widgets.Button(description="Save rating", button_style="success", icon="check")
    out = widgets.Output()

    def _v(change=None):
        r = stars.value
        txt, col = (("shown as an EXEMPLAR to imitate", "#16a34a") if r >= 4 else
                    ("shown as an ANTI-PATTERN to avoid", "#dc2626") if r <= 2 else
                    ("recorded, but not injected into the prompt", "#a16207"))
        verdict.value = f"<span style='color:{col}'>{r}/5 — {txt}</span>"
    stars.observe(_v, names="value"); _v()

    def _save(_):
        traj.human_rating = HumanRating(
            rating=stars.value, rater=rater.value or "unnamed",
            label=label.value or traj.trajectory_id, critique=critique.value,
            rated_at=now_iso())
        path = save_trajectory(traj)
        with out:
            clear_output()
            n = len(load_rated())
            print(f"saved -> {path.relative_to(HERE)}")
            print(f"{n} rated trajector{'y' if n == 1 else 'ies'} on disk; "
                  f"the next run's system prompt now includes this one.")

    save.on_click(_save)
    box = widgets.VBox([
        widgets.HTML("<h4 style='margin:2px'>Rate this trajectory</h4>"),
        widgets.HTML(trace_html(traj)),
        stars, verdict, rater, label, critique, save, out])
    display(box)
    return box


def browse():
    """Everything rated so far."""
    rated = load_rated()
    if not rated:
        print("No rated trajectories yet. Run ui.ask() then ui.rate().")
        return
    rows = "".join(
        f"<tr><td>{t.human_rating.rating}/5</td><td>{_esc(t.human_rating.label)}</td>"
        f"<td>{t.metadata.total_steps}</td><td>{t.metadata.total_revisions}</td>"
        f"<td>{_esc(t.outcome.selection_sensitivity.verdict) if t.outcome else '—'}</td>"
        f"<td style='font-size:11px'>{_esc(t.human_rating.critique[:110])}</td></tr>"
        for t in sorted(rated, key=lambda t: -t.human_rating.rating))
    display(HTML(f"""<table style="font-size:12px;border-collapse:collapse">
      <tr style="text-align:left"><th>rating</th><th>label</th><th>steps</th>
      <th>revisions</th><th>verdict</th><th>critique</th></tr>{rows}</table>"""))
    return rated
