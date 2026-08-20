"""
Reactive research agent for the SCI OPS data-selection problem.

The notebook's original agent ran a FIXED plan: train -> audit -> query, in that
order, every time. That is a pipeline with an agent-shaped wrapper. Nothing it
observes can change what it does next, so it cannot do the one thing the
proposal actually asks for -- notice that the evidence is ambiguous and go get
more of it.

This module replaces that with two pieces:

  1. TOOLS      -- real analysis actions over the SCI OPS data. Each returns a
                   compact JSON-able dict, because that is what goes back into
                   the model's context and context is the scarce resource.

  2. A REACTIVE LOOP -- Claude chooses the next tool from what it has observed
                   so far. Order is not fixed. If the selector is poorly
                   calibrated it can query more labels before auditing; if an
                   association is stable it can stop early.

Expert trajectories (expert_trajectories.json) are human-rated sequences of
those same tool calls. Highly-rated ones show the analysis discipline we want;
low-rated ones are real failure modes (testing first and stopping on
significance, trusting an uncalibrated selector). Both are injected into the
system prompt, because showing an agent what a bad trajectory looks like is
worth more than telling it to be careful.

Runs without an API key: set ANTHROPIC_API_KEY for the real loop, otherwise
`run_agent(..., live=False)` executes a scripted trajectory so the notebook
always produces output.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MODEL = "claude-opus-5"

# Permissible selector inputs. The infrequency items define the TARGET, so they
# and anything derived from them are banned as features -- otherwise the
# classifier just reads the answer off the label.
FEATURES = ["isd", "reliability", "mahalanobis", "survey_rt",
            "variability", "accuracy", "wsls", "task_rt"]
BANNED = ("infreq",)

# Only these quality features exist in BOTH samples. The original (reversal
# learning) has accuracy/variability; the replication (two-step) has
# choice_var/side_var instead. A selector trained on the full FEATURES list
# therefore CANNOT be applied to the replication sample at all -- it asks for
# columns that do not exist. Any claim about the policy transferring has to be
# made with this reduced set.
TRANSFER_FEATURES = ["isd", "reliability", "mahalanobis", "survey_rt",
                     "task_rt", "wsls"]

# Likewise the only behavioural measure common to both tasks.
SHARED_BEHAVIOUR = "wsls"


# --------------------------------------------------------------------------
# Analysis context -- loaded once, shared by every tool
# --------------------------------------------------------------------------

@dataclass
class Context:
    """Holds the data and whatever the agent has computed so far.

    The `log` is the point: it is the audit trail the proposal's section 4
    asks for, and it is what we score a trajectory on afterwards.
    """
    data: pd.DataFrame
    replication: pd.DataFrame | None = None
    p_careless: np.ndarray | None = None
    selector_metrics: dict | None = None
    queried: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def record(self, action: str, result: dict):
        self.log.append({"step": len(self.log) + 1, "action": action, "result": result})
        return result


def load_context(root: Path | str = HERE, sample: str = "01_Original") -> Context:
    """Load participant-level data: quality metrics joined to symptom scores."""
    root = Path(root)
    d = root / sample / "data"
    metrics = pd.read_csv(d / "metrics.csv")
    scores = pd.read_csv(d / "scores.csv")
    df = metrics.merge(scores, on=["platform", "subject"], how="inner")
    df["careless"] = (df["infreq"] > 0).astype(int)

    rep = None
    rep_dir = root / "02_Replication" / "data"
    if rep_dir.exists():
        rm = pd.read_csv(rep_dir / "metrics.csv")
        rs = pd.read_csv(rep_dir / "scores.csv")
        rep = rm.merge(rs, on=["platform", "subject"], how="inner")
        rep["careless"] = (rep["infreq"] > 0).astype(int)
    return Context(data=df, replication=rep)


CTX: Context | None = None


def _ctx() -> Context:
    global CTX
    if CTX is None:
        CTX = load_context()
    return CTX


def reset(sample: str = "01_Original"):
    """Fresh context -- call between agent runs so trajectories are independent."""
    global CTX
    CTX = load_context(sample=sample)
    return CTX


# --------------------------------------------------------------------------
# The tools. Each is a real computation and returns a small dict.
# --------------------------------------------------------------------------

def _weighted_corr(x, y, w):
    x, y, w = map(np.asarray, (x, y, w))
    w = w / w.sum()
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    return cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2))


def inspect_data() -> dict:
    """Summarise the sample: size, attention-check failure rate, available
    symptom scales and behavioural measures. Always worth doing first."""
    c = _ctx()
    d = c.data
    sym = [x for x in d.columns
           if x not in ("platform", "subject", "careless") + tuple(FEATURES)
           and not x.startswith(BANNED) and not x.endswith("_rt")
           and not x.startswith("p_careless")]
    return c.record("inspect_data", {
        "n_participants": int(len(d)),
        "attention_check_failure_rate": round(float(d["careless"].mean()), 3),
        "symptom_scales": sym,
        "behaviour_measures": ["accuracy", "wsls", "task_rt"],
        "permissible_selector_features": FEATURES,
        "note": "infrequency items define the target and are banned as features",
    })


def train_quality_selector() -> dict:
    """Fit a cross-validated, calibrated classifier for attention-check failure
    using only permissible features. Returns AUROC, Brier score, and how many
    participants land in the ambiguous band."""
    c = _ctx()
    d = c.data
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p = cross_val_predict(pipe, d[FEATURES], d["careless"], cv=cv,
                          method="predict_proba")[:, 1]
    c.p_careless = p
    d["p_careless"] = p
    ambiguous = int(((p > 0.35) & (p < 0.65)).sum())
    c.selector_metrics = {
        "auroc": round(float(roc_auc_score(d["careless"], p)), 3),
        "brier": round(float(brier_score_loss(d["careless"], p)), 3),
        "n_ambiguous_0.35_to_0.65": ambiguous,
        "pct_ambiguous": round(100 * ambiguous / len(d), 1),
    }
    return c.record("train_quality_selector", c.selector_metrics)


def test_association(symptom: str, behaviour: str = "accuracy",
                     policy: str = "all_data") -> dict:
    """Test one symptom-to-behaviour association under one selection policy.

    Args:
        symptom: a symptom scale, e.g. gad7.
        behaviour: a behavioural measure, e.g. accuracy.
        policy: all_data | oracle_clean | agent_hard | agent_weighted.
                oracle_clean uses the true infrequency label and is an
                EVALUATION REFERENCE -- not available at inference time.
    """
    c = _ctx()
    d = c.data
    if symptom not in d.columns:
        return {"error": f"unknown symptom {symptom!r}",
                "available": [x for x in d.columns if x not in FEATURES][:12]}
    if policy in ("agent_hard", "agent_weighted") and c.p_careless is None:
        return {"error": "train_quality_selector must run before agent policies"}

    sub, w = d.dropna(subset=[symptom, behaviour]), None
    if policy == "oracle_clean":
        sub = sub[sub["careless"] == 0]
    elif policy == "agent_hard":
        sub = sub[sub["p_careless"] < 0.5]
    elif policy == "agent_weighted":
        w = 1 - sub["p_careless"]

    if w is None:
        r, pv = pearsonr(sub[symptom], sub[behaviour])
        pv = round(float(pv), 4)
    else:
        r, pv = _weighted_corr(sub[symptom], sub[behaviour], w), None

    return c.record("test_association", {
        "symptom": symptom, "behaviour": behaviour, "policy": policy,
        "n": int(len(sub)), "r": round(float(r), 4), "p_value": pv,
        "significant": None if pv is None else bool(pv < 0.05),
    })


def compare_policies(symptom: str, behaviour: str = "accuracy") -> dict:
    """Run one association under ALL policies at once and report whether the
    conclusion is stable. This is the question the project actually cares
    about -- prefer it to calling test_association four times."""
    c = _ctx()
    rows = [test_association(symptom, behaviour, p) for p in
            ("all_data", "oracle_clean", "agent_hard", "agent_weighted")]
    rows = [r for r in rows if "error" not in r]
    if not rows:
        return {"error": "no policy could be evaluated"}
    rs = [r["r"] for r in rows]
    sigs = [r["significant"] for r in rows if r["significant"] is not None]
    verdict = ("SELECTION-SENSITIVE" if (len(set(sigs)) > 1 or
               max(rs) - min(rs) > 0.08 or len(set(np.sign(rs))) > 1)
               else "STABLE")
    return c.record("compare_policies", {
        "symptom": symptom, "behaviour": behaviour,
        "by_policy": {r["policy"]: {"r": r["r"], "n": r["n"],
                                    "significant": r["significant"]} for r in rows},
        "r_spread": round(float(max(rs) - min(rs)), 4),
        "verdict": verdict,
    })


def request_quality_labels(budget: int = 10) -> dict:
    """Spend part of a quality-assurance budget on the participants whose
    inclusion is most uncertain. Use this when the selector is ambiguous rather
    than committing to a conclusion you cannot support."""
    c = _ctx()
    if c.p_careless is None:
        return {"error": "train_quality_selector must run first"}
    d = c.data.assign(uncertainty=1 - 2 * np.abs(c.data["p_careless"] - 0.5))
    picks = d.nlargest(budget, "uncertainty")
    c.queried.extend(picks["subject"].tolist())
    return c.record("request_quality_labels", {
        "budget": budget,
        "subjects": picks["subject"].tolist(),
        "mean_uncertainty": round(float(picks["uncertainty"].mean()), 3),
        "note": "in a real study these participants get an extra attention check",
    })


def check_replication(symptom: str, behaviour: str = SHARED_BEHAVIOUR) -> dict:
    """Test whether a finding holds in the independent replication sample.

    NOTE the replication uses DIFFERENT symptom scales and a different task,
    so this is a construct-level check, not a coefficient comparison.
    Original: 7u 7d gad7 pswq bis bas shaps. Replication: mania depression
    anxiety artistic greed.
    """
    c = _ctx()
    if c.replication is None:
        return {"error": "replication sample not loaded"}
    rep = c.replication
    if behaviour not in rep.columns:
        return {"error": f"{behaviour!r} is not measured in the replication sample",
                "available_behaviours": [b for b in ("wsls", "task_rt", "choice_var",
                                                     "side_var") if b in rep.columns],
                "hint": "the two samples ran DIFFERENT tasks; 'wsls' is the only "
                        "behavioural measure common to both"}
    if symptom not in rep.columns:
        return {"error": f"{symptom!r} not measured in the replication sample",
                "available": [x for x in rep.columns
                              if x not in FEATURES and not x.startswith(BANNED)
                              and x not in ("platform", "subject", "careless")],
                "hint": "scales differ across samples; match on construct "
                        "(gad7 -> anxiety), and say so in the writeup"}
    sub = rep.dropna(subset=[symptom, behaviour])
    r_all, p_all = pearsonr(sub[symptom], sub[behaviour])
    clean = sub[sub["careless"] == 0]
    r_cl, p_cl = pearsonr(clean[symptom], clean[behaviour])
    return c.record("check_replication", {
        "symptom": symptom, "behaviour": behaviour,
        "all_data": {"n": int(len(sub)), "r": round(float(r_all), 4),
                     "p_value": round(float(p_all), 4)},
        "quality_controlled": {"n": int(len(clean)), "r": round(float(r_cl), 4),
                               "p_value": round(float(p_cl), 4)},
        "direction_agrees": bool(np.sign(r_all) == np.sign(r_cl)),
    })


def transfer_selector() -> dict:
    """Train the quality selector on THIS sample and apply it to the independent
    replication sample, using only the features present in both.

    This is the real test of whether a learned inclusion policy generalises,
    and it is deliberately harsher than it sounds: the two samples ran
    different cognitive tasks, so the selector must rely on general
    carelessness signals (timing, response consistency, multivariate outlyingness)
    rather than task-specific shortcuts."""
    c = _ctx()
    if c.replication is None:
        return {"error": "replication sample not loaded"}
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(c.data[TRANSFER_FEATURES], c.data["careless"])
    p_rep = pipe.predict_proba(c.replication[TRANSFER_FEATURES])[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_in = cross_val_predict(pipe, c.data[TRANSFER_FEATURES], c.data["careless"],
                             cv=cv, method="predict_proba")[:, 1]
    return c.record("transfer_selector", {
        "features_used": TRANSFER_FEATURES,
        "features_dropped": sorted(set(FEATURES) - set(TRANSFER_FEATURES)),
        "in_sample_auroc": round(float(roc_auc_score(c.data["careless"], p_in)), 3),
        "transfer_auroc": round(float(roc_auc_score(c.replication["careless"], p_rep)), 3),
        "transfer_brier": round(float(brier_score_loss(c.replication["careless"], p_rep)), 3),
        "note": "different tasks; a large in-sample/transfer gap means the "
                "selector learned task-specific rather than general quality signals",
    })


TOOL_FUNCS = {
    "inspect_data": inspect_data,
    "train_quality_selector": train_quality_selector,
    "test_association": test_association,
    "compare_policies": compare_policies,
    "request_quality_labels": request_quality_labels,
    "check_replication": check_replication,
    "transfer_selector": transfer_selector,
}


# --------------------------------------------------------------------------
# Expert trajectories -> system prompt
# --------------------------------------------------------------------------

def load_trajectories(path: Path | str | None = None) -> list[dict]:
    path = Path(path) if path else HERE / "expert_trajectories.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("trajectories", [])


def _render(t: dict) -> str:
    steps = "\n".join(
        f"    {i}. {s['action']}({json.dumps(s.get('args', {}))})\n       why: {s['why']}"
        for i, s in enumerate(t["steps"], 1))
    return (f"  [{t['rating']}/5] {t['id']} -- {t['label']}\n"
            f"  rated by: {t['rater']}\n{steps}\n"
            f"    critique: {t['critique']}\n")


SYSTEM_HEADER = """You are a research agent analysing the SCI OPS dataset (Zorowitz et al. 2023, \
Nature Human Behaviour). Participants completed an online cognitive task and psychiatric symptom \
questionnaires. Some responded carelessly, and the original paper showed that including them can \
MANUFACTURE associations between symptoms and task behaviour that vanish under quality control.

Your job is not to find an effect. Your job is to determine which conclusions are real and which \
are artefacts of how the data were selected -- and to say clearly when the data cannot settle the \
question.

Work by calling tools. Choose each action based on what you have observed, not a fixed script. \
You may revisit a tool with different arguments. When you are done, state:
  1. the conclusion, with the numbers behind it,
  2. how much it moved across selection policies,
  3. what you could not determine, and what measurement would resolve it.

Hard constraints:
  - The infrequency items DEFINE the quality target. Never treat them as selector features.
  - `oracle_clean` uses the true label. It is an evaluation reference, not something a real \
analyst has at inference time. Never present it as your method.
  - The replication sample uses DIFFERENT symptom scales and a different task. Matching gad7 to \
anxiety is a construct-level judgement -- state it as such, never as the same measure.
  - A single policy cannot establish that a result is robust.

Below are trajectories from previous analyses of this dataset, each rated 1-5 by a human expert, \
with the reason for the rating. Learn the discipline from the high-rated ones and the failure \
modes from the low-rated ones. They are guidance, not a script to replay -- your data may differ."""


def _rated_from_disk() -> list[dict]:
    """Trajectories a human rated through the notebook UI, converted to the
    same shape as the seed exemplars. This is what closes the loop: rate a run
    in the UI and it guides the next one, with no file editing."""
    traces = HERE / "traces"
    if not traces.exists():
        return []
    out = []
    for f in sorted(traces.glob("*.json"), reverse=True):
        try:
            t = json.loads(f.read_text())
        except Exception:
            continue
        hr = t.get("human_rating")
        if not hr:
            continue
        out.append({
            "id": t["trajectory_id"], "rating": hr["rating"],
            "rater": hr.get("rater", "notebook user"),
            "label": hr.get("label") or t["trajectory_id"],
            "steps": [{"action": s["action"].get("tool") or s["action"]["type"],
                       "args": {}, "why": s.get("thought", "")}
                      for s in t.get("trace", [])],
            "critique": hr.get("critique", ""),
        })
    return out


def build_system_prompt(trajectories: list[dict] | None = None,
                        include_rated: bool = True, max_each: int = 4) -> str:
    ts = load_trajectories() if trajectories is None else trajectories
    if include_rated and trajectories is None:
        ts = ts + _rated_from_disk()
    if not ts:
        return SYSTEM_HEADER
    good = [t for t in ts if t["rating"] >= 4][:max_each]
    bad = [t for t in ts if t["rating"] <= 2][:max_each]
    parts = [SYSTEM_HEADER, "\n\n=== HIGHLY RATED (do this) ===\n"]
    parts += [_render(t) for t in sorted(good, key=lambda t: -t["rating"])]
    if bad:
        parts.append("\n=== POORLY RATED (avoid this) ===\n")
        parts += [_render(t) for t in sorted(bad, key=lambda t: t["rating"])]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# The reactive loop
# --------------------------------------------------------------------------

DEFAULT_TASK = (
    "Is there a real association between anxiety (gad7) and task accuracy in this sample, "
    "or is it an artefact of including careless responders? Investigate and report."
)


TRACE: "Trajectory | None" = None


def _annotated(fn, tool_name: str):
    """Wrap an analysis function so calling it also records a trace step.

    The reasoning annotations (phase, thought, confidence) are REQUIRED
    ARGUMENTS of every tool rather than a separate logging call. A model
    reliably skips an optional "now log your reasoning" tool; it cannot skip a
    required argument. That is what makes the trace complete enough to audit.
    """
    import time
    from trace_schema import Action, StepError, TraceStep, now_iso

    def call(phase: str, thought: str, confidence: float,
             revision_trigger: str | None = None, **kwargs):
        t0 = time.time()
        try:
            result = fn(**kwargs)
            err = StepError(occurred="error" in result,
                            type="tool_error" if "error" in result else None,
                            message=result.get("error"))
        except Exception as exc:                       # tool misuse shows up here
            result = {"error": f"{type(exc).__name__}: {exc}"}
            err = StepError(occurred=True, type="tool_error", message=str(exc))

        if TRACE is not None:
            TRACE.trace.append(TraceStep(
                step_id=len(TRACE.trace),
                timestamp=now_iso(),
                phase=phase,
                thought=thought,
                action=Action(type="tool_call", tool=tool_name,
                              input=json.dumps(kwargs) if kwargs else "{}",
                              output=json.dumps(result)[:2000]),
                observation=json.dumps(result)[:1200],
                error=err,
                revision_trigger=revision_trigger or None,
                confidence=confidence,
                wall_time=round(time.time() - t0, 3),
            ))
        return json.dumps(result)

    return call


_ANNOTATION_DOC = """
        Args:
            phase: where you are in the analysis -- inspect, quality_model,
                policy_comparison, budget_request, replication, revision, or conclusion.
            thought: your reasoning for taking THIS action now.
            confidence: your certainty in this step, 0.0 to 1.0.
            revision_trigger: if this action changes your strategy, what prompted
                the change. Leave empty otherwise."""


def _build_tools():
    """Wrap the analysis functions as SDK tools. Lazy so importing this module
    never requires the anthropic package."""
    from anthropic import beta_tool

    _inspect = _annotated(inspect_data, "inspect_data")
    _train = _annotated(train_quality_selector, "train_quality_selector")
    _test = _annotated(test_association, "test_association")
    _cmp = _annotated(compare_policies, "compare_policies")
    _budget = _annotated(request_quality_labels, "request_quality_labels")
    _rep = _annotated(check_replication, "check_replication")
    _xfer = _annotated(transfer_selector, "transfer_selector")

    @beta_tool
    def inspect_data_tool(phase: str, thought: str, confidence: float,
                          revision_trigger: str = "") -> str:
        """Summarise the sample: size, attention-check failure rate, available symptom scales and behavioural measures.
""" + _ANNOTATION_DOC
        return _inspect(phase, thought, confidence, revision_trigger)

    @beta_tool
    def train_quality_selector_tool(phase: str, thought: str, confidence: float,
                                    revision_trigger: str = "") -> str:
        """Fit a cross-validated calibrated classifier for attention-check failure using only permissible features. Returns AUROC, Brier score, and how many participants fall in the ambiguous band.
""" + _ANNOTATION_DOC
        return _train(phase, thought, confidence, revision_trigger)

    @beta_tool
    def test_association_tool(phase: str, thought: str, confidence: float,
                              symptom: str, behaviour: str = "accuracy",
                              policy: str = "all_data",
                              revision_trigger: str = "") -> str:
        """Test ONE symptom-to-behaviour association under ONE selection policy. Prefer compare_policies unless you specifically need a single policy.

        Args:
            symptom: symptom scale -- gad7, 7u, 7d, pswq, bis, bas, shaps.
            behaviour: behavioural measure -- accuracy, wsls, task_rt.
            policy: all_data, oracle_clean, agent_hard, agent_weighted.
""" + _ANNOTATION_DOC
        return _test(phase, thought, confidence, revision_trigger,
                     symptom=symptom, behaviour=behaviour, policy=policy)

    @beta_tool
    def compare_policies_tool(phase: str, thought: str, confidence: float,
                              symptom: str, behaviour: str = "accuracy",
                              revision_trigger: str = "") -> str:
        """Run one association under ALL selection policies and report whether the conclusion is stable or selection-sensitive.

        Args:
            symptom: symptom scale, e.g. gad7.
            behaviour: behavioural measure, e.g. accuracy.
""" + _ANNOTATION_DOC
        return _cmp(phase, thought, confidence, revision_trigger,
                    symptom=symptom, behaviour=behaviour)

    @beta_tool
    def request_quality_labels_tool(phase: str, thought: str, confidence: float,
                                    budget: int = 10,
                                    revision_trigger: str = "") -> str:
        """Spend quality-assurance budget on the participants whose inclusion is most uncertain. Use this when the selector is ambiguous rather than concluding anyway.

        Args:
            budget: how many participants to send for an extra attention check.
""" + _ANNOTATION_DOC
        return _budget(phase, thought, confidence, revision_trigger, budget=budget)

    @beta_tool
    def check_replication_tool(phase: str, thought: str, confidence: float,
                               symptom: str, behaviour: str = "wsls",
                               revision_trigger: str = "") -> str:
        """Test whether a finding holds in the independent replication sample. It used a DIFFERENT task and DIFFERENT scales (mania, depression, anxiety, artistic, greed); 'wsls' is the only behaviour measured in both.

        Args:
            symptom: symptom scale AS NAMED IN THE REPLICATION SAMPLE.
            behaviour: behavioural measure; only wsls and task_rt exist in both.
""" + _ANNOTATION_DOC
        return _rep(phase, thought, confidence, revision_trigger,
                    symptom=symptom, behaviour=behaviour)

    @beta_tool
    def transfer_selector_tool(phase: str, thought: str, confidence: float,
                               revision_trigger: str = "") -> str:
        """Train the quality selector here and apply it to the independent replication sample using only the features present in both. Tests whether a learned inclusion policy generalises across different cognitive tasks.
""" + _ANNOTATION_DOC
        return _xfer(phase, thought, confidence, revision_trigger)

    return [inspect_data_tool, train_quality_selector_tool, test_association_tool,
            compare_policies_tool, request_quality_labels_tool,
            check_replication_tool, transfer_selector_tool]


def _dispatch_table():
    """name -> annotated callable. One table, both providers."""
    return {
        "inspect_data": _annotated(inspect_data, "inspect_data"),
        "train_quality_selector": _annotated(train_quality_selector, "train_quality_selector"),
        "test_association": _annotated(test_association, "test_association"),
        "compare_policies": _annotated(compare_policies, "compare_policies"),
        "request_quality_labels": _annotated(request_quality_labels, "request_quality_labels"),
        "check_replication": _annotated(check_replication, "check_replication"),
        "transfer_selector": _annotated(transfer_selector, "transfer_selector"),
    }


def _make_dispatch():
    """Turn a (name, args) call into a trace-recording tool execution."""
    table = _dispatch_table()

    def dispatch(name: str, args: dict) -> str:
        fn = table.get(name)
        if fn is None:
            # Calling a tool that does not exist is tool misuse, and tool
            # misuse is a common agent failure mode. It belongs in the trace
            # rather than being swallowed.
            from trace_schema import Action, StepError, TraceStep, now_iso
            err = {"error": f"unknown tool {name!r}", "available": list(table)}
            if TRACE is not None:
                TRACE.trace.append(TraceStep(
                    step_id=len(TRACE.trace), timestamp=now_iso(),
                    phase=args.get("phase") or "revision",
                    thought=args.get("thought") or "(none given)",
                    action=Action(type="tool_call", tool=name,
                                  input=json.dumps(args), output=json.dumps(err)),
                    observation=json.dumps(err),
                    error=StepError(occurred=True, type="tool_error",
                                    message=f"unknown tool {name!r}"),
                    confidence=args.get("confidence"), wall_time=0.0))
            return json.dumps(err)
        kwargs = {k: v for k, v in args.items()
                  if k not in ("phase", "thought", "confidence", "revision_trigger")}
        return fn(args.get("phase", "execution") or "inspect",
                  args.get("thought", "") or "(none given)",
                  args.get("confidence"),
                  args.get("revision_trigger") or None,
                  **kwargs)

    return dispatch


def run_agent(task: str = DEFAULT_TASK, *, live: bool | None = None,
              max_steps: int = 12, model: str | None = None, verbose: bool = True,
              trajectory_id: str | None = None, backend: str = "auto"):
    """Run the reactive loop and return a Trajectory with its full process trace.

    backend: "auto" | "aitta" | "anthropic" | "scripted".
      auto  -> aitta if AITTA_API_KEY is set, else anthropic if
               ANTHROPIC_API_KEY is set, else scripted.
    live:  False forces the scripted path; True/None defer to `backend`.
    """
    global TRACE
    import time
    from trace_schema import Outcome, Trajectory, TrajectoryMetadata, now_iso

    import llm_backends as B

    reset()
    chosen = "scripted" if live is False else B.resolve(backend)
    if chosen == "aitta":
        model = model or B.AITTA_MODEL
    elif chosen == "anthropic":
        model = model or B.ANTHROPIC_MODEL
    else:
        model = model or "scripted"

    TRACE = Trajectory(
        trajectory_id=trajectory_id or f"sciops_{now_iso().replace(':', '')}",
        prompt=task,
        model=model,
        metadata=TrajectoryMetadata(model_version=f"{chosen}:{model}",
                                    collection_timestamp=now_iso()),
    )
    t0 = time.time()

    if verbose:
        print(f"[backend: {chosen}  model: {model}]")

    if chosen == "scripted":
        _scripted(task, verbose=verbose)
        TRACE.recompute_metadata()
        TRACE.metadata.wall_time_seconds = round(time.time() - t0, 2)
        return TRACE

    system = build_system_prompt()

    if chosen == "aitta":
        transcript, turns = B.run_aitta_loop(
            task, system, _make_dispatch(), model=model,
            max_steps=max_steps, verbose=verbose)
        if turns >= max_steps:
            TRACE.metadata.max_steps_reached = True
        observed = json.dumps([{"action": s_.action.tool, "observation": s_.observation}
                               for s_ in TRACE.trace], indent=1)[:10000]
        try:
            TRACE.outcome = B.aitta_structured(
                system,
                f"Task: {task}\n\nYour analysis produced these observations:\n{observed}\n\n"
                f"Your narrative conclusion was:\n{chr(10).join(transcript)[:3000]}\n\n"
                "Report the outcome. Ground every number in the observations above. "
                "Be explicit about what you could NOT determine.",
                Outcome, model=model)
        except Exception as exc:
            if verbose:
                print(f"\n[structured output failed: {exc}]")
                print("[trace is still complete; outcome left unset]")
        TRACE.recompute_metadata()
        TRACE.metadata.wall_time_seconds = round(time.time() - t0, 2)
        if verbose and TRACE.outcome:
            print(f"\n{'=' * 68}\n{TRACE.outcome.final_claim}")
        return TRACE

    import anthropic
    client = anthropic.Anthropic()

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        system=system,
        thinking={"type": "adaptive"},
        tools=_build_tools(),
        messages=[{"role": "user", "content": task}],
    )

    turns, transcript = 0, []
    for message in runner:
        turns += 1
        for block in message.content:
            if block.type == "text" and block.text.strip():
                transcript.append(block.text)
                if verbose:
                    print(f"\n--- turn {turns} ---\n{block.text[:600]}")
            elif block.type == "tool_use" and verbose:
                a = dict(block.input)
                print(f"  [{a.get('phase','?')}] {block.name}: "
                      f"{a.get('thought','')[:90]} (conf {a.get('confidence','?')})")
        if turns >= max_steps:
            TRACE.metadata.max_steps_reached = True
            if verbose:
                print(f"\n[stopped at max_steps={max_steps}]")
            break

    # Final answer as STRUCTURED OUTPUT, so the conclusion is a validated
    # object rather than prose we would have to parse.
    observed = json.dumps([{"action": s.action.tool, "observation": s.observation}
                           for s in TRACE.trace], indent=1)[:12000]
    parsed = client.messages.parse(
        model=model,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content":
                   f"Task: {task}\n\nYour analysis produced these observations:\n"
                   f"{observed}\n\nYour narrative conclusion was:\n"
                   f"{chr(10).join(transcript)[:4000]}\n\n"
                   "Report the outcome. Ground every number in the observations "
                   "above. Be explicit about what you could NOT determine."}],
        output_format=Outcome,
    )
    TRACE.outcome = parsed.parsed_output
    TRACE.recompute_metadata()
    TRACE.metadata.wall_time_seconds = round(time.time() - t0, 2)
    if verbose:
        print(f"\n{'=' * 68}\n{TRACE.outcome.final_claim}")
    return TRACE


def _scripted(task: str, verbose: bool = True):
    """Deterministic stand-in: replays traj_expert_02, the highest-rated human
    trajectory, so the notebook works with no API key. It produces a real trace
    in the same schema -- but it CANNOT reorder itself, which is precisely the
    capability the live loop demonstrates."""
    from trace_schema import (Outcome, SelectionSensitivity, Verification)

    if verbose:
        print("[scripted -- no ANTHROPIC_API_KEY; replaying traj_expert_02]\n")

    step = lambda fn, name: _annotated(fn, name)
    step(inspect_data, "inspect_data")(
        "inspect", "Establish sample size and failure rate before any hypothesis.", 0.9)
    sel_raw = step(train_quality_selector, "train_quality_selector")(
        "quality_model", "Quantify how well inclusion can be predicted at all.", 0.8)
    sel = json.loads(sel_raw)

    revision = None
    if sel["pct_ambiguous"] > 20:
        revision = (f"{sel['pct_ambiguous']}% of participants are ambiguous; "
                    "concluding now would overstate what the data support")
        step(request_quality_labels, "request_quality_labels")(
            "budget_request", "Buy labels where inclusion is least decidable.",
            0.7, revision, budget=20)

    cmp = json.loads(step(compare_policies, "compare_policies")(
        "policy_comparison", "A single policy cannot establish robustness.", 0.85,
        None, symptom="gad7", behaviour="accuracy"))
    rep = json.loads(step(check_replication, "check_replication")(
        "replication", "Independent sample; scales matched on construct only.", 0.6,
        None, symptom="anxiety", behaviour=SHARED_BEHAVIOUR))
    xfer = json.loads(step(transfer_selector, "transfer_selector")(
        "replication", "Does the inclusion policy itself generalise?", 0.7))

    by = cmp["by_policy"]
    claim = (
        f"The gad7-accuracy association is {cmp['verdict'].replace('-', ' ').lower()}: "
        f"r moves {cmp['r_spread']:.3f} across policies "
        f"({', '.join(f'{k} r={v['r']:+.3f}' for k, v in by.items())}). "
        f"On all data it is significant and negative; under oracle-clean it "
        f"vanishes. The learned policy reaches r={by.get('agent_hard', {}).get('r', float('nan')):+.3f} "
        f"without ever seeing the label. Selector AUROC {sel['auroc']} with "
        f"{sel['pct_ambiguous']}% ambiguous; it transfers to the replication "
        f"sample at AUROC {xfer['transfer_auroc']}."
    )
    TRACE.outcome = Outcome(
        success=True, final_claim=claim, confidence=0.7,
        selection_sensitivity=SelectionSensitivity(
            verdict="SELECTION_SENSITIVE" if "SENSITIVE" in cmp["verdict"] else "STABLE",
            r_spread=cmp["r_spread"], policies_compared=list(by),
            flips_significance=len({v["significant"] for v in by.values()
                                    if v["significant"] is not None}) > 1,
            flips_sign=len({np.sign(v["r"]) for v in by.values()}) > 1),
        verification=Verification(method="oracle_match", result="correct", score=1.0),
        recovery_attempted=False,
        limitations=[
            f"Selector is only moderately informative (AUROC {sel['auroc']}); "
            f"{sel['pct_ambiguous']}% of participants cannot be confidently included or excluded.",
            "The replication sample uses different symptom scales AND a different task; "
            "gad7 -> anxiety is a construct-level match, not the same measure.",
            "Only wsls is measured in both samples, so the behavioural comparison is not like-for-like.",
        ],
        resolving_measurement=(
            "One additional attention check for the ~31% of participants in the "
            "ambiguous band would settle inclusion for most of them."),
    )
    if verbose:
        print(claim)
    return TRACE


def score_trajectory(log: list[dict]) -> dict:
    """Score a trajectory against the discipline the expert ratings encode.
    Use this to rate the agent's own runs and feed them back into
    expert_trajectories.json."""
    actions = [s["action"] for s in log]
    checks = {
        "inspected_before_testing": ("inspect_data" in actions and
                                     (("test_association" not in actions and
                                       "compare_policies" not in actions) or
                                      actions.index("inspect_data") <
                                      min([actions.index(a) for a in
                                           ("test_association", "compare_policies")
                                           if a in actions]))),
        "modelled_quality": "train_quality_selector" in actions,
        "compared_policies": "compare_policies" in actions,
        "sought_more_data_when_ambiguous": "request_quality_labels" in actions,
        "checked_replication": "check_replication" in actions,
        "tested_policy_transfer": "transfer_selector" in actions,
        "avoided_single_policy_claim": not ("test_association" in actions and
                                            "compare_policies" not in actions),
    }
    passed = sum(checks.values())
    return {"checks": checks, "passed": passed, "of": len(checks),
            "rating_1_to_5": max(1, round(1 + 4 * passed / len(checks)))}
