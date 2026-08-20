"""
Process-trace schema for the agent loop.

Scoring only an agent's FINAL OUTPUT cannot tell systematic reasoning apart
from a lucky guess, cannot localise where an analysis went wrong, and cannot be
audited. Two runs here can both conclude "the gad7 association is
selection-sensitive" while one compared four policies and bought extra
attention checks and the other guessed. The conclusion field cannot
distinguish them; a per-step trace can.

So every step records nine fields:

    step_id, timestamp, phase, thought, action{type,tool,input,output},
    observation, error{occurred,type,message}, revision_trigger, confidence

plus a trajectory-level `outcome` and `metadata` block. Errors are classified
as tool_misuse | reasoning_error | hallucination, which separates "called the
tool wrong" from "reasoned wrong" -- different problems needing different
fixes.

Two fields are specific to this project:

  * `phase` uses a data-selection vocabulary -- inspect, quality_model,
    policy_comparison, budget_request, replication, revision, conclusion --
    rather than a generic discovery one. PHASE_MAP translates to the generic
    names if traces ever need to be compared against other corpora.
  * `outcome.selection_sensitivity` records whether a claim survives
    alternative defensible data selections. That is this project's whole
    question and nothing else in the schema captures it.

`human_rating` is filled in through the notebook UI, not by hand-editing JSON.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- vocabularies

Phase = Literal[
    "inspect",             # look at the data before forming any expectation
    "quality_model",       # model which observations are trustworthy
    "policy_comparison",   # test the claim under alternative selections
    "budget_request",      # buy more information instead of concluding
    "replication",         # independent sample / transfer
    "revision",            # strategy change after an observation
    "conclusion",          # state the claim and its limits
]

# Generic phase names, for comparing traces against other corpora.
PHASE_MAP = {
    "inspect": "literature_review",
    "quality_model": "experiment_design",
    "policy_comparison": "execution",
    "budget_request": "experiment_design",
    "replication": "analysis",
    "revision": "revision",
    "conclusion": "conclusion",
}

FailureType = Literal["tool_misuse", "reasoning_error", "hallucination"]
ActionType = Literal["tool_call", "reasoning", "conclude"]


# ---------------------------------------------------------------- step

class Action(BaseModel):
    type: ActionType
    tool: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None


class StepError(BaseModel):
    occurred: bool = False
    type: Optional[str] = None       # tool_error | api_error | parse_error | ...
    message: Optional[str] = None


class TraceStep(BaseModel):
    """One step: the nine trace fields, plus wall_time."""
    step_id: int
    timestamp: str
    phase: Phase
    thought: str = Field(description="the agent's reasoning at this step")
    action: Action
    observation: Optional[str] = None
    error: StepError = Field(default_factory=StepError)
    revision_trigger: Optional[str] = Field(
        default=None, description="what prompted a strategy change; null if none")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="self-reported certainty")
    wall_time: Optional[float] = None


# ---------------------------------------------------------------- outcome

class Verification(BaseModel):
    method: Literal["oracle_match", "llm_judge", "human", "pending"] = "pending"
    result: Literal["correct", "incorrect", "partial", "pending"] = "pending"
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SelectionSensitivity(BaseModel):
    """A claim in this project is not just right or wrong -- it is stable or
    not stable across defensible ways of choosing the data."""
    verdict: Literal["STABLE", "SELECTION_SENSITIVE", "UNDETERMINED"]
    r_spread: Optional[float] = Field(
        default=None, description="max-min effect size across policies")
    policies_compared: list[str] = Field(default_factory=list)
    flips_significance: Optional[bool] = None
    flips_sign: Optional[bool] = None


class Outcome(BaseModel):
    """What the agent concluded. This is the structured-output target."""
    success: Optional[bool] = None
    final_claim: str = Field(
        description="the concluding statement, with the numbers behind it")
    confidence: float = Field(ge=0.0, le=1.0)
    selection_sensitivity: SelectionSensitivity
    verification: Verification = Field(default_factory=Verification)
    failure_type: Optional[FailureType] = None
    recovery_attempted: bool = False
    recovery_successful: Optional[bool] = None
    limitations: list[str] = Field(
        default_factory=list,
        description="what this analysis could NOT determine")
    resolving_measurement: Optional[str] = Field(
        default=None,
        description="what additional measurement would settle the open question")


# ---------------------------------------------------------------- trajectory

class TrajectoryMetadata(BaseModel):
    total_steps: int = 0
    total_tool_calls: int = 0
    total_failures: int = 0
    total_revisions: int = 0
    wall_time_seconds: float = 0.0
    max_steps_reached: bool = False
    model_version: Optional[str] = None
    collection_timestamp: Optional[str] = None


class HumanRating(BaseModel):
    """Human-in-the-loop rating, collected through the notebook UI."""
    rating: int = Field(ge=1, le=5)
    rater: str = ""
    label: str = ""
    critique: str = ""
    rated_at: Optional[str] = None


class Trajectory(BaseModel):
    trajectory_id: str
    task_id: str = "sciops_selection"
    domain: str = "psychiatry_behavioural"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    prompt: str
    model: Optional[str] = None
    trace: list[TraceStep] = Field(default_factory=list)
    outcome: Optional[Outcome] = None
    metadata: TrajectoryMetadata = Field(default_factory=TrajectoryMetadata)
    human_rating: Optional[HumanRating] = None

    def recompute_metadata(self):
        m = self.metadata
        m.total_steps = len(self.trace)
        m.total_tool_calls = sum(1 for s in self.trace if s.action.type == "tool_call")
        m.total_failures = sum(1 for s in self.trace if s.error.occurred)
        m.total_revisions = sum(1 for s in self.trace if s.revision_trigger)
        m.wall_time_seconds = round(sum(s.wall_time or 0 for s in self.trace), 2)
        return self

    def summary(self) -> str:
        m = self.metadata
        rating = f"  human {self.human_rating.rating}/5" if self.human_rating else ""
        sens = self.outcome.selection_sensitivity.verdict if self.outcome else "-"
        return (f"{self.trajectory_id}: {m.total_steps} steps, "
                f"{m.total_tool_calls} tool calls, {m.total_failures} errors, "
                f"{m.total_revisions} revisions | {sens}{rating}")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
