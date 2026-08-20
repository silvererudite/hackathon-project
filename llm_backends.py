"""
Pluggable LLM backends for the reactive loop.

Two providers, one tool set:

  aitta      CSC's hosted inference for the summer school. OpenAI-COMPATIBLE,
             so it uses the `openai` client with a custom base_url -- not the
             Anthropic SDK. This is the default when AITTA_API_KEY is set,
             because it is the key you actually have and it needs no GPU
             allocation or Slurm job.
  anthropic  Claude via the Anthropic SDK's tool runner.
  scripted   No API at all; replays the best-rated human trajectory.

TOOL_SPECS below is the single source of truth for the tool surface. Both
providers are derived from it, so adding a tool means editing one list rather
than two provider-specific definitions that drift apart.

Aitta docs: https://github.com/marlon-tobaben/lumi-aif-ellis-summer-school-2026
Token:      https://aitta-auth.csc.fi/myToken   ->   export AITTA_API_KEY=...
"""
from __future__ import annotations

import json
import os
from typing import Callable

AITTA_BASE_URL = "https://aitta-api.csc.fi/openai/v1"

# gpt-oss-120b is the model the summer-school repo documents, and it is the
# right one here for a specific reason: it accepts `tools`/`tool_choice` on
# /v1/chat/completions. A model without function calling cannot drive this
# loop at all -- it could only narrate what it would do.
AITTA_MODEL = "openai/gpt-oss-120b"

# Ranked fallbacks, best-first, from what Aitta actually serves (checked
# 2026-08-20). Only chat models with a real function-calling implementation are
# listed -- the loop cannot run without one. Probe them with:
#     python check_aitta.py --probe
AITTA_TOOL_MODELS = [
    "openai/gpt-oss-120b",              # 117B MoE, built for agentic/tool use. Default.
    "Qwen/Qwen3.6-35B-A3B",             # MoE, only ~3B active -> fast; strong tool calling
    "meta-llama/Llama-3.3-70B-Instruct",  # dependable, widely-tested tool calling
    "MiniMaxAI/MiniMax-M2.7",           # designed for agentic workflows
    "Qwen/Qwen3.6-27B",                 # dense Qwen 3.6
    "Qwen/Qwen3-Coder-Next",            # excellent tool calling, but coder-tuned
]

# Served by Aitta but NOT usable for this loop, and why. Kept so nobody wastes
# time trying them.
AITTA_UNSUITABLE = {
    "lightonai/modernbert-embed-large": "embedding model, no chat endpoint",
    "intfloat/multilingual-e5-large": "embedding model, no chat endpoint",
    "Unbabel/Tower-Plus-9B": "translation-specialised",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "1.1B; too small to hold a tool schema",
    "TurkuNLP/gpt3-finnish-small": "small Finnish base model, not tuned for tools",
    "AI-Sweden/gpt-sw3-20b-instruct": "Swedish-focused; no tool-calling support",
    "NCSR-Demokritos/kaLlamaki": "Greek-focused; no tool-calling support",
    "LumiOpen/Poro-34B-chat": "Finnish/English chat; no tool-calling support",
    "Qwen/Qwen3-VL-30B-A3B-Thinking": "vision-language; tools unreliable here",
}
ANTHROPIC_MODEL = "claude-opus-5"


# ---------------------------------------------------------------- tool specs

def _annot(extra: dict | None = None, required: list[str] | None = None) -> dict:
    """Every tool carries the reasoning annotations as REQUIRED parameters, so
    the trace is complete by construction rather than by the model's goodwill."""
    props = {
        "phase": {"type": "string",
                  "enum": ["inspect", "quality_model", "policy_comparison",
                           "budget_request", "replication", "revision", "conclusion"],
                  "description": "where you are in the analysis"},
        "thought": {"type": "string",
                    "description": "your reasoning for taking THIS action now"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                       "description": "your certainty in this step"},
        "revision_trigger": {"type": "string",
                             "description": "if this changes your strategy, what "
                                            "prompted it; empty string otherwise"},
    }
    props.update(extra or {})
    return {"type": "object", "properties": props,
            "required": ["phase", "thought", "confidence"] + (required or []),
            "additionalProperties": False}


SYMPTOM = {"type": "string", "description": "symptom scale, e.g. gad7, 7u, 7d, pswq, bis, bas, shaps"}
BEHAVIOUR = {"type": "string", "description": "behavioural measure: accuracy, wsls, or task_rt"}

TOOL_SPECS = [
    ("inspect_data",
     "Summarise the sample: size, attention-check failure rate, available symptom "
     "scales and behavioural measures. Worth doing before forming any expectation.",
     _annot()),
    ("train_quality_selector",
     "Fit a cross-validated calibrated classifier for attention-check failure using "
     "only permissible features. Returns AUROC, Brier score, and how many "
     "participants fall in the ambiguous band.",
     _annot()),
    ("test_association",
     "Test ONE symptom-to-behaviour association under ONE selection policy. Prefer "
     "compare_policies unless you specifically need a single policy.",
     _annot({"symptom": SYMPTOM, "behaviour": BEHAVIOUR,
             "policy": {"type": "string",
                        "enum": ["all_data", "oracle_clean", "agent_hard", "agent_weighted"],
                        "description": "oracle_clean uses the true label -- an "
                                       "evaluation reference, not your method"}},
            ["symptom"])),
    ("compare_policies",
     "Run one association under ALL selection policies and report whether the "
     "conclusion is stable or selection-sensitive. This is usually what you want.",
     _annot({"symptom": SYMPTOM, "behaviour": BEHAVIOUR}, ["symptom"])),
    ("request_quality_labels",
     "Spend quality-assurance budget on the participants whose inclusion is most "
     "uncertain. Use this when the selector is ambiguous rather than concluding anyway.",
     _annot({"budget": {"type": "integer", "minimum": 1, "maximum": 100,
                        "description": "how many participants to send for an extra check"}})),
    ("check_replication",
     "Test whether a finding holds in the independent replication sample. It ran a "
     "DIFFERENT task with DIFFERENT scales (mania, depression, anxiety, artistic, "
     "greed); 'wsls' is the only behaviour measured in both samples.",
     _annot({"symptom": {"type": "string",
                         "description": "scale AS NAMED IN THE REPLICATION SAMPLE"},
             "behaviour": {"type": "string",
                           "description": "only wsls and task_rt exist in both"}},
            ["symptom"])),
    ("transfer_selector",
     "Train the quality selector here and apply it to the replication sample using "
     "only the features present in both. Tests whether the inclusion policy itself "
     "generalises across different cognitive tasks.",
     _annot()),
]


def openai_tools() -> list[dict]:
    return [{"type": "function",
             "function": {"name": n, "description": d, "parameters": p}}
            for n, d, p in TOOL_SPECS]


# ---------------------------------------------------------------- resolution

def resolve(prefer: str | None = None) -> str:
    """Which backend to use. Aitta wins by default -- it is the key you have."""
    if prefer and prefer != "auto":
        return prefer
    if os.environ.get("AITTA_API_KEY"):
        return "aitta"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "scripted"


def aitta_client():
    import openai
    key = os.environ.get("AITTA_API_KEY")
    if not key:
        raise RuntimeError(
            "AITTA_API_KEY is not set. Get a token at https://aitta-auth.csc.fi/myToken "
            "then:  export AITTA_API_KEY='...'")
    return openai.OpenAI(api_key=key, base_url=AITTA_BASE_URL)


def list_aitta_models() -> list[str]:
    """What Aitta is actually serving right now."""
    try:
        return sorted(m.id for m in aitta_client().models.list().data)
    except Exception as exc:
        return [f"<could not list models: {type(exc).__name__}: {exc}>"]


# ---------------------------------------------------------------- aitta loop

def run_aitta_loop(task: str, system: str, dispatch: Callable[[str, dict], str],
                   *, model: str = AITTA_MODEL, max_steps: int = 12,
                   verbose: bool = True) -> tuple[list[str], int]:
    """Manual OpenAI-style tool-calling loop.

    Written by hand rather than with a helper because the OpenAI-compatible
    surface has no equivalent of the Anthropic tool runner. The loop is the
    standard one: send messages -> if the reply carries tool_calls, execute
    each and append a `tool` message per call -> repeat until it stops asking.

    `dispatch(name, args) -> str` runs the tool and records the trace step.
    """
    client = aitta_client()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task}]
    tools = openai_tools()
    transcript, turns = [], 0

    for turns in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
            tool_choice="auto", temperature=0.2, max_tokens=4000,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            transcript.append(msg.content)
            if verbose:
                print(f"\n--- turn {turns} ---\n{msg.content[:600]}")

        calls = msg.tool_calls or []
        if not calls:
            break

        for call in calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                args = {}
                if verbose:
                    print(f"  ! could not parse arguments for {name}: {exc}")
            if verbose:
                print(f"  [{args.get('phase','?')}] {name}: "
                      f"{str(args.get('thought',''))[:80]} "
                      f"(conf {args.get('confidence','?')})")
            result = dispatch(name, args)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "name": name, "content": result})

    return transcript, turns


def aitta_structured(system: str, prompt: str, schema_model, *,
                     model: str = AITTA_MODEL, max_retries: int = 2):
    """Ask for a final answer conforming to a Pydantic model.

    Tries the OpenAI `json_schema` response format first. Not every
    OpenAI-compatible server implements it, so on failure we fall back to
    asking for JSON in the prompt and validating locally -- which is why the
    schema is also inlined into the text.
    """
    client = aitta_client()
    schema = schema_model.model_json_schema()
    ask = (f"{prompt}\n\nRespond with ONLY a JSON object matching this schema. "
           f"No prose, no markdown fence.\n\n{json.dumps(schema, indent=1)[:6000]}")

    attempts = [
        dict(response_format={"type": "json_schema",
                              "json_schema": {"name": schema_model.__name__,
                                              "schema": schema, "strict": False}}),
        dict(response_format={"type": "json_object"}),
        dict(),
    ]
    last = None
    for opts in attempts[:max_retries + 1]:
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=3000,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": ask}],
                **opts)
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json").strip()
            return schema_model.model_validate_json(text)
        except Exception as exc:
            last = exc
            continue
    raise RuntimeError(f"structured output failed after {max_retries + 1} attempts: {last}")
