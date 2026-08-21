"""
Pluggable LLM backends for the reactive loop.

Two providers, one tool set:

  aitta         CSC's hosted inference for the summer school. OpenAI-COMPATIBLE,
                so it uses the `openai` client with a custom base_url -- not the
                Anthropic SDK. This is the default when AITTA_API_KEY is set.
  transformers  Local Hugging Face inference on BlueBEAR (or any GPU node).
                Used when backend="transformers" or BLUEBEAR_LLM=1 is set.
                Requires a Slurm GPU job -- do not run on the login node.
  anthropic     Claude via the Anthropic SDK's tool runner.
  scripted      No API at all; replays the best-rated human trajectory.

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

# Local inference on BlueBEAR via module load Transformers/4.42.0-foss-2023a.
# Qwen2.5-7B-Instruct: strong tool calling, open weights (no HF token), ~14 GB
# VRAM in bf16 on one A100 40 GB. Matches the Qwen family used on Aitta and is
# appropriate for this agent loop -- NOT TinyLlama, which is too small for tools.
TRANSFORMERS_MODEL = os.environ.get("TRANSFORMERS_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Larger alternative if you have headroom: "Qwen/Qwen2.5-14B-Instruct" (~28 GB bf16).
# Gated models (Meta Llama) need HF_TOKEN and acceptance on huggingface.co first.
TRANSFORMERS_MODEL_ALTERNATIVES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


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
    """Which backend to use."""
    if prefer and prefer != "auto":
        return prefer
    if os.environ.get("BLUEBEAR_LLM") or os.environ.get("TRANSFORMERS_BACKEND"):
        return "transformers"
    if os.environ.get("AITTA_API_KEY"):
        return "aitta"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "scripted"


def aitta_client(timeout: float = 120.0, max_retries: int = 1):
    """Aitta client.

    The SDK defaults to a 600s timeout with 2 retries -- up to 30 minutes of
    apparent hang on a model that is cold-starting or not actually up. We cap
    it much lower so a stuck model fails fast and says so.
    """
    import openai
    key = os.environ.get("AITTA_API_KEY")
    if not key:
        raise RuntimeError(
            "AITTA_API_KEY is not set. Get a token at https://aitta-auth.csc.fi/myToken "
            "then:  export AITTA_API_KEY='...'")
    return openai.OpenAI(api_key=key, base_url=AITTA_BASE_URL,
                         timeout=timeout, max_retries=max_retries)


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
            brace = text.find("{")
            if brace > 0:
                text = text[brace:]
            return schema_model.model_validate_json(text)
        except Exception as exc:
            last = exc
            continue
    raise RuntimeError(f"structured output failed after {max_retries + 1} attempts: {last}")


# ---------------------------------------------------------------- transformers (BlueBEAR / local GPU)

_TRANSFORMERS_MODEL = None
_TRANSFORMERS_TOKENIZER = None


def load_transformers_model(*, model: str | None = None, verbose: bool = True):
    """Load tokenizer + causal LM once per process. Call only inside a GPU job."""
    global _TRANSFORMERS_MODEL, _TRANSFORMERS_TOKENIZER
    if _TRANSFORMERS_MODEL is not None:
        return _TRANSFORMERS_MODEL, _TRANSFORMERS_TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = model or os.environ.get("TRANSFORMERS_MODEL", TRANSFORMERS_MODEL)
    if verbose:
        print(f"Loading {name} …")
        print(f"  PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        mdl = mdl.to("cpu")
    _TRANSFORMERS_MODEL, _TRANSFORMERS_TOKENIZER = mdl, tok
    return mdl, tok


def _parse_qwen_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Extract Qwen2.5 <tool_call> blocks. Returns (assistant prose, [{name, arguments}])."""
    import re

    prose_parts, calls = [], []
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    last = 0
    for match in pattern.finditer(text):
        prose_parts.append(text[last:match.start()].strip())
        payload = match.group(1).strip()
        try:
            obj = json.loads(payload)
            name = obj.get("name") or obj.get("function") or ""
            arguments = obj.get("arguments") or obj.get("parameters") or obj
            if name and isinstance(arguments, dict) and "name" in arguments:
                arguments = {k: v for k, v in arguments.items() if k != "name"}
            calls.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})
        except json.JSONDecodeError:
            calls.append({"name": "", "arguments": {"raw": payload}})
        last = match.end()
    prose_parts.append(text[last:].strip())
    prose = "\n".join(p for p in prose_parts if p).strip()
    return prose, calls


def _transformers_generate(messages: list, tools: list[dict], *, model, tokenizer,
                           max_new_tokens: int = 2048) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def run_transformers_loop(task: str, system: str, dispatch: Callable[[str, dict], str],
                          *, model: str | None = None, max_steps: int = 12,
                          verbose: bool = True) -> tuple[list[str], int]:
    """Tool-calling loop using local Transformers + Qwen2.5 chat template."""
    mdl, tok = load_transformers_model(model=model, verbose=verbose)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    tools = openai_tools()
    transcript, turns = [], 0

    for turns in range(1, max_steps + 1):
        raw = _transformers_generate(messages, tools, model=mdl, tokenizer=tok)
        prose, calls = _parse_qwen_tool_calls(raw)
        # Keep raw assistant text (incl. tool_call tags) for multi-turn chat template.
        assistant_msg: dict = {"role": "assistant", "content": raw.strip() or prose}
        if calls:
            assistant_msg["tool_calls"] = [
                {"id": f"call_{turns}_{i}", "type": "function",
                 "function": {"name": c["name"],
                              "arguments": json.dumps(c["arguments"])}}
                for i, c in enumerate(calls)
            ]
        messages.append(assistant_msg)

        if prose:
            transcript.append(prose)
            if verbose:
                print(f"\n--- turn {turns} ---\n{prose[:600]}")

        if not calls:
            break

        for i, call in enumerate(calls):
            name = call["name"]
            args = call["arguments"] if isinstance(call["arguments"], dict) else {}
            if not name:
                if verbose:
                    print(f"  ! malformed tool call (no name), skipping: {str(args)[:120]}")
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": "malformed tool call from model"}),
                    "name": "unknown",
                    "tool_call_id": f"call_{turns}_{i}",
                })
                continue
            if verbose:
                print(f"  [{args.get('phase', '?')}] {name}: "
                      f"{str(args.get('thought', ''))[:80]} "
                      f"(conf {args.get('confidence', '?')})")
            result = dispatch(name, args)
            messages.append({
                "role": "tool",
                "content": result,
                "name": name,
                "tool_call_id": f"call_{turns}_{i}",
            })

    return transcript, turns


def transformers_structured(system: str, prompt: str, schema_model, *,
                            model: str | None = None, max_retries: int = 2):
    """Ask the local model for JSON matching a Pydantic schema."""
    mdl, tok = load_transformers_model(model=model, verbose=False)
    schema = schema_model.model_json_schema()
    ask = (f"{prompt}\n\nRespond with ONLY a JSON object matching this schema. "
           f"No prose, no markdown fence.\n\n{json.dumps(schema, indent=1)[:6000]}")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": ask}]
    last = None
    for _ in range(max_retries + 1):
        try:
            text = _transformers_generate(messages, tools=[], model=mdl, tokenizer=tok,
                                          max_new_tokens=3000)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json").strip()
            brace = text.find("{")
            if brace > 0:
                text = text[brace:]
            end = text.rfind("}")
            if end >= 0:
                text = text[: end + 1]
            return schema_model.model_validate_json(text)
        except Exception as exc:
            last = exc
            continue
    raise RuntimeError(f"structured output failed after {max_retries + 1} attempts: {last}")
