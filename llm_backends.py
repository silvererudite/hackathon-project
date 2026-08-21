"""LLM providers for the tool-free, code-generating ReAct loop.

No provider receives function schemas or a ``tools`` argument. Every model
response is ordinary JSON containing either Python code to execute or a final
structured conclusion.
"""
from __future__ import annotations

import json
import os
from typing import Any

AITTA_BASE_URL = "https://aitta-api.csc.fi/openai/v1"
AITTA_MODEL = "openai/gpt-oss-120b"
ANTHROPIC_MODEL = "claude-sonnet-4-5"


def _config_value(name: str) -> str | None:
    try:
        import config
        value = getattr(config, name, None)
    except (ImportError, AttributeError):
        return None
    return value if isinstance(value, str) and value.strip() else None


OPENAI_MODEL = _config_value("OPENAI_MODEL") or "gpt-4.1-mini"
HF_MODEL_PATH = _config_value("HF_MODEL_PATH")
HF_DEVICE = _config_value("HF_DEVICE") or "auto"
HF_MAX_TOKENS = int(_config_value("HF_MAX_TOKENS") or "4000")


def resolve(prefer: str | None = None) -> str:
    if prefer and prefer != "auto":
        return prefer
    if os.environ.get("AITTA_API_KEY"):
        return "aitta"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY") or _config_value("OPENAI_API_KEY"):
        return "openai"
    if HF_MODEL_PATH:
        return "huggingface"
    return "scripted"


def aitta_client():
    import openai
    key = os.environ.get("AITTA_API_KEY")
    if not key:
        raise RuntimeError("AITTA_API_KEY is not set.")
    return openai.OpenAI(api_key=key, base_url=AITTA_BASE_URL)


def openai_client():
    import openai
    key = os.environ.get("OPENAI_API_KEY") or _config_value("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment or config.py.")
    return openai.OpenAI(api_key=key)


def list_aitta_models() -> list[str]:
    try:
        return sorted(model.id for model in aitta_client().models.list().data)
    except Exception as exc:
        return [f"<could not list models: {type(exc).__name__}: {exc}>"]


def model_for(backend: str) -> str:
    return {
        "openai": OPENAI_MODEL,
        "aitta": AITTA_MODEL,
        "anthropic": ANTHROPIC_MODEL,
        "huggingface": HF_MODEL_PATH or "huggingface",
        "scripted": "scripted-code",
    }[backend]


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].removeprefix("json").strip()
    return text


def chat_json(
    backend: str,
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 5000,
) -> dict[str, Any]:
    """Return one JSON object from a provider, without any tool/function API."""
    if backend in {"openai", "aitta"}:
        client = openai_client() if backend == "openai" else aitta_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
        return json.loads(_strip_json_fence(text))

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conversation = [m for m in messages if m["role"] != "system"]
        response = client.messages.create(
            model=model, system=system, messages=conversation,
            temperature=temperature, max_tokens=max_tokens,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return json.loads(_strip_json_fence(text))

    if backend == "huggingface":
        from transformers import pipeline
        generator = pipeline("text-generation", model=model, device_map=HF_DEVICE)
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        result = generator(
            prompt, max_new_tokens=min(max_tokens, HF_MAX_TOKENS),
            do_sample=temperature > 0, temperature=max(temperature, 0.01),
            return_full_text=False,
        )[0]["generated_text"]
        return json.loads(_strip_json_fence(result))

    raise ValueError(f"Backend {backend!r} cannot generate a live response.")
