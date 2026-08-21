#!/usr/bin/env python3
"""Verify Aitta connectivity and ordinary JSON/code generation."""
from __future__ import annotations

import json
import os
import sys

import llm_backends as backends


def check_model(model: str) -> bool:
    try:
        response = backends.chat_json(
            "aitta",
            [
                {"role": "system", "content": "Return JSON only. Do not call tools."},
                {"role": "user", "content":
                 "Return {\"action\":\"execute\",\"code\":\"result = 2 + 2\"}."},
            ],
            model=model,
            max_tokens=200,
        )
        ok = response.get("action") == "execute" and isinstance(response.get("code"), str)
        print(f"  {'JSON OK' if ok else 'INVALID'}  {model}")
        return ok
    except Exception as exc:
        print(f"  FAILED   {model} ({type(exc).__name__}: {str(exc)[:100]})")
        return False


def main() -> int:
    if not os.environ.get("AITTA_API_KEY"):
        print("AITTA_API_KEY is not set.")
        return 1
    print(f"base_url : {backends.AITTA_BASE_URL}")
    print(f"model    : {backends.AITTA_MODEL}")
    print("mode     : ordinary JSON/code generation (no function tools)\n")
    if "--probe" in sys.argv:
        models = backends.list_aitta_models()
        candidates = [m for m in models if not m.startswith("<")][:12]
        return 0 if any(check_model(model) for model in candidates) else 1
    models = backends.list_aitta_models()
    print(f"models available: {len(models)}")
    return 0 if check_model(backends.AITTA_MODEL) else 1


if __name__ == "__main__":
    raise SystemExit(main())
