#!/usr/bin/env python
"""Verify the Aitta connection before running the agent.  python check_aitta.py"""
import json
import os
import sys

import llm_backends as B


def probe(models=None) -> int:
    """Ask each candidate for a tool call and report which actually comply.

    Worth running once: a served model is not necessarily a model whose
    deployment has tool parsing switched on, and that is invisible until you
    try. A model that answers in prose here cannot drive the agent loop.
    """
    client = B.aitta_client()
    tools = B.openai_tools()[:1]
    models = models or B.AITTA_TOOL_MODELS
    print(f"probing {len(models)} models for tool calling...\n")
    ok = []
    for m in models:
        try:
            r = client.chat.completions.create(
                model=m, max_tokens=300, tools=tools, tool_choice="auto",
                messages=[{"role": "user",
                           "content": "Summarise the sample before doing anything else."}])
            calls = r.choices[0].message.tool_calls or []
            if calls:
                args = json.loads(calls[0].function.arguments or "{}")
                missing = [k for k in ("phase", "thought", "confidence") if k not in args]
                flag = f"  (omitted {missing})" if missing else ""
                print(f"  TOOLS OK   {m}{flag}")
                ok.append(m)
            else:
                print(f"  prose only {m}  <- cannot drive the loop")
        except Exception as exc:
            print(f"  FAILED     {m}  ({type(exc).__name__}: {str(exc)[:70]})")
    if ok:
        print(f"\nUsable: {', '.join(ok)}")
        print(f"Default {B.AITTA_MODEL}"
              f"{' is OK' if B.AITTA_MODEL in ok else ' NOT working -- switch'}")
    return 0 if ok else 1


def main():
    if "--probe" in sys.argv:
        if not os.environ.get("AITTA_API_KEY"):
            print("AITTA_API_KEY not set.")
            return 1
        return probe()

    print(f"base_url : {B.AITTA_BASE_URL}")
    print(f"model    : {B.AITTA_MODEL}")
    key = os.environ.get("AITTA_API_KEY")
    if not key:
        print("\nAITTA_API_KEY not set.")
        print("  1. get a token at https://aitta-auth.csc.fi/myToken")
        print("  2. export AITTA_API_KEY='...'")
        return 1
    print(f"key      : set ({len(key)} chars)\n")

    print("1. listing models…")
    models = B.list_aitta_models()
    for m in models[:25]:
        mark = "  <- ours" if m == B.AITTA_MODEL else ""
        print(f"     {m}{mark}")
    if B.AITTA_MODEL not in models and not any(x.startswith("<") for x in models):
        print(f"\n  ! {B.AITTA_MODEL} is NOT in the list. Pick one above and set "
              f"llm_backends.AITTA_MODEL, or pass model=... to run_agent().")

    print("\n2. chat completion…")
    client = B.aitta_client()
    r = client.chat.completions.create(
        model=B.AITTA_MODEL, max_tokens=40,
        messages=[{"role": "user", "content": "Reply with exactly: ready"}])
    print(f"     {r.choices[0].message.content!r}")

    print("\n3. tool calling (the loop depends on this)…")
    r = client.chat.completions.create(
        model=B.AITTA_MODEL, max_tokens=300, tools=B.openai_tools(),
        tool_choice="auto",
        messages=[{"role": "user",
                   "content": "Summarise the sample before doing anything else."}])
    calls = r.choices[0].message.tool_calls or []
    if not calls:
        print("     ! no tool call returned — this model may not support tools.")
        print("       The agent loop will not work. Try another model.")
        return 1
    for c in calls:
        args = json.loads(c.function.arguments or "{}")
        print(f"     {c.function.name}({', '.join(f'{k}={v!r}' for k, v in list(args.items())[:3])})")
    print("\nOK — aitta backend is ready. Run the notebook with backend='aitta'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
