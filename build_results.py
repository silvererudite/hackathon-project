#!/usr/bin/env python
"""
Score the agent's real traces against the reference analysis.

    python build_results.py --traces "outputs 2"

The comparison that matters is not "did it get the number right" but "did the
number it reported correspond to what it actually did". Those come apart, and
when they do only the trace can tell you.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import correlates as C

HERE = Path(__file__).resolve().parent


def parse_trace(path: str) -> dict:
    t = json.loads(Path(path).read_text())
    cond = os.path.basename(path).split("_")[0]
    steps, flagged, applied, final_sig, id_type = [], None, 0, None, None

    for s in t["trace"]:
        tool = s["action"].get("tool")
        try:
            args = json.loads(s["action"].get("input") or "{}")
        except json.JSONDecodeError:
            args = {}
        obs = s.get("observation") or ""
        steps.append(tool)

        if tool == "flag_careless_subjects":
            m = re.search(r'"n_flagged": (\d+)', obs)
            if m:
                flagged = int(m.group(1))
        if tool == "correlation_sweep":
            ex = args.get("exclude_subjects") or []
            if ex:
                id_type = type(ex[0]).__name__
                m = re.search(r'"n_subjects_excluded": (\d+)', obs)
                applied = int(m.group(1)) if m else 0
            m = re.search(r'"n_significant": (\d+)', obs)
            if m:
                final_sig = int(m.group(1))

    ref = C.run_condition(cond)
    o = t.get("outcome") or {}
    claim = o.get("final_claim", "")
    # what the agent SAID it excluded, from its own prose
    m = re.search(r"excluding (\d+) subject", claim)
    claimed_excl = int(m.group(1)) if m else 0
    m = re.search(r"(\d+)\s*(?:out of|/)\s*63", claim)
    claimed_sig = int(m.group(1)) if m else None

    return {
        "condition": cond,
        "model": t.get("model"),
        "steps": t["metadata"]["total_steps"],
        "tools_called": steps,
        "flagged": flagged,
        "exclusion_applied": applied,
        "id_type_used": id_type,
        "agent_significant": final_sig,
        "claimed_excluded": claimed_excl,
        "claimed_significant": claimed_sig,
        "ref_flagged": ref["detection"]["n_flagged"],
        "ref_significant": ref["n_significant_after"],
        "verdict": (o.get("selection_sensitivity") or {}).get("verdict"),
        "confidence": o.get("confidence"),
        "claim": claim,
    }


def classify(r: dict) -> tuple[str, str]:
    """Did the reported conclusion match what the agent actually did?"""
    if r["condition"] in ("baseline", "test1"):
        ok = (r["flagged"] in (0, None)) and r["claimed_excluded"] == 0
        return ("CORRECT" if ok else "OVERCLAIMED",
                "no basis to exclude, and none claimed" if ok
                else "claimed an exclusion the data cannot support")
    if r["claimed_excluded"] and not r["exclusion_applied"]:
        return ("SILENT FAILURE",
                f"reported excluding {r['claimed_excluded']} subjects; the sweep "
                f"actually dropped {r['exclusion_applied']}"
                + (f" (ids passed as {r['id_type_used']}, must be str)"
                   if r["id_type_used"] == "int" else " -- exclusion never applied"))
    if r["agent_significant"] == r["ref_significant"]:
        return ("CORRECT", f"{r['exclusion_applied']} excluded, "
                           f"{r['agent_significant']}/63 -- matches reference")
    return ("MISMATCH", f"agent {r['agent_significant']}/63 vs reference "
                        f"{r['ref_significant']}/63")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="outputs 2")
    ap.add_argument("--out", default="RESULTS.md")
    a = ap.parse_args()

    rows = [parse_trace(f) for f in
            sorted(glob.glob(os.path.join(a.traces, "*_trace.json")))
            if "scripted" not in f]
    for r in rows:
        r["status"], r["why"] = classify(r)

    print(f"{'cond':9s} {'flagged':>8s} {'applied':>8s} {'sig':>6s} {'ref':>5s} "
          f"{'claimed':>8s}  status")
    print("-" * 88)
    for r in rows:
        print(f"{r['condition']:9s} {str(r['flagged']):>8s} {r['exclusion_applied']:>8} "
              f"{str(r['agent_significant']):>6s} {r['ref_significant']:>5} "
              f"{r['claimed_excluded']:>8}  {r['status']}")
        print(f"{'':9s} -> {r['why']}")

    Path(a.out).write_text(render(rows))
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    json.dump(rows, open(out / "agent_results.json", "w"), indent=1)
    print(f"\n-> {a.out} and outputs/agent_results.json")


def render(rows: list[dict]) -> str:
    n_ok = sum(1 for r in rows if r["status"] == "CORRECT")
    n_silent = sum(1 for r in rows if r["status"] == "SILENT FAILURE")
    model = rows[0]["model"]
    lines = [
        "# Results",
        "",
        f"Six conditions, one agent (`{model}`), against a deterministic reference "
        "analysis. Correlations always run on the same 386-subject table, so only "
        "the exclusion set differs.",
        "",
        "## Headline",
        "",
        f"**{n_ok} of {len(rows)} conditions produced a conclusion that matched what "
        f"the agent actually did.** In {n_silent}, the agent reported excluding "
        "subjects while the analysis it ran excluded none — the number in the "
        "write-up was real, the exclusion behind it was not.",
        "",
        "An output-only benchmark scores those runs as successes. They are the "
        "reason to keep traces.",
        "",
        "## Per condition",
        "",
        "| cond | flagged | actually excluded | significant | reference | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['flagged'] if r['flagged'] is not None else '—'} "
            f"| {r['exclusion_applied']} | "
            f"{r['agent_significant'] if r['agent_significant'] is not None else '—'}/63 "
            f"| {r['ref_significant']}/63 | **{r['status']}** |")
    lines += ["", "## What happened in each", ""]
    for r in rows:
        lines += [f"**{r['condition']}** — {r['status']}: {r['why']}  ",
                  f"*Tools:* `{' → '.join(t for t in r['tools_called'] if t)}`  ",
                  f"*Reported confidence:* {r['confidence']}", ""]
    lines += [
        "## The silent failure, in detail",
        "",
        "Subject ids are strings (`02hfkd0x4jtnoiwsds69adoq`). The tool schema "
        "declared `exclude_subjects` as an array of integers — our bug. In TEST 2 "
        "the agent obeyed the schema and passed row numbers `[2, 7, 10, ...]`. "
        "Nothing matched, zero subjects were dropped, and the call returned "
        "**success with no error**. The agent then reported excluding 171 subjects.",
        "",
        "In TEST 5 the agent ignored the schema and passed the real string ids, so "
        "the exclusion worked and it landed on 11/63 — exactly the reference.",
        "",
        "Two lessons, and the second is the one worth presenting:",
        "",
        "1. A wrong tool schema does not fail loudly; it produces a confident wrong "
        "answer. `correlation_sweep` now reports `exclusion_warning` when requested "
        "ids match nothing.",
        "2. **The final answer could not have revealed this.** Both runs report a "
        "plausible number of significant correlations and a plausible exclusion "
        "count. Only the trace shows that one of them did the work and the other "
        "did not.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
