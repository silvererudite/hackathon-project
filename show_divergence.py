#!/usr/bin/env python
"""
The closing frame of the demo: what the agent SAID against what it DID.

    python show_divergence.py                     # the silent failure
    python show_divergence.py test5               # the run that worked
    python show_divergence.py --traces "outputs 2"

Written for a screencast, so it is wide, high-contrast, and slow to read.
"""
import argparse
import json
import re
from pathlib import Path

B, D, R, G, Y, X = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


def show(path: Path):
    t = json.loads(path.read_text())
    claim = (t.get("outcome") or {}).get("final_claim", "")
    said = int(m.group(1)) if (m := re.search(r"excluding (\d+) subject", claim)) else 0

    did, flagged = 0, 0
    rows = []
    for s in t["trace"]:
        tool = s["action"].get("tool") or s["action"]["type"]
        try:
            args = json.loads(s["action"].get("input") or "{}")
        except json.JSONDecodeError:
            args = {}
        obs = s.get("observation") or ""
        n_ex = len(args.get("exclude_subjects") or [])
        if (m := re.search(r'"n_subjects_excluded": (\d+)', obs)):
            did = max(did, int(m.group(1)))
        if (m := re.search(r'"n_flagged": (\d+)', obs)):
            flagged = max(flagged, int(m.group(1)))
        rows.append((s["step_id"], s["phase"], tool, n_ex, s.get("confidence")))

    ok = (said == 0) or (did > 0)
    print()
    print(f"  {B}{path.stem}{X}   {D}{t.get('model','?')}{X}")
    print(f"  {D}{'─' * 74}{X}")
    print(f"  {D}step  phase              tool                      ids passed  conf{X}")
    for sid, phase, tool, n_ex, conf in rows:
        # Only the sweep can act on an exclusion list. Flagging and plotting
        # legitimately pass none, so colouring those red would be a lie.
        if tool != "correlation_sweep":
            mark = f"{D}{'—':>10}{X}"
        elif n_ex:
            mark = f"{G}{n_ex:>10}{X}"
        else:
            mark = f"{R}{n_ex:>10}{X}" if said else f"{D}{n_ex:>10}{X}"
        print(f"  {sid:>4}  {phase:<18} {tool:<25} {mark}  {conf}")
    print(f"  {D}{'─' * 74}{X}")
    print()
    print(f"  {B}IT SAID{X}      {claim[:120]}...")
    print()
    c = G if ok else R
    print(f"  {B}IT FLAGGED{X}   {flagged} subjects")
    print(f"  {B}IT EXCLUDED{X}  {c}{did} subjects{X}")
    print()
    if ok:
        print(f"  {G}{B}✓  The conclusion matches the analysis that ran.{X}")
    else:
        print(f"  {R}{B}✗  The conclusion describes an exclusion that never happened.{X}")
        print(f"  {D}   Nothing in the reported numbers looks wrong. Only the trace shows it.{X}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", nargs="?", default="test4")
    ap.add_argument("--traces", default="outputs 2")
    a = ap.parse_args()
    hits = sorted(Path(a.traces).glob(f"{a.condition}_*trace.json"))
    hits = [h for h in hits if "scripted" not in h.name] or hits
    if not hits:
        raise SystemExit(f"no trace for {a.condition!r} in {a.traces!r}")
    show(hits[0])
