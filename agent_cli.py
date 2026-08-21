#!/usr/bin/env python3
"""Run the SCI OPS agent from a terminal and persist its process trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_agent import DEFAULT_TASK, run_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the code-generating SCI OPS ReAct agent, stream its exploration, "
            "and save the complete trajectory as JSON."
        )
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=DEFAULT_TASK,
        help="Research question for the agent (defaults to the built-in demo question).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "openai", "aitta", "anthropic", "huggingface", "scripted"),
        default="auto",
        help="LLM backend. 'auto' uses the key found in the environment or config.py.",
    )
    parser.add_argument("--model", help="Override the backend's default model.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Maximum number of agent turns (default: 12).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path. By default, writes traces/<trajectory-id>.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not stream exploration details; only print the result path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be at least 1")

    trajectory = run_agent(
        task=args.task,
        backend=args.backend,
        model=args.model,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )

    output = args.output or Path("traces") / f"{trajectory.trajectory_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nTrajectory saved to: {output.resolve()}")
    print(
        f"Recorded {trajectory.metadata.total_steps} steps, "
        f"{trajectory.metadata.total_code_executions} code executions, and "
        f"{trajectory.metadata.total_failures} failures."
    )
    artifacts = []
    for step in trajectory.trace:
        try:
            observation = json.loads(step.observation or "{}")
        except json.JSONDecodeError:
            continue
        artifacts.extend(observation.get("artifacts") or [])
    for artifact in artifacts:
        print(f"Saved {artifact.get('type', 'artifact')}: {artifact.get('path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
