#!/usr/bin/env python
"""Verify data/ layout — SCI OPS CSVs or correlates_common_subjects.csv."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEATURES = ["isd", "reliability", "mahalanobis", "survey_rt",
            "variability", "accuracy", "wsls", "task_rt"]
JOIN_KEYS = ["platform", "subject"]


def check_sciops_dir(label: str, data_dir: Path) -> list[str]:
    errors = []
    print(f"\n=== {label} (SCI OPS format) ===")
    print(f"Path: {data_dir}")
    for name in ("metrics.csv", "scores.csv"):
        if not (data_dir / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return errors

    import pandas as pd
    metrics = pd.read_csv(data_dir / "metrics.csv")
    scores = pd.read_csv(data_dir / "scores.csv")
    for col in JOIN_KEYS + ["infreq"] + FEATURES:
        if col not in metrics.columns:
            errors.append(f"metrics.csv missing {col!r}")
    merged = metrics.merge(scores, on=JOIN_KEYS, how="inner")
    print(f"  OK — {len(merged)} participants (full agent supported)")
    return errors


def main() -> int:
    from data_loader import CORRELATES_FILE, describe_correlates, find_correlates_csv
    from sciops_agent import resolve_data_dirs

    root = HERE
    print(f"Project root: {root}")

    correlates = find_correlates_csv(root)
    if correlates:
        print(f"\n=== data/{CORRELATES_FILE} ===")
        describe_correlates(correlates)

    try:
        orig, rep = resolve_data_dirs(root)
        all_errors = check_sciops_dir("original", orig)
        if rep:
            all_errors += check_sciops_dir("replication", rep)
        if correlates:
            print("\nNote: correlates file present AND SCI OPS CSVs found — agent uses SCI OPS.")
        return 1 if all_errors else 0
    except FileNotFoundError:
        if correlates:
            print("\nPartial mode: correlates file only.")
            print("  Works: inspect_data, test/compare association (all_data policy)")
            print("  Needs download_data.sh: quality selector, oracle/agent policies, replication")
            return 0
        print("\nNo data found in data/ or 01_Original/data/")
        return 1


if __name__ == "__main__":
    sys.exit(main())
