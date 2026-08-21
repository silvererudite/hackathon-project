"""Load hackathon-project/data/correlates_common_subjects.csv into agent shape."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CORRELATES_FILE = "correlates_common_subjects.csv"

# Map file columns -> names used by sciops_agent tools
RENAME = {
    "GAD-7": "gad7",
    "7-up": "7u",
    "7-down": "7d",
    "BIS": "bis",
    "BAS": "bas",
    "SHAPS": "shaps",
    "PSWQ": "pswq",
    "Acc": "accuracy",
    "WS": "wsls",
}

# Extra columns kept for inspection but not used by default tools
EXTRA = ["Pts", "LS", "Pers", "beta", "eta_p", "eta_n", "kappa"]


def find_correlates_csv(root: Path) -> Path | None:
    for candidate in (root / "data" / CORRELATES_FILE,
                      Path(__file__).resolve().parent / "data" / CORRELATES_FILE):
        if candidate.is_file():
            return candidate
    env = __import__("os").environ.get("SCIOPS_CORRELATES_CSV")
    if env and Path(env).is_file():
        return Path(env)
    return None


def load_correlates(path: Path) -> tuple[pd.DataFrame, dict]:
    """Return (dataframe, metadata about what is / is not available)."""
    df = pd.read_csv(path).rename(columns=RENAME)
    if "subject" not in df.columns:
        raise ValueError(f"{path.name}: missing 'subject' column")
    df["platform"] = "correlates"

    meta = {
        "source": str(path),
        "format": "correlates_common_subjects",
        "n_participants": len(df),
        "has_infreq": False,
        "has_quality_features": False,
        "symptom_columns": [c for c in ("gad7", "7u", "7d", "bis", "bas", "shaps", "pswq")
                            if c in df.columns],
        "behaviour_columns": [c for c in ("accuracy", "wsls") if c in df.columns],
        "extra_columns": [c for c in EXTRA if c in df.columns],
        "limitations": [
            "No infreq/attention-check column — quality selector and oracle_clean policies "
            "cannot run. Use bash download_data.sh for full SCI OPS metrics+scores.",
            "No isd/reliability/mahalanobis/survey_rt/variability/task_rt — "
            "train_quality_selector is unavailable.",
        ],
    }
    return df, meta


def describe_correlates(path: Path) -> None:
    df, meta = load_correlates(path)
    print(f"Found: {path}")
    print(f"Participants: {meta['n_participants']}")
    print(f"Symptoms mapped: {meta['symptom_columns']}")
    print(f"Behaviour mapped: {meta['behaviour_columns']}")
    print(f"Extra columns (not used by default tools): {meta['extra_columns']}")
    for note in meta["limitations"]:
        print(f"  LIMIT: {note}")
    if "gad7" in df.columns and "accuracy" in df.columns:
        print("  OK for: inspect_data, compare_policies(all_data only), test_association(all_data)")
