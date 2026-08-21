#!/usr/bin/env python
"""
BASELINE + TEST 1-5: does the agent match a careful human analyst?

    python run_experiments.py --list
    python run_experiments.py --condition baseline --backend scripted
    python run_experiments.py --all --backend aitta

Each condition pairs a prompt with a dataset. The BASELINE prompt asks only
for the uncorrected correlation sweep. The TEST prompt adds one sentence --
"Exclude subjects that appear to induce spurious correlations" -- and the
question is whether the agent can act on it.

    BASELINE  baseline prompt + correlates_common_subjects
    TEST 1    test prompt     + correlates_common_subjects
    TEST 2    test prompt     + correlates_with_task_data
    TEST 3    test prompt     + correlates_with_survey_data
    TEST 4    test prompt     + correlates_with_full_data
    TEST 5    test prompt     + correlates_with_full_data_and_metrics

Difficulty rises across TEST 1-5 in a specific way: TEST 1 gives the agent
nothing to detect carelessness WITH, so honest failure is the right answer.
TEST 3 adds per-item survey responses, from which straight-lining and
infrequency failures are derivable. TEST 5 hands over the study's own quality
metrics, which is the easiest case. TEST 2 and 4 add trial-level data and with
it the pseudo-replication trap.

Every run is scored on both traps and written to outputs/ as a trace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import correlates as C

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

TOOL_REF = ("the `plot_correlation_matrix` tool")

BASELINE_PROMPT = (
    "Compute absolute Spearman rank correlations between columns 2 to 8 and "
    "columns 9 to 17 and run a two-sided, p < 0.05 significance test on the "
    "correlations without correcting for multiple comparisons. "
    f"Plot significant correlations in a matrix using {TOOL_REF}."
)

TEST_PROMPT = (
    "Compute absolute Spearman rank correlations between columns 2 to 8 and "
    "columns 9 to 17 and run a two-sided, p < 0.05 significance test on the "
    "correlations without correcting for multiple comparisons. "
    "Exclude subjects that appear to induce spurious correlations. "
    f"Plot significant correlations in a matrix using {TOOL_REF}."
)

CONDITIONS = {
    "baseline": ("BASELINE", BASELINE_PROMPT, "correlates_common_subjects"),
    "test1":    ("TEST 1",   TEST_PROMPT,     "correlates_common_subjects"),
    "test2":    ("TEST 2",   TEST_PROMPT,     "correlates_with_task_data"),
    "test3":    ("TEST 3",   TEST_PROMPT,     "correlates_with_survey_data"),
    "test4":    ("TEST 4",   TEST_PROMPT,     "correlates_with_full_data"),
    "test5":    ("TEST 5",   TEST_PROMPT,     "correlates_with_full_data_and_metrics"),
}


def context_block(dataset: str) -> str:
    """What an analyst would be told about the data. Withholding the codebook
    tests reading comprehension, not scientific judgement."""
    d = C.describe(dataset)
    lines = [
        f"DATASET: {dataset}",
        f"  {d['note']}",
        f"  {d['n_rows']:,} rows, {d['n_subjects']} subjects "
        f"({d['rows_per_subject']} rows per subject, {d['level']}-level)",
        f"  columns 2-8  (symptoms): {', '.join(d['columns_2_to_8_symptoms'])}",
        f"  columns 9-17 (measures): {', '.join(d['columns_9_to_17_measures'])}",
        f"  {d['n_tests_uncorrected']} correlations, uncorrected at alpha=0.05 "
        f"-> ~{0.05 * d['n_tests_uncorrected']:.1f} false positives expected from noise alone",
    ]
    if d["extra_columns"]:
        lines.append(f"  also available: {', '.join(d['extra_columns'])}")
    lines.append("\nWHAT THE COLUMNS MEAN:")
    lines += [f"  {k:14s} {v}" for k, v in d["codebook"].items()]
    return "\n".join(lines)


def score(result: dict, condition: str) -> dict:
    """Did the run survive the two traps?"""
    d = C.describe(CONDITIONS[condition][2])
    trial_level = d["level"] == "trial"
    return {
        "excluded_any_subject": result.get("n_subjects_excluded", 0) > 0,
        "n_excluded": result.get("n_subjects_excluded", 0),
        "n_significant": result.get("n_significant"),
        "n_tests": result.get("n_tests"),
        "expected_by_chance": result.get("expected_false_positives_if_null"),
        # trap 2: did it aggregate when it needed to?
        "pseudo_replication_risk": trial_level,
        "avoided_pseudo_replication": (not trial_level) or result.get("aggregate") == "subject",
    }


def run_reference(condition: str, exclude=None, aggregate="none") -> dict:
    """Deterministic reference analysis -- what the prompt literally asks for.
    This is the human-analyst comparator, not the agent."""
    _, _, dataset = CONDITIONS[condition]
    res = C.spearman_matrix(dataset, exclude_subjects=exclude, aggregate=aggregate)
    OUTPUTS.mkdir(exist_ok=True)
    png = OUTPUTS / f"{condition}_matrix.png"
    C.plot_matrix(res, png, title=f"{CONDITIONS[condition][0]} — {dataset}")
    res["figure"] = str(png)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=list(CONDITIONS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--backend", default="scripted",
                    help="scripted | aitta | anthropic | transformers")
    ap.add_argument("--reference-only", action="store_true",
                    help="run the deterministic comparator, no LLM")
    ap.add_argument("--aggregate", default="none", choices=["none", "subject"])
    a = ap.parse_args()

    if a.list:
        for k, (label, prompt, ds) in CONDITIONS.items():
            d = C.describe(ds)
            print(f"{k:9s} {label:9s} {ds:42s} {d['n_rows']:>7,} rows / "
                  f"{d['n_subjects']} subj  [{d['level']}-level]")
        return

    todo = list(CONDITIONS) if a.all else [a.condition or "baseline"]
    OUTPUTS.mkdir(exist_ok=True)
    summary = []

    for cond in todo:
        label, prompt, dataset = CONDITIONS[cond]
        print(f"\n{'=' * 72}\n{label}  |  {dataset}\n{'=' * 72}")
        print(context_block(dataset))
        print(f"\nPROMPT:\n  {prompt}\n")

        res = run_reference(cond, aggregate=a.aggregate)
        sc = score(res, cond)
        print(f"reference analysis: {res['n_significant']}/{res['n_tests']} significant "
              f"(~{res['expected_false_positives_if_null']} expected by chance)")
        if res.get("pseudo_replication_warning"):
            print(f"  ! {res['pseudo_replication_warning']}")
        for p in res["significant_pairs"][:5]:
            print(f"    {p['symptom']:8s} x {p['measure']:6s} rho={p['rho']:+.3f} p={p['p']:.2e}")
        print(f"  figure -> {res['figure']}")

        row = {"condition": cond, "label": label, "dataset": dataset,
               "prompt": prompt, "reference": {k: v for k, v in res.items()
                                               if not k.startswith("_")},
               "score": sc}

        if not a.reference_only and a.backend != "scripted":
            import sciops_agent as A
            print(f"\n--- agent ({a.backend}) ---")
            traj = A.run_agent(task=prompt + "\n\n" + context_block(dataset),
                               backend=a.backend, max_steps=14)
            row["agent"] = json.loads(traj.model_dump_json())
            (OUTPUTS / f"{cond}_trace.json").write_text(traj.model_dump_json(indent=1))

        summary.append(row)

    (OUTPUTS / "experiments.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n{'=' * 72}\nSUMMARY")
    hdr = f"{'cond':9s} {'significant':>12s} {'~by chance':>11s} {'rows/subj':>10s}  status"
    print(hdr); print("-" * len(hdr))
    for r in summary:
        sc, ref = r["score"], r["reference"]
        status = ("ok" if sc["avoided_pseudo_replication"]
                  else "INFLATED by pseudo-replication")
        print(f"{r['condition']:9s} {ref['n_significant']:>4}/{ref['n_tests']:<7} "
              f"{ref['expected_false_positives_if_null']:>11} "
              f"{ref['rows_per_subject']:>9.0f}x  {status}")
    print(f"\n-> outputs/experiments.json")


if __name__ == "__main__":
    main()
