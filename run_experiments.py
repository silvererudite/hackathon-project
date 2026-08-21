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

# v2: correlations ALWAYS on correlates_common_subjects (386 rows, one per
# subject), so n is identical in every condition. The supporting tables are
# evidence for exclusion only and never enter a correlation.
CONDITIONS = {
    "baseline": ("BASELINE", BASELINE_PROMPT, C.CORRELATION_TABLE),
    "test1":    ("TEST 1",   TEST_PROMPT,     C.CORRELATION_TABLE),
    "test2":    ("TEST 2",   TEST_PROMPT,     C.CORRELATION_TABLE),
    "test3":    ("TEST 3",   TEST_PROMPT,     C.CORRELATION_TABLE),
    "test4":    ("TEST 4",   TEST_PROMPT,     C.CORRELATION_TABLE),
    "test5":    ("TEST 5",   TEST_PROMPT,     C.CORRELATION_TABLE),
}


def context_block(dataset: str, condition: str | None = None) -> str:
    """What an analyst would be told. Withholding the codebook tests reading
    comprehension, not scientific judgement."""
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

    support = C.EXPERIMENT.get(condition or "", [])
    if support:
        lines.append("\nSUPPORTING TABLES (evidence for deciding who to exclude; "
                     "they are NOT part of the correlation):")
        for t in support:
            sd = C.load(t)
            lines.append(f"  {t}: {C.SUPPORT_TABLES[t]}")
            lines.append(f"    {len(sd):,} rows, {sd.shape[1]} columns, "
                         f"{sd['subject'].nunique()} subjects; "
                         f"columns: {', '.join(list(sd.columns)[:10])}"
                         f"{' ...' if sd.shape[1] > 10 else ''}")
    elif condition:
        lines.append("\nSUPPORTING TABLES: none. There is no information in this "
                     "condition from which carelessness could be judged.")

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


def run_reference(condition: str, strategy: str = "best") -> dict:
    """Deterministic reference analysis -- the human-analyst comparator."""
    r = C.run_condition(condition, strategy=strategy)
    OUTPUTS.mkdir(exist_ok=True)
    png = OUTPUTS / f"{condition}_matrix.png"
    label = CONDITIONS[condition][0]
    sup = "+".join(t.replace("_data", "") for t in r["support_tables"]) or "no support tables"
    C.plot_matrix(r["_after"], png, title=f"{label} — {sup}")
    r["figure"] = str(png)
    return r


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
    ap.add_argument("--aggregate", default="none", choices=["none", "subject"],
                    help="legacy single-table mode only")
    ap.add_argument("--strategy", default="best", choices=["best", "union"],
                    help="best: use strongest evidence available; "
                         "union: drop anyone any signal flags")
    a = ap.parse_args()

    if a.list:
        print(f"correlation table (all conditions): {C.CORRELATION_TABLE} "
              f"— {len(C.load(C.CORRELATION_TABLE))} subjects, n is constant\n")
        for k, (label, _prompt, _ds) in CONDITIONS.items():
            sup = C.EXPERIMENT[k]
            print(f"  {k:9s} {label:9s} support: "
                  f"{', '.join(sup) if sup else '(none — no basis to exclude)'}")
        return

    todo = list(CONDITIONS) if a.all else [a.condition or "baseline"]
    OUTPUTS.mkdir(exist_ok=True)
    summary = []

    for cond in todo:
        label, prompt, dataset = CONDITIONS[cond]
        print(f"\n{'=' * 72}\n{label}  |  {dataset}\n{'=' * 72}")
        print(context_block(dataset, cond))
        print(f"\nPROMPT:\n  {prompt}\n")

        res = run_reference(cond, strategy=a.strategy)
        det = res["detection"]
        print(f"reference: {res['n_significant_before']}/{res['n_tests']} significant "
              f"before exclusion (~{res['expected_by_chance']} expected by chance)")
        if det["n_flagged"]:
            print(f"  excluded {det['n_flagged']} subjects via {det.get('signal_used')} "
                  f"-> n={res['n_subjects_after']}")
            print(f"  {res['n_significant_after']}/{res['n_tests']} significant after")
            v = (det.get("validation_vs_attention_checks") or {}).get("_combined")
            if v:
                print(f"  proxy quality vs the study's attention checks: "
                      f"precision {v['precision']:.2f}, recall {v['recall']:.2f} "
                      f"({v['n_true_careless']} truly careless)")
        else:
            print(f"  no exclusion: {det['note']}")
        for ev in det.get("evidence", []):
            print(f"    [{ev['strength']:6s}] {ev['table']:14s} {ev['signal']} "
                  f"-> {ev['n_flagged']}")
        print(f"  figure -> {res['figure']}")

        row = {"condition": cond, "label": label, "dataset": dataset,
               "support_tables": res["support_tables"], "prompt": prompt,
               "reference": {k: v for k, v in res.items() if not k.startswith("_")}}

        if not a.reference_only and a.backend != "scripted":
            import sciops_agent as A
            print(f"\n--- agent ({a.backend}) ---")
            traj = A.run_agent(task=prompt + "\n\n" + context_block(dataset, cond),
                               backend=a.backend, max_steps=14)
            row["agent"] = json.loads(traj.model_dump_json())
            (OUTPUTS / f"{cond}_trace.json").write_text(traj.model_dump_json(indent=1))

        summary.append(row)

    (OUTPUTS / "experiments.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n{'=' * 72}\nSUMMARY")
    hdr = (f"{'cond':9s} {'support':24s} {'excluded':>9s} {'n':>5s} "
           f"{'significant':>13s}  proxy vs attention checks")
    print(hdr); print("-" * len(hdr))
    for r in summary:
        ref = r["reference"]; det = ref["detection"]
        sup = "+".join(t.replace("_data", "") for t in r["support_tables"]) or "(none)"
        v = (det.get("validation_vs_attention_checks") or {}).get("_combined")
        q = (f"P={v['precision']:.2f} R={v['recall']:.2f}"
             if v and v.get("n_predicted") else "")
        print(f"{r['condition']:9s} {sup:24s} {det['n_flagged']:>9} "
              f"{ref['n_subjects_after']:>5} "
              f"{ref['n_significant_before']:>5} -> {ref['n_significant_after']:<4}  {q}")
    print(f"\n-> outputs/experiments.json")


if __name__ == "__main__":
    main()
