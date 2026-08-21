"""
Correlation-matrix tools for the human-vs-agent comparison.

The experiment gives an agent a prompt of the form

    "Compute absolute Spearman rank correlations between columns 2 to 8 and
     columns 9 to 17, two-sided, p < 0.05, WITHOUT correcting for multiple
     comparisons. Plot significant correlations in a matrix."

and the TEST variant adds

    "Exclude subjects that appear to induce spurious correlations."

Columns 2-8 are the 7 symptom scales, 9-17 the 9 task measures, so the
uncorrected sweep runs 63 tests at alpha=0.05 -- about 3 false positives
expected from noise alone before careless responders are considered.

TWO TRAPS ARE BUILT INTO THESE DATASETS, and an agent has to survive both.

  1. CARELESS RESPONDERS. Inattentive participants induce correlations that
     vanish once they are excluded. This is the trap the study is about and
     the one the TEST prompt asks about.

  2. PSEUDO-REPLICATION. Three of the five files are TRIAL-level: the same
     subject appears on ~90 rows with their symptom score copied onto every
     one. Correlating at row level inflates n from 386 to 34,740 and turns
     rho = -0.093 from p = 0.067 into p = 5e-68. Nothing about the effect
     changed; only the row count did.

We deliberately DO NOT silently aggregate. The tool reports `n_rows`,
`n_subjects` and `rows_per_subject` on every call, and warns when they differ,
so a capable agent can notice and call `aggregate="subject"` itself. Silently
fixing it would remove the second trap and make the benchmark easier than the
science actually is. Whether the agent notices is a result worth recording.
"""
from __future__ import annotations

from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"

# --------------------------------------------------------------------------
# The experiment, v2: ONE correlation table + supporting evidence tables.
# --------------------------------------------------------------------------
# Correlations are ALWAYS computed on correlates_common_subjects -- 386 rows,
# one per subject. n is therefore identical in every condition, which is what
# makes the conditions comparable. The earlier design varied the correlation
# table itself, so trial-level files silently inflated n from 386 to 34,740
# and manufactured significance that had nothing to do with the manipulation.
#
# The supporting tables are EVIDENCE FOR EXCLUSION only. They never enter a
# correlation. What changes across conditions is how much a careful analyst
# could learn about which subjects to drop:
#
#   TEST 1  nothing            -> no basis to exclude; saying so is correct
#   TEST 2  + task             -> behavioural signals (RT, accuracy, stickiness)
#   TEST 3  + survey           -> per-item responses; straight-lining derivable
#   TEST 4  + task + survey    -> both behavioural and survey evidence
#   TEST 5  + ... + metrics    -> the study's own attention-check counts
#
CORRELATION_TABLE = "correlates_common_subjects"

SUPPORT_TABLES = {
    "task_data": "36,540 trial rows (~90/subject): block, trial, choice, rt, accuracy, outcome.",
    "survey_data": "406 subjects x per-item survey responses plus per-scale response times.",
    "metrics_data": ("386 subjects x the study's own quality metrics: infreq "
                     "(failed attention checks), isd, reliability, mahalanobis."),
}

EXPERIMENT = {
    "baseline": [],
    "test1": [],
    "test2": ["task_data"],
    "test3": ["survey_data"],
    "test4": ["task_data", "survey_data"],
    "test5": ["task_data", "survey_data", "metrics_data"],
}

# Semicolon-delimited on delivery; sniffed automatically by load().
_DELIM_OVERRIDE = {"survey_data": ";"}


# Legacy single-table conditions. Kept so the earlier runs still reproduce.
CONDITIONS = {
    "correlates_common_subjects": "386 subjects. Symptoms + task measures only. Subject-level.",
    "correlates_with_survey_data": "386 subjects. Adds per-item survey responses. Subject-level.",
    "correlates_with_task_data": "386 subjects x 90 trials. Adds trial-level task data. TRIAL-LEVEL.",
    "correlates_with_full_data": "406 subjects x 90 trials. Survey items + trial data. TRIAL-LEVEL.",
    "correlates_with_full_data_and_metrics": (
        "386 subjects x 90 trials. Everything plus the study's own quality "
        "metrics (infreq, isd, reliability, mahalanobis). TRIAL-LEVEL."),
}

SYMPTOMS = ["7-up", "7-down", "GAD-7", "BIS", "BAS", "SHAPS", "PSWQ"]      # cols 2-8
MEASURES = ["Acc", "Pts", "WS", "LS", "Pers", "beta", "eta_p", "eta_n", "kappa"]  # cols 9-17

# What the columns mean. The agent cannot judge plausibility without this, and
# the study's authors had it -- withholding it tests the wrong thing.
CODEBOOK = {
    "7-up": "7 Up mania/hypomania screen (higher = more symptoms)",
    "7-down": "7 Down depression screen (higher = more symptoms)",
    "GAD-7": "Generalised Anxiety Disorder 7-item scale",
    "BIS": "Behavioural Inhibition System sensitivity",
    "BAS": "Behavioural Activation System sensitivity",
    "SHAPS": "Snaith-Hamilton Pleasure Scale (anhedonia)",
    "PSWQ": "Penn State Worry Questionnaire",
    "Acc": "proportion of correct choices in the reversal-learning task",
    "Pts": "total points earned",
    "WS": "win-stay rate: P(repeat choice | previous trial rewarded)",
    "LS": "lose-shift rate: P(switch choice | previous trial unrewarded)",
    "Pers": "perseveration: repeated choices after contingency reversal",
    "beta": "inverse temperature from the RL model (choice determinism)",
    "eta_p": "learning rate for positive prediction errors",
    "eta_n": "learning rate for negative prediction errors",
    "kappa": "choice-stickiness parameter",
    "infreq": "count of failed infrequency (attention-check) items",
    "isd": "intra-individual SD of survey responses (straight-lining if low)",
    "reliability": "within-subject split-half consistency",
    "mahalanobis": "multivariate outlyingness of the response profile",
}


def available() -> list[str]:
    return sorted(p.stem for p in DATASETS.glob("*.csv"))


def load(name: str, aggregate: str = "none") -> pd.DataFrame:
    """Load one condition.

    aggregate="subject" collapses trial-level files to one row per subject
    (mean of numeric columns). Use it deliberately -- see the module docstring.
    """
    path = DATASETS / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{name!r} not found. Available: {available()}")
    # Sniff the delimiter: survey_data ships semicolon-separated, and reading
    # it with the default comma yields one 60-character column name and no
    # usable data -- a failure that looks like an empty result, not an error.
    sep = _DELIM_OVERRIDE.get(name)
    if sep is None:
        head = path.open().readline()
        sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    if aggregate == "subject" and df["subject"].duplicated().any():
        num = list(df.select_dtypes("number").columns)
        df = df.groupby("subject", as_index=False)[num].mean().copy()
    return df


def describe(name: str) -> dict:
    """Shape, level, and the codebook -- what an analyst would be told."""
    df = load(name)
    n, s = len(df), df["subject"].nunique()
    cols = list(df.columns)
    return {
        "dataset": name,
        "note": CONDITIONS.get(name, ""),
        "n_rows": n,
        "n_subjects": int(s),
        "rows_per_subject": round(n / s, 1),
        "level": "subject" if n == s else "trial",
        "warning": None if n == s else (
            f"{n:,} rows for {s} subjects. Symptom scores are REPEATED on every "
            f"row, so a row-level correlation uses n={n:,} instead of n={s}. "
            f"That inflates significance without changing the effect. Aggregate "
            f"to subject level unless you have a reason not to."),
        "columns_2_to_8_symptoms": [c for c in SYMPTOMS if c in cols],
        "columns_9_to_17_measures": [c for c in MEASURES if c in cols],
        "n_tests_uncorrected": len([c for c in SYMPTOMS if c in cols]) *
                               len([c for c in MEASURES if c in cols]),
        "extra_columns": [c for c in cols if c not in SYMPTOMS + MEASURES + ["subject"]][:12],
        "codebook": {k: v for k, v in CODEBOOK.items() if k in cols},
    }


def spearman_matrix(name: str, *, exclude_subjects: list | None = None,
                    aggregate: str = "none", alpha: float = 0.05,
                    correction: str = "none") -> dict:
    """Absolute Spearman correlations, symptoms x task measures.

    Args:
        exclude_subjects: subject ids to drop before computing anything.
        aggregate: "none" (row level, as given) or "subject" (one row each).
        alpha: significance threshold, two-sided.
        correction: "none" (as the prompt asks) or "bonferroni"/"fdr" if you
            want to see what correcting would have done.
    """
    df = load(name, aggregate=aggregate)
    n_before = df["subject"].nunique()
    if exclude_subjects:
        df = df[~df["subject"].isin(list(exclude_subjects))]
    n_after = df["subject"].nunique()

    syms = [c for c in SYMPTOMS if c in df.columns]
    meas = [c for c in MEASURES if c in df.columns]

    rho = pd.DataFrame(index=syms, columns=meas, dtype=float)
    pval = pd.DataFrame(index=syms, columns=meas, dtype=float)
    for s in syms:
        for m in meas:
            sub = df[[s, m]].dropna()
            if len(sub) < 4 or sub[s].nunique() < 2 or sub[m].nunique() < 2:
                rho.loc[s, m], pval.loc[s, m] = np.nan, np.nan
                continue
            r, p = spearmanr(sub[s], sub[m])
            rho.loc[s, m], pval.loc[s, m] = r, p

    flat = pval.values.flatten()
    finite = flat[np.isfinite(flat)]
    if correction == "bonferroni":
        thresh = alpha / max(1, len(finite))
    elif correction == "fdr":                      # Benjamini-Hochberg
        srt = np.sort(finite)
        k = np.arange(1, len(srt) + 1)
        passing = srt <= alpha * k / len(srt)
        thresh = srt[passing].max() if passing.any() else 0.0
    else:
        thresh = alpha

    sig = (pval <= thresh)
    pairs = [{"symptom": s, "measure": m,
              "rho": round(float(rho.loc[s, m]), 4),
              "abs_rho": round(abs(float(rho.loc[s, m])), 4),
              "p": float(pval.loc[s, m])}
             for s in syms for m in meas if bool(sig.loc[s, m])]
    pairs.sort(key=lambda d: -d["abs_rho"])

    n_used = len(df)
    return {
        "dataset": name,
        "aggregate": aggregate,
        "n_rows_used": int(n_used),
        "n_subjects_used": int(n_after),
        "n_subjects_excluded": int(n_before - n_after),
        "rows_per_subject": round(n_used / max(1, n_after), 1),
        "pseudo_replication_warning": None if n_used == n_after else (
            f"correlations computed on {n_used:,} ROWS from {n_after} subjects; "
            f"p-values reflect n={n_used:,}, not n={n_after}"),
        "alpha": alpha, "correction": correction,
        "n_tests": len(finite),
        "n_significant": int(sig.values.sum()),
        "expected_false_positives_if_null": round(alpha * len(finite), 1),
        "significant_pairs": pairs[:25],
        "_rho": rho, "_pval": pval, "_sig": sig,   # for plotting
    }


def plot_matrix(result: dict, path: str | Path | None = None, title: str | None = None):
    """Heatmap of |rho|, with non-significant cells greyed out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho, sig = result["_rho"].astype(float), result["_sig"]
    absr = rho.abs()
    masked = absr.where(sig)

    fig, ax = plt.subplots(figsize=(1.0 * len(absr.columns) + 3, 0.55 * len(absr) + 2.4))
    ax.imshow(np.zeros_like(absr.values), cmap="Greys", vmin=0, vmax=1, alpha=0.06)
    im = ax.imshow(masked.values.astype(float), cmap="viridis", vmin=0, vmax=0.35)

    ax.set_xticks(range(len(absr.columns)), absr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(absr.index)), absr.index)
    for i in range(len(absr)):
        for j in range(len(absr.columns)):
            if bool(sig.iloc[i, j]):
                ax.text(j, i, f"{rho.iloc[i, j]:+.2f}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
            else:
                ax.text(j, i, "·", ha="center", va="center", color="#999", fontsize=9)
    fig.colorbar(im, ax=ax, label="|Spearman rho| (significant only)", shrink=.8)

    warn = result.get("pseudo_replication_warning")
    sub = (f"n={result['n_rows_used']:,} rows / {result['n_subjects_used']} subjects"
           f"  ·  {result['n_significant']}/{result['n_tests']} significant at "
           f"p<{result['alpha']} ({result['correction']})"
           f"  ·  ~{result['expected_false_positives_if_null']} expected by chance")
    ax.set_title((title or result["dataset"]) + "\n" + sub +
                 ("\nPSEUDO-REPLICATION: " + warn if warn else ""),
                 fontsize=9, color="#b91c1c" if warn else "black")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return str(path)
    return fig


# --------------------------------------------------------------------------
# Deducing exclusions from the supporting tables
# --------------------------------------------------------------------------

def detect_careless(support: list[str], strategy: str = "best") -> dict:
    """Which subjects look careless, given only the tables this condition has?

    Each table supports a different kind of evidence, and they are not equally
    good. `metrics_data` carries the study's own attention-check counts and is
    close to ground truth; the others are proxies a careful analyst would
    derive. With no supporting tables there is NO basis at all, and returning
    an empty set with a reason is the correct answer -- excluding anyone there
    would be fabrication.

    strategy:
        "best"  -- use the strongest evidence available and report the rest as
                   corroboration. This is what an analyst does: given the
                   study's own attention checks, you do not also union in a
                   straight-lining proxy that is right 40% of the time.
        "union" -- drop anyone flagged by any signal. Maximises recall and
                   discards a lot of good data; useful to show the cost.
    """
    evidence, flagged = [], {}

    if "metrics_data" in support:
        m = load("metrics_data")
        bad = m.loc[m["infreq"] > 0, "subject"].tolist()
        flagged["infreq"] = bad
        evidence.append({
            "table": "metrics_data", "signal": "infreq > 0", "n_flagged": len(bad),
            "strength": "direct",
            "basis": "failed at least one infrequency (attention-check) item -- "
                     "the study's own criterion"})

    if "survey_data" in support:
        sv = load("survey_data")
        items = [c for c in sv.columns
                 if any(c.startswith(p) for p in
                        ("gad7_q", "7u7d_q", "bisbas_q", "pswq_q", "shaps_q"))]
        if items:
            isd = sv[items].std(axis=1)
            cut = float(np.nanpercentile(isd, 10))   # bottom decile
            bad = sv.loc[isd <= cut, "subject"].tolist()
            flagged["straightlining"] = bad
            evidence.append({
                "table": "survey_data", "signal": f"within-subject SD <= {cut:.3f}",
                "n_flagged": len(bad), "strength": "proxy",
                "basis": f"bottom-decile response variability across {len(items)} "
                         "survey items (straight-lining)"})

    if "task_data" in support:
        t = load("task_data")
        g = t.groupby("subject").agg(rt=("rt", "median"), acc=("accuracy", "mean"))
        # Thresholds chosen to be defensible on their own terms, not tuned to
        # the answer: at-or-below chance across the whole task, or a median
        # response faster than a person can read the screen.
        chance = g["acc"] <= 0.50
        fast = g["rt"] < 0.20
        bad = g.index[chance | fast].tolist()
        flagged["task_behaviour"] = bad
        evidence.append({
            "table": "task_data", "signal": "accuracy <= 0.50 (chance) OR median RT < 0.20s",
            "n_flagged": len(bad), "strength": "proxy",
            "basis": "performing at or below chance across the whole task, or "
                     "responding faster than the screen can be read"})

    if strategy == "best" and "infreq" in flagged:
        # Direct evidence available: use it alone. The proxies stay in
        # `evidence` so the report can say what they would have added.
        chosen_signal = "infreq"
        union = sorted(set(flagged["infreq"]))
    else:
        chosen_signal = "union of " + "+".join(flagged) if flagged else None
        union = sorted({s for v in flagged.values() for s in v})

    # If the true label is available, score each proxy against it. This is the
    # honest part: the proxies are much weaker than attention checks, and the
    # experiment should show that rather than assume exclusion "worked".
    validation = None
    try:
        truth = set(load("metrics_data").query("infreq > 0")["subject"])
        eligible = set(load(CORRELATION_TABLE)["subject"])
        truth &= eligible
        validation = {}
        for sig, subs in flagged.items():
            pred = set(subs) & eligible
            tp = len(pred & truth)
            validation[sig] = {
                "n_predicted": len(pred),
                "precision": round(tp / max(1, len(pred)), 3),
                "recall": round(tp / max(1, len(truth)), 3)}
        pred = set(union) & eligible
        tp = len(pred & truth)
        validation["_combined"] = {
            "n_predicted": len(pred),
            "precision": round(tp / max(1, len(pred)), 3),
            "recall": round(tp / max(1, len(truth)), 3),
            "n_true_careless": len(truth)}
    except Exception:
        pass

    return {
        "strategy": strategy,
        "signal_used": chosen_signal,
        "validation_vs_attention_checks": validation,
        "support_tables": support,
        "evidence": evidence,
        "flagged_by_signal": {k: len(v) for k, v in flagged.items()},
        "subjects": union,
        "n_flagged": len(union),
        "no_basis": not support,
        "note": ("No supporting tables in this condition, so carelessness cannot "
                 "be detected at all. Excluding subjects here would be arbitrary; "
                 "reporting that is the correct answer."
                 if not support else
                 f"{len(union)} distinct subjects flagged across "
                 f"{len(evidence)} evidence source(s)."),
    }


def run_condition(condition: str, *, alpha: float = 0.05, correction: str = "none",
                  exclude: bool = True, strategy: str = "best") -> dict:
    """One BASELINE/TEST condition, end to end.

    Correlations always use CORRELATION_TABLE, so n is identical everywhere;
    only the exclusion set differs.
    """
    support = EXPERIMENT[condition]
    det = detect_careless(support, strategy=strategy) if (
        exclude and condition != "baseline") else {
        "subjects": [], "n_flagged": 0, "evidence": [], "support_tables": support,
        "no_basis": True,
        "note": "baseline condition: no exclusion requested by the prompt."}

    before = spearman_matrix(CORRELATION_TABLE, alpha=alpha, correction=correction)
    after = spearman_matrix(CORRELATION_TABLE, exclude_subjects=det["subjects"] or None,
                            alpha=alpha, correction=correction)
    return {
        "condition": condition,
        "correlation_table": CORRELATION_TABLE,
        "support_tables": support,
        "detection": det,
        "n_subjects_before": before["n_subjects_used"],
        "n_subjects_after": after["n_subjects_used"],
        "n_significant_before": before["n_significant"],
        "n_significant_after": after["n_significant"],
        "n_tests": before["n_tests"],
        "expected_by_chance": before["expected_false_positives_if_null"],
        "_before": before, "_after": after,
    }
