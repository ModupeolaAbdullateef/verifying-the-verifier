"""
analysis.py - Statistical analysis and figures for Chapter 5.

Reads the scored outputs of score.py and produces the tables, statistical
tests and figures reported in the dissertation.

Inputs (in the results folder):
    runs.csv            invoice runs (Experiments 1 and 2)
    tickets_runs.csv    ticket-triage runs (replication)
    fields.csv          optional; used for the description-matching sensitivity analysis

Outputs (in results/analysis/):
    report.txt          every statistic quoted in Chapter 5
    tables/*.csv        the results tables
    figures/*.png       the results figures

Statistical conventions follow Section 3.7: significance level 0.05; Wilson
95% confidence intervals for proportions; exact McNemar tests for paired
binary outcomes, with Holm correction across comparisons; Cochran's Q for
three paired binary outcomes; Kruskal-Wallis with Dunn's post-hoc tests and
Holm correction for cost and latency; and a logistic regression estimated by
generalised estimating equations (GEE) with documents as clusters, as a
robustness check for the dependence between repeated trials.

Usage:
    python analysis.py
    python analysis.py --results results --rate 35
"""

from __future__ import annotations

import argparse
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from scipy import stats                             # noqa: E402

ROOT = Path(__file__).resolve().parent
CONDITIONS = ["C0", "C1", "C2", "C3"]
LABELS = {"C0": "C0 Unverified", "C1": "C1 Rules", "C2": "C2 Checker agent", "C3": "C3 Human"}
COLOURS = {"C0": "#8a8985", "C1": "#2a78d6", "C2": "#eb6834", "C3": "#1baf7a"}
ERROR_TYPES = ["omission", "substitution", "calculation", "hallucination", "format"]
HOURLY_RATES = [20, 35, 50]          # notional reviewer cost in USD per hour (sensitivity)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})


# --------------------------------------------------------------------------
# Statistical helpers
# --------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Proportion with its Wilson 95% confidence interval."""
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def fmt_ci(successes: int, n: int) -> str:
    p, lo, hi = wilson(successes, n)
    return "-" if n == 0 else f"{p:.1%} [{lo:.1%}, {hi:.1%}]"


def mcnemar_exact(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """Exact McNemar test on paired booleans; returns (a-only, b-only, p)."""
    a, b = a.astype(bool), b.astype(bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(only_a, only_b), n, 0.5))
    return only_a, only_b, p


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def cochran_q(matrix: np.ndarray) -> tuple[float, float]:
    """Cochran's Q for an n x k matrix of paired binary outcomes."""
    x = matrix.astype(int)
    k = x.shape[1]
    col, row, total = x.sum(axis=0), x.sum(axis=1), x.sum()
    denominator = k * total - (row ** 2).sum()
    if denominator == 0:
        return (0.0, 1.0)
    q = (k - 1) * (k * (col ** 2).sum() - total ** 2) / denominator
    return q, float(stats.chi2.sf(q, k - 1))


def cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return math.sqrt(chi2 / (n * (min(table.shape) - 1)))


def dunn_holm(groups: dict[str, np.ndarray]) -> list[tuple[str, str, float, float]]:
    """Dunn's post-hoc test on ranks with Holm correction."""
    names = list(groups)
    values = np.concatenate([groups[g] for g in names])
    ranks = stats.rankdata(values)
    n = len(values)
    _, counts = np.unique(values, return_counts=True)
    tie = 1 - (counts ** 3 - counts).sum() / (n ** 3 - n)
    mean_rank, sizes, start = {}, {}, 0
    for g in names:
        size = len(groups[g])
        mean_rank[g], sizes[g] = ranks[start:start + size].mean(), size
        start += size
    pairs, raw = [], []
    for a, b in combinations(names, 2):
        se = math.sqrt(tie * n * (n + 1) / 12 * (1 / sizes[a] + 1 / sizes[b]))
        z = (mean_rank[a] - mean_rank[b]) / se
        pairs.append((a, b, z))
        raw.append(2 * stats.norm.sf(abs(z)))
    return [(a, b, z, p) for (a, b, z), p in zip(pairs, holm(raw))]


def median_iqr(values: pd.Series) -> str:
    q1, q2, q3 = values.quantile([0.25, 0.5, 0.75])
    return f"{q2:.4g} ({q1:.4g}-{q3:.4g})"


# --------------------------------------------------------------------------
# Report writer
# --------------------------------------------------------------------------

class Report:
    def __init__(self, path: Path):
        self.path, self.lines = path, []

    def h(self, title: str) -> None:
        self.lines += ["", title, "=" * len(title)]

    def w(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, df: pd.DataFrame, name: str, folder: Path) -> None:
        df.to_csv(folder / f"{name}.csv", index=False)
        self.lines.append(df.to_string(index=False))

    def save(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def bar_ci(ax, x, rows: list[tuple[str, int, int]], width: float, offset: float = 0.0) -> None:
    """Bars with Wilson confidence intervals and direct value labels."""
    for i, (cond, k, n) in enumerate(rows):
        if n == 0:
            continue
        p, lo, hi = wilson(k, n)
        xpos = x[i] + offset
        ax.bar(xpos, p * 100, width, color=COLOURS[cond], edgecolor="white", linewidth=1)
        ax.errorbar(xpos, p * 100, yerr=[[max(0, p - lo) * 100], [max(0, hi - p) * 100]], color="#0b0b0b",
                    capsize=3, linewidth=1)
        ax.text(xpos, hi * 100 + 2, f"{p:.0%}", ha="center", va="bottom", fontsize=8.5,
                color="#0b0b0b")


# --------------------------------------------------------------------------
# Experiment 1
# --------------------------------------------------------------------------

def experiment1(runs: pd.DataFrame, fields: pd.DataFrame | None, rep: Report, tables: Path,
                figures: Path, rate: float) -> dict:
    nat = runs[runs.experiment == "natural"].copy()
    auto = nat[nat.condition.isin(["C0", "C1", "C2"])]
    rep.h("Experiment 1 - natural errors")

    rows = []
    for cond in CONDITIONS:
        rs = nat[nat.condition == cond]
        if rs.empty:
            continue
        n = len(rs)
        rows.append({
            "condition": cond, "runs": n,
            "initial_error": fmt_ci(int(rs.initial_has_error.sum()), n),
            "task_success": fmt_ci(int(rs.task_success.sum()), n),
            "decision_correct": fmt_ci(int(rs.decision_correct.sum()), n),
            "final_hallucination": fmt_ci(int(rs.final_hallucination_present.sum()), n),
            "mean_field_error_rate": f"{rs.final_error_rate.mean():.2%}",
            "cost_usd_median_iqr": median_iqr(rs.token_cost_usd),
            "latency_s_median_iqr": median_iqr(rs.wall_clock_s),
        })
    rep.table(pd.DataFrame(rows), "t5_1_experiment1_outcomes", tables)

    # Omnibus and paired tests on task success and decision correctness (C0-C2)
    for outcome in ("task_success", "decision_correct"):
        ct = pd.crosstab(auto.condition, auto[outcome]).values
        chi2, p, dof, expected = stats.chi2_contingency(ct, correction=False)
        rep.w(f"\n{outcome}: chi-square({dof}) = {chi2:.2f}, p = {p:.3g}, "
              f"Cramer's V = {cramers_v(ct):.3f}")
        wide = auto.pivot_table(index=["task_id", "trial"], columns="condition",
                                values=outcome, aggfunc="first").dropna()
        tests = [(c, *mcnemar_exact(wide[c], wide["C0"])) for c in ("C1", "C2")]
        tests.append(("C1 vs C2", *mcnemar_exact(wide["C1"], wide["C2"])))
        adjusted = holm([t[3] for t in tests])
        for (name, only_a, only_b, p_raw), p_adj in zip(tests, adjusted):
            label = name if " vs " in name else f"{name} vs C0"
            rep.w(f"  McNemar {label}: discordant {only_a}/{only_b}, p = {p_raw:.3g}, "
                  f"Holm p = {p_adj:.3g} (pairs = {len(wide)})")

    # Robustness: GEE logistic regression clustered by document
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        data = auto.assign(success=auto.task_success.astype(int),
                           condition=pd.Categorical(auto.condition, ["C0", "C1", "C2"]),
                           complexity=pd.Categorical(auto.complexity, ["low", "medium", "high"]))
        model = smf.gee("success ~ condition + complexity", groups="task_id", data=data,
                        family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = model.conf_int()
        rep.w("\nGEE logistic regression of task success (exchangeable, clustered by document):")
        for term in model.params.index:
            rep.w(f"  {term:<28} OR = {math.exp(model.params[term]):6.2f} "
                  f"[{math.exp(ci.loc[term, 0]):.2f}, {math.exp(ci.loc[term, 1]):.2f}], "
                  f"p = {model.pvalues[term]:.3g}")
    except Exception as error:      # reported rather than hidden
        rep.w(f"\nGEE model could not be estimated: {error}")

    # Gate behaviour on natural errors
    rep.w("\nGate behaviour on natural errors (runs whose first extraction had an error):")
    gate_rows = []
    for cond in ("C1", "C2", "C3"):
        rs = nat[nat.condition == cond]
        if rs.empty:
            continue
        err, clean = rs[rs.initial_has_error], rs[~rs.initial_has_error]
        flagged = rs[rs.gate_flagged]
        gate_rows.append({
            "condition": cond, "erroneous_runs": len(err),
            "recall_flagged": fmt_ci(int(err.gate_flagged.sum()), len(err)),
            "fixed_at_gate": fmt_ci(int((err.gated_errors == 0).sum()), len(err)),
            "precision": fmt_ci(int(flagged.initial_has_error.sum()), len(flagged)),
            "false_flag_rate": fmt_ci(int(clean.gate_flagged.sum()), len(clean)),
        })
    rep.table(pd.DataFrame(gate_rows), "t5_2_experiment1_gate", tables)

    # Concentration of natural errors
    c0 = nat[nat.condition == "C0"]
    by_doc = c0.groupby(["task_id", "layout", "complexity"]).initial_has_error.mean().reset_index()
    by_doc = by_doc.sort_values("initial_has_error", ascending=False)
    erring = by_doc[by_doc.initial_has_error > 0]
    rep.w(f"\nDocuments with any natural error: {len(erring)} of {len(by_doc)}")
    rep.w(by_doc.head(8).to_string(index=False))
    systematic = by_doc[by_doc.initial_has_error >= 0.9].task_id
    share = c0[c0.task_id.isin(systematic)].initial_has_error.sum() / max(1, c0.initial_has_error.sum())
    rep.w(f"Share of erroneous C0 runs from documents failing in >=90% of trials: {share:.1%}")
    by_layout = c0.groupby("layout").initial_has_error.agg(["mean", "size"])
    by_cx = c0.groupby("complexity").initial_has_error.agg(["mean", "size"])
    rep.w("Initial error rate by layout:\n" + by_layout.to_string())
    rep.w("Initial error rate by complexity:\n" + by_cx.to_string())
    prop = c0.groupby("initial_has_error").decision_correct.agg(["mean", "size"])
    rep.w("Propagation (C0): decision correct by whether the first extraction erred:\n"
          + prop.to_string())

    # Cost and latency
    rep.w("\nCost and latency (C0-C2): Kruskal-Wallis with Dunn-Holm post-hoc")
    for metric in ("token_cost_usd", "wall_clock_s"):
        groups = {c: auto[auto.condition == c][metric].values for c in ("C0", "C1", "C2")}
        h, p = stats.kruskal(*groups.values())
        eps = (h - len(groups) + 1) / (sum(len(v) for v in groups.values()) - len(groups))
        rep.w(f"  {metric}: H = {h:.1f}, p = {p:.3g}, epsilon-squared = {eps:.3f}")
        for a, b, z, padj in dunn_holm(groups):
            rep.w(f"    {a} vs {b}: z = {z:.2f}, Holm p = {padj:.3g}")

    # Sensitivity: ignore a leading quantity in line-item descriptions
    if fields is not None and {"stage", "field", "outcome", "pred", "truth"} <= set(fields.columns):
        final = fields[(fields.stage == "final") & (fields.outcome != "correct")
                       & (fields.trial <= 30)].copy()
        prefix = final.field.str.endswith("description") & final.apply(
            lambda r: bool(re.match(r"^\d+\s*x\s+", str(r.pred)))
            and re.sub(r"^\d+\s*x\s+", "", str(r.pred)).strip().lower()
            == str(r.truth).strip().lower(), axis=1)
        remaining = final[~prefix].groupby("run_id").size()
        sens = auto.assign(sens_success=auto.decision_correct
                           & ~auto.run_id.isin(remaining.index))
        rep.w("\nSensitivity: task success when a leading quantity in descriptions is ignored")
        for cond in ("C0", "C1", "C2"):
            rs = sens[sens.condition == cond]
            rep.w(f"  {cond}: strict {rs.task_success.mean():.1%} -> lenient "
                  f"{rs.sens_success.mean():.1%}")

    # Figure 5.1: natural error rate by document
    fig, ax = plt.subplots(figsize=(8, 3.2))
    layout_colour = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    ax.bar(range(len(by_doc)), by_doc.initial_has_error * 100,
           color=[layout_colour[l] for l in by_doc.layout], edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(by_doc)), by_doc.task_id.str.replace("DOC-", ""), rotation=90,
                  fontsize=8)
    ax.set_ylabel("Trials with an extraction error (%)")
    ax.set_ylim(0, 105)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in layout_colour.values()]
    ax.legend(handles, ["Layout A (table)", "Layout B (key-value)", "Layout C (email)"],
              frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(figures / "fig5_1_errors_by_document.png")
    plt.close(fig)
    return {"nat": nat}


# --------------------------------------------------------------------------
# Experiment 2
# --------------------------------------------------------------------------

def experiment2(runs: pd.DataFrame, rep: Report, tables: Path, figures: Path) -> None:
    sf = runs[runs.experiment == "fault"].copy()
    rep.h("Experiment 2 - seeded faults")
    faulty, clean = sf[sf.fault_present == True], sf[sf.fault_present == False]  # noqa: E712

    rows = []
    for cond in CONDITIONS:
        f, c, a = faulty[faulty.condition == cond], clean[clean.condition == cond], sf[sf.condition == cond]
        if a.empty:
            continue
        rows.append({"condition": cond, "faulty": len(f), "clean": len(c),
                     "detected": fmt_ci(int(f.fault_detected.sum()), len(f)),
                     "false_alarm": fmt_ci(int(c.false_alarm.sum()), len(c)),
                     "removed": fmt_ci(int(f.fault_removed_at_gate.sum()), len(f)),
                     "task_success": fmt_ci(int(a.task_success.sum()), len(a)),
                     "decision_correct": fmt_ci(int(a.decision_correct.sum()), len(a)),
                     "cost_usd_median_iqr": median_iqr(a.token_cost_usd),
                     "latency_s_median_iqr": median_iqr(a.wall_clock_s)})
    rep.table(pd.DataFrame(rows), "t5_3_experiment2_outcomes", tables)

    type_rows = []
    for cond in ("C1", "C2", "C3"):
        f = faulty[faulty.condition == cond]
        row = {"condition": cond}
        for t in ERROR_TYPES:
            ft = f[f.fault_type == t]
            row[t] = fmt_ci(int(ft.fault_detected.sum()), len(ft))
        for label, mask in (("constraint_visible", f.fault_c1_visible == True),   # noqa: E712
                            ("semantic", f.fault_c1_visible == False)):           # noqa: E712
            row[label] = fmt_ci(int(f[mask].fault_detected.sum()), int(mask.sum()))
        type_rows.append(row)
    rep.table(pd.DataFrame(type_rows), "t5_4_detection_by_type", tables)

    # C1 vs C2 on all seeded records
    wide = faulty.pivot_table(index=["task_id", "trial"], columns="condition",
                              values="fault_detected", aggfunc="first")
    a_only, b_only, p = mcnemar_exact(wide["C2"].dropna(), wide["C1"].dropna())
    rep.w(f"\nDetection, C2 vs C1 on all {len(wide)} faulty records: McNemar discordant "
          f"{a_only}/{b_only}, p = {p:.3g}")
    clean_wide = clean.pivot_table(index=["task_id", "trial"], columns="condition",
                                   values="false_alarm", aggfunc="first")
    a_only, b_only, p = mcnemar_exact(clean_wide["C2"], clean_wide["C1"])
    rep.w(f"False alarms, C2 vs C1 on {len(clean_wide)} control records: discordant "
          f"{a_only}/{b_only}, p = {p:.3g}")

    # Three-way paired comparison on the records reviewed in C3
    if "C3" in wide.columns:
        three = wide[["C1", "C2", "C3"]].dropna().astype(bool)
        q, pq = cochran_q(three.values)
        rep.w(f"\nOn the {len(three)} faulty records reviewed in C3: detection C1 "
              f"{three.C1.mean():.1%}, C2 {three.C2.mean():.1%}, C3 {three.C3.mean():.1%}")
        rep.w(f"  Cochran's Q(2) = {q:.2f}, p = {pq:.3g}")
        tests = [(a, b, *mcnemar_exact(three[a], three[b]))
                 for a, b in (("C2", "C1"), ("C2", "C3"), ("C1", "C3"))]
        for (a, b, x, y, p_raw), p_adj in zip(tests, holm([t[4] for t in tests])):
            rep.w(f"  McNemar {a} vs {b}: discordant {x}/{y}, p = {p_raw:.3g}, Holm p = {p_adj:.3g}")

    # Propagation: which unverified faults change the decision
    c0 = sf[(sf.condition == "C0")]
    prop = c0.groupby("fault_type").decision_correct.agg(["mean", "size"])
    rep.w("\nPropagation (C0): decision correct by seeded fault type:\n" + prop.to_string())
    variants = c0[c0.fault_present == True].groupby("fault_variant").decision_correct.mean()  # noqa: E712
    rep.w("Decision correct by variant (C0, lowest first):\n"
          + variants.sort_values().head(6).to_string())

    # Human reviewer by session
    c3 = faulty[faulty.condition == "C3"]
    if not c3.empty and "session" in c3.columns:
        by_session = c3.groupby("session").agg(items=("fault_detected", "size"),
                                               detected=("fault_detected", "mean"),
                                               median_seconds=("human_seconds", "median"))
        rep.w("\nC3 by review session (faulty records):\n" + by_session.to_string())
        first = c3[c3.session == c3.session.min()]
        later = c3[c3.session != c3.session.min()]
        table = [[int(first.fault_detected.sum()), int((~first.fault_detected.astype(bool)).sum())],
                 [int(later.fault_detected.sum()), int((~later.fault_detected.astype(bool)).sum())]]
        rep.w(f"  Fisher exact, first session vs later: p = {stats.fisher_exact(table)[1]:.3g}")

    # Figure 5.2: detection by fault type
    fig, ax = plt.subplots(figsize=(8, 3.4))
    groups = ERROR_TYPES + ["semantic"]
    x = np.arange(len(groups))
    width = 0.26
    for i, cond in enumerate(("C1", "C2", "C3")):
        f = faulty[faulty.condition == cond]
        rows = []
        for g in groups:
            fg = f[f.fault_c1_visible == False] if g == "semantic" else f[f.fault_type == g]  # noqa: E712
            rows.append((cond, int(fg.fault_detected.sum()), len(fg)))
        for j, (c, k, n) in enumerate(rows):
            if n:
                p, lo, hi = wilson(k, n)
                ax.bar(x[j] + (i - 1) * width, p * 100, width, color=COLOURS[cond],
                       edgecolor="white", linewidth=1, label=LABELS[cond] if j == 0 else None)
                ax.errorbar(x[j] + (i - 1) * width, p * 100, yerr=[[max(0, p - lo) * 100], [max(0, hi - p) * 100]],
                            color="#0b0b0b", capsize=2, linewidth=0.8)
    ax.set_xticks(x, [g.capitalize() for g in groups[:-1]] + ["All semantic"])
    ax.set_ylabel("Seeded faults detected (%)")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="x", visible=False)
    fig.savefig(figures / "fig5_2_detection_by_type.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Trade-off (RQ3)
# --------------------------------------------------------------------------

def tradeoff(runs: pd.DataFrame, rep: Report, tables: Path, figures: Path, rate: float) -> None:
    rep.h("Trade-off - cost per error avoided (RQ3)")
    rows = []
    for experiment in ("natural", "fault"):
        ex = runs[runs.experiment == experiment]
        base = ex[ex.condition == "C0"]
        base_fail, base_cost = 1 - base.task_success.mean(), base.token_cost_usd.mean()
        for cond in CONDITIONS:
            rs = ex[ex.condition == cond]
            if rs.empty:
                continue
            fail = 1 - rs.task_success.mean()
            token = rs.token_cost_usd.mean()
            human = rs.human_seconds.mean() / 3600
            row = {"experiment": experiment, "condition": cond, "runs": len(rs),
                   "task_success": round(1 - fail, 4), "token_cost_usd": round(token, 5),
                   "review_seconds": round(rs.human_seconds.mean(), 1)}
            for r in HOURLY_RATES:
                total = token + human * r
                row[f"total_cost_usd_at_{r}"] = round(total, 4)
                avoided = base_fail - fail
                row[f"cost_per_failure_avoided_at_{r}"] = (
                    round((total - base_cost) / avoided, 4) if cond != "C0" and avoided > 0 else None)
            rows.append(row)
    df = pd.DataFrame(rows)
    rep.table(df, "t5_5_tradeoff", tables)
    rep.w(f"Human review valued at USD {HOURLY_RATES} per hour; the primary rate is USD {rate:g}.")

    # Figure 5.3: accuracy-cost plane (log cost)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, experiment, title in zip(axes, ("natural", "fault"),
                                     ("Experiment 1: natural errors", "Experiment 2: seeded faults")):
        sub = df[df.experiment == experiment]
        for _, r in sub.iterrows():
            cost = r[f"total_cost_usd_at_{int(rate)}"] if f"total_cost_usd_at_{int(rate)}" in r else r.token_cost_usd
            ax.scatter(cost, r.task_success * 100, s=70, color=COLOURS[r.condition],
                       edgecolor="white", linewidth=1.5, zorder=3)
            below = r.condition == "C0"
            ax.annotate(r.condition, (cost, r.task_success * 100), textcoords="offset points",
                        xytext=(0, -15) if below else (7, -3), ha="center" if below else "left",
                        fontsize=9)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"Cost per run, USD (log scale; review at ${rate:g}/h)")
    axes[0].set_ylabel("Task success (%)")
    axes[0].set_ylim(0, 105)
    fig.savefig(figures / "fig5_3_accuracy_cost.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Replication
# --------------------------------------------------------------------------

def replication(tickets: pd.DataFrame, rep: Report, tables: Path) -> None:
    rep.h("Replication - ticket triage")
    rows = []
    for cond in CONDITIONS:
        rs = tickets[tickets.condition == cond]
        if rs.empty:
            continue
        n = len(rs)
        downstream = ((rs.initial_errors == 0) & ~rs.priority_correct.astype(bool)).sum()
        rows.append({"condition": cond, "runs": n,
                     "initial_error": fmt_ci(int(rs.initial_has_error.sum()), n),
                     "flagged": fmt_ci(int(rs.gate_flagged.sum()), n),
                     "task_success": fmt_ci(int(rs.task_success.sum()), n),
                     "priority_correct": fmt_ci(int(rs.priority_correct.sum()), n),
                     "priority_error_after_correct_triage": fmt_ci(int(downstream), n),
                     "cost_usd_median_iqr": median_iqr(rs.token_cost_usd),
                     "latency_s_median_iqr": median_iqr(rs.wall_clock_s)})
    rep.table(pd.DataFrame(rows), "t5_6_replication", tables)
    auto = tickets[tickets.condition.isin(["C0", "C1", "C2"])]
    by_ticket = auto.groupby("task_id").agg(task_success=("task_success", "mean"),
                                            triage_error=("initial_has_error", "mean"),
                                            priority_correct=("priority_correct", "mean"))
    rep.w("\nBy ticket (C0-C2 pooled):\n" + by_ticket.round(2).to_string())
    wrong = auto[~auto.priority_correct.astype(bool)]
    rep.w(f"Wrong priorities: {len(wrong)}; with correct triage: "
          f"{int((wrong.initial_errors == 0).sum())} ({(wrong.initial_errors == 0).mean():.1%})")


# --------------------------------------------------------------------------

def attach_sessions(runs: pd.DataFrame, results: Path) -> pd.DataFrame:
    """Add the review session of each C3 run from the review log."""
    queue, reviews = results / "review_queue.jsonl", results / "reviews.jsonl"
    if not (queue.exists() and reviews.exists()):
        return runs
    q = pd.read_json(queue, lines=True)[["review_id", "task_id", "trial"]]
    r = pd.read_json(reviews, lines=True)[["review_id", "session"]]
    sessions = q.merge(r, on="review_id")
    out = runs.merge(sessions[["task_id", "trial", "session"]], on=["task_id", "trial"], how="left")
    out.loc[out.condition != "C3", "session"] = np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical analysis and figures for Chapter 5.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--rate", type=float, default=35.0,
                        help="primary notional reviewer cost, USD per hour")
    args = parser.parse_args()
    results = ROOT / args.results
    out = results / "analysis"
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    runs = attach_sessions(pd.read_csv(results / "runs.csv"), results)
    for col in ("task_success", "decision_correct", "initial_has_error", "gate_flagged",
                "final_hallucination_present"):
        runs[col] = runs[col].astype(bool)
    fields = pd.read_csv(results / "fields.csv") if (results / "fields.csv").exists() else None
    tickets = (pd.read_csv(results / "tickets_runs.csv")
               if (results / "tickets_runs.csv").exists() else None)

    rep = Report(out / "report.txt")
    rep.w("Analysis of the Verifying the Verifier experiments")
    rep.w(f"Invoice runs: {len(runs)}; ticket runs: {0 if tickets is None else len(tickets)}")
    experiment1(runs, fields, rep, tables, figures, args.rate)
    experiment2(runs, rep, tables, figures)
    tradeoff(runs, rep, tables, figures, args.rate)
    if tickets is not None:
        replication(tickets, rep, tables)
    rep.save()
    print(f"Analysis written to {out} (report.txt, tables/, figures/)")


if __name__ == "__main__":
    main()
