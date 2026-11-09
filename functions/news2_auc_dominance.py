"""Pairwise complete dominance of partial NEWS2 scores using subset AUCs.

Each pair of parameters is compared on the same 32 subsets of the other
five parameters, including the empty subset (AUC = 0.5). All subset AUCs
must use the same patients, folds, and aggregation. Results are descriptive,
not statistical significance tests. No patient-level data are required.
"""

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .news2_auc_shapley import PARAMETER_LABELS, OUTCOME_LABELS


def compute_news2_auc_dominance(results, feature_order, tolerance=1e-10):
    """Return win-fraction matrix, pairwise summary, and all comparisons.

    Strict complete dominance means the row parameter's incremental AUC
    exceeds the column parameter's in EVERY common context by more than
    tolerance. Numerical ties do not count as wins. Dominance need not exist.
    """
    features = list(feature_order)
    if len(features) != 7 or len(set(features)) != 7:
        raise ValueError("Provide seven unique NEWS2 parameters.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative.")
    required = {"features_json", "auc"}
    if not required.issubset(results.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(results.columns))}")

    aucs = {frozenset(): 0.5}
    for row in results.itertuples(index=False):
        names = json.loads(row.features_json)
        subset = frozenset(names)
        auc = float(row.auc)
        if (not subset or not subset.issubset(features)
                or len(names) != len(subset) or subset in aucs):
            raise ValueError(f"Invalid or duplicate subset: {names}")
        if not np.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError(f"Invalid AUC: {auc}")
        aucs[subset] = auc
    expected = {frozenset(s) for k in range(8)
                for s in combinations(features, k)}
    if set(aucs) != expected:
        raise ValueError("All 127 non-empty subset AUCs are required.")

    comparisons = []
    summaries = []
    wins_matrix = pd.DataFrame(np.nan, index=features, columns=features)
    for first, second in combinations(features, 2):
        others = [f for f in features if f not in (first, second)]
        differences = []
        for size in range(6):
            for context in combinations(others, size):
                base = frozenset(context)
                gain_first = aucs[base | {first}] - aucs[base]
                gain_second = aucs[base | {second}] - aucs[base]
                difference = gain_first - gain_second
                differences.append(difference)
                comparisons.append({
                    "parameter_a": first, "parameter_b": second,
                    "context_json": json.dumps(list(context)),
                    "context_size": size, "baseline_auc": aucs[base],
                    "gain_a": gain_first, "gain_b": gain_second,
                    "gain_difference_a_minus_b": difference,
                })
        differences = np.asarray(differences)
        count = len(differences)
        wins = int(np.sum(differences > tolerance))
        losses = int(np.sum(differences < -tolerance))
        ties = count - wins - losses
        wins_matrix.loc[first, second] = wins / count
        wins_matrix.loc[second, first] = losses / count
        for a, b, w, loss, minimum in (
            (first, second, wins, losses, differences.min()),
            (second, first, losses, wins, -differences.max()),
        ):
            summaries.append({
                "row_parameter": a, "column_parameter": b,
                "n_contexts": count, "wins": w, "losses": loss,
                "ties": ties, "win_fraction": w / count,
                "complete_dominance": w == count,
                "minimum_gain_advantage": float(minimum),
            })
    return wins_matrix, pd.DataFrame(summaries), pd.DataFrame(comparisons)


def plot_news2_auc_dominance(results, feature_order, cohort, outcome,
                            output_dir=None, tolerance=1e-10):
    """Plot one 7x7 matrix; export figure and auditable comparisons.

    Colors encode the fraction of 32 contexts where the row wins.
    CD identifies strict complete dominance (32 wins out of 32).
    Returns (pairwise_summary, figure).
    """
    matrix, summary, comparisons = compute_news2_auc_dominance(
        results, feature_order, tolerance=tolerance,
    )
    for table in (summary, comparisons):
        table["cohort"] = cohort
        table["outcome"] = outcome

    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        figure, ax = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)
        cmap = plt.get_cmap("turbo").copy()
        cmap.set_bad("#F1F5F9")
        image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(range(7), PARAMETER_LABELS, rotation=40, ha="right")
        ax.set_yticks(range(7), PARAMETER_LABELS)
        ax.set_xlabel("Compared parameter (column)")
        ax.set_ylabel("Candidate parameter (row)")
        ax.set_title(
            f"{cohort} | {OUTCOME_LABELS.get(outcome, outcome)}\n"
            "Dominance analysis of NEWS2 discrimination",
            loc="left", fontsize=14, pad=16,
        )
        lookup = summary.set_index(["row_parameter", "column_parameter"])
        features = list(feature_order)
        for i, first in enumerate(features):
            for j, second in enumerate(features):
                if i == j:
                    text, color = "—", "#64748B"
                else:
                    row = lookup.loc[(first, second)]
                    text = f"{int(row.wins)}/32"
                    if row.complete_dominance:
                        text += "\nCD"
                    color = "white" if row.win_fraction < 0.2 or row.win_fraction > 0.8 else "#0F172A"
                ax.text(j, i, text, ha="center", va="center", color=color,
                        fontsize=10, fontweight="bold" if "CD" in text else "normal")
        bar = figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        bar.set_label("Fraction of contexts where the row adds more AUC")
        figure.suptitle(
            "Cells: wins / 32 common contexts. CD: strict complete dominance.\n"
            "Includes the empty context (AUC = 0.5); ties are not wins. "
            "Descriptive comparisons, without uncertainty intervals.",
            fontsize=10, y=1.03,
        )
        if output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            slug = "".join(c if c.isalnum() else "_"
                           for c in f"{cohort}_{outcome}".lower())
            stem = f"{slug}_auc_dominance"
            for extension in ("png", "pdf"):
                figure.savefig(destination / f"{stem}.{extension}",
                               dpi=300, bbox_inches="tight")
            summary.to_csv(destination / f"{stem}_summary.csv", index=False)
            comparisons.to_csv(destination / f"{stem}_contexts.csv", index=False)
    return summary, figure
