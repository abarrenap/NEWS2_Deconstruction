"""Exact Shapley attribution of NEWS2 discrimination above chance.

Uses the 127 subset AUCs; no patient data or SHAP package is required.
All subsets must use the same patients, folds, and AUC aggregation.
"""

import json
from itertools import combinations
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARAMETER_LABELS = (
    "Respiratory rate", "Oxygen saturation", "Supplemental oxygen",
    "Systolic blood pressure", "Heart rate", "Consciousness", "Temperature",
)
OUTCOME_LABELS = {"M30D": "30-day mortality", "UVI": "ICU admission"}


def decompose_news2_auc(results, feature_order):
    """
    Exact Shapley decomposition of AUC above chance.

    Required columns:
        features_json: JSON list of parameters in each subset
        auc: AUC calculated consistently across all subsets

    The empty subset is assigned AUC = 0.5.
    """
    features = list(feature_order)
    n = len(features)

    if n != 7 or len(set(features)) != n:
        raise ValueError("Provide the seven unique NEWS2 parameters.")

    auc_by_subset = {frozenset(): 0.5}

    for row in results.itertuples(index=False):
        subset = frozenset(json.loads(row.features_json))
        auc = float(row.auc)

        if not subset or not subset.issubset(features):
            raise ValueError(f"Invalid subset: {subset}")
        if subset in auc_by_subset:
            raise ValueError(f"Duplicate subset: {subset}")
        if not np.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError(f"Invalid AUC for {subset}: {auc}")

        auc_by_subset[subset] = auc

    expected = {
        frozenset(subset)
        for size in range(n + 1)
        for subset in combinations(features, size)
    }

    if set(auc_by_subset) != expected:
        raise ValueError("Results must contain all 127 non-empty subsets.")

    full_auc = auc_by_subset[frozenset(features)]
    auc_above_chance = full_auc - 0.5

    if auc_above_chance <= 0:
        raise ValueError(
            "Percentage attribution requires full-score AUC above 0.5."
        )

    rows = []

    for feature in features:
        others = [name for name in features if name != feature]
        contribution = 0.0

        for size in range(n):
            weight = (
                factorial(size)
                * factorial(n - size - 1)
                / factorial(n)
            )

            for subset_tuple in combinations(others, size):
                subset = frozenset(subset_tuple)
                expanded = subset | {feature}

                contribution += weight * (
                    auc_by_subset[expanded] - auc_by_subset[subset]
                )

        rows.append({
            "parameter": feature,
            "shapley_auc": contribution,
            "percentage": 100 * contribution / auc_above_chance,
            "full_auc": full_auc,
        })

    attribution = pd.DataFrame(rows)

    np.testing.assert_allclose(
        attribution["shapley_auc"].sum(),
        auc_above_chance,
        atol=1e-10,
    )

    return attribution


def plot_news2_auc_attribution(results, feature_order, cohort, outcome,
                               output_dir=None, xlim=None):
    """Draw one experiment's AUC attribution and export PNG, PDF, and CSV.

    Bars retain the same parameter order across experiments. Negative
    contributions are retained. Axes adapt to each experiment unless xlim
    is supplied; percentages always refer to that experiment's full AUC.
    Returns (attribution, figure).
    """
    attribution = decompose_news2_auc(results, feature_order)
    attribution["cohort"] = cohort
    attribution["outcome"] = outcome
    percentages = attribution["percentage"].to_numpy()
    full_auc = attribution["full_auc"].iloc[0]

    with plt.rc_context({"font.size": 11, "axes.titlesize": 14,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        positions = np.arange(len(attribution))
        colors = np.where(percentages >= 0, "#2878B5", "#C44E52")
        ax.barh(positions, percentages, color=colors, height=0.65)
        ax.set_yticks(positions, PARAMETER_LABELS)
        ax.invert_yaxis()
        ax.axvline(0, color="#475569", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(
            f"{cohort} | {OUTCOME_LABELS.get(outcome, outcome)}\n"
            f"Shapley decomposition of discrimination (full-score AUC = {full_auc:.3f})",
            loc="left", pad=16,
        )
        ax.set_xlabel("Share of full-score AUC above chance (%)", labelpad=10)
        for position, value in zip(positions, percentages):
            ax.annotate(f"{value:.1f}%", (value, position),
                        xytext=(5 if value >= 0 else -5, 0),
                        textcoords="offset points", va="center",
                        ha="left" if value >= 0 else "right", fontsize=10)
        if xlim is None:
            low = min(0.0, float(percentages.min()))
            high = max(0.0, float(percentages.max()))
            padding = max(4.0, (high - low) * 0.18)
            ax.set_xlim(low - padding, high + padding)
        else:
            ax.set_xlim(*xlim)

        if output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            slug = "".join(c if c.isalnum() else "_"
                           for c in f"{cohort}_{outcome}".lower())
            stem = f"{slug}_shapley_auc"
            for extension in ("png", "pdf"):
                figure.savefig(destination / f"{stem}.{extension}",
                               dpi=300, bbox_inches="tight")
            attribution.to_csv(destination / f"{stem}.csv", index=False)

    return attribution, figure
