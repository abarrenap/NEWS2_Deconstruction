"""Fold assignment and plots used by NEWS2 scale deconstruction."""

from __future__ import annotations
from typing import Sequence
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import StratifiedKFold

TURBO_FEATURE_COLORS = [
    "#30123B", "#4661D6", "#35A7C5", "#62D84E",
    "#E5E419", "#F68D3D", "#7A0403",
]

def add_stratified_folds(
    data: pd.DataFrame,
    *,
    target: str,
    fold_column: str = "cv_fold",
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add one deterministic, target-stratified fold assignment to ``data``."""
    if set(data[target].dropna().unique()) != {0, 1}:
        raise ValueError(f"{target!r} must be binary and encoded as 0/1.")
    if data[target].value_counts().min() < n_splits:
        raise ValueError(f"Each class needs at least {n_splits} observations.")

    folded = data.copy()
    folded[fold_column] = -1
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    for fold, (_, test_positions) in enumerate(splitter.split(folded, folded[target])):
        folded.iloc[test_positions, folded.columns.get_loc(fold_column)] = fold
    return folded

def make_branching_subset_plot(
    results: pd.DataFrame,
    feature_order: Sequence[str],
    *,
    news2_auc: float | None = None,
    title: str = "NEWS2 deconstruction",
) -> go.Figure:
    """Create an interactive subset lattice with optional NEWS2 AUC reference."""
    expected_subsets = 2 ** len(feature_order) - 1
    if len(results) != expected_subsets:
        raise ValueError(f"Expected {expected_subsets} subsets; found {len(results)}.")

    plot_results = results.copy()
    plot_results["feature_set"] = plot_results["features_json"].map(
        lambda value: frozenset(json.loads(value))
    )
    if news2_auc is not None and not 0 <= float(news2_auc) <= 1:
        raise ValueError("news2_auc must be between 0 and 1.")
    # Horizontal position is the number of retained measurements; vertical
    # position is the actual outer-CV AUC, not a within-layer rank.
    node_positions = {
        row.subset_key: (row.n_features, row.auc)
        for row in plot_results.itertuples()
    }

    edge_x, edge_y = [], []
    for parent in plot_results.itertuples():
        if parent.n_features == 1:
            continue
        x0, y0 = node_positions[parent.subset_key]
        for removed_feature in parent.feature_set:
            child_set = parent.feature_set - {removed_feature}
            child_key = "|".join(
                feature for feature in feature_order if feature in child_set
            )
            x1, y1 = node_positions[child_key]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    ba_min, ba_max = _color_range(plot_results["balanced_accuracy"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="skip",
            line=dict(color="rgba(100, 116, 139, 0.28)", width=1),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[node_positions[key][0] for key in plot_results["subset_key"]],
            y=[node_positions[key][1] for key in plot_results["subset_key"]],
            mode="markers",
            marker=dict(
                size=12,
                color=plot_results["balanced_accuracy"],
                colorscale="Turbo",
                cmin=ba_min,
                cmax=ba_max,
                colorbar=dict(title="Balanced<br>accuracy"),
                line=dict(color="#1f2937", width=0.5),
            ),
            customdata=np.column_stack(
                [
                    plot_results["subset_label"],
                    plot_results["auc"],
                    plot_results["balanced_accuracy"],
                    plot_results["sensitivity"],
                    plot_results["specificity"],
                    plot_results["threshold_mean"],
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "AUC: %{customdata[1]:.3f}<br>"
                "Balanced accuracy: %{customdata[2]:.3f}<br>"
                "Sensitivity: %{customdata[3]:.3f}<br>"
                "Specificity: %{customdata[4]:.3f}<br>"
                "Mean inner-selected threshold: %{customdata[5]:.3f}<extra></extra>"
            ),
        )
    )
    y_values = plot_results["auc"].tolist()
    if news2_auc is not None:
        y_values.append(float(news2_auc))
    figure.update_layout(
        title=f"{title}: scale performance across all NEWS2 feature subsets",
        xaxis=dict(
            title="Number of NEWS2 measurements retained",
            tickmode="array",
            tickvals=list(range(1, len(feature_order) + 1)),
            ticktext=[str(value) for value in range(1, len(feature_order) + 1)],
            range=[len(feature_order) + 0.35, 0.65],
        ),
        yaxis=dict(
            title="Mean outer-CV AUC",
            range=[
                max(0, min(y_values) - 0.015),
                min(1, max(y_values) + 0.015),
            ],
            tickformat=".2f",
            gridcolor="rgba(148, 163, 184, 0.35)",
            zeroline=False,
        ),
        template="plotly_white",
        height=780,
        margin=dict(l=35, r=120, t=80, b=65),
        showlegend=False,
    )
    if news2_auc is not None:
        figure.add_hline(
            y=float(news2_auc),
            line=dict(color="#7A0403", width=2, dash="dash"),
            annotation_text=f"NEWS2 AUC = {float(news2_auc):.3f}",
            annotation_position="top right",
            annotation_font_color="#7A0403",
        )
    return figure

def _prepare_subset_results(results: pd.DataFrame) -> pd.DataFrame:
    """Add parsed feature sets and a stable ordering to subset results."""
    required_columns = {
        "subset_key",
        "subset_label",
        "features_json",
        "n_features",
        "auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    }
    missing = sorted(required_columns - set(results.columns))
    if missing:
        raise ValueError(f"Results are missing required columns: {missing}")
    prepared = results.copy().sort_values(
        ["balanced_accuracy", "auc", "n_features", "subset_label"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    prepared["feature_set"] = prepared["features_json"].map(
        lambda value: frozenset(json.loads(value))
    )
    return prepared

def _color_range(values: Sequence[float] | np.ndarray | pd.Series) -> tuple[float, float]:
    """Return the data-driven limits for a continuous Plotly color scale."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("A continuous color scale requires at least one finite value.")

    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    if np.isclose(minimum, maximum):
        # Plotly needs a non-zero interval when every displayed value is identical.
        return minimum - 1e-6, maximum + 1e-6
    return minimum, maximum

def make_subset_performance_matrix(
    results: pd.DataFrame, feature_order: Sequence[str]
) -> go.Figure:
    """Plot feature membership alongside AUC, BA, sensitivity, and specificity."""
    prepared = _prepare_subset_results(results)
    subset_labels = prepared["subset_label"].tolist()
    membership = np.array(
        [
            [int(feature in feature_set) for feature in feature_order]
            for feature_set in prepared["feature_set"]
        ]
    )
    metric_columns = ["auc", "balanced_accuracy", "sensitivity", "specificity"]
    metric_labels = ["AUC", "BA", "Sensitivity", "Specificity"]
    metric_values = prepared.loc[:, metric_columns].to_numpy(dtype=float)
    metric_min, metric_max = _color_range(metric_values)
    height = max(720, 18 * len(prepared) + 170)

    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.56, 0.44],
        horizontal_spacing=0.025,
        subplot_titles=("NEWS2 measurements included", "Outer-CV performance"),
    )
    figure.add_trace(
        go.Heatmap(
            z=membership,
            x=list(feature_order),
            y=subset_labels,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#F1F5F9"], [0.499, "#F1F5F9"], [0.5, "#F68D3D"], [1, "#F68D3D"]],
            showscale=False,
            customdata=np.repeat(
                prepared[["subset_label", "auc", "balanced_accuracy"]].to_numpy()[:, None, :],
                len(feature_order),
                axis=1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Measurement: %{x}<br>Included: %{z}<br>"
                "AUC: %{customdata[1]:.3f}<br>BA: %{customdata[2]:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=metric_values,
            x=metric_labels,
            y=subset_labels,
            zmin=metric_min,
            zmax=metric_max,
            colorscale="Turbo",
            colorbar=dict(title="Metric", x=1.02),
            customdata=np.repeat(
                prepared[["subset_label"]].to_numpy()[:, None, :],
                len(metric_columns),
                axis=1,
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x}: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        title="All NEWS2 subsets: composition and nested-CV performance",
        template="plotly_white",
        height=height,
        margin=dict(l=250, r=100, t=80, b=50),
    )
    figure.update_yaxes(autorange="reversed", tickfont=dict(size=9))
    figure.update_xaxes(tickangle=-35)
    return figure

def make_pareto_frontier_plot(
    results: pd.DataFrame,
    *,
    news2_auc: float | None = None,
) -> go.Figure:
    """Plot the AUC/measurement-count trade-off and label Pareto-optimal subsets."""
    prepared = _prepare_subset_results(results)
    if news2_auc is not None and not 0 <= float(news2_auc) <= 1:
        raise ValueError("news2_auc must be between 0 and 1.")

    frontier_rows = []
    best_auc_so_far = -np.inf
    for n_features, group in prepared.groupby("n_features", sort=True):
        best = group.nlargest(1, "auc").iloc[0]
        if best["auc"] > best_auc_so_far + 1e-12:
            frontier_rows.append(best)
            best_auc_so_far = best["auc"]
    frontier = pd.DataFrame(frontier_rows)
    ba_min, ba_max = _color_range(prepared["balanced_accuracy"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=prepared["n_features"],
            y=prepared["auc"],
            mode="markers",
            marker=dict(
                size=10,
                color=prepared["balanced_accuracy"],
                colorscale="Turbo",
                cmin=ba_min,
                cmax=ba_max,
                colorbar=dict(title="BA"),
                line=dict(color="#1F2937", width=0.5),
            ),
            customdata=prepared[
                ["subset_label", "balanced_accuracy", "sensitivity", "specificity"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Measurements: %{x}<br>AUC: %{y:.3f}<br>"
                "BA: %{customdata[1]:.3f}<br>"
                "Sensitivity: %{customdata[2]:.3f}<br>"
                "Specificity: %{customdata[3]:.3f}<extra></extra>"
            ),
            name="All subsets",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frontier["n_features"],
            y=frontier["auc"],
            mode="lines+markers+text",
            text=frontier["subset_label"],
            textposition="top center",
            cliponaxis=False,
            marker=dict(size=13, color="#E84A20", symbol="diamond"),
            line=dict(color="#E84A20", width=2),
            hovertemplate="<b>Pareto-optimal</b><br>%{text}<br>AUC: %{y:.3f}<extra></extra>",
            name="Pareto frontier",
        )
    )
    y_values = prepared["auc"].tolist()
    if news2_auc is not None:
        y_values.append(float(news2_auc))
    figure.update_layout(
        title="AUC versus number of retained NEWS2 measurements",
        template="plotly_white",
        xaxis=dict(
            title="Number of retained measurements",
            tickmode="array",
            tickvals=sorted(prepared["n_features"].unique()),
        ),
        yaxis=dict(
            title="Mean outer-CV AUC",
            range=[max(0, min(y_values) - 0.015), min(1, max(y_values) + 0.04)],
            tickformat=".2f",
            gridcolor="rgba(148, 163, 184, 0.35)",
        ),
        height=680,
        margin=dict(l=70, r=115, t=95, b=160),
        legend=dict(orientation="h", y=1.04, x=0),
    )
    if news2_auc is not None:
        figure.add_hline(
            y=float(news2_auc),
            line=dict(color="#7A0403", width=2, dash="dash"),
            annotation_text=f"NEWS2 AUC = {float(news2_auc):.3f}",
            annotation_position="top left",
        )
    return figure

def make_marginal_ablation_plot(
    results: pd.DataFrame, feature_order: Sequence[str], title: str = "NEWS2 deconstruction"
) -> go.Figure:
    """Show the AUC change caused by removing each measurement across parents."""
    prepared = _prepare_subset_results(results)
    by_key = prepared.set_index("subset_key", drop=False)
    ablations = []
    for parent in prepared.itertuples():
        if parent.n_features == 1:
            continue
        for removed_feature in parent.feature_set:
            child_key = "|".join(
                feature
                for feature in feature_order
                if feature in parent.feature_set and feature != removed_feature
            )
            if child_key not in by_key.index:
                continue
            child = by_key.loc[child_key]
            ablations.append(
                {
                    "removed_feature": removed_feature,
                    "parent_subset": parent.subset_label,
                    "child_subset": child.subset_label,
                    "delta_balanced_accuracy": (
                        child.balanced_accuracy - parent.balanced_accuracy
                    ),
                    "delta_auc": child.auc - parent.auc,
                }
            )
    ablation_data = pd.DataFrame(ablations)
    if ablation_data.empty:
        raise ValueError("At least one parent/child subset pair is required.")

    figure = go.Figure()
    for feature_position, feature in enumerate(feature_order):
        feature_data = ablation_data.loc[
            ablation_data["removed_feature"].eq(feature)
        ]
        feature_color = TURBO_FEATURE_COLORS[
            feature_position % len(TURBO_FEATURE_COLORS)
        ]
        figure.add_trace(
            go.Box(
                x=[feature] * len(feature_data),
                y=feature_data["delta_auc"],
                name=feature,
                boxpoints="all",
                jitter=0.28,
                pointpos=0,
                marker=dict(color=feature_color, size=4, opacity=0.45),
                line=dict(color=feature_color),
                customdata=feature_data[["parent_subset", "child_subset"]].to_numpy(),
                hovertemplate=(
                    "Removed: %{x}<br>Δ AUC: %{y:.3f}<br>"
                    "Parent: %{customdata[0]}<br>Child: %{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.update_layout(
        title=f"Marginal effect of removing each NEWS2 measurement on {title}",
        template="plotly_white",
        xaxis_title="Removed measurement",
        yaxis_title="Δ AUC after removal (child − parent)",
        height=600,
        margin=dict(l=85, r=35, t=80, b=110),
    )
    figure.add_hline(y=0, line=dict(color="#475569", width=1.5, dash="dash"))
    return figure
