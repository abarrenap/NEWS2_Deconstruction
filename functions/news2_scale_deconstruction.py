"""Nested-CV deconstruction of NEWS2 using NEWS2 component points.

Each raw measurement is converted to its standard NEWS2 component score.  A
subset's continuous prediction is the sum of the retained component scores.
There is no fitted estimator: nested cross-validation is used only to select a
classification threshold on the outer-training folds before final evaluation
on the untouched outer-test fold.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence
import json
import os
from shutil import move

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from .news2_scale_utils import make_branching_subset_plot


def score_news2_components(
    data: pd.DataFrame,
    feature_order: Sequence[str],
) -> pd.DataFrame:
    """Return the seven NEWS2 component-point columns.

    ``feature_order`` follows the same positional contract as
    :func:`functions.news.calculate_NEWS2`: respiratory rate, oxygen
    saturation, supplemental oxygen, systolic blood pressure, heart rate,
    consciousness, and temperature.  In this project consciousness is encoded
    as 1 for alert and 0 for non-alert, matching the existing NEWS2 function
    and cohort construction in ``NEWS2_deconstruction.ipynb``.
    """
    if len(feature_order) != 7 or len(set(feature_order)) != 7:
        raise ValueError("feature_order must contain seven unique NEWS2 columns.")
    missing = [column for column in feature_order if column not in data.columns]
    if missing:
        raise ValueError(f"Missing NEWS2 columns: {missing}")

    rr, spo2, oxygen, sbp, heart_rate, consciousness, temperature = [
        pd.to_numeric(data[column], errors="coerce") for column in feature_order
    ]
    component_values = [
        np.select(
            [
                rr <= 8,
                rr.between(9, 11),
                rr.between(12, 20),
                rr.between(21, 24),
                rr >= 25,
            ],
            [3, 1, 0, 2, 3],
            default=np.nan,
        ),
        np.select(
            [spo2 <= 91, (spo2 > 91) & (spo2 < 94), spo2.between(94, 95), spo2 >= 96],
            [3, 2, 1, 0],
            default=np.nan,
        ),
        np.select([oxygen == 0, oxygen == 1], [0, 2], default=np.nan),
        np.select(
            [
                sbp <= 90,
                sbp.between(91, 100),
                sbp.between(101, 110),
                sbp.between(111, 219),
                sbp >= 220,
            ],
            [3, 2, 1, 0, 3],
            default=np.nan,
        ),
        np.select(
            [
                heart_rate <= 40,
                heart_rate.between(41, 50),
                heart_rate.between(51, 90),
                heart_rate.between(91, 110),
                heart_rate.between(111, 130),
                heart_rate >= 131,
            ],
            [3, 1, 0, 1, 2, 3],
            default=np.nan,
        ),
        np.select(
            [consciousness == 1, consciousness == 0],
            [0, 3],
            default=np.nan,
        ),
        np.select(
            [
                temperature <= 35,
                (temperature > 35) & (temperature <= 36),
                (temperature > 36) & (temperature <= 38),
                (temperature > 38) & (temperature <= 39),
                temperature > 39,
            ],
            [3, 1, 0, 1, 2],
            default=np.nan,
        ),
    ]
    return pd.DataFrame(
        np.column_stack(component_values),
        columns=list(feature_order),
        index=data.index,
        dtype=float,
    )


def evaluate_component_raw_deltas(
    prehospital_data: pd.DataFrame,
    hospital_data: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    targets: Sequence[str] = ("M30D", "UVI"),
    id_column: str = "#ID",
) -> pd.DataFrame:
    """Evaluate prehospital-to-hospital raw-measurement changes.

    Each patient's prehospital value is subtracted from their paired hospital
    value. AUC is then calculated separately for every raw-measurement
    change/outcome combination.
    """
    required = {id_column, *feature_order, *targets}
    for name, frame in (("prehospital", prehospital_data), ("hospital", hospital_data)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data are missing columns: {missing}")
        if frame[id_column].duplicated().any():
            raise ValueError(f"{name} data contain duplicated patient IDs.")

    pre = prehospital_data.loc[:, [id_column, *targets, *feature_order]].copy()
    hosp = hospital_data.loc[:, [id_column, *feature_order]].copy()
    paired = pre.merge(hosp, on=id_column, how="inner", suffixes=("_pre", "_hosp"))

    rows = []
    for feature in feature_order:
        delta = paired[f"{feature}_hosp"] - paired[f"{feature}_pre"]
        for target in targets:
            valid = pd.DataFrame({"delta": delta, "target": paired[target]}).dropna()
            if valid["target"].nunique() != 2:
                raise ValueError(f"{target!r} must contain both classes after pairing.")
            raw_auc = float(roc_auc_score(valid["target"], valid["delta"]))
            rows.append(
                {
                    "feature": feature,
                    "outcome": target,
                    "auc": max(raw_auc, 1 - raw_auc),
                    "raw_auc": raw_auc,
                    "risk_direction": (
                        "higher hospital-minus-prehospital change"
                        if raw_auc >= 0.5
                        else "lower hospital-minus-prehospital change"
                    ),
                    "n_patients": int(len(valid)),
                }
            )
    return pd.DataFrame(rows)


def make_component_delta_auc_plot(delta_results: pd.DataFrame) -> go.Figure:
    """Plot each raw-measurement delta's AUC for M30D and UVI."""
    required = {"feature", "outcome", "auc"}
    if missing := required - set(delta_results.columns):
        raise ValueError(f"Delta results are missing columns: {sorted(missing)}")
    figure = go.Figure()
    colors = {"M30D": "#4661D6", "UVI": "#7A0403"}
    for outcome in ("M30D", "UVI"):
        values = delta_results.loc[delta_results["outcome"].eq(outcome)]
        figure.add_trace(go.Bar(
            name=outcome, x=values["feature"], y=values["auc"],
            marker_color=colors[outcome],
            hovertemplate="%{x}<br>" + outcome + " AUC: %{y:.3f}<extra></extra>",
        ))
    figure.add_hline(y=0.5, line=dict(color="#475569", dash="dash"))
    figure.update_layout(
        title="AUC of prehospital-to-hospital raw-measurement changes",
        template="plotly_white", barmode="group", height=520,
        xaxis_title="Measurement", yaxis_title="AUC from raw-value change",
        yaxis=dict(range=[0, 1], tickformat=".2f"),
    )
    return figure


def optimize_scale_threshold(
    y_true: Sequence[int],
    scores: Sequence[float],
    fold_labels: Sequence[object],
    *,
    reference_threshold: float = 5.0,
) -> tuple[float, float]:
    """Maximize mean inner-fold balanced accuracy over all score cutoffs.

    Inner folds contribute equally.  When several cutoffs are equivalent, the
    one closest to the conventional NEWS2 escalation threshold of 5 is used;
    any remaining tie is resolved in favor of the lower cutoff.
    """
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    folds = np.asarray(fold_labels)
    if not (len(y) == len(values) == len(folds)):
        raise ValueError("y_true, scores, and fold_labels must have equal length.")
    if len(y) == 0 or not np.isfinite(values).all():
        raise ValueError("Threshold optimization requires finite, non-empty scores.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y_true must be binary and encoded as 0/1.")

    unique_folds = list(pd.unique(folds))
    candidates = np.r_[np.unique(values), np.nextafter(values.max(), np.inf)]
    candidate_results: list[tuple[float, float]] = []
    for threshold in candidates:
        fold_balanced_accuracies = []
        for fold in unique_folds:
            mask = folds == fold
            if set(np.unique(y[mask])) != {0, 1}:
                raise ValueError("Every inner fold must contain both target classes.")
            predictions = (values[mask] >= threshold).astype(int)
            fold_balanced_accuracies.append(
                balanced_accuracy_score(y[mask], predictions)
            )
        candidate_results.append(
            (float(threshold), float(np.mean(fold_balanced_accuracies)))
        )

    best_score = max(score for _, score in candidate_results)
    tied_thresholds = [
        threshold
        for threshold, score in candidate_results
        if np.isclose(score, best_score, rtol=0, atol=1e-12)
    ]
    best_threshold = min(
        tied_thresholds,
        key=lambda threshold: (abs(threshold - reference_threshold), threshold),
    )
    return float(best_threshold), float(best_score)


def _outer_metrics(
    y_true: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    predictions = (values >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "auc": float(roc_auc_score(y, values)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_scale_subset(
    data: pd.DataFrame,
    *,
    component_scores: pd.DataFrame,
    features: Sequence[str],
    target: str,
    fold_column: str,
    feature_order: Sequence[str],
    experiment_id: str,
    reference_threshold: float = 5.0,
) -> dict[str, object]:
    """Evaluate one partial NEWS2 sum with nested threshold selection."""
    features = tuple(features)
    unknown = sorted(set(features) - set(component_scores.columns))
    if unknown:
        raise ValueError(f"Missing component scores for: {unknown}")
    required = {target, fold_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not component_scores.index.equals(data.index):
        raise ValueError("component_scores and data must have identical indexes.")

    subset_score = component_scores.loc[:, list(features)].sum(axis=1, min_count=len(features))
    if subset_score.isna().any() or data[[target, fold_column]].isna().any().any():
        raise ValueError("Scale deconstruction requires complete scores, target, and folds.")
    if set(data[target].astype(int).unique()) != {0, 1}:
        raise ValueError(f"{target!r} must be binary and encoded as 0/1.")

    folds = sorted(data[fold_column].unique())
    if len(folds) < 3:
        raise ValueError("Nested scale CV requires at least three folds.")
    fold_rows = []
    for outer_fold in folds:
        train_mask = data[fold_column] != outer_fold
        test_mask = ~train_mask
        threshold, inner_best_ba = optimize_scale_threshold(
            data.loc[train_mask, target],
            subset_score.loc[train_mask],
            data.loc[train_mask, fold_column],
            reference_threshold=reference_threshold,
        )
        metrics = _outer_metrics(
            data.loc[test_mask, target], subset_score.loc[test_mask], threshold
        )
        metrics.update(
            {
                "outer_fold": int(outer_fold),
                "threshold": threshold,
                "inner_best_balanced_accuracy": inner_best_ba,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        )
        fold_rows.append(metrics)

    fold_metrics = pd.DataFrame(fold_rows)
    result: dict[str, object] = {
        "experiment_id": experiment_id,
        "subset_key": "|".join(features),
        "subset_label": " + ".join(features),
        "features_json": json.dumps(features),
        "n_features": len(features),
        "n_outer_folds": len(folds),
        "n_inner_folds": len(folds) - 1,
        "scoring_method": "partial_news2_sum",
        "threshold_selection": "mean_inner_fold_balanced_accuracy",
        "reference_threshold": float(reference_threshold),
    }
    for metric in ("auc", "balanced_accuracy", "sensitivity", "specificity"):
        result[metric] = float(fold_metrics[metric].mean())
        result[f"{metric}_sd"] = float(fold_metrics[metric].std(ddof=1))
    result["threshold_mean"] = float(fold_metrics["threshold"].mean())
    result["threshold_sd"] = float(fold_metrics["threshold"].std(ddof=1))
    result["inner_best_balanced_accuracy_mean"] = float(
        fold_metrics["inner_best_balanced_accuracy"].mean()
    )
    result["outer_fold_metrics_json"] = fold_metrics.to_json(orient="records")
    result.update({feature: int(feature in features) for feature in feature_order})
    return result


def all_nonempty_subsets(feature_order: Sequence[str]) -> list[tuple[str, ...]]:
    """Return all measurement subsets from seven retained features down to one."""
    return [
        subset
        for size in range(len(feature_order), 0, -1)
        for subset in combinations(feature_order, size)
    ]


def _save_checkpoint(results: pd.DataFrame, results_path: Path) -> pd.DataFrame:
    ordered = results.sort_values(
        ["n_features", "balanced_accuracy", "subset_key"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = results_path.with_suffix(".tmp.csv")
    ordered.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    os.replace(temporary_path, results_path)
    return ordered


def run_resumable_scale_search(
    data: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    target: str,
    fold_column: str,
    experiment_id: str,
    results_path: str | Path,
    reference_threshold: float = 5.0,
    n_jobs: int = 1,
    progress: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Evaluate all 127 partial NEWS2 scales and checkpoint each result."""
    results_path = Path(results_path)
    component_scores = score_news2_components(data, feature_order)
    subsets = all_nonempty_subsets(feature_order)
    if results_path.exists():
        saved = pd.read_csv(results_path, encoding="utf-8-sig")
        if "experiment_id" not in saved.columns:
            # Older scale-deconstruction exports used semicolons as delimiters,
            # decimal commas, and no checkpoint metadata. Convert them in
            # memory so completed subsets are retained rather than rerun.
            legacy = pd.read_csv(
                results_path, sep=";", decimal=",", encoding="utf-8-sig"
            )
            legacy = legacy.loc[:, ~legacy.columns.str.startswith("Unnamed:")]
            if {"subset_label", "n_features", "auc", "balanced_accuracy"} <= set(
                legacy.columns
            ):
                legacy["subset_label"] = legacy["subset_label"].str.replace(
                    "bold_oxy", "blod_oxy", regex=False
                )
                legacy = legacy.rename(columns={"bold_oxy": "blod_oxy"})
                legacy["subset_key"] = legacy["subset_label"].str.replace(
                    " + ", "|", regex=False
                )
                legacy["features_json"] = legacy["subset_label"].map(
                    lambda label: json.dumps(label.split(" + "))
                )
                legacy["experiment_id"] = experiment_id
                for column in legacy.columns:
                    if column not in {
                        "experiment_id",
                        "subset_label",
                        "subset_key",
                        "features_json",
                    }:
                        legacy[column] = pd.to_numeric(legacy[column], errors="coerce")
                saved = legacy
                progress("Loaded 127 completed subsets from the legacy checkpoint.")
            else:
                backup_path = results_path.with_suffix(".incompatible.csv")
                if backup_path.exists():
                    backup_path = results_path.with_suffix(
                        f".incompatible.{pd.Timestamp.now():%Y%m%d%H%M%S}.csv"
                    )
                move(results_path, backup_path)
                progress(
                    f"Saved incompatible checkpoint as {backup_path.name}; "
                    "starting a new checkpoint."
                )
                saved = pd.DataFrame()
        else:
            saved = saved.loc[saved["experiment_id"].eq(experiment_id)].copy()
            # Preserve previously completed checkpoints after correcting the
            # oxygen-saturation feature name from ``bold_oxy`` to ``blod_oxy``.
            if "bold_oxy" in saved.columns:
                saved = saved.rename(columns={"bold_oxy": "blod_oxy"})
            for column in ("subset_key", "subset_label", "features_json"):
                if column in saved:
                    saved[column] = saved[column].astype(str).str.replace(
                        "bold_oxy", "blod_oxy", regex=False
                    )
            saved = saved.drop_duplicates("subset_key", keep="last")
    else:
        saved = pd.DataFrame()

    completed = set(saved.get("subset_key", pd.Series(dtype=str)))
    pending = [subset for subset in subsets if "|".join(subset) not in completed]
    progress(f"Completed: {len(completed)}/{len(subsets)} subsets")
    progress(f"Pending: {len(pending)} subsets")

    with tqdm(total=len(subsets), initial=len(completed), desc="NEWS2 subsets") as bar:
        for size in range(len(feature_order), 0, -1):
            level = [subset for subset in pending if len(subset) == size]
            if not level:
                continue
            with ThreadPoolExecutor(max_workers=max(1, int(n_jobs))) as executor:
                futures = {
                    executor.submit(
                        evaluate_scale_subset,
                        data,
                        component_scores=component_scores,
                        features=subset,
                        target=target,
                        fold_column=fold_column,
                        feature_order=feature_order,
                        experiment_id=experiment_id,
                        reference_threshold=reference_threshold,
                    ): subset
                    for subset in level
                }
                for future in as_completed(futures):
                    subset = futures[future]
                    try:
                        saved = pd.concat(
                            [saved, pd.DataFrame([future.result()])], ignore_index=True
                        )
                        saved = _save_checkpoint(saved, results_path)
                        bar.update(1)
                        bar.set_postfix_str(" + ".join(subset), refresh=False)
                    except Exception as error:
                        raise RuntimeError(f"Failed subset: {subset}") from error
    return _save_checkpoint(saved, results_path)


def make_scale_branching_subset_plot(
    results: pd.DataFrame,
    feature_order: Sequence[str],
    *,
    news2_auc: float | None = None,
    title: str = "NEWS2 scale deconstruction",
):
    """Create the existing branching plot with scale-specific wording."""
    figure = make_branching_subset_plot(
        results,
        feature_order,
        news2_auc=news2_auc,
        title=title,
    )
    figure.update_layout(
        title=f"{title}: partial NEWS2 scale performance across all subsets"
    )
    return figure
