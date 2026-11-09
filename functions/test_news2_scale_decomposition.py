import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from functions.news import calculate_NEWS2
from functions.news2_scale_utils import add_stratified_folds
from functions.news2_scale_deconstruction import (
    evaluate_scale_subset,
    optimize_scale_threshold,
    score_news2_components,
)


FEATURES = ["rr", "spo2", "oxygen", "sbp", "hr", "alert", "temp"]


def test_component_sum_matches_existing_news2_calculation():
    data = pd.DataFrame(
        {
            "rr": [8, 10, 16, 23, 26],
            "spo2": [90, 92, 94, 96, 98],
            "oxygen": [1, 0, 1, 0, 1],
            "sbp": [85, 95, 105, 150, 225],
            "hr": [38, 45, 80, 120, 140],
            "alert": [1, 0, 1, 0, 1],
            "temp": [34.5, 35.5, 37, 38.5, 40],
        }
    )
    expected = calculate_NEWS2(data, FEATURES)["NEWS2"]
    components = score_news2_components(data, FEATURES)
    pd.testing.assert_series_equal(
        components.sum(axis=1), expected, check_names=False
    )


def test_threshold_optimizer_maximizes_mean_fold_balanced_accuracy():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scores = np.array([0, 4, 1, 3, 2, 5, 4, 2], dtype=float)
    folds = np.repeat(np.arange(4), 2)
    threshold, best_score = optimize_scale_threshold(y, scores, folds)

    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    brute_force = []
    for candidate in candidates:
        per_fold = [
            balanced_accuracy_score(
                y[folds == fold], (scores[folds == fold] >= candidate).astype(int)
            )
            for fold in np.unique(folds)
        ]
        brute_force.append(np.mean(per_fold))
    assert np.isclose(best_score, max(brute_force))
    assert threshold in candidates


def test_outer_test_labels_do_not_affect_selected_threshold():
    rng = np.random.default_rng(9)
    n = 120
    data = pd.DataFrame(
        {
            "rr": rng.integers(7, 30, n),
            "spo2": rng.integers(88, 100, n),
            "oxygen": rng.integers(0, 2, n),
            "sbp": rng.integers(80, 230, n),
            "hr": rng.integers(35, 145, n),
            "alert": rng.integers(0, 2, n),
            "temp": rng.uniform(34, 40, n),
        }
    )
    components = score_news2_components(data, FEATURES)
    data["target"] = (components.sum(axis=1) + rng.normal(0, 3, n) >= 7).astype(int)
    data = add_stratified_folds(
        data, target="target", fold_column="fold", n_splits=4, random_state=4
    )
    components = score_news2_components(data, FEATURES)

    original = evaluate_scale_subset(
        data,
        component_scores=components,
        features=FEATURES,
        target="target",
        fold_column="fold",
        feature_order=FEATURES,
        experiment_id="test",
    )
    changed = data.copy()
    changed.loc[changed["fold"] == 0, "target"] = 1 - changed.loc[
        changed["fold"] == 0, "target"
    ]
    changed_result = evaluate_scale_subset(
        changed,
        component_scores=components,
        features=FEATURES,
        target="target",
        fold_column="fold",
        feature_order=FEATURES,
        experiment_id="test",
    )

    original_folds = pd.DataFrame(json.loads(original["outer_fold_metrics_json"]))
    changed_folds = pd.DataFrame(json.loads(changed_result["outer_fold_metrics_json"]))
    original_fold_zero = original_folds.set_index("outer_fold").loc[0]
    changed_fold_zero = changed_folds.set_index("outer_fold").loc[0]
    assert original_fold_zero["threshold"] == changed_fold_zero["threshold"]
    assert (
        original_fold_zero["inner_best_balanced_accuracy"]
        == changed_fold_zero["inner_best_balanced_accuracy"]
    )
