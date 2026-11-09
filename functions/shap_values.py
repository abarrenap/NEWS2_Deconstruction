"""Optional patient-level explanations of NEWS2 points (not AUC)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from tqdm.notebook import tqdm


from .news2_auc_shapley import decompose_news2_auc  # Backward-compatible import.
from .news2_scale_deconstruction import score_news2_components


class NEWS2Predictor(RegressorMixin, BaseEstimator):
    """Expose the fixed NEWS2 scoring rule through sklearn's interface."""

    def __init__(self, feature_order):
        self.feature_order = feature_order

    def fit(self, X, y=None):
        # No coefficients or outcome relationships are learned.
        self.n_features_in_ = len(self.feature_order)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")

        frame = pd.DataFrame(
            np.asarray(X, dtype=np.float64),
            columns=list(self.feature_order),
        )

        components = score_news2_components(
            frame,
            self.feature_order,
        )

        component_array = components.to_numpy(dtype=np.float64)

        if not np.isfinite(component_array).all():
            raise ValueError(
                "Some measurements could not be converted to finite "
                "NEWS2 component points."
            )

        return component_array.sum(axis=1)


def plot_news2_shap(
    data,
    feature_order,
    cohort_name,
    output_dir="shap_plots",
    background_size=100,
    max_patients=500,
    batch_size=50,
    random_state=42,
):
    """
    Explain total NEWS2 points using raw measurements as SHAP inputs.

    Produces:
        - Beeswarm plot: PNG and PDF.
        - Mean absolute SHAP bar plot: PNG and PDF.
        - Parameter importance table: CSV.

    Set max_patients=None to explain every eligible patient.

    feature_order must contain the seven parameters in this order:
        respiratory rate, oxygen saturation, supplemental oxygen,
        systolic blood pressure, heart rate, consciousness, temperature.

    SHAP values explain NEWS2 score contributions, not outcome risk.
    """
    feature_order = list(feature_order)

    if len(feature_order) != 7 or len(set(feature_order)) != 7:
        raise ValueError(
            "feature_order must contain seven unique NEWS2 columns."
        )

    missing = [
        column for column in feature_order
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Missing NEWS2 columns: {missing}")

    for name, value in (
        ("background_size", background_size),
        ("batch_size", batch_size),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    if max_patients is not None:
        if (
            not isinstance(max_patients, (int, np.integer))
            or max_patients < 1
        ):
            raise ValueError(
                "max_patients must be a positive integer or None."
            )

    # Standard float64 prevents object/nullable dtype errors in SHAP.
    X = (
        data.loc[:, feature_order]
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float64)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if X.empty:
        raise ValueError(
            "No patients have complete finite measurements."
        )

    predictor = NEWS2Predictor(feature_order).fit(X)
    predictor.predict(X)

    background = X.sample(
        n=min(background_size, len(X)),
        random_state=random_state,
    )

    if max_patients is None or len(X) <= max_patients:
        X_explain = X.copy()
    else:
        X_explain = X.sample(
            n=max_patients,
            random_state=random_state,
        )

    # Pass explicit float64 arrays to both the masker and explainer.
    background_array = background.to_numpy(dtype=np.float64)
    explain_array = X_explain.to_numpy(dtype=np.float64)

    masker = shap.maskers.Independent(
        background_array,
        max_samples=len(background_array),
    )

    explainer = shap.ExactExplainer(
        predictor.predict,
        masker,
    )

    values = []
    baselines = []

    with tqdm(
        total=len(explain_array),
        desc=f"{cohort_name}: SHAP",
        unit="patient",
    ) as progress:
        for start in range(0, len(explain_array), batch_size):
            batch = explain_array[start:start + batch_size]

            explanation = explainer(
                batch,
                silent=True,
            )

            values.append(
                np.asarray(explanation.values, dtype=np.float64)
            )
            baselines.append(
                np.asarray(
                    explanation.base_values,
                    dtype=np.float64,
                ).reshape(-1)
            )
            progress.update(len(batch))

    labels = [
        "Respiratory rate",
        "Oxygen saturation",
        "Supplemental oxygen",
        "Systolic blood pressure",
        "Heart rate",
        "Consciousness",
        "Temperature",
    ]

    shap_values = shap.Explanation(
        values=np.concatenate(values, axis=0),
        base_values=np.concatenate(baselines),
        data=explain_array,
        feature_names=labels,
    )

    # Contributions plus baseline must reconstruct the NEWS2 score.
    np.testing.assert_allclose(
        shap_values.base_values + shap_values.values.sum(axis=1),
        predictor.predict(explain_array),
        rtol=0,
        atol=1e-6,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "".join(
        character if character.isalnum() else "_"
        for character in str(cohort_name).lower()
    ).strip("_") or "cohort"

    def save_figure(figure, stem):
        for extension in ("png", "pdf"):
            figure.savefig(
                output_dir / f"{stem}.{extension}",
                dpi=300,
                bbox_inches="tight",
            )

    # Beeswarm: individual patient contributions.
    shap.plots.beeswarm(
        shap_values,
        max_display=7,
        show=False,
    )
    figure = plt.gcf()
    plt.title(f"{cohort_name}: contributions to NEWS2 score")
    plt.xlabel("SHAP value (NEWS2 points)")
    figure.tight_layout()
    save_figure(figure, f"{prefix}_shap_beeswarm")
    plt.show()
    plt.close(figure)

    # Bar: mean absolute contribution across explained patients.
    shap.plots.bar(
        shap_values,
        max_display=7,
        show=False,
    )
    figure = plt.gcf()
    plt.title(f"{cohort_name}: mean absolute NEWS2 contribution")
    plt.xlabel("Mean absolute SHAP value (NEWS2 points)")
    figure.tight_layout()
    save_figure(figure, f"{prefix}_shap_importance")
    plt.show()
    plt.close(figure)

    importance = (
        pd.DataFrame({
            "parameter": labels,
            "mean_absolute_shap": np.abs(
                shap_values.values
            ).mean(axis=0),
        })
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance.to_csv(
        output_dir / f"{prefix}_shap_importance.csv",
        index=False,
    )

    return shap_values, importance
