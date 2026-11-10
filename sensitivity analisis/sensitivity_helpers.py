"""Shared preparation and resumable patient bootstrap for NEWS2 sensitivity analyses.

This module performs no analysis on import. Notebook cells initiate execution.
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from functions.news2_scale_deconstruction import (
    score_news2_components, evaluate_scale_subset,
)

FEATURES = ["resp_rate", "blod_oxy", "oxigen", "blod_pres",
            "heart_rate", "consciousness", "temp"]
OUTCOMES = ["M30D", "UVI"]
SUBSETS = [s for k in range(1, 8) for s in combinations(FEATURES, k)]


def load_original_data(path):
    data = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    if data["#ID"].isna().any() or data["#ID"].duplicated().any():
        raise ValueError("One non-missing unique #ID per original patient is required.")
    data = data.reset_index(drop=True)
    data["_source_patient"] = np.arange(len(data))
    return data


def numeric(series):
    return pd.to_numeric(
        series.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    ).astype(float)


def prepare_cohort(raw, suffix, gcs_cutoff=15):
    """3 consciousness points if GCS < cutoff; 0 otherwise; missing stays missing.

    Cutoff 15 reproduces the baseline for valid GCS totals (3–15).
    Cutoff 13 assigns 0 points at GCS 13, 14, and 15.
    All variants use identical complete-measurement eligibility rules.
    """
    result = pd.DataFrame(index=raw.index)
    result["#ID"] = raw["#ID"]
    result["_source_patient"] = raw["_source_patient"]
    for feature, prefix in (
        ("resp_rate", "FR"), ("blod_oxy", "SpO2"),
        ("blod_pres", "TAS"), ("heart_rate", "FC"), ("temp", "TT"),
    ):
        result[feature] = numeric(raw[f"{prefix}-{suffix}"])
    fio2 = numeric(raw[f"FiO2-{suffix}"])
    result["oxigen"] = np.where(fio2.notna(), (fio2 > 0.21).astype(float), np.nan)
    components = pd.concat(
        [numeric(raw[f"GCS.{part}-{suffix}"]) for part in ("O", "V", "M")],
        axis=1,
    )
    gcs = components.sum(axis=1, min_count=3)
    # Out-of-range totals cannot be interpreted as valid GCS scores.
    # Treat them as missing so complete-case filtering excludes them.
    gcs = gcs.where(gcs.between(3, 15))
    result["gcs_total"] = gcs
    result["consciousness"] = np.where(
        gcs.notna(), (gcs >= gcs_cutoff).astype(float), np.nan,
    )
    for outcome, column in (("M30D", "Mortalidad a 30 dias"), ("UVI", "UVI")):
        result[outcome] = numeric(raw[column])
        invalid = result[outcome].dropna()
        if not invalid.isin([0, 1]).all():
            raise ValueError(f"{outcome} must contain only 0, 1, or missing values.")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    points = score_news2_components(result, FEATURES)
    if points.isna().any().any():
        raise ValueError("Some eligible measurements cannot be scored; review units/coding.")
    result["NEWS2"] = points.sum(axis=1)
    return result.reset_index(drop=True)


def assign_grouped_folds(data, outcome, n_splits, seed):
    """Keep all bootstrap copies of a source patient in a single fold."""
    result = data.dropna(subset=[outcome]).copy().reset_index(drop=True)
    result[outcome] = result[outcome].astype(int)
    unique = result.drop_duplicates("_source_patient")
    counts = unique[outcome].value_counts()
    if set(counts.index) != {0, 1} or counts.min() < n_splits:
        raise ValueError("Not enough unique patients in each class for the requested folds.")
    # Retry fold assignment, not patient resampling, if grouped stratification
    # produces a one-class validation fold. Never silently discard a bootstrap.
    for attempt in range(100):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + attempt,
        )
        folds = np.full(len(result), -1, dtype=int)
        for fold, (_, test) in enumerate(splitter.split(
            result[FEATURES], result[outcome], groups=result["_source_patient"],
        )):
            folds[test] = fold
        if all(result.loc[folds == f, outcome].nunique() == 2 for f in range(n_splits)):
            result["cv_fold"] = folds
            return result
    raise ValueError("Could not construct grouped folds containing both classes.")


def evaluate_all_subsets(data, outcome, experiment_id, n_jobs=1, progress=None):
    """Reuse the original evaluator, threshold tie rules, and fold averaging."""
    points = score_news2_components(data, FEATURES)

    def evaluate(subset):
        return evaluate_scale_subset(
            data, component_scores=points, features=subset, target=outcome,
            fold_column="cv_fold", feature_order=FEATURES,
            experiment_id=experiment_id, reference_threshold=5.0,
        )

    def collect(iterator):
        rows = []
        for result in iterator:
            rows.append(result)
            if progress is not None:
                progress.update(1)
        return pd.DataFrame(rows)

    if n_jobs == 1:
        return collect(map(evaluate, SUBSETS))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return collect(pool.map(evaluate, SUBSETS))


def run_patient_bootstrap(raw, output_dir, n_bootstraps=2000,
                          n_splits=10, random_state=42, n_jobs=1):
    """Ordinary patient bootstrap from the original raw dataset.

    Each replicate samples N original patients with replacement, preserving
    paired prehospital/hospital measurements and outcomes. Cohort and outcome
    eligibility are applied after sampling. Each replicate reruns grouped CV
    and inner-fold threshold selection for all 127 subsets and four experiments.
    Replicate files are atomic checkpoints. The original estimate uses the same
    grouped-CV procedure. No bootstrap copy can cross an outer fold boundary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy().tobytes()
        + json.dumps(list(raw.columns)).encode()
    ).hexdigest()
    settings = {
        "method_version": 2, "dataset_sha256": fingerprint,
        "n_bootstraps": n_bootstraps, "n_splits": n_splits,
        "random_state": random_state, "features": FEATURES,
        "sampling": "ordinary_patient_bootstrap_before_cohort_filtering",
        "threshold": "inner_fold_mean_balanced_accuracy",
        "interval": "pointwise_percentile_95", "gcs_cutoff": 15,
        "invalid_gcs": "exclude_outside_3_to_15",
    }
    manifest = output_dir / "bootstrap_settings.json"
    if manifest.exists():
        saved_settings = json.loads(manifest.read_text())
        # Confidence level changes summaries, not the stored bootstrap draws.
        saved_settings.pop("interval", None)
        resampling_settings = {k: v for k, v in settings.items() if k != "interval"}
        if saved_settings != resampling_settings:
            raise ValueError("Checkpoint settings/data differ. Use a new output directory.")
    manifest.write_text(json.dumps(settings, indent=2) + "\n")
    keep = ["subset_key", "subset_label", "features_json", "n_features",
            "auc", "balanced_accuracy", "threshold_mean"]

    def evaluate_raw(sample, replicate, subset_progress=None):
        rows = []
        seed = int(np.random.SeedSequence([random_state, replicate + 1, 867]).generate_state(1)[0])
        for setting, suffix in (("Prehospital", "1"), ("Hospital", "H")):
            cohort = prepare_cohort(sample, suffix, gcs_cutoff=15)
            for outcome in OUTCOMES:
                folded = assign_grouped_folds(cohort, outcome, n_splits, seed)
                result = evaluate_all_subsets(
                    folded, outcome, f"bootstrap_{replicate}_{setting}_{outcome}", n_jobs,
                    progress=subset_progress,
                )[keep].copy()
                result["cohort"] = setting
                result["outcome"] = outcome
                result["n_patients_with_copies"] = len(folded)
                result["n_unique_patients"] = folded["_source_patient"].nunique()
                result["bootstrap_id"] = replicate
                rows.append(result)
        return pd.concat(rows, ignore_index=True)

    def save_atomic(frame, destination):
        temporary = destination.with_suffix(".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)

    original_path = output_dir / "original_estimates.csv"
    if not original_path.exists():
        with tqdm(total=4 * len(SUBSETS), desc="Original dataset: subset evaluations",
                  unit="subset") as baseline_progress:
            save_atomic(evaluate_raw(raw, -1, subset_progress=baseline_progress), original_path)
    completed = {i for i in range(n_bootstraps)
                 if (output_dir / f"bootstrap_{i:04d}.csv").exists()}
    with tqdm(total=n_bootstraps, initial=len(completed),
              desc="Bootstrap datasets", unit="dataset") as bar:
        for replicate in range(n_bootstraps):
            if replicate in completed:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([random_state, replicate]))
            indices = rng.integers(0, len(raw), size=len(raw))
            sample = raw.iloc[indices].reset_index(drop=True)
            try:
                result = evaluate_raw(sample, replicate)
            except ValueError as error:
                raise ValueError(
                    f"Bootstrap {replicate} failed; no replicate was silently discarded. "
                    "Review class counts/fold feasibility. Completed checkpoints are retained."
                ) from error
            save_atomic(result, output_dir / f"bootstrap_{replicate:04d}.csv")
            bar.update()
    return summarize_bootstrap(output_dir, n_bootstraps)


def summarize_bootstrap(output_dir, n_bootstraps=2000):
    """Pointwise 2.5th/97.5th percentiles across 2,000 replicate estimates."""
    output_dir = Path(output_dir)
    frames = []
    for i in tqdm(range(n_bootstraps), desc="Reading checkpoints", unit="file"):
        path = output_dir / f"bootstrap_{i:04d}.csv"
        if not path.exists():
            raise ValueError(f"Missing replicate: {path.name}")
        frame = pd.read_csv(path)
        keys = ["cohort", "outcome", "subset_key"]
        if len(frame) != 508 or frame.duplicated(keys).any():
            raise ValueError(f"Incomplete/duplicated replicate: {path.name}")
        if not np.isfinite(frame[["auc", "balanced_accuracy"]].to_numpy()).all():
            raise ValueError(f"Nonfinite metrics: {path.name}")
        frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    keys = ["cohort", "outcome", "subset_key"]
    grouped = samples.groupby(keys)
    if not grouped.size().eq(n_bootstraps).all() or len(grouped) != 508:
        raise ValueError("Every experiment/subset must have all bootstrap estimates.")
    summary = pd.read_csv(output_dir / "original_estimates.csv")
    summary = summary.rename(columns={"auc": "auc_estimate", "balanced_accuracy": "ba_estimate"})
    for metric, prefix in (("auc", "auc"), ("balanced_accuracy", "ba")):
        limits = grouped[metric].quantile([.025, .975]).unstack()
        limits.columns = [f"{prefix}_ci95_low", f"{prefix}_ci95_high"]
        summary = summary.merge(limits.reset_index(), on=keys, validate="one_to_one")
    summary["n_bootstraps"] = n_bootstraps
    summary.to_csv(output_dir / "bootstrap_95ci_all_subsets.csv", index=False)
    return summary


def audit_gcs(raw):
    """Report aggregate GCS eligibility before any computation."""
    rows = []
    for cohort, suffix in (("Prehospital", "1"), ("Hospital", "H")):
        components = pd.concat(
            [numeric(raw[f"GCS.{part}-{suffix}"]) for part in ("O", "V", "M")],
            axis=1,
        )
        total = components.sum(axis=1, min_count=3)
        invalid = total.notna() & ~total.between(3, 15)
        rows.append({
            "cohort": cohort,
            "original_patients": len(raw),
            "missing_gcs_total": int(total.isna().sum()),
            "invalid_gcs_total_excluded": int(invalid.sum()),
            "valid_gcs_total": int(total.between(3, 15).sum()),
            "invalid_total_counts": total.loc[invalid].value_counts().sort_index().to_json(),
        })
    return pd.DataFrame(rows)


def plot_gcs_delta_auc_by_size(comparison, setting, outcome, output_dir):
    """Show matched AUC changes across the 64 subsets containing consciousness."""
    contains_gcs = comparison["subset_key"].str.split("|").map(
        lambda names: "consciousness" in names
    )
    all_gcs = comparison.loc[contains_gcs].copy()
    selected = all_gcs.loc[
        all_gcs["cohort"].eq(setting) & all_gcs["outcome"].eq(outcome)
    ].copy()
    expected_counts = [1, 6, 15, 20, 15, 6, 1]
    counts = selected.groupby("n_features").size().reindex(range(1, 8), fill_value=0)
    if counts.tolist() != expected_counts:
        raise ValueError("Expected 64 unique GCS-containing subsets with counts 1,6,15,20,15,6,1.")
    if selected["subset_key"].duplicated().any():
        raise ValueError("Duplicate subsets in the selected experiment.")
    if not np.isfinite(all_gcs["delta_auc"].to_numpy()).all():
        raise ValueError("AUC differences must be finite.")

    colors = plt.get_cmap("turbo")(np.linspace(0, 1, 7))
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for size in range(1, 8):
            values = selected.loc[selected["n_features"].eq(size)].sort_values(
                "subset_key"
            )["delta_auc"].to_numpy()
            color = colors[size - 1]
            if len(values) > 1:
                ax.boxplot(
                    [values], positions=[size], widths=.52, patch_artist=True,
                    manage_ticks=False, showfliers=False,
                    boxprops={"facecolor": color, "alpha": .45, "edgecolor": "#334155"},
                    medianprops={"color": "#0F172A", "linewidth": 1.8},
                    whiskerprops={"color": "#475569"},
                    capprops={"color": "#475569"},
                )
                offsets = np.linspace(-.17, .17, len(values))
            else:
                offsets = np.zeros(1)
            ax.scatter(
                size + offsets, values, s=38 if len(values) > 1 else 65,
                color=color, edgecolors="#334155", linewidths=.5, zorder=3,
            )
        lower = min(0.0, float(all_gcs["delta_auc"].min()))
        upper = max(0.0, float(all_gcs["delta_auc"].max()))
        padding = max(.002, .12 * (upper - lower))
        ax.set_ylim(lower - padding, upper + padding)
        ax.axhline(0, color="#475569", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, 8), [f"{k}\n(n={n})" for k,n in enumerate(expected_counts, 1)])
        ax.set_xlim(.5, 7.5)
        ax.set_xlabel("Number of parameters in the subset (including consciousness)")
        ax.set_ylabel("ΔAUC: GCS < 13 minus baseline")
        label = {"M30D": "30-day mortality", "UVI": "ICU admission"}.get(outcome, outcome)
        ax.set_title(f"{setting} | {label}\nEffect of changing the consciousness threshold", loc="left", pad=14)
        ax.grid(axis="y", color="#E2E8F0", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{setting.lower()}_{outcome.lower()}_gcs13_delta_auc_by_subset_size"
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
        selected.to_csv(output_dir / f"{stem}.csv", index=False)
    return fig