"""Diagnosis-selected oxygen saturation sensitivity analysis. No work on import."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from sensitivity_helpers import FEATURES, SUBSETS, numeric, prepare_cohort
from functions.news2_scale_deconstruction import score_news2_components, evaluate_scale_subset


def oxygen_components(data, use_diagnosis_scale2=False):
    """Retain original Scale 1 except Dg #1 == 10 in the sensitivity variant.

    Scale 2 uses whole-percent chart lower boundaries, extended to fractional
    measurements as [84,86), [86,88), [88,93), etc.; no rounding is performed.
    Values above 100 are accepted without clipping and use the highest band
    on oxygen, or zero saturation points on room air.
    Supplemental oxygen status is always observed for Scale 2 scoring, even
    when its separate +2-point component is omitted from a partial score.
    """
    points = score_news2_components(data, FEATURES)
    if use_diagnosis_scale2:
        selected = data["diagnosis_code"].eq(10)
        spo2 = data.loc[selected, "blod_oxy"].to_numpy(dtype=float)
        oxygen = data.loc[selected, "oxigen"].to_numpy(dtype=float)
        invalid = ~np.isfinite(spo2) | (spo2 < 0)
        if invalid.any():
            columns = [c for c in ["#ID", "blod_oxy", "diagnosis_code", "oxigen"]
                       if c in data.columns]
            affected = data.loc[selected, columns].iloc[np.flatnonzero(invalid)]
            raise ValueError(
                "Invalid SpO2 among patients with Dg #1 == 10. "
                "Expected finite, nonnegative values. Values above 100 are accepted without clipping.\n"
                + affected.to_string(index=False)
                + "\nCheck the original SpO2-1 or SpO2-H coding before rerunning."
            )
        scale2 = np.select(
            [spo2 < 84, spo2 < 86, spo2 < 88,
             (spo2 < 93) | (oxygen == 0), spo2 < 95, spo2 < 97],
            [3, 2, 1, 0, 1, 2], default=3,
        )
        points.loc[selected, "blod_oxy"] = scale2
    if points.isna().any().any():
        raise ValueError("Nonfinite component scores; inspect measurement coding.")
    return points


def prepare_oxygen_cohort(raw, suffix):
    """Consciousness uses the baseline GCS cutoff 15 in both variants."""
    if "Dg #1" not in raw.columns:
        raise ValueError("The dataset must contain the exact column 'Dg #1'.")
    cohort = prepare_cohort(raw, suffix, gcs_cutoff=15)
    lookup = pd.Series(numeric(raw["Dg #1"]).to_numpy(), index=raw["_source_patient"])
    cohort["diagnosis_code"] = cohort["_source_patient"].map(lookup)
    cohort["uses_scale2"] = cohort["diagnosis_code"].eq(10)
    return cohort


def run_oxygen_subset_search(data, outcome, variant, results_path, experiment_id,
                            n_jobs=1):
    """Evaluate all subsets using explicitly supplied, variant-specific points.

    Do not use the standard search here: it would silently rescore with Scale 1.
    Checkpoints include a fingerprint of scored data, target, and folds.
    """
    results_path=Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    points=oxygen_components(data, use_diagnosis_scale2=(variant == "diagnosis_scale2"))
    relevant=pd.concat([points, data[[outcome, "cv_fold", "_source_patient"]]], axis=1)
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(relevant,index=True).to_numpy().tobytes()).hexdigest()
    settings={"version":1,"variant":variant,"fingerprint":fingerprint,
              "experiment_id":experiment_id,"reference_threshold":5.0}
    manifest=results_path.with_suffix(".settings.json")
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("Dataset/scoring/folds changed. Choose a new output directory.")
    if results_path.exists() and not manifest.exists():
        raise ValueError("Checkpoint has no settings manifest; choose a new output directory.")
    manifest.write_text(json.dumps(settings,indent=2)+"\n")
    rows=pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    done={row["subset_key"] for row in rows}
    expected={"|".join(s) for s in SUBSETS}
    if len(done)!=len(rows) or not done.issubset(expected):
        raise ValueError("Invalid or duplicate checkpoint subsets.")
    pending=[s for s in SUBSETS if "|".join(s) not in done]

    def evaluate(subset):
        result=evaluate_scale_subset(
            data, component_scores=points, features=subset, target=outcome,
            fold_column="cv_fold", feature_order=FEATURES,
            experiment_id=experiment_id, reference_threshold=5.0,
        )
        result["oxygen_scoring_variant"]=variant
        return result

    def save(result):
        rows.append(result)
        temporary=results_path.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(temporary,index=False)
        temporary.replace(results_path)

    with tqdm(total=len(SUBSETS),initial=len(done),
              desc=f"{experiment_id}: subsets",unit="subset") as progress:
        if n_jobs==1:
            for subset in pending:
                save(evaluate(subset)); progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                futures=[pool.submit(evaluate,s) for s in pending]
                for future in as_completed(futures):
                    save(future.result()); progress.update(1)
    return pd.DataFrame(rows).sort_values(["n_features","subset_key"]).reset_index(drop=True)


def plot_oxygen_delta_auc_by_size(comparison, setting, outcome, output_dir):
    """Show matched AUC changes across the 64 subsets containing oxygen saturation."""
    contains_gcs = comparison["subset_key"].str.split("|").map(
        lambda names: "blod_oxy" in names
    )
    all_gcs = comparison.loc[contains_gcs].copy()
    selected = all_gcs.loc[
        all_gcs["cohort"].eq(setting) & all_gcs["outcome"].eq(outcome)
    ].copy()
    expected_counts = [1, 6, 15, 20, 15, 6, 1]
    counts = selected.groupby("n_features").size().reindex(range(1, 8), fill_value=0)
    if counts.tolist() != expected_counts:
        raise ValueError("Expected 64 unique SpO2-containing subsets with counts 1,6,15,20,15,6,1.")
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
        ax.set_xlabel("Number of parameters in the subset (including oxygen saturation)")
        ax.set_ylabel("ΔAUC: Diagnosis-selected Scale 2 minus baseline")
        label = {"M30D": "30-day mortality", "UVI": "ICU admission"}.get(outcome, outcome)
        ax.set_title(f"{setting} | {label}\nEffect of changing the oxygen saturation scale", loc="left", pad=14)
        ax.grid(axis="y", color="#E2E8F0", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{setting.lower()}_{outcome.lower()}_scale2_delta_auc_by_subset_size"
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
        selected.to_csv(output_dir / f"{stem}.csv", index=False)
    return fig