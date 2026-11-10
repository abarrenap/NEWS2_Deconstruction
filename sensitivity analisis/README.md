# Sensitivity analyses

Run the notebooks locally from this folder or the repository root. The original dataset must be available at `data/NEWs2.csv` in the NEWS2 repository. No analyses were executed during authoring.

- `01_bootstrap_2000_95ci.ipynb`: 2,000 ordinary patient bootstrap samples, repeated grouped cross-validation and threshold selection, 95% pointwise percentile intervals for all 127 subsets and all four experiments. Checkpoints support resumption. This is a substantial computation.
- `02_consciousness_gcs_below_13.ipynb`: modified consciousness score (3 points for GCS < 13, 0 otherwise), compared with a newly computed matched baseline on identical patients/folds. Includes Shapley and dominance figures.
- `sensitivity_helpers.py`: shared preparation and bootstrap implementation, reusing the original scale evaluator.

Missing FiO2/GCS measurements stay missing rather than receiving implicit category assignments. GCS totals outside 3–15 are also excluded, with aggregate counts shown before execution. Both notebooks document this eligibility difference from the original preparation. Outputs stay under `data/outputs/sensitivity/`, which is excluded from Git by the existing `data/` ignore rule. Keep patient-level files private.

## Oxygen saturation Scale 2 sensitivity

`03_oxygen_saturation_scale2_diagnosis10.ipynb` compares the all-Scale 1 baseline with Scale 2 selected by `Dg #1 == 10`, in both settings and outcomes. The GCS cutoff remains 15. Includes matched AUC/BA results, Shapley charts, dominance matrices, and four delta-AUC boxplots grouped by saturation-containing subset size. `oxygen_scale_sensitivity_helpers.py` supplies explicit variant-specific scoring and resumable searches. The notebook documents the official Scale 2 chart, diagnosis-rule assumption, missing diagnosis handling, and fractional saturation bins. Outputs remain under `data/outputs/sensitivity/oxygen_scale2_diagnosis10/`.
