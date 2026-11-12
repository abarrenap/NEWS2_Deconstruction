# Sensitivity analyses

Run the notebooks locally from this folder or the repository root. The original dataset must be available at `data/NEWs2.csv` in the NEWS2 repository.

- `01_bootstrap_2000_95ci.ipynb`: 2,000 ordinary patient bootstrap samples, repeated grouped cross-validation and threshold selection, 95% pointwise percentile intervals for all 127 subsets and all four experiments. Checkpoints support resumption. This is a substantial computation.
- `02_consciousness_gcs_below_13.ipynb`: modified consciousness score (3 points for GCS < 13, 0 otherwise), compared with a newly computed matched baseline on identical patients/folds. 
- `03_oxygen_saturation_scale2_diagnosis10.ipynb` compares the all-Scale 1 baseline with Scale 2 selected by `Dg #1 == 10`, in both settings and outcomes.
- `sensitivity_helpers.py`: shared preparation and bootstrap implementation, reusing the original scale evaluator.



### Additional report intervals

The final section of notebook 1 uses `bootstrap_report_intervals.py` to read saved replicate CSVs and export paired subset-minus-full intervals/noninferiority, Shapley intervals, parameter-removal intervals, dominance stability, and selection frequencies into `boostrap/`. It does not rerun bootstrap experiments. These use stored mean-CV AUCs, which differ from the manuscript's whole-cohort AUC definition.

