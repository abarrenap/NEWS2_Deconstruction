# NEWS2 Deconstruction

## Overview

This repository provides the analysis code for decomposing NEWS2 into its constituent parameters and evaluating their predictive contribution in prehospital and hospital care.

The workflow evaluates all **127 non-empty combinations** of the seven NEWS2 parameters. Each subset is scored by summing the original NEWS2 component points for the retained parameters. Classification thresholds are selected within the training data and evaluated on held-out folds.

The seven parameters are:

- Respiratory rate
- Peripheral oxygen saturation
- Supplemental oxygen
- Systolic blood pressure
- Heart rate
- Consciousness
- Temperature

The analysis examines two outcomes: **30-day mortality** and **intensive care admission**, represented in the code as `M30D` and `UVI`, respectively.

## Interactive report

The combined interactive report is included in the repository root:

**[NEWS2 scale deconstruction report](NEWS2_scale_deconstruction_report.html)**

Download the HTML file and open it in a web browser to explore the results. The report includes interactive subset and ablation figures, four Shapley AUC charts, and four dominance matrices. The latter eight figures are embedded images, so the report remains self-contained.

## Analysis workflow

1. Prepare the prehospital and hospital cohorts.
2. Convert physiological measurements into NEWS2 component points.
3. Generate every non-empty subset of the seven parameters.
4. Evaluate each subset using stratified outer cross-validation.
5. Select classification thresholds using inner folds of the outer-training data.
6. Evaluate performance on the untouched outer-test fold.
7. Compare subset performance, parameter removal effects, and component-level AUC differences.
8. Decompose AUC above chance using exact Shapley attribution.
9. Compare parameters through pairwise complete-dominance analysis.
10. Generate figures and a combined self-contained HTML report.

The notebook configures **10 outer folds** and a random seed of **42**. Evaluation includes ROC AUC, balanced accuracy, sensitivity, and specificity.

The prediction score is the sum of the retained NEWS2 component points. No predictive model is fitted.

## Shapley decomposition of discrimination

For each setting and outcome, exact Shapley values attribute the full NEWS2 score’s discrimination above chance to its seven parameters. A parameter’s contribution is its incremental AUC gain, averaged across all possible orders in which parameters could be added.

The analysis assigns AUC = 0.5 to the empty subset and uses the mean outer-fold AUC for every non-empty subset. Contributions sum to **full-score AUC − 0.5**. Bars express each contribution as a percentage of that total and sum to 100%, including any negative contributions. Percentage attribution requires the full-score AUC to exceed 0.5.

A contribution of 30% means 30% of the discrimination above chance, not an AUC increase of 0.30. Percentages are relative to each experiment’s full-score AUC; use the exported absolute AUC contributions when comparing magnitudes between experiments.

## Dominance analysis

For each pair of parameters, the analysis compares the AUC gain from adding either parameter to the **same subset** of the other five parameters. There are **32 common contexts per pair**, including the empty subset.

Each experiment has its own matrix:

- **Rows:** candidate parameters; **columns:** comparison parameters.
- **Cell counts:** contexts where the row parameter adds more AUC than the column parameter.
- **32/32 · CD:** strict complete dominance; the row parameter contributes more in every common context.
- **Intermediate counts:** superiority depends on the combination; complete dominance is not established.
- **Ties:** differences within `1e-10` are treated as numerical ties and are not counted as wins.

All four matrices use the Turbo color scale over the same 0–1 range. The color represents the fraction of contexts won, not the magnitude of the AUC advantage. Complete dominance can establish a partial hierarchy without ordering every parameter. A larger incremental contribution can also mean a smaller AUC reduction when both additions reduce discrimination.

These are descriptive comparisons of fixed NEWS2 partial scores, without significance tests or uncertainty intervals. All subsets must use the same patients, folds, and AUC aggregation. The Shapley contributions correspond to general dominance weights for this AUC-based game; ranking first by Shapley contribution does not establish complete dominance.

## Parameter-removal and patient-level plots

The marginal ablation plots show **AUC after removal minus AUC before removal** for 63 pairs per parameter, excluding pairs that leave an empty score. Negative values indicate reduced discrimination after removal. These 63 removal pairs differ from the 32 common contexts used to compare a pair of parameters in the dominance matrices.

The optional patient-level SHAP beeswarm and bar plots explain contributions to individual **NEWS2 point totals**, relative to a sampled background. They do not attribute outcome discrimination and are separate from the Shapley AUC and dominance analyses. The HTML’s eight new attribution figures are the four AUC charts and four dominance matrices; optional patient-level SHAP figures are not embedded by this export workflow.

## Repository contents

```text
.
├── NEWS2_deconstruction_scale.ipynb
├── NEWS2_scale_deconstruction_report.html
├── functions/
│   ├── __init__.py
│   ├── news.py
│   ├── news2_scale_deconstruction.py
│   ├── news2_scale_utils.py
│   ├── news2_scale_html_report.py
│   ├── news2_auc_shapley.py
│   ├── news2_auc_dominance.py
│   ├── shap_values.py
│   └── test_news2_scale_decomposition.py
├── requirements.txt
└── README.md
```

| File | Purpose |
| --- | --- |
| `NEWS2_deconstruction_scale.ipynb` | Main analysis notebook: cohort preparation, subset evaluation, comparisons, and report generation. |
| `NEWS2_scale_deconstruction_report.html` | Combined interactive report containing analysis results and figures. |
| `functions/news.py` | Reference NEWS2 score calculation. |
| `functions/news2_scale_deconstruction.py` | Component scoring, threshold optimization, subset evaluation, resumable searches, and component comparisons. |
| `functions/news2_scale_utils.py` | Stratified fold assignment and subset visualization utilities. |
| `functions/news2_scale_html_report.py` | Interactive HTML report generation. |
| `functions/test_news2_scale_decomposition.py` | Tests for component scoring, threshold selection, and separation of training and test labels. |
| `functions/news2_auc_shapley.py` | Exact Shapley decomposition of AUC above chance and attribution charts. |
| `functions/news2_auc_dominance.py` | Pairwise dominance matrices, summary counts, and context-specific AUC gains. |
| `functions/shap_values.py` | Optional patient-level SHAP explanations of NEWS2 points. |
| `requirements.txt` | Python dependencies. |

## Installation

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Or on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data preparation

The analysis requires the study dataset. Repository code alone does not provide access to patient-level data.

The notebook expects a semicolon-delimited CSV file with UTF-8 BOM-compatible encoding. Its cohort preparation code uses the following columns, with suffix `1` for prehospital measurements and `H` for hospital measurements:

| Measurement | Expected columns |
| --- | --- |
| Patient identifier | `#ID` |
| Respiratory rate | `FR-1`, `FR-H` |
| Oxygen saturation | `SpO2-1`, `SpO2-H` |
| Inspired oxygen fraction | `FiO2-1`, `FiO2-H` |
| Systolic blood pressure | `TAS-1`, `TAS-H` |
| Heart rate | `FC-1`, `FC-H` |
| Temperature | `TT-1`, `TT-H` |
| Glasgow Coma Scale components | `GCS.O-1`, `GCS.V-1`, `GCS.M-1`, and corresponding `-H` columns |
| 30-day mortality | `Mortalidad a 30 dias` |
| Intensive care admission | `UVI` |

Supplemental oxygen is derived from `FiO2 > 0.21`. Consciousness is encoded as alert when the total Glasgow Coma Scale score equals 15. Outcomes must be encoded as binary values, `0` and `1`.

The scoring implementation uses NEWS2 oxygen saturation **Scale 1**. The cohort preparation excludes rows with missing values in the seven analysis parameters.

Before running the notebook, update the input CSV path and `OUTPUT_DIR` to match your local folder structure. If using another dataset, adapt the column mapping and verify measurement units, missing-value handling, and outcome definitions.

## Running the analysis

Start Jupyter from the repository root:

```bash
python -m jupyterlab
```

Open `NEWS2_deconstruction_scale.ipynb` and run the cells in order.

The notebook performs four cohort–outcome analyses:

| Care setting | Outcome |
| --- | --- |
| Prehospital | 30-day mortality |
| Prehospital | Intensive care admission |
| Hospital | 30-day mortality |
| Hospital | Intensive care admission |

Subset searches support checkpointing so interrupted computations can resume from saved results. Use separate output directories when changing the dataset or analysis settings to avoid reusing incompatible checkpoints.

## Outputs

The workflow produces subset-level results, checkpoint files, interactive visualizations, and an HTML report.

The visualizations include:

- Subset performance across the number of retained parameters
- Changes in AUC after removing individual parameters
- Component-level AUC comparisons
- Four Shapley AUC attribution charts, one per setting and outcome
- Four pairwise dominance matrices, one per setting and outcome

The published report is stored in the repository root as `NEWS2_scale_deconstruction_report.html`.

When rerunning the analysis, the destination of the newly generated report is determined by the report-generation cell in the notebook. To save it directly to the repository root, use `Path("NEWS2_scale_deconstruction_report.html")` as the report destination when running Jupyter from that folder.

### Figure exports and HTML generation

Each experiment’s Shapley AUC chart and dominance matrix appears after its experiment in the notebook. With the current `OUTPUT_DIR`, exports are organized as:

```text
data/outputs/news2_scale_deconstruction/
├── shapley_auc/      # Four charts (PNG/PDF) and attribution tables (CSV)
├── dominance_auc/    # Four matrices (PNG/PDF), summaries and contexts (CSV)
└── NEWS2_shapley_auc.csv
```

Run all four Shapley chart cells and all four dominance matrix cells **before** the HTML export cell. The export passes `figures_dir=OUTPUT_DIR` and embeds the eight saved PNG files as image data inside the HTML. It does not recalculate these figures and raises an error if an expected image is missing. Regenerate the figures after changing data or settings to avoid embedding stale images.

The published HTML can be viewed without the dataset or output folders. Regenerating the complete report requires the experiment result tables and saved figures. Patient-level SHAP cells are optional and are not required for this export.

## Validation

Run the included tests from the repository root:

```bash
python -m pytest functions/test_news2_scale_decomposition.py
```

The tests check that component scores reproduce the reference NEWS2 total, threshold optimization maximizes the specified inner-fold criterion, and outer-test labels do not influence threshold selection.

## Reproducibility

For a reproducible analysis, retain the dataset version, code revision, Python and package versions, configuration settings, and generated outputs.

The supplied dependency list does not pin package versions. Record the environment used for the final manuscript analysis:

```bash
python -m pip freeze > requirements-frozen.txt
```

