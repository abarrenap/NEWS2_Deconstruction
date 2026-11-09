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

Download the HTML file and open it in a web browser to explore the results and interactive figures.

## Analysis workflow

1. Prepare the prehospital and hospital cohorts.
2. Convert physiological measurements into NEWS2 component points.
3. Generate every non-empty subset of the seven parameters.
4. Evaluate each subset using stratified outer cross-validation.
5. Select classification thresholds using inner folds of the outer-training data.
6. Evaluate performance on the untouched outer-test fold.
7. Compare subset performance, parameter removal effects, and component-level AUC differences.
8. Generate interactive figures and a combined HTML report.

The notebook configures **10 outer folds** and a random seed of **42**. Evaluation includes ROC AUC, balanced accuracy, sensitivity, and specificity.

The prediction score is the sum of the retained NEWS2 component points. No predictive model is fitted.

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
│   └── test_news2_scale_deconstruction.py
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
| `functions/test_news2_scale_deconstruction.py` | Tests for component scoring, threshold selection, and separation of training and test labels. |
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
- Performance matrices and Pareto frontiers
- Changes in AUC after removing individual parameters
- Component-level AUC comparisons

The published report is stored in the repository root as `NEWS2_scale_deconstruction_report.html`.

When rerunning the analysis, the destination of the newly generated report is determined by the report-generation cell in the notebook. To save it directly to the repository root, use `Path("NEWS2_scale_deconstruction_report.html")` as the report destination when running Jupyter from that folder.

## Validation

Run the included tests from the repository root:

```bash
python -m pytest functions/test_news2_scale_deconstruction.py
```

The tests check that component scores reproduce the reference NEWS2 total, threshold optimization maximizes the specified inner-fold criterion, and outer-test labels do not influence threshold selection.

## Reproducibility

For a reproducible analysis, retain the dataset version, code revision, Python and package versions, configuration settings, and generated outputs.

The supplied dependency list does not pin package versions. Record the environment used for the final manuscript analysis:

```bash
python -m pip freeze > requirements-frozen.txt
```

