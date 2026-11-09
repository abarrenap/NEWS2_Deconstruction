"""Standalone interactive HTML report for NEWS2 scale deconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Sequence

import pandas as pd
import plotly.io as pio

from .news2_scale_utils import make_marginal_ablation_plot
from .news2_scale_deconstruction import (
    make_component_delta_auc_plot,
    make_scale_branching_subset_plot,
)


@dataclass(frozen=True)
class ScaleExperimentReport:
    """One cohort/outcome result set for the combined report."""

    cohort: str
    outcome: str
    results: pd.DataFrame
    news2_auc: float


def _best_subset(results: pd.DataFrame) -> pd.Series:
    return results.sort_values(
        ["balanced_accuracy", "auc", "n_features", "subset_label"],
        ascending=[False, False, True, True],
    ).iloc[0]


def _complete_scale(results: pd.DataFrame) -> pd.Series:
    maximum_features = int(results["n_features"].max())
    complete = results.loc[results["n_features"].eq(maximum_features)]
    if len(complete) != 1:
        raise ValueError("Results must contain exactly one complete NEWS2 scale.")
    return complete.iloc[0]


def _plot_html(figure, *, include_plotlyjs: bool) -> str:
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs="inline" if include_plotlyjs else False,
        config={"responsive": True, "displaylogo": False},
    )


def _metric(value: float) -> str:
    return "—" if pd.isna(value) else f"{float(value):.3f}"


def _panel(
    experiment: ScaleExperimentReport,
    feature_order: Sequence[str],
    *,
    include_plotlyjs: bool,
) -> str:
    best = _best_subset(experiment.results)
    complete = _complete_scale(experiment.results)
    title = f"{experiment.outcome} — {experiment.cohort}"
    branch = make_scale_branching_subset_plot(
        experiment.results,
        feature_order,
        news2_auc=experiment.news2_auc,
        title=title,
    )
    ablation = make_marginal_ablation_plot(
        experiment.results,
        feature_order,
        title=title,
    )
    return f"""
    <article class="experiment-panel">
      <header>
        <p class="eyebrow">{escape(experiment.cohort)}</p>
        <h3>{escape(experiment.outcome)}</h3>
      </header>
      <p class="summary">
        The highest-balanced-accuracy partial scale retained
        <strong>{escape(str(best.subset_label))}</strong> and used a mean
        inner-selected threshold of <strong>{_metric(best.threshold_mean)}</strong>.
        The complete seven-component scale achieved BA {_metric(complete.balanced_accuracy)}
        with mean threshold {_metric(complete.threshold_mean)}.
      </p>
      <h4>Best partial scale</h4>
      <dl class="metrics">
        <div><dt>AUC</dt><dd>{_metric(best.auc)}</dd></div>
        <div><dt>Balanced accuracy</dt><dd>{_metric(best.balanced_accuracy)}</dd></div>
        <div><dt>Sensitivity</dt><dd>{_metric(best.sensitivity)}</dd></div>
        <div><dt>Specificity</dt><dd>{_metric(best.specificity)}</dd></div>
        <div><dt>Mean threshold</dt><dd>{_metric(best.threshold_mean)}</dd></div>
      </dl>
      <div class="figure">{_plot_html(branch, include_plotlyjs=include_plotlyjs)}</div>
      <div class="figure">{_plot_html(ablation, include_plotlyjs=False)}</div>
    </article>
    """


def create_news2_scale_html(
    experiments: Sequence[ScaleExperimentReport],
    feature_order: Sequence[str],
    output_path: str | Path,
    *,
    title: str = "NEWS2 scale deconstruction",
    component_delta_results: pd.DataFrame | None = None,
) -> Path:
    """Write one self-contained report covering all four experiments."""
    if len(experiments) != 4:
        raise ValueError("Exactly four cohort/outcome experiments are required.")
    if {item.cohort for item in experiments} != {"Prehospital", "Hospital"}:
        raise ValueError("Experiments must include Prehospital and Hospital.")
    if {item.outcome for item in experiments} != {"M30D", "UVI"}:
        raise ValueError("Experiments must include M30D and UVI.")

    ordered = sorted(
        experiments,
        key=lambda item: (item.cohort != "Prehospital", item.outcome != "M30D"),
    )
    overview_rows = []
    best_auc_rows = []
    for item in ordered:
        best = _best_subset(item.results)
        complete = _complete_scale(item.results)
        overview_rows.append(
            "<tr>"
            f"<td>{escape(item.cohort)}</td>"
            f"<td>{escape(item.outcome)}</td>"
            f"<td>{escape(str(best.subset_label))}</td>"
            f"<td>{_metric(best.auc)}</td>"
            f"<td>{_metric(best.balanced_accuracy)}</td>"
            f"<td>{_metric(best.sensitivity)}</td>"
            f"<td>{_metric(best.specificity)}</td>"
            f"<td>{_metric(best.threshold_mean)}</td>"
            f"<td>{_metric(complete.balanced_accuracy)}</td>"
            "</tr>"
        )
        best_auc = item.results.sort_values(
            ["auc", "balanced_accuracy", "n_features"],
            ascending=[False, False, True],
        ).iloc[0]
        best_auc_rows.append(
            "<tr>"
            f"<td>{escape(item.cohort)}</td><td>{escape(item.outcome)}</td>"
            f"<td>{escape(str(best_auc.subset_label))}</td><td>{_metric(best_auc.auc)}</td>"
            "</tr>"
        )

    delta_section = ""
    if component_delta_results is not None:
        delta_table = component_delta_results.copy()
        delta_table["auc"] = delta_table["auc"].map(_metric)
        delta_rows = "".join(
            f"<tr><td>{escape(str(row.feature))}</td><td>{escape(str(row.outcome))}</td>"
            f"<td>{row.auc}</td><td>{int(row.n_patients):,}</td></tr>"
            for row in delta_table.itertuples(index=False)
        )
        delta_section = f"""
  <section>
    <h2>Prehospital-to-hospital raw-measurement changes</h2>
    <p class="lede">For each paired patient, the hospital raw measurement minus the prehospital raw measurement was used as the predictor. AUCs below 0.5 are inverted so the displayed value represents discrimination strength regardless of direction.</p>
    <div class="figure">{_plot_html(make_component_delta_auc_plot(component_delta_results), include_plotlyjs=False)}</div>
    <div class="overview"><table><thead><tr><th>Component</th><th>Outcome</th><th>AUC</th><th>Paired patients</th></tr></thead><tbody>{delta_rows}</tbody></table></div>
  </section>"""

    panels = []
    include_plotlyjs = True
    for item in ordered:
        panels.append(
            _panel(item, feature_order, include_plotlyjs=include_plotlyjs)
        )
        include_plotlyjs = False

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ --ink:#172033; --muted:#5b6678; --line:#dce3eb; --paper:#fff; --wash:#f5f7fa; --accent:#1f4e79; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--wash); color:var(--ink); font:16px/1.55 Arial, sans-serif; }}
    main {{ max-width:1900px; margin:auto; padding:3rem clamp(1rem,3vw,4rem) 5rem; }}
    h1,h2,h3,h4 {{ font-family:Georgia,serif; line-height:1.2; }}
    h1 {{ font-size:clamp(2rem,4vw,3.2rem); margin:0 0 .7rem; }}
    h2 {{ margin-top:3rem; border-bottom:2px solid var(--accent); padding-bottom:.5rem; }}
    h3 {{ font-size:1.7rem; margin:.1rem 0; }}
    h4 {{ margin:1.4rem 0 .6rem; }}
    .lede,.summary {{ color:var(--muted); }}
    .method {{ background:#eaf2f8; border-left:4px solid var(--accent); padding:1rem 1.2rem; margin:1.5rem 0; }}
    .overview,.experiment-panel {{ background:var(--paper); border:1px solid var(--line); padding:1.25rem; }}
    .overview {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
    th,td {{ border-bottom:1px solid var(--line); padding:.65rem; text-align:left; }}
    th {{ color:var(--muted); text-transform:uppercase; font-size:.75rem; letter-spacing:.04em; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1.5rem; align-items:start; }}
    .eyebrow {{ color:var(--accent); font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:.78rem; margin:0; }}
    .metrics {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.6rem; }}
    .metrics div {{ background:var(--wash); border:1px solid var(--line); padding:.7rem; }}
    dt {{ color:var(--muted); font-size:.75rem; text-transform:uppercase; }}
    dd {{ margin:.15rem 0 0; font-size:1.25rem; font-weight:700; }}
    .figure {{ min-width:0; overflow:hidden; }}
    footer {{ color:var(--muted); margin-top:2rem; font-size:.85rem; }}
    @media (max-width:1100px) {{ .grid {{ grid-template-columns:1fr; }} }}
    @media (max-width:700px) {{ .metrics {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
<main>
  <header>
    <p class="eyebrow">Nested cross-validation report</p>
    <h1>{escape(title)}</h1>
    <p class="lede">Direct evaluation of every non-empty subset of the seven NEWS2 components in prehospital and hospital cohorts for 30-day mortality and UVI.</p>
  </header>
  <section class="method">
    <strong>Method.</strong> Raw measurements are converted to standard NEWS2 points and summed. No predictive model is fitted. AUC uses the partial NEWS2 sum directly; the classification cutoff is chosen only from each outer-training set by maximizing mean inner-fold balanced accuracy, then applied to the untouched outer-test fold.
  </section>
  <section>
    <h2>Overview</h2>
    <div class="overview">
      <h3>Maximum Balanced Accuracy subset</h3>
      <table>
        <thead><tr><th>Cohort</th><th>Outcome</th><th>Best BA subset</th><th>AUC</th><th>BA</th><th>Sensitivity</th><th>Specificity</th><th>Threshold</th><th>Complete-scale BA</th></tr></thead>
        <tbody>{''.join(overview_rows)}</tbody>
      </table>
      <h3>Maximum AUC subset</h3>
      <table>
        <thead><tr><th>Cohort</th><th>Outcome</th><th>Best AUC subset</th><th>AUC</th></tr></thead>
        <tbody>{''.join(best_auc_rows)}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Experiment details</h2>
    <div class="grid">{''.join(panels)}</div>
  </section>
  {delta_section}
  <footer>Generated from the nested-CV NEWS2 scale deconstruction results.</footer>
</main>
</body>
</html>"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
