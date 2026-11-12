"""Paired, two-sided DeLong tests of fixed NEWS2 subsets versus full NEWS2.

Run: python 'sensitivity analisis/compute_delong_comparisons.py'
Uses original patient data and baseline preparation from sensitivity_helpers.
Outputs whole-cohort AUCs, NOT mean cross-validation AUCs; does not edit the
manuscript or its point estimates. Tests are pointwise, without multiplicity
or subset-selection adjustment. Reference: DeLong et al. (1988), Biometrics
44:837–845; structural-component covariance with half credit for tied scores.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def paired_delong(y, subset_score, full_score):
    """Return AUCs, subset-minus-full difference, SE, z, p and log10(p).

    Both scores must refer to the SAME patients in the SAME order. Higher
    scores must indicate greater event risk. At least two of each class are
    required. Degenerate covariance is explicitly identified.
    """
    y = np.asarray(y)
    scores = np.asarray([subset_score, full_score], dtype=float)
    if y.ndim != 1 or scores.shape != (2, len(y)):
        raise ValueError('Expected binary labels and two aligned score vectors.')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Labels must be binary and scores finite; no silent filtering.')
    m, n = int(np.sum(y == 1)), int(np.sum(y == 0))
    if min(m, n) < 2:
        raise ValueError('Need at least two events and two non-events.')
    positive_components, negative_components = [], []
    for score in scores:
        pos, neg = score[y == 1], score[y == 0]
        sp, sn = np.sort(pos), np.sort(neg)
        # Per-event probability of beating a non-event, with ties worth 1/2.
        vp = (np.searchsorted(sn, pos, side='left') +
              np.searchsorted(sn, pos, side='right')) / (2 * n)
        # Per-non-event probability of being beaten by an event.
        vn = 1 - (np.searchsorted(sp, neg, side='left') +
                  np.searchsorted(sp, neg, side='right')) / (2 * m)
        positive_components.append(vp)
        negative_components.append(vn)
    vp, vn = np.asarray(positive_components), np.asarray(negative_components)
    auc = vp.mean(axis=1)
    delta = float(auc[0] - auc[1])
    variance = float(np.var(vp[0] - vp[1], ddof=1) / m +
                     np.var(vn[0] - vn[1], ddof=1) / n)
    se = np.sqrt(variance)
    if variance == 0:
        identical = np.array_equal(scores[0], scores[1])
        z = 0.0 if delta == 0 else np.copysign(np.inf, delta)
        p = 1.0 if delta == 0 else np.nan
        logp = 0.0 if delta == 0 else np.nan
        status = 'identical_scores' if identical else 'zero_variance_no_wald_inference'
    else:
        z = delta / se
        logp = (np.log(2) + norm.logsf(abs(z))) / np.log(10)
        p = float(10 ** logp)
        status = 'ok'
    return dict(auc_subset=auc[0], auc_full=auc[1], delta_auc=delta,
                delta_auc_se=se, z=z, p_value=p, log10_p_value=logp,
                status=status, n_patients=len(y), n_events=m, n_non_events=n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/NEWs2.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'data/outputs/sensitivity/delong')
    args = parser.parse_args()
    from sensitivity_helpers import load_original_data, prepare_cohort, FEATURES, SUBSETS
    from functions.news2_scale_deconstruction import score_news2_components
    raw = load_original_data(args.data)
    results = []
    with tqdm(total=4 * 126, desc='Paired DeLong tests', unit='test') as progress:
        for setting, suffix in [('Prehospital', '1'), ('Hospital', 'H')]:
            prepared = prepare_cohort(raw, suffix, gcs_cutoff=15)
            for outcome in ['M30D', 'UVI']:
                cohort = prepared.dropna(subset=[outcome]).reset_index(drop=True)
                points = score_news2_components(cohort, FEATURES)
                full = points[FEATURES].sum(axis=1).to_numpy()
                y = cohort[outcome].to_numpy()
                for subset in SUBSETS:
                    if len(subset) == len(FEATURES):
                        continue
                    result = paired_delong(y, points[list(subset)].sum(axis=1), full)
                    result.update(cohort=setting, outcome=outcome,
                                  subset_key='|'.join(subset), n_features=len(subset),
                                  is_core=set(subset) == {'consciousness', 'blod_oxy', 'oxigen'},
                                  auc_method='whole_cohort', test='paired_two_sided_delong',
                                  multiplicity_adjusted=False, selection_adjusted=False,
                                  preparation='sensitivity_helpers baseline; GCS <15; complete cases')
                    results.append(result)
                    progress.update()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.output_dir / 'delong_all_subsets_vs_full.csv', index=False)
    core = frame.loc[frame.is_core]
    core.to_csv(args.output_dir / 'delong_core_vs_full.csv', index=False)
    print(core[['cohort', 'outcome', 'n_patients', 'delta_auc', 'p_value', 'status']].to_string(index=False))
    print(f'Saved 504 paired tests and four core comparisons to {args.output_dir}')
    print('Baseline eligibility follows sensitivity_helpers; reconcile cohort counts before citing.')


if __name__ == '__main__':
    main()
