"""Report statistics from saved, paired bootstrap subset evaluations.

No resampling or scale evaluations are performed. All AUCs are the stored
mean outer-fold AUCs, not whole-cohort AUCs. Importing performs no analysis.
"""

from itertools import combinations
from math import factorial
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sensitivity_helpers import FEATURES

EXPERIMENTS = [("Prehospital", "M30D"), ("Prehospital", "UVI"),
               ("Hospital", "M30D"), ("Hospital", "UVI")]
SUBSETS = [tuple(s) for k in range(1, 8) for s in combinations(FEATURES, k)]
SUBSET_KEYS = ["|".join(s) for s in SUBSETS]
SUBSET_INDEX = {frozenset(s): i + 1 for i, s in enumerate(SUBSETS)}
SUBSET_INDEX[frozenset()] = 0
SIZES = np.array([len(s) for s in SUBSETS])


def load_report_bootstrap(replicate_dir, n_bootstraps=2000):
    """Load complete checkpoints in a common order; never rerun a dataset."""
    if not isinstance(n_bootstraps, (int, np.integer)) or n_bootstraps < 2:
        raise ValueError("At least two complete replicate files are required.")
    replicate_dir=Path(replicate_dir)
    expected=pd.MultiIndex.from_tuples(
        [(cohort, outcome, key) for cohort,outcome in EXPERIMENTS for key in SUBSET_KEYS],
        names=["cohort", "outcome", "subset_key"],
    )
    metrics=["auc", "balanced_accuracy"]

    def read(path, bootstrap_id=None):
        frame=pd.read_csv(path)
        keys=list(expected.names)
        required=set(keys + metrics + ["features_json", "n_features"])
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns in {path.name}: {sorted(required-set(frame.columns))}")
        if len(frame)!=508 or frame.duplicated(keys).any():
            raise ValueError(f"Expected 508 unique subset evaluations in {path.name}")
        indexed=frame.set_index(keys)
        if set(indexed.index)!=set(expected):
            raise ValueError(f"Wrong subset/experiment keys in {path.name}")
        indexed=indexed.reindex(expected)
        for key, size, encoded in zip(indexed.index.get_level_values("subset_key"),
                                      indexed["n_features"], indexed["features_json"]):
            names=key.split("|")
            if list(json.loads(encoded))!=names or size!=len(names):
                raise ValueError(f"Subset metadata mismatch in {path.name}: {key}")
        if bootstrap_id is not None and "bootstrap_id" in indexed:
            if not indexed["bootstrap_id"].eq(bootstrap_id).all():
                raise ValueError(f"Wrong bootstrap ID inside {path.name}")
        values=indexed[metrics].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
            raise ValueError(f"Invalid AUC/BA values in {path.name}")
        return values.reshape(4,127,2)

    original=read(replicate_dir.parent / "original_estimates.csv")
    values=np.empty((n_bootstraps,4,127,2),dtype=float)
    for i in tqdm(range(n_bootstraps),desc="Load saved bootstrap datasets",unit="CSV"):
        path=replicate_dir / f"bootstrap_{i:04d}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}; this section never creates replicates.")
        values[i]=read(path,bootstrap_id=i)
    return {"original":original,"replicates":values,"n_bootstraps":n_bootstraps,
            "summary_dir":replicate_dir.parent}


def export_report_bootstrap(loaded, noninferiority_margin=0.02, tolerance=1e-10):
    """Export 95% pointwise percentile intervals and bootstrap stability.

    Bootstrap proportions are empirical stability/selection frequencies, not
    confidence that a hypothesis is true. Selection frequencies are separate
    from intervals for subsets fixed using the original estimates.
    """
    if not 0 < noninferiority_margin < 1:
        raise ValueError("Provide a noninferiority margin between 0 and 1.")
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError("tolerance must be finite and nonnegative.")
    original=loaded["original"]
    boot=loaded["replicates"]
    count=loaded["n_bootstraps"]
    destination=Path(loaded["summary_dir"])
    destination.mkdir(parents=True,exist_ok=True)
    records={name:[] for name in ["paired_differences_95ci", "shapley_auc_95ci",
             "parameter_removal_pairs_95ci", "parameter_removal_summary_95ci",
             "dominance_bootstrap_stability", "subset_selection_frequencies",
             "selected_subsets_95ci"]}

    def interval(values):
        return np.quantile(values,[.025,.975],axis=0)

    def metadata(experiment):
        cohort,outcome=EXPERIMENTS[experiment]
        return {"cohort":cohort,"outcome":outcome,"n_bootstraps":count,
                "auc_method":"mean_outer_fold_auc", "confidence_level":.95}

    # Exact seven-feature Shapley linear operator, including empty AUC = 0.5.
    weights=np.zeros((128,7))
    for feature_index,feature in enumerate(FEATURES):
        others=[f for f in FEATURES if f!=feature]
        for size in range(7):
            weight=factorial(size)*factorial(6-size)/factorial(7)
            for base in combinations(others,size):
                base=frozenset(base)
                weights[SUBSET_INDEX[base|{feature}],feature_index]+=weight
                weights[SUBSET_INDEX[base],feature_index]-=weight

    # Whole-subset ranking uses deterministic, explicit secondary tie rules.
    lexical_rank=np.empty(127,dtype=int)
    lexical_rank[np.argsort(np.array(SUBSET_KEYS))]=np.arange(127)

    with tqdm(total=4,desc="Summarize report statistics",unit="experiment") as progress:
        for e in range(4):
            meta=metadata(e)
            auc=np.column_stack([np.full(count,.5),boot[:,e,:,0]])
            auc_original=np.r_[.5,original[e,:,0]]

            # Paired deltas use the SAME bootstrap ID for subset and full score.
            for j,key in enumerate(SUBSET_KEYS):
                row={**meta,"subset_key":key,"n_features":int(SIZES[j]),
                     "noninferiority_margin":noninferiority_margin}
                for metric,index in (("auc",0),("ba",1)):
                    estimates=boot[:,e,j,index]-boot[:,e,126,index]
                    low,high=interval(estimates)
                    row.update({f"delta_{metric}_estimate":original[e,j,index]-original[e,126,index],
                                f"delta_{metric}_ci95_low":low,f"delta_{metric}_ci95_high":high})
                row["auc_noninferior_pointwise"]=bool(row["delta_auc_ci95_low"]>-noninferiority_margin)
                row["is_full_score"]=bool(SIZES[j]==7)
                row["selection_adjusted"]=False
                records["paired_differences_95ci"].append(row)

            # Absolute and normalized attribution. Do not clip negative values.
            phi=auc @ weights
            phi_original=auc_original @ weights
            denom=auc[:,-1]-.5
            valid=denom>0
            pct=np.full_like(phi,np.nan)
            pct[valid]=100*phi[valid]/denom[valid,None]
            pct_original=100*phi_original/(auc_original[-1]-.5) if auc_original[-1]>.5 else np.full(7,np.nan)
            for j,feature in enumerate(FEATURES):
                low,high=interval(phi[:,j])
                p_low,p_high=interval(pct[valid,j]) if valid.any() else (np.nan,np.nan)
                records["shapley_auc_95ci"].append({
                    **meta,"parameter":feature,"shapley_auc_estimate":phi_original[j],
                    "shapley_auc_ci95_low":low,"shapley_auc_ci95_high":high,
                    "percentage_estimate":pct_original[j],"percentage_ci95_low":p_low,
                    "percentage_ci95_high":p_high,"percentage_valid_replicates":int(valid.sum()),
                    "full_auc_not_above_chance_replicates":int((~valid).sum()),
                    "percentage_ci_conditional_on_positive_full_auc":bool((~valid).any()),
                })

            # Removal effect: without minus with. Exclude empty child => 63 pairs.
            for feature in FEATURES:
                others=[f for f in FEATURES if f!=feature]
                removal=[]; original_removal=[]; parent_sizes=[]
                for size in range(1,7):
                    for base_tuple in combinations(others,size):
                        base=frozenset(base_tuple)
                        expanded=base|{feature}
                        child,parent=SUBSET_INDEX[base],SUBSET_INDEX[expanded]
                        values=auc[:,child]-auc[:,parent]
                        estimate=auc_original[child]-auc_original[parent]
                        low,high=interval(values)
                        parent_key="|".join(f for f in FEATURES if f in expanded)
                        records["parameter_removal_pairs_95ci"].append({
                            **meta,"parameter":feature,"parent_subset_key":parent_key,
                            "child_subset_key":"|".join(base_tuple),"parent_n_features":size+1,
                            "delta_auc_estimate":estimate,"delta_auc_ci95_low":low,
                            "delta_auc_ci95_high":high,
                        })
                        removal.append(values);original_removal.append(estimate);parent_sizes.append(size+1)
                removal=np.column_stack(removal)
                original_removal=np.asarray(original_removal)
                parent_sizes=np.asarray(parent_sizes)
                for label,mask in [("all_63_pairs",np.ones(63,dtype=bool))]+[
                    (f"parent_size_{k}",parent_sizes==k) for k in range(2,8)
                ]:
                    # CI of a summary across contexts, not spread of the 63 dots.
                    for statistic in ("mean","median"):
                        reduction=np.mean if statistic=="mean" else np.median
                        sample_stat=reduction(removal[:,mask],axis=1)
                        low,high=interval(sample_stat)
                        records["parameter_removal_summary_95ci"].append({
                            **meta,"parameter":feature,"contexts":label,
                            "n_pairs":int(mask.sum()),"statistic":statistic,
                            "delta_auc_estimate":float(reduction(original_removal[mask])),
                            "delta_auc_ci95_low":low,"delta_auc_ci95_high":high,
                        })

            # Pairwise complete dominance over 32 common contexts, strict >tol.
            for first,second in combinations(FEATURES,2):
                others=[f for f in FEATURES if f not in (first,second)]
                first_ids=[];second_ids=[]
                for size in range(6):
                    for base in combinations(others,size):
                        base=frozenset(base)
                        first_ids.append(SUBSET_INDEX[base|{first}])
                        second_ids.append(SUBSET_INDEX[base|{second}])
                delta=auc[:,first_ids]-auc[:,second_ids]
                delta_original=auc_original[first_ids]-auc_original[second_ids]
                for a,b,d,od in [(first,second,delta,delta_original),
                                (second,first,-delta,-delta_original)]:
                    wins=(d>tolerance).sum(axis=1)
                    low,high=interval(wins/32)
                    records["dominance_bootstrap_stability"].append({
                        **meta,"row_parameter":a,"column_parameter":b,
                        "n_contexts":32,"original_wins":int((od>tolerance).sum()),
                        "original_complete_dominance":bool(np.all(od>tolerance)),
                        "complete_dominance_replicates":int((wins==32).sum()),
                        "complete_dominance_bootstrap_fraction":float((wins==32).mean()),
                        "win_fraction_ci95_low":low,"win_fraction_ci95_high":high,
                        "tolerance":tolerance,
                    })

            # Reselect independently in each bootstrap; frequency is descriptive.
            for criterion,metric_index,secondary_index in [("auc",0,1),("balanced_accuracy",1,0)]:
                for size in [None,*range(1,8)]:
                    candidates=np.flatnonzero(SIZES==size) if size is not None else np.arange(127)
                    primary=boot[:,e,candidates,metric_index]
                    secondary=boot[:,e,candidates,secondary_index]
                    ranks=np.lexsort((np.broadcast_to(lexical_rank[candidates],primary.shape),
                                      np.broadcast_to(SIZES[candidates],primary.shape),
                                      -secondary,-primary),axis=1)
                    selected=candidates[ranks[:,0]]
                    order=np.lexsort((lexical_rank[candidates],SIZES[candidates],
                                      -original[e,candidates,secondary_index],
                                      -original[e,candidates,metric_index]))
                    original_best=int(candidates[order[0]])
                    scope="all_sizes" if size is None else f"size_{size}"
                    counts=np.bincount(selected,minlength=127)
                    for j in candidates:
                        records["subset_selection_frequencies"].append({
                            **meta,"criterion":criterion,"candidate_scope":scope,
                            "subset_key":SUBSET_KEYS[j],"n_features":int(SIZES[j]),
                            "selected_count":int(counts[j]),"selected_fraction":float(counts[j]/count),
                            "selected_on_original":bool(j==original_best),
                            "tie_break":"other_metric_desc_then_n_features_asc_then_subset_key_asc",
                        })
                    for label,j in [(f"original_best_{criterion}_{scope}",original_best)]:
                        row={**meta,"selection_label":label,"subset_key":SUBSET_KEYS[j],
                             "n_features":int(SIZES[j]),"selection_adjusted":False}
                        for metric,index in [("auc",0),("ba",1)]:
                            low,high=interval(boot[:,e,j,index])
                            row.update({f"{metric}_estimate":original[e,j,index],
                                        f"{metric}_ci95_low":low,f"{metric}_ci95_high":high})
                        records["selected_subsets_95ci"].append(row)
            # Prespecified core, full score, and singleton rows for manuscript tables.
            predefined=[("full_score",126),("three_parameter_core",
                SUBSET_INDEX[frozenset(["blod_oxy","oxigen","consciousness"]) ]-1)]
            predefined += [(f"single_{f}",SUBSET_INDEX[frozenset([f])]-1) for f in FEATURES]
            for label,j in predefined:
                row={**meta,"selection_label":label,"subset_key":SUBSET_KEYS[j],
                     "n_features":int(SIZES[j]),"selection_adjusted":False}
                for metric,index in [("auc",0),("ba",1)]:
                    low,high=interval(boot[:,e,j,index])
                    row.update({f"{metric}_estimate":original[e,j,index],
                                f"{metric}_ci95_low":low,f"{metric}_ci95_high":high})
                records["selected_subsets_95ci"].append(row)
            progress.update(1)

    files={}
    for name,rows in records.items():
        path=destination / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path,index=False)
        files[name]=str(path)
    status=[
        ("paired_delta_and_noninferiority", "computed", "Paired mean-CV AUC differences; lower 95% bound > -0.02. Pointwise, no multiplicity/selection adjustment."),
        ("shapley", "computed", "Absolute AUC and normalized percentage percentile intervals; invalid percentage denominators counted explicitly."),
        ("dominance_stability", "computed", "Proportion of resamples with complete dominance; not a posterior probability or significance test."),
        ("parameter_removal", "computed", "Paired 63-context removal CIs, plus mean/median and full-score removal effects."),
        ("selection", "computed", "Reselect best AUC and best BA subset in each replicate. Fixed-original-subset CIs are not selection-adjusted."),
    ]
    path=destination / "bootstrap_report_calculation_status.csv"
    pd.DataFrame(status,columns=["statistic","status","note"]).to_csv(path,index=False)
    files["calculation_status"]=str(path)
    return pd.DataFrame([{"output":key,"csv_path":value} for key,value in files.items()])
