#!/usr/bin/env python3
"""Command-line workflow for summarize results in the TF2.9 multi-backbone engine."""

from __future__ import annotations

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def load(root):
 rows=[]
 for p in sorted(root.glob("*/seed_*/fold_*/metrics.json")):
  r=json.loads(p.read_text()); flat={k:v for k,v in r.items() if k not in {"metrics","runtime"}}; flat.update({"metric_"+k:v for k,v in r.get("metrics",{}).items()}); rows.append(flat)
 return pd.DataFrame(rows)
p=argparse.ArgumentParser(); p.add_argument("--outdir",type=Path,default=Path("outputs")); p.add_argument("--backbone",required=True); p.add_argument("--task",required=True); a=p.parse_args(); root=a.outdir/a.backbone/a.task; f=load(root)
if f.empty: raise SystemExit("No metrics")
f.to_csv(root/"all_fold_metrics.csv",index=False)
seed=f.groupby(["variant","base_seed"],as_index=False).agg(mean_primary=("primary_metric","mean"),fold_std=("primary_metric",lambda x:x.std(ddof=0)),folds=("fold","count")); seed.to_csv(root/"seed_level_summary.csv",index=False)
model=seed.groupby("variant",as_index=False).agg(mean_of_seed_means=("mean_primary","mean"),seed_std_sample=("mean_primary",lambda x:x.std(ddof=1)),seeds=("base_seed","count")); model.to_csv(root/"model_summary.csv",index=False)
out={"model_summary":model.to_dict("records")};
if {"baseline","materon"}.issubset(set(f.variant)):
 b=f[f.variant=="baseline"]; m=f[f.variant=="materon"]; q=b.merge(m,on=["base_seed","fold"],suffixes=("_baseline","_materon"),validate="one_to_one"); classification=str(f.iloc[0].task_type)=="classification"; q["improvement"]=(q.primary_metric_materon-q.primary_metric_baseline)*(1 if classification else -1); q["materon_wins"]=q.improvement>0; q.to_csv(root/"paired_seed_fold_results.csv",index=False)
 s=q.groupby("base_seed",as_index=False).agg(baseline_mean=("primary_metric_baseline","mean"),materon_mean=("primary_metric_materon","mean"),mean_improvement=("improvement","mean"),wins=("materon_wins","sum")); s.to_csv(root/"paired_seed_summary.csv",index=False)
 imp=s.mean_improvement.to_numpy(float); desc=q.improvement.to_numpy(float)
 stats={"main_unit":"five seed-level five-fold means","num_seeds":len(imp),"mean_seed_level_improvement":float(imp.mean()),"std_seed_level_improvement":float(imp.std(ddof=1)),"seed_wins":int((imp>0).sum()),"descriptive_seed_fold_pairs":len(desc),"descriptive_pair_wins":int((desc>0).sum()),"descriptive_pair_mean":float(desc.mean())}
 try:
  from scipy.stats import ttest_1samp,wilcoxon
  stats["seed_level_ttest_p_two_sided"]=float(ttest_1samp(imp,0).pvalue); stats["seed_level_wilcoxon_p_two_sided"]=float(wilcoxon(imp).pvalue)
 except Exception: stats["seed_level_ttest_p_two_sided"]=None; stats["seed_level_wilcoxon_p_two_sided"]=None
 out["paired"]=stats
(root/"overall_summary.json").write_text(json.dumps(out,indent=2)); print(model.to_string(index=False)); print(json.dumps(out.get("paired",{}),indent=2))
