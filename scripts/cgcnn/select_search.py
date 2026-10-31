#!/usr/bin/env python3
"""Rank gate configurations x recipes per outer fold by inner-validation score and keep the top n."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inner-csv", required=True)
    p.add_argument("--stages", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--std-weight", type=float, default=0.10)
    p.add_argument("--configs")
    a = p.parse_args()
    stages = [x for x in a.stages.split(",") if x]
    df = pd.read_csv(a.inner_csv)
    df = df[df.stage.isin(stages)].copy()
    if a.configs:
        df = df[df.config.isin([x for x in a.configs.split(",") if x])]
    if df.empty:
        raise RuntimeError("No rows for stages %s" % stages)
    agg = df.groupby(["config", "recipe", "outer_fold"], as_index=False).agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"), runs=("mae", "size"),
        median_epoch=("selected_epoch", "median"), boundary_rate=("boundary_hit", "mean"))
    agg["selection_score"] = agg.mean_mae + a.std_weight * agg.std_mae.fillna(0)
    per_fold = {}
    for fold in sorted(agg.outer_fold.unique()):
        g = agg[agg.outer_fold == fold].sort_values(["selection_score", "mean_mae", "config", "recipe"])
        per_fold[str(int(fold))] = [{"config": r.config, "recipe": r.recipe, "selected_epoch": int(round(r.median_epoch)),
                                     "selection_score": float(r.selection_score), "mean_mae": float(r.mean_mae),
                                     "boundary_rate": float(r.boundary_rate)} for _, r in g.head(a.top_n).iterrows()]
    payload = {"stages": stages, "top_n": a.top_n, "std_weight": a.std_weight, "per_fold": per_fold}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    agg.to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
