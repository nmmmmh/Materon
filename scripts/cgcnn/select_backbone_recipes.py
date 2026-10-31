#!/usr/bin/env python3
"""Keep the top-n training recipes per outer fold for CGCNN and no-gate physics CGCNN (successive halving).

Score = mean inner-validation MAE + std_weight * SD over repeats/seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BACKBONES = ["cgcnn_baseline", "physics_full20"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inner-csv", required=True)
    p.add_argument("--stages", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-n", type=int, required=True)
    p.add_argument("--std-weight", type=float, default=0.10)
    a = p.parse_args()
    stages = a.stages.split(",")
    df = pd.read_csv(a.inner_csv)
    df = df[df.stage.isin(stages) & df.config.isin(BACKBONES)].copy()
    if df.empty:
        raise RuntimeError("No backbone rows")
    agg = df.groupby(["config", "recipe", "outer_fold"], as_index=False).agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"), runs=("mae", "size"),
        median_epoch=("selected_epoch", "median"), boundary_rate=("boundary_hit", "mean"))
    agg["selection_score"] = agg.mean_mae + a.std_weight * agg.std_mae.fillna(0)
    per = {}
    for cfg in BACKBONES:
        per[cfg] = {}
        for fold in range(5):
            g = agg[(agg.config == cfg) & (agg.outer_fold == fold)].sort_values(["selection_score", "mean_mae", "recipe"])
            per[cfg][str(fold)] = [{"recipe": r.recipe, "selected_epoch": int(round(r.median_epoch)),
                                    "selection_score": float(r.selection_score), "boundary_rate": float(r.boundary_rate)}
                                   for _, r in g.head(a.top_n).iterrows()]
    payload = {"stages": stages, "top_n": a.top_n, "per_fold": per}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    agg.to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
