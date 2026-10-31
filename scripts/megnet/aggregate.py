#!/usr/bin/env python3
"""MEGNet Matbench summaries: per task/model MAE, paired fold wins versus MEGNet, cross-fold ranking stability."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from materon.megnet.config import load_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, help="output root of train_job.py")
    p.add_argument("--reports", required=True)
    a = p.parse_args()
    q = load_protocol()
    root, reports = Path(a.outdir), Path(a.reports)
    reports.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in q["task_order"]:
        for model in q["models"]:
            for fold in q["outer_folds"]:
                path = root / task / model / f"outer_fold_{fold}" / "seed_42" / "metrics.json"
                if path.exists():
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    df.to_csv(reports / "run_level_results.csv", index=False)
    if len(df):
        df.groupby(["task", "model", "paper_name"]).agg(
            mean_mae=("mae", "mean"), fold_sd_mae=("mae", "std"), min_mae=("mae", "min"), max_mae=("mae", "max"),
            mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"), mean_parameters=("parameter_count", "mean"),
        ).reset_index().to_csv(reports / "task_model_summary.csv", index=False)
        pair = []
        for task in q["task_order"]:
            base = df[(df.task == task) & (df.model == "megnet_baseline")].set_index("fold")
            for model in ["materon_pca_megnet", "materon_explain_megnet"]:
                cur = df[(df.task == task) & (df.model == model)].set_index("fold")
                for fold in q["outer_folds"]:
                    if fold in base.index and fold in cur.index:
                        b, c = float(base.loc[fold, "mae"]), float(cur.loc[fold, "mae"])
                        pair.append({"task": task, "model": model, "fold": fold, "baseline_mae": b, "mae": c,
                                     "delta_mae": c - b, "relative_improvement": (b - c) / b, "win_vs_baseline": bool(c < b)})
        pair_df = pd.DataFrame(pair)
        pair_df.to_csv(reports / "paired_vs_baseline.csv", index=False)
        if len(pair_df):
            pair_df.groupby(["task", "model"]).agg(
                mean_baseline_mae=("baseline_mae", "mean"), mean_mae=("mae", "mean"), mean_delta_mae=("delta_mae", "mean"),
                mean_relative_improvement=("relative_improvement", "mean"), fold_wins=("win_vs_baseline", "sum"),
                folds=("fold", "count"),
            ).reset_index().to_csv(reports / "paired_vs_baseline_summary.csv", index=False)

    stability = []
    for task in q["task_order"]:
        scores = []
        for fold in q["outer_folds"]:
            path = root / task / "materon_explain_megnet" / f"outer_fold_{fold}" / "seed_42" / "interpretability" / "feature_scores.csv"
            if path.exists():
                scores.append(pd.read_csv(path).set_index("feature")["mean_abs_actual_delta"])
        if len(scores) == 5:
            m = pd.concat(scores, axis=1)
            corr, o5, o10 = [], [], []
            for i, j in itertools.combinations(range(5), 2):
                corr.append(float(spearmanr(m.iloc[:, i], m.iloc[:, j])[0]))
                o5.append(len(set(m.iloc[:, i].nlargest(5).index) & set(m.iloc[:, j].nlargest(5).index)) / 5)
                o10.append(len(set(m.iloc[:, i].nlargest(10).index) & set(m.iloc[:, j].nlargest(10).index)) / 10)
            stability.append({"task": task, "mean_pairwise_spearman": float(np.mean(corr)),
                              "min_pairwise_spearman": float(np.min(corr)), "mean_top5_overlap": float(np.mean(o5)),
                              "mean_top10_overlap": float(np.mean(o10))})
    pd.DataFrame(stability).to_csv(reports / "interpretability_stability.csv", index=False)
    print(df.groupby(["task", "model"])["mae"].mean().to_string() if len(df) else "no runs found")


if __name__ == "__main__":
    main()
