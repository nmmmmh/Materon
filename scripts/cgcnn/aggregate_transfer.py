#!/usr/bin/env python3
"""Summaries of the CGCNN transfer runs: per task/model MAE, paired comparison with CGCNN, selected settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from materon.cgcnn.config import transfer_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    protocol = transfer_protocol()
    rows = []
    for path in Path(a.root).glob("**/metrics.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("protocol_id") != protocol["protocol_id"]:
            continue
        rows.append({"task": record["task"], "config": record["config"], "paper_name": record["paper_name"],
                     "fold": int(record["outer_fold"]), "seed": int(record["seed"]),
                     "mae": float(record["metrics"]["mae"]), "rmse": float(record["metrics"]["rmse"]),
                     "parameters": int(record["parameter_count"]), "base_recipe": record["base_recipe"],
                     "selected_learning_rate": float(record["selected_learning_rate"]),
                     "selected_epoch": int(record["selected_epoch"])})
    if not rows:
        raise RuntimeError("No completed transfer runs found")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["task", "config", "fold", "seed"])
    frame.to_csv(out / "run_level_results.csv", index=False)
    max_epoch = max(protocol["optimization_calibration"]["checkpoint_epochs"])
    summary = frame.groupby(["task", "config", "paper_name"], as_index=False).agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"), mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
        mean_parameters=("parameters", "mean"), median_selected_lr=("selected_learning_rate", "median"),
        median_selected_epoch=("selected_epoch", "median"),
        boundary_hits=("selected_epoch", lambda v: int((v == max_epoch).sum())), runs=("mae", "size"))
    summary["task_mad"] = summary["task"].map({n: m["mad"] for n, m in protocol["tasks"].items()})
    summary["normalized_mae"] = summary["mean_mae"] / summary["task_mad"]
    summary.to_csv(out / "task_model_summary.csv", index=False)

    baseline = frame[frame["config"] == "fixed_cgcnn_baseline"][["task", "fold", "seed", "mae"]].rename(
        columns={"mae": "baseline_mae"})
    paired = frame.merge(baseline, on=["task", "fold", "seed"], how="left", validate="many_to_one")
    paired["delta_mae"] = paired["mae"] - paired["baseline_mae"]
    paired["relative_improvement"] = (paired["baseline_mae"] - paired["mae"]) / paired["baseline_mae"]
    paired.to_csv(out / "paired_vs_baseline.csv", index=False)
    paired.groupby(["task", "config", "paper_name"], as_index=False).agg(
        mean_delta_mae=("delta_mae", "mean"), std_delta_mae=("delta_mae", "std"),
        mean_relative_improvement=("relative_improvement", "mean"),
        wins=("delta_mae", lambda v: int((v < 0).sum())), ties=("delta_mae", lambda v: int((v == 0).sum())),
        losses=("delta_mae", lambda v: int((v > 0).sum())), pairs=("delta_mae", "size"),
    ).to_csv(out / "paired_vs_baseline_summary.csv", index=False)
    frame.groupby(["task", "config", "paper_name", "selected_learning_rate", "selected_epoch"], as_index=False).size(
    ).rename(columns={"size": "fold_count"}).to_csv(out / "selected_optimization_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
