#!/usr/bin/env python3
"""Collect the metrics.json files of one phase (inner or final) into a CSV and a per-setting summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = ["phase", "stage", "config", "resolved_config", "match_to", "source_config", "recipe", "outer_fold", "repeat",
        "seed", "selected_epoch", "best_epoch", "boundary_hit", "epochs_ran", "train_seconds", "parameter_count",
        "target_parameter_count", "parameter_match_units", "test_evaluations", "max_epochs", "fit_seed", "split_seed",
        "parameter_match_relative_gap", "validation_used", "early_stopping_used"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--phase", choices=["inner", "final"], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = []
    for path in Path(a.root).rglob("metrics.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("phase") != a.phase or data.get("status") != "complete":
            continue
        m = data.get("metrics", {})
        row = {k: data.get(k) for k in KEYS}
        row.update({"mae": m.get("mae"), "rmse": m.get("rmse"), "metrics_path": str(path)})
        rows.append(row)
    if not rows:
        raise RuntimeError("No completed %s metrics found" % a.phase)
    frame = pd.DataFrame(rows).sort_values(["stage", "config", "recipe", "outer_fold", "seed", "repeat"])
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summary = frame.groupby(["stage", "config", "recipe", "outer_fold"], dropna=False, as_index=False).agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"), mean_rmse=("rmse", "mean"), runs=("mae", "size"),
        median_epoch=("selected_epoch", "median"), mean_parameters=("parameter_count", "mean")
    ).sort_values(["stage", "mean_mae", "std_mae"])
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
