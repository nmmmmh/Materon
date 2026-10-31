#!/usr/bin/env python3
"""CGCNN phonons result tables (Table I, Table S7 and ablations) from the aggregated runs.

Inputs are the CSVs of aggregate_results.py (inner and final phases) and the
formal settings of select_formal_settings.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from materon.cgcnn.config import recipes


def _safe_std(series):
    value = series.std()
    return float(0.0 if pd.isna(value) else value)


def _read_oof_predictions(final):
    frames = []
    for row in final.itertuples(index=False):
        pred_path = Path(str(getattr(row, "metrics_path", ""))).parent / "predictions.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        columns = [("config", row.config), ("resolved_config", getattr(row, "resolved_config", row.config)),
                   ("recipe", row.recipe), ("outer_fold", int(row.outer_fold)), ("seed", int(row.seed)),
                   ("selected_epoch", int(row.selected_epoch))]
        for i, (key, value) in enumerate(columns):
            pred.insert(i, key, value)
        frames.append(pred)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _factorial_tables(final, tables):
    """2x2x2 factorial over shared expert (s), low-rank gate (l) and dimension gate (d)."""
    factor = final[final.config.str.match(r"^factorial_s[01]_l[01]_d[01]$")].copy()
    # legacy_input is the S0-L1-D0 cell.
    if "factorial_s0_l1_d0" not in set(factor.config) and "legacy_input" in set(final.config):
        alias = final[final.config == "legacy_input"].copy()
        alias["config"] = "factorial_s0_l1_d0"
        factor = pd.concat([factor, alias], ignore_index=True)
    if factor.empty:
        return
    factor[["shared", "low_rank", "dimension"]] = factor.config.str.extract(
        r"factorial_s([01])_l([01])_d([01])").astype(int)
    cells = factor.groupby(["shared", "low_rank", "dimension"], as_index=False).agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"), runs=("mae", "size"))
    cells.to_csv(tables / "factorial_cells.csv", index=False)
    effects = []
    for name in ["shared", "low_rank", "dimension"]:
        on = float(cells.loc[cells[name] == 1, "mean_mae"].mean())
        off = float(cells.loc[cells[name] == 0, "mean_mae"].mean())
        effects.append({"effect": name, "mae_on": on, "mae_off": off, "delta_on_minus_off": on - off})
    pd.DataFrame(effects).to_csv(tables / "factorial_effects.csv", index=False)
    interactions = []
    for x, y in [("shared", "low_rank"), ("shared", "dimension"), ("low_rank", "dimension")]:
        means = cells.groupby([x, y]).mean_mae.mean()
        value = float(means.loc[(1, 1)] - means.loc[(1, 0)] - means.loc[(0, 1)] + means.loc[(0, 0)])
        interactions.append({"interaction": f"{x}:{y}", "difference_of_differences": value})
    three_way = 0.0
    for _, r in cells.iterrows():
        three_way += (-1.0 if int(r.shared + r.low_rank + r.dimension) % 2 == 0 else 1.0) * float(r.mean_mae)
    interactions.append({"interaction": "shared:low_rank:dimension", "difference_of_differences": three_way})
    interactions.append({"interaction": "grand_mean_reference", "difference_of_differences": float(cells.mean_mae.mean())})
    pd.DataFrame(interactions).to_csv(tables / "factorial_interactions.csv", index=False)


SUBSETS = {
    "architecture_ablation": ["physics_full20", "simple_input", "legacy_simple_input", "legacy_input", "refined_input",
                              "materon_best_consensus", "materon_best_nested"],
    "context_ablation": ["legacy_input", "context_coord", "context_qeff", "context_emag", "context_all"],
    "feature_truth_controls": ["physics_full20", "physics_grouped_pca", "physics_shuffled_full20",
                               "materon_best_consensus", "materon_best_grouped_pca", "materon_best_shuffled_full20"],
    "gate_capacity_results": ["materon_best_legacy_capacity", "materon_best_refined_capacity", "legacy_input",
                              "refined_input", "materon_best_consensus"],
    "parameter_count_controls": ["physics_full20", "physics_parameter_matched", "materon_best_consensus"],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inner-csv", required=True)
    p.add_argument("--final-csv", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    tables = Path(a.output_dir)
    tables.mkdir(parents=True, exist_ok=True)
    inner = pd.read_csv(a.inner_csv)
    final = pd.read_csv(a.final_csv)
    settings = json.loads(Path(a.settings).read_text(encoding="utf-8"))

    summary = final.groupby("config", as_index=False).agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"), mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
        runs=("mae", "size"), mean_parameters=("parameter_count", "mean"), median_epoch=("selected_epoch", "median"),
    ).sort_values(["mean_mae", "std_mae", "config"])
    summary.to_csv(tables / "main_results.csv", index=False)
    final.groupby(["config", "seed"], as_index=False).agg(
        five_fold_mean_mae=("mae", "mean"), five_fold_mean_rmse=("rmse", "mean"),
        median_epoch=("selected_epoch", "median")).to_csv(tables / "seed_level_five_fold_results.csv", index=False)
    final.groupby(["config", "outer_fold"], as_index=False).agg(
        three_seed_mean_mae=("mae", "mean"), three_seed_std_mae=("mae", "std"),
        selected_epoch=("selected_epoch", "median")).to_csv(tables / "fold_level_results.csv", index=False)
    columns = [c for c in ["stage", "config", "resolved_config", "recipe", "outer_fold", "repeat", "seed",
                           "selected_epoch", "boundary_hit", "max_epochs", "mae", "rmse", "parameter_count"]
               if c in inner.columns]
    inner[columns].to_csv(tables / "hyperparameter_search.csv", index=False)

    rows = []
    for fold, items in settings["per_fold"].items():
        for label, item in items.items():
            recipe = recipes()[item["recipe"]]
            rows.append({"outer_fold": int(fold), "model": label, "resolved_config": item.get("resolved_config", label),
                         "recipe": item["recipe"], "selected_epoch": item["selected_epoch"],
                         **{k: recipe[k] for k in ["hidden_dim", "depth", "batch_size", "learning_rate", "schedule",
                                                   "dropout"]}})
    pd.DataFrame(rows).to_csv(tables / "fold_selected_hyperparameters.csv", index=False)

    _factorial_tables(final, tables)
    for name, ids in SUBSETS.items():
        summary[summary.config.isin(ids)].to_csv(tables / f"{name}.csv", index=False)

    pair_rows = []
    for target in ["materon_best_consensus", "materon_best_nested"]:
        for base in ["cgcnn_baseline", "physics_full20", "physics_parameter_matched"]:
            paired = final[final.config == target][["outer_fold", "seed", "mae"]].merge(
                final[final.config == base][["outer_fold", "seed", "mae"]], on=["outer_fold", "seed"],
                suffixes=("_target", "_base"))
            if paired.empty:
                continue
            delta = paired.mae_base - paired.mae_target
            pair_rows.append({"target": target, "baseline": base, "pairs": len(paired),
                              "mean_improvement": float(delta.mean()), "std_improvement": _safe_std(delta),
                              "relative_improvement_percent": float(100.0 * delta.mean() / paired.mae_base.mean()),
                              "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                              "losses": int((delta < 0).sum())})
    pd.DataFrame(pair_rows).to_csv(tables / "paired_improvements_summary.csv", index=False)

    oof = _read_oof_predictions(final)
    if not oof.empty:
        oof.to_csv(tables / "all_oof_predictions.csv", index=False)
        ensemble = oof.groupby(["config", "outer_fold", "index"], as_index=False).agg(
            target=("target", "first"), prediction=("prediction", "mean"), seed_count=("seed", "nunique"))
        ensemble["absolute_error"] = (ensemble.target - ensemble.prediction).abs()
        ensemble.groupby("config", as_index=False).agg(
            ensemble_oof_mae=("absolute_error", "mean"), samples=("index", "size")
        ).sort_values("ensemble_oof_mae").to_csv(tables / "ensemble_oof_results.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
