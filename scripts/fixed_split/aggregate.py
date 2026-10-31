#!/usr/bin/env python3
"""Tables of the fixed-split 2D study (Table S1 and Sec. S6/S9 statistics).

* performance_summary.csv, paired_test_bootstrap.csv (paired per-structure improvements, 3000 resamples)
* cross_backbone_{feature,node,intervention}_consistency.csv (CGCNN vs MEGNet readouts)
* interpretability_fidelity_summary.csv, chemical_role_interventions_all.csv, pathway_interventions_all.csv
* case_studies/<dataset>_selected_cases.csv (deterministic 12-case selection)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def safe_corr(func, a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(func(a, b)[0])


def bootstrap_mean_ci(values, repeats=2000, seed=42, alpha=0.05):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    draws = np.empty(int(repeats), dtype=float)
    for i in range(int(repeats)):
        draws[i] = float(np.mean(values[rng.randint(0, len(values), len(values))]))
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def load_predictions(path):
    """Test predictions with absolute errors recomputed in float64."""
    frame = pd.read_csv(path)
    frame["absolute_error"] = np.abs(frame["target"].to_numpy(np.float64) - frame["prediction"].to_numpy(np.float64))
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    a = p.parse_args()
    from materon.fixed_split.config import load_protocol

    protocol = load_protocol()
    out = Path(a.output_root)
    tables, cases = out / "tables", out / "case_studies"
    tables.mkdir(parents=True, exist_ok=True)
    cases.mkdir(parents=True, exist_ok=True)

    runs = pd.DataFrame([json.loads((out / "runs" / d / m / "seed_42" / "metrics.json").read_text(encoding="utf-8"))
                         for d in protocol["datasets"] for m in protocol["models"]])
    runs.to_csv(tables / "run_level_performance.csv", index=False)
    perf = runs[["dataset", "model", "backend", "role", "mae", "rmse", "r2", "best_epoch", "epochs_ran",
                 "parameter_count"]].sort_values(["dataset", "backend", "mae"])
    perf.to_csv(tables / "performance_summary.csv", index=False)

    paired_rows = []
    for dataset in protocol["datasets"]:
        for backbone in ["cgcnn", "megnet"]:
            base_pred = load_predictions(out / "runs" / dataset / f"{backbone}_baseline" / "seed_42" / "predictions.csv")
            for model_id in [f"materon_nogate_{backbone}", f"materon_pca_{backbone}", f"materon_explain_{backbone}"]:
                model_pred = load_predictions(out / "runs" / dataset / model_id / "seed_42" / "predictions.csv")
                merged = base_pred.merge(model_pred, on=["sample_id", "target"], suffixes=("_baseline", "_model"),
                                         validate="one_to_one")
                improvement = merged["absolute_error_baseline"].to_numpy(float) - merged["absolute_error_model"].to_numpy(float)
                lo, hi = bootstrap_mean_ci(improvement, repeats=3000, seed=42)
                base_mae = float(np.mean(merged["absolute_error_baseline"]))
                model_mae = float(np.mean(merged["absolute_error_model"]))
                paired_rows.append({"dataset": dataset, "backbone": backbone, "model": model_id, "n_test": len(merged),
                                    "baseline_mae": base_mae, "model_mae": model_mae,
                                    "absolute_improvement": base_mae - model_mae,
                                    "relative_improvement_percent": 100.0 * (base_mae - model_mae) / base_mae,
                                    "paired_sample_win_fraction": float(np.mean(improvement > 0)),
                                    "paired_mean_improvement_ci95_low": lo, "paired_mean_improvement_ci95_high": hi})
    pd.DataFrame(paired_rows).to_csv(tables / "paired_test_bootstrap.csv", index=False)

    cross_rows, node_rows, consistency, fidelity, roles_all, pathways_all = [], [], [], [], [], []
    for dataset in protocol["datasets"]:
        croot = out / "interpretability" / dataset / "cgcnn"
        mroot = out / "interpretability" / dataset / "megnet"
        cf = pd.read_csv(croot / "feature_scores.csv").sort_values("feature_index")
        mf = pd.read_csv(mroot / "feature_scores.csv").sort_values("feature_index")
        for metric in ["mean_abs_actual_delta", "mean_signed_actual_delta", "dimension_gate", "mean_abs_promotion",
                       "mean_abs_inhibition"]:
            order_c = list(cf.sort_values(metric, ascending=False)["feature_index"])
            order_m = list(mf.sort_values(metric, ascending=False)["feature_index"])
            row = {"dataset": dataset, "level": "feature", "metric": metric,
                   "spearman": safe_corr(spearmanr, cf[metric], mf[metric]),
                   "kendall": safe_corr(kendalltau, cf[metric], mf[metric])}
            for k in [5, 10]:
                row[f"top{k}_overlap"] = len(set(order_c[:k]) & set(order_m[:k])) / k
            cross_rows.append(row)

        merged = pd.read_csv(croot / "node_scores.csv").merge(
            pd.read_csv(mroot / "node_scores.csv"), on=["sample_id", "site_index", "atomic_number", "element", "role"],
            suffixes=("_cgcnn", "_megnet"), validate="one_to_one")
        for sid, group in merged.groupby("sample_id"):
            k = max(1, int(math.ceil(0.1 * len(group))))
            ctop = set(group.nlargest(k, "node_score_cgcnn")["site_index"])
            mtop = set(group.nlargest(k, "node_score_megnet")["site_index"])
            node_rows.append({"dataset": dataset, "sample_id": sid, "n_nodes": len(group),
                              "spearman": safe_corr(spearmanr, group["node_score_cgcnn"], group["node_score_megnet"]),
                              "top10percent_overlap": len(ctop & mtop) / k})

        for backbone, root in [("cgcnn", croot), ("megnet", mroot)]:
            for name, target in [("role_interventions.csv", roles_all), ("pathway_interventions.csv", pathways_all)]:
                frame = pd.read_csv(root / name)
                frame.insert(0, "backbone", backbone)
                frame.insert(0, "dataset", dataset)
                target.append(frame)
            for level, file, key in [("feature", "feature_interventions.csv", "gate_modulation_mask"),
                                     ("node", "node_interventions.csv", "node_gate_mask")]:
                subset = pd.read_csv(root / file)
                subset = subset[subset["intervention_type"] == key]
                top = subset[subset["intervention"] == "top"].iloc[0]
                bottom = subset[subset["intervention"] == "bottom"].iloc[0]
                random = subset[subset["intervention"].astype(str).str.startswith("random_")]
                rnd = float(random["mean_abs_prediction_change"].mean())
                fidelity.append({"dataset": dataset, "backbone": backbone, "level": level,
                                 "top_delta_mae": float(top["delta_mae"]), "bottom_delta_mae": float(bottom["delta_mae"]),
                                 "random_mean_delta_mae": float(random["delta_mae"].mean()),
                                 "top_prediction_change": float(top["mean_abs_prediction_change"]),
                                 "bottom_prediction_change": float(bottom["mean_abs_prediction_change"]),
                                 "random_mean_prediction_change": rnd,
                                 "top_to_random_change_ratio": float(top["mean_abs_prediction_change"]) / max(rnd, 1e-12),
                                 "top_to_bottom_change_ratio": float(top["mean_abs_prediction_change"])
                                 / max(float(bottom["mean_abs_prediction_change"]), 1e-12)})
        for filename in ["feature_interventions.csv", "node_interventions.csv", "role_interventions.csv",
                         "pathway_interventions.csv"]:
            mm = pd.read_csv(croot / filename).merge(pd.read_csv(mroot / filename), on=["intervention_type", "intervention"],
                                                     suffixes=("_cgcnn", "_megnet"), validate="one_to_one")
            consistency.append({"dataset": dataset, "file": filename, "n": len(mm),
                                "delta_mae_spearman": safe_corr(spearmanr, mm["delta_mae_cgcnn"], mm["delta_mae_megnet"]),
                                "prediction_change_spearman": safe_corr(spearmanr, mm["mean_abs_prediction_change_cgcnn"],
                                                                        mm["mean_abs_prediction_change_megnet"]),
                                "signed_prediction_change_spearman": safe_corr(spearmanr,
                                                                               mm["signed_mean_prediction_change_cgcnn"],
                                                                               mm["signed_mean_prediction_change_megnet"])})
    nodec = pd.DataFrame(node_rows)
    pd.DataFrame(cross_rows).to_csv(tables / "cross_backbone_feature_consistency.csv", index=False)
    nodec.to_csv(tables / "cross_backbone_node_consistency.csv", index=False)
    pd.DataFrame(consistency).to_csv(tables / "cross_backbone_intervention_consistency.csv", index=False)
    pd.DataFrame(fidelity).to_csv(tables / "interpretability_fidelity_summary.csv", index=False)
    pd.concat(roles_all, ignore_index=True).to_csv(tables / "chemical_role_interventions_all.csv", index=False)
    pd.concat(pathways_all, ignore_index=True).to_csv(tables / "pathway_interventions_all.csv", index=False)

    # Deterministic selection of 12 illustrative test structures per dataset.
    for dataset in protocol["datasets"]:
        cp = load_predictions(out / "runs" / dataset / "materon_explain_cgcnn" / "seed_42" / "predictions.csv").rename(
            columns={"prediction": "prediction_cgcnn", "absolute_error": "absolute_error_cgcnn"})
        mp = load_predictions(out / "runs" / dataset / "materon_explain_megnet" / "seed_42" / "predictions.csv").rename(
            columns={"prediction": "prediction_megnet", "absolute_error": "absolute_error_megnet"})
        frame = cp.merge(mp[["sample_id", "prediction_megnet", "absolute_error_megnet"]], on="sample_id", validate="one_to_one")
        frame["mean_absolute_error"] = (frame["absolute_error_cgcnn"] + frame["absolute_error_megnet"]) / 2
        nc = nodec[nodec["dataset"] == dataset][["sample_id", "spearman", "top10percent_overlap"]].rename(
            columns={"spearman": "node_explanation_spearman"})
        frame = frame.merge(nc, on="sample_id", how="left", validate="one_to_one")
        cg_med, mg_med = float(frame["absolute_error_cgcnn"].median()), float(frame["absolute_error_megnet"].median())
        frame["error_category"] = np.select(
            [(frame["absolute_error_cgcnn"] <= cg_med) & (frame["absolute_error_megnet"] <= mg_med),
             (frame["absolute_error_cgcnn"] > cg_med) & (frame["absolute_error_megnet"] > mg_med),
             (frame["absolute_error_cgcnn"] <= cg_med) & (frame["absolute_error_megnet"] > mg_med)],
            ["both_accurate", "both_difficult", "cgcnn_better"], default="megnet_better")
        frame.to_csv(cases / f"{dataset}_test_error_and_consistency.csv", index=False)
        chosen = []

        def add(items, reason, count):
            existing, added = {x[0] for x in chosen}, 0
            for sid in items:
                if sid not in existing:
                    chosen.append((sid, reason))
                    existing.add(sid)
                    added += 1
                if added >= count:
                    break

        for label, part in zip(["target_low", "target_mid", "target_high"], np.array_split(frame.sort_values("target"), 3)):
            add(part.sort_values("mean_absolute_error")["sample_id"], label, 2)
        add(frame.sort_values("mean_absolute_error")["sample_id"], "lowest_error", 2)
        add(frame.sort_values("mean_absolute_error", ascending=False)["sample_id"], "highest_error", 2)
        add(frame.sort_values("node_explanation_spearman", ascending=False)["sample_id"], "most_cross_backbone_consistent", 1)
        add(frame.sort_values("node_explanation_spearman")["sample_id"], "most_cross_backbone_divergent", 1)
        if len(chosen) < 12:
            add(frame.sort_values("sample_id")["sample_id"], "deterministic_fill", 12 - len(chosen))
        selected = pd.DataFrame(chosen[:12], columns=["sample_id", "selection_reason"]).merge(frame, on="sample_id",
                                                                                               validate="one_to_one")
        selected.insert(0, "dataset", dataset)
        selected.to_csv(cases / f"{dataset}_selected_cases.csv", index=False)
    print(perf.to_string(index=False))


if __name__ == "__main__":
    main()
