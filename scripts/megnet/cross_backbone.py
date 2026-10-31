#!/usr/bin/env python3
"""CGCNN-MEGNet agreement of the Matbench E/I readouts.

Feature level: Spearman/Kendall correlation and Top-5/Top-10 overlap of the
mean |net modulation| per attribute (dielectric and JDFT2D per fold; phonons
as the 5-fold MEGNet mean versus the CGCNN 15-run stability table of
scripts/cgcnn/feature_stability.py). With --cgcnn-detailed-root
(scripts/megnet/export_cgcnn_detailed.py) also per-structure node agreement
and fidelity ratios.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def safe_corr(fun, a, b):
    v = fun(a, b)[0]
    return float(v) if np.isfinite(v) else np.nan


def rank_metrics(names, mv, cv, task, fold):
    names, mv, cv = np.asarray(names), np.asarray(mv, float), np.asarray(cv, float)
    return {"task": task, "fold": fold, "spearman": safe_corr(spearmanr, mv, cv), "kendall": safe_corr(kendalltau, mv, cv),
            "top5_overlap": len(set(names[np.argsort(-mv)[:5]]) & set(names[np.argsort(-cv)[:5]])) / 5,
            "top10_overlap": len(set(names[np.argsort(-mv)[:10]]) & set(names[np.argsort(-cv)[:10]])) / 10}


def ratio(path):
    d = pd.read_csv(path)
    top = float(d.loc[d.intervention == "top", "mean_abs_prediction_change"].iloc[0])
    rnd = float(d[d.intervention.str.startswith("random_")].mean_abs_prediction_change.mean())
    return top / (rnd + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--megnet-output-root", required=True)
    p.add_argument("--cgcnn-transfer-root", required=True, help="output root of scripts/cgcnn/run_transfer.py")
    p.add_argument("--cgcnn-detailed-root", default="")
    p.add_argument("--cgcnn-phonons-stability", default="",
                   help="materon_best_consensus_feature_stability.csv from scripts/cgcnn/feature_stability.py")
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    mroot, croot = Path(a.megnet_output_root), Path(a.cgcnn_transfer_root)
    detailed = Path(a.cgcnn_detailed_root) if a.cgcnn_detailed_root else None
    feature_rows, node_rows, fidelity_rows = [], [], []
    for task in ["matbench_dielectric", "matbench_jdft2d"]:
        for fold in range(5):
            interp = mroot / task / "materon_explain_megnet" / f"outer_fold_{fold}" / "seed_42" / "interpretability"
            cs = croot / task / "fixed_materon_explain" / f"outer_fold_{fold}" / "seed_42" / "analysis_summary.json"
            if not (interp / "feature_scores.csv").exists() or not cs.exists():
                continue
            m = pd.read_csv(interp / "feature_scores.csv").set_index("feature")
            d = json.loads(cs.read_text())["input_gate"]
            names, cvals = d["feature_names"], np.asarray(d["actual_total_delta_mean_abs"])
            common = [x for x in names if x in m.index]
            row = rank_metrics(common, m.loc[common, "mean_abs_actual_delta"].to_numpy(),
                               np.asarray([cvals[names.index(x)] for x in common]), task, fold)
            if detailed and (detailed / task / f"outer_fold_{fold}" / "feature_scores.csv").exists():
                cdet = pd.read_csv(detailed / task / f"outer_fold_{fold}" / "feature_scores.csv").set_index("feature").loc[common]
                mm = m.loc[common]
                row.update({
                    "signed_direction_agreement": float(np.mean(np.sign(mm.mean_signed_actual_delta) == np.sign(cdet.mean_signed_actual_delta))),
                    "promotion_spearman": safe_corr(spearmanr, mm.mean_abs_promotion, cdet.mean_abs_promotion),
                    "inhibition_spearman": safe_corr(spearmanr, mm.mean_abs_inhibition, cdet.mean_abs_inhibition),
                    "ei_dominance_agreement": float(np.mean((mm.mean_abs_promotion > mm.mean_abs_inhibition)
                                                            == (cdet.mean_abs_promotion > cdet.mean_abs_inhibition)))})
            feature_rows.append(row)
            if not detailed:
                continue
            cn = detailed / task / f"outer_fold_{fold}" / "node_scores.csv.gz"
            mn = interp / "node_scores.csv.gz"
            if not (mn.exists() and cn.exists()):
                continue
            j = pd.read_csv(mn).merge(pd.read_csv(cn), on=["sample_id", "site_index", "atomic_number"],
                                      suffixes=("_megnet", "_cgcnn"))
            for sid, g in j.groupby("sample_id"):
                if len(g) < 2:
                    continue
                a1, a2 = g.actual_delta_norm_megnet.to_numpy(), g.actual_delta_norm_cgcnn.to_numpy()
                k = max(1, int(np.ceil(len(g) * 0.1)))
                node_rows.append({
                    "task": task, "fold": fold, "sample_id": sid, "n_nodes": len(g),
                    "spearman": safe_corr(spearmanr, a1, a2), "kendall": safe_corr(kendalltau, a1, a2),
                    "promotion_spearman": safe_corr(spearmanr, g.promotion_norm_megnet, g.promotion_norm_cgcnn),
                    "inhibition_spearman": safe_corr(spearmanr, g.inhibition_norm_megnet, g.inhibition_norm_cgcnn),
                    "top10pct_overlap": len(set(np.argsort(-a1)[:k]) & set(np.argsort(-a2)[:k])) / k,
                    "top_element_agreement": int(g.iloc[np.argmax(a1)].element_megnet == g.iloc[np.argmax(a2)].element_cgcnn),
                    "ei_dominance_agreement": float(np.mean((g.promotion_weight_megnet > g.inhibition_weight_megnet)
                                                            == (g.promotion_weight_cgcnn > g.inhibition_weight_cgcnn)))})
            files = [interp / "feature_fidelity.csv", cn.parent / "feature_fidelity.csv",
                     interp / "node_fidelity.csv", cn.parent / "node_fidelity.csv"]
            if all(f.exists() for f in files):
                fidelity_rows.append({"task": task, "fold": fold,
                                      "megnet_feature_fidelity_ratio": ratio(files[0]),
                                      "cgcnn_feature_fidelity_ratio": ratio(files[1]),
                                      "megnet_node_fidelity_ratio": ratio(files[2]),
                                      "cgcnn_node_fidelity_ratio": ratio(files[3])})
    phonon_files = [mroot / "matbench_phonons" / "materon_explain_megnet" / f"outer_fold_{f}" / "seed_42"
                    / "interpretability" / "feature_scores.csv" for f in range(5)]
    if a.cgcnn_phonons_stability and all(x.exists() for x in phonon_files):
        mm = pd.concat([pd.read_csv(x).set_index("feature")["mean_abs_actual_delta"] for x in phonon_files], axis=1).mean(axis=1)
        cc = pd.read_csv(a.cgcnn_phonons_stability).set_index("feature_name")["mean_abs_gate_delta"]
        common = [x for x in mm.index if x in cc.index]
        feature_rows.append(rank_metrics(common, mm.loc[common].to_numpy(), cc.loc[common].to_numpy(),
                                         "matbench_phonons", "aggregate_5fold_vs_cgcnn_15run"))
    pd.DataFrame(feature_rows).to_csv(out / "feature_consistency.csv", index=False)
    pd.DataFrame(node_rows).to_csv(out / "node_consistency.csv", index=False)
    pd.DataFrame(fidelity_rows).to_csv(out / "fidelity_consistency.csv", index=False)
    if feature_rows:
        fdf = pd.DataFrame(feature_rows)
        summary = [{"task": task, "feature_comparisons": len(g), "mean_spearman": float(g.spearman.mean()),
                    "mean_kendall": float(g.kendall.mean()), "mean_top5_overlap": float(g.top5_overlap.mean()),
                    "mean_top10_overlap": float(g.top10_overlap.mean())} for task, g in fdf.groupby("task")]
        pd.DataFrame(summary).to_csv(out / "feature_consistency_summary.csv", index=False)
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
