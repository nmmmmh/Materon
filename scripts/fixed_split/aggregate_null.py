#!/usr/bin/env python3
"""Compare trained, untrained and input-magnitude attribute rankings (Table S15).

Inputs: <raw-dir>/<dataset>__<backbone>__{untrained,trained_check}.json from
null_ranking.py and the trained feature_scores.csv of explain_job.py.
Conventions: scipy Spearman on the 20 scores, Top-K overlap = |TopK(A) & TopK(B)|
(descending score, stable order), SD over seed pairs with ddof = 1.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

DATASETS = ["hoip_bandgap", "jdft2d_exfoliation"]
BACKBONES = ["cgcnn", "megnet"]


def topk(v, k):
    return set(np.argsort(-np.asarray(v, float), kind="stable")[:k].tolist())


def compare(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (20,) or b.shape != (20,):
        raise ValueError("expected two 20-vectors")
    if np.std(a) == 0 or np.std(b) == 0:
        rho = tau = float("nan")
    else:
        rho, tau = float(spearmanr(a, b).correlation), float(kendalltau(a, b).correlation)
    t10, t5 = len(topk(a, 10) & topk(b, 10)), len(topk(a, 5) & topk(b, 5))
    return {"spearman": rho, "kendall": tau, "top10_overlap_count": t10, "top10_overlap_fraction": t10 / 10,
            "top5_overlap_count": t5, "top5_overlap_fraction": t5 / 5}


def summarize(rows, labels):
    out = {"n_pairs": len(rows), "pairs": [dict(pair=p, **r) for p, r in zip(labels, rows)]}
    for key in ["spearman", "kendall", "top10_overlap_count", "top5_overlap_count"]:
        v = np.asarray([r[key] for r in rows], float)
        out[f"{key}_mean"] = float(np.mean(v))
        out[f"{key}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out[f"{key}_min"], out[f"{key}_max"] = float(np.min(v)), float(np.max(v))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--interpretability-root", required=True, help="<output-root>/interpretability of explain_job.py")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    raw, interp = Path(a.raw_dir), Path(a.interpretability_root)
    payload = {"datasets": {}}
    for ds in DATASETS:
        trained, tchk, untr = {}, {}, {}
        for bb in BACKBONES:
            trained[bb] = pd.read_csv(interp / ds / bb / "feature_scores.csv").sort_values("feature_index")
            tchk[bb] = json.loads((raw / f"{ds}__{bb}__trained_check.json").read_text())
            untr[bb] = json.loads((raw / f"{ds}__{bb}__untrained.json").read_text())
        seeds = [r["seed"] for r in untr["cgcnn"]["runs"]]
        T = {bb: trained[bb]["mean_abs_actual_delta"].to_numpy(float) for bb in BACKBONES}
        X = {bb: np.asarray(untr[bb]["input_mean_abs_x"], float) for bb in BACKBONES}
        if not np.allclose(X["cgcnn"], X["megnet"], rtol=0, atol=1e-6):
            raise RuntimeError(f"input-only mismatch between CGCNN and MEGNet caches for {ds}")
        Xin = X["cgcnn"]
        U = {bb: [np.asarray(r["mean_abs_actual_delta"], float) for r in untr[bb]["runs"]] for bb in BACKBONES}
        n = len(seeds)
        c = {"trained_cgcnn_vs_trained_megnet": compare(T["cgcnn"], T["megnet"]),
             "input_only_vs_trained_cgcnn": compare(Xin, T["cgcnn"]),
             "input_only_vs_trained_megnet": compare(Xin, T["megnet"]),
             "untrained_cgcnn_vs_untrained_megnet_matched_seed": summarize(
                 [compare(U["cgcnn"][i], U["megnet"][i]) for i in range(n)],
                 [(f"cgcnn_s{s}", f"megnet_s{s}") for s in seeds])}
        cross = list(itertools.product(range(n), range(n)))
        c["untrained_cgcnn_vs_untrained_megnet_all_pairs"] = summarize(
            [compare(U["cgcnn"][i], U["megnet"][j]) for i, j in cross],
            [(f"cgcnn_s{seeds[i]}", f"megnet_s{seeds[j]}") for i, j in cross])
        for bb in BACKBONES:
            c[f"untrained_{bb}_vs_trained_{bb}"] = summarize([compare(U[bb][i], T[bb]) for i in range(n)],
                                                             [(f"{bb}_s{s}", "trained") for s in seeds])
        c["untrained_cgcnn_vs_trained_megnet"] = summarize([compare(U["cgcnn"][i], T["megnet"]) for i in range(n)],
                                                           [(f"cgcnn_s{s}", "trained_megnet") for s in seeds])
        c["untrained_megnet_vs_trained_cgcnn"] = summarize([compare(U["megnet"][i], T["cgcnn"]) for i in range(n)],
                                                           [(f"megnet_s{s}", "trained_cgcnn") for s in seeds])
        for bb in BACKBONES:
            c[f"untrained_{bb}_vs_input_only"] = summarize([compare(U[bb][i], Xin) for i in range(n)],
                                                           [(f"{bb}_s{s}", "input_only") for s in seeds])
            pairs = list(itertools.combinations(range(n), 2))
            c[f"untrained_{bb}_seed_vs_seed"] = summarize([compare(U[bb][i], U[bb][j]) for i, j in pairs],
                                                          [(f"s{seeds[i]}", f"s{seeds[j]}") for i, j in pairs])
        # Decomposition of the trained readout into its dimension-gate and E/I terms.
        Dterm = {bb: np.asarray(tchk[bb]["runs"][0]["mean_abs_dimension_term"], float) for bb in BACKBONES}
        Eterm = {bb: np.asarray(tchk[bb]["runs"][0]["mean_abs_ei_term"], float) for bb in BACKBONES}
        dims = {bb: trained[bb]["dimension_gate"].to_numpy(float) for bb in BACKBONES}
        decomposition = {
            "dimension_term_cgcnn_vs_megnet": compare(Dterm["cgcnn"], Dterm["megnet"]),
            "ei_term_cgcnn_vs_megnet": compare(Eterm["cgcnn"], Eterm["megnet"]),
            "dimension_gate_cgcnn_vs_megnet": compare(dims["cgcnn"], dims["megnet"]),
            **{f"dimension_term_{bb}_vs_trained_{bb}": compare(Dterm[bb], T[bb]) for bb in BACKBONES},
            **{f"ei_term_{bb}_vs_trained_{bb}": compare(Eterm[bb], T[bb]) for bb in BACKBONES},
            **{f"ei_term_{bb}_vs_input_only": compare(Eterm[bb], Xin) for bb in BACKBONES},
            "share_of_sum_mean_abs_from_dimension_term": {
                bb: float(Dterm[bb].sum() / (Dterm[bb].sum() + Eterm[bb].sum())) for bb in BACKBONES},
            "trained_check_max_abs_diff_vs_feature_scores": {
                bb: float(np.max(np.abs(np.asarray(tchk[bb]["runs"][0]["mean_abs_actual_delta"]) - T[bb])))
                for bb in BACKBONES},
        }
        payload["datasets"][ds] = {"comparisons": c, "trained_decomposition": decomposition}
        for key, value in c.items():
            if "spearman_mean" in value:
                print(f"{ds} {key:55s} rho={value['spearman_mean']:.3f}+-{value['spearman_sd']:.3f}")
            else:
                print(f"{ds} {key:55s} rho={value['spearman']:.3f}")
    Path(a.output).write_text(json.dumps(payload, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
