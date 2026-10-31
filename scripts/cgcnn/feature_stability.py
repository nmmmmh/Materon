#!/usr/bin/env python3
"""Attribute-modulation stability of the formal Materon-CGCNN phonons models.

For materon_best_consensus and materon_best_nested: run run_faithfulness.py on
every seed-42 fold model and summarize the mean |gate correction| per
attribute over all fold/seed models (<config>_feature_stability.csv). The
consensus table is the CGCNN phonons reference of scripts/megnet/cross_backbone.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CONFIGS = ["materon_best_consensus", "materon_best_nested"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="output root of run_trial.py")
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--random-repeats", type=int, default=30)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    faithfulness_script = Path(__file__).with_name("run_faithfulness.py")
    report = {}
    for cfg in CONFIGS:
        faith = []
        for path in sorted((root / "official_final" / cfg).glob("recipe_*/outer_fold_*/seed_42/final/metrics.json")):
            fold = json.loads(path.read_text())["outer_fold"]
            target = out / f"{cfg}_fold{fold}_faithfulness.json"
            subprocess.run([sys.executable, str(faithfulness_script), "--metrics", str(path),
                            "--cache-root", a.cache_root, "--random-repeats", str(a.random_repeats),
                            "--output", str(target)], check=True)
            faith.append(json.loads(target.read_text()))
        summaries = []
        for ap in sorted((root / "official_final" / cfg).glob("recipe_*/outer_fold_*/seed_*/final/analysis_summary.json")):
            gate = json.loads(ap.read_text()).get("input_gate", {})
            scores = gate.get("actual_total_delta_mean_abs") or gate.get("relative_actual_modulation_mean_abs")
            if scores:
                summaries.append({"scores": np.asarray(scores, float), "names": gate.get("feature_names", [])})
        stability = {"runs": len(summaries)}
        if summaries:
            length = min(len(x["scores"]) for x in summaries)
            matrix = np.stack([x["scores"][:length] for x in summaries])
            names = summaries[0]["names"][:length]
            mean, std = matrix.mean(0), matrix.std(0)
            rows = [{"feature_index": int(i), "feature_name": names[i] if i < len(names) else str(i),
                     "mean_abs_gate_delta": float(mean[i]), "std_abs_gate_delta": float(std[i]),
                     "mean_rank": float(np.mean([np.where(np.argsort(-r) == i)[0][0] + 1 for r in matrix]))}
                    for i in np.argsort(-mean)]
            pd.DataFrame(rows).to_csv(out / f"{cfg}_feature_stability.csv", index=False)
            k = min(5, length)
            pairs = [(i, j) for i in range(len(matrix)) for j in range(i + 1, len(matrix))]
            correlations = [float(spearmanr(matrix[i], matrix[j]).correlation) for i, j in pairs]
            overlaps = [len(set(np.argsort(-matrix[i])[:k]) & set(np.argsort(-matrix[j])[:k])) / float(k)
                        for i, j in pairs]
            stability.update({"feature_count": length,
                              "mean_pairwise_spearman": float(np.nanmean(correlations)) if correlations else None,
                              "mean_top5_overlap": float(np.mean(overlaps)) if overlaps else None,
                              "top_features": rows[:10]})
        report[cfg] = {"faithfulness": faith, "stability": stability}
    (out / "interpretability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
