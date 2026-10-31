#!/usr/bin/env python3
"""Top-k versus random attribute masking of a trained Materon-CGCNN phonons model.

Attributes are ranked by mean |actual gate correction| on the outer test fold;
the k top-ranked corrections are disabled (feature mask 0) and the mean
absolute prediction change is compared with k random attributes
(RandomState(20260726), --random-repeats draws).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, help="metrics.json of a final Materon run")
    p.add_argument("--cache-root", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--random-repeats", type=int, default=30)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    import tensorflow as tf
    from matbench.bench import MatbenchBenchmark
    from materon.cgcnn.cache import cache_class_for_config
    from materon.cgcnn.config import get_config, get_recipe
    from materon.cgcnn.models import build_paired_model
    from materon.cgcnn.training import official_positions, set_global_seed

    meta = json.loads(Path(a.metrics).read_text())
    cfg = get_config(meta.get("resolved_config") or meta["config"])
    recipe = get_recipe(meta["recipe"])
    if cfg["model_type"] != "materon" or cfg.get("input_gate") is None:
        raise ValueError("Faithfulness requires a Materon input gate")
    task = next(iter(MatbenchBenchmark(subset=["matbench_phonons"], autoload=False).tasks))
    task.load()
    cache = cache_class_for_config(cfg)(task, a.cache_root, cfg["feature_mode"], False)
    _, _, test_pos, _, _, _ = official_positions(task, cache, int(meta["outer_fold"]))
    x, y = cache.subset_positions(test_pos)
    set_global_seed(int(meta["seed"]))
    model, _, _ = build_paired_model(cfg, recipe, [v[:min(4, len(y))] for v in x], int(meta["seed"]))
    model.load_weights(str(Path(a.metrics).parent / "model.weights.h5"))
    baseline = model.predict(x, verbose=0).reshape(-1)
    _, analysis = model(x, training=False, return_analysis=True)
    delta = analysis["input_gate"]["actual_total_delta"]
    values = delta.flat_values.numpy() if isinstance(delta, tf.RaggedTensor) else delta.numpy()
    score = np.mean(np.abs(values), axis=0)
    k = min(a.k, len(score))
    top = np.argsort(-score)[:k]
    gate = model.input_gate

    def masked_change(indices):
        mask = np.ones(len(score), np.float32)
        mask[indices] = 0
        gate.set_feature_mask(mask)
        change = float(np.mean(np.abs(model.predict(x, verbose=0).reshape(-1) - baseline)))
        gate.reset_feature_mask()
        return change

    top_change = masked_change(top)
    rng = np.random.RandomState(20260726)
    random_changes = [masked_change(rng.choice(len(score), k, replace=False)) for _ in range(a.random_repeats)]
    names = cache.metadata.get("feature_names", [])
    payload = {"config": cfg["id"], "outer_fold": meta["outer_fold"], "seed": meta["seed"], "k": k,
               "top_feature_indices": top.tolist(),
               "top_feature_names": [names[i] if i < len(names) else str(i) for i in top],
               "top_k_prediction_change": top_change, "random_k_mean_change": float(np.mean(random_changes)),
               "random_k_std_change": float(np.std(random_changes)),
               "faithfulness_ratio": top_change / (float(np.mean(random_changes)) + 1e-12)}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
