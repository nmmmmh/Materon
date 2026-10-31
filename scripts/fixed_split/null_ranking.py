#!/usr/bin/env python3
"""Null references for the E/I attribute ranking of the fixed-split study (Table S15; no training).

The readout is the one of explain_job.py: feature score_j = mean over test
atoms of |actual_total_delta_j|, with actual_total_delta = (d - 1) x + s (w+ v+ - w- v-).

  untrained      models built exactly as for training (same seed procedure and
                 backbone pairing), no weights loaded (seeds 42-46)
  trained_check  the trained best.weights.h5 loaded; reproduces feature_scores.csv

At initialization d_j = sigmoid(1) = 0.731 for every attribute, so the (d - 1) x
term is a uniform rescaling of |x|. The reported null runs used CPU execution.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["hoip_bandgap", "jdft2d_exfoliation"])
    p.add_argument("--backbone", required=True, choices=["cgcnn", "megnet"])
    p.add_argument("--mode", required=True, choices=["untrained", "trained_check"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--output-root", required=True, help="output root of the fixed-split study")
    p.add_argument("--cgcnn-cache-root", required=True)
    p.add_argument("--megnet-cache-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import numpy as np
    import tensorflow as tf
    from materon.attributes import RAW_FEATURE_COLUMNS
    from materon.fixed_split.cache_access import cgcnn_source_config, open_cgcnn_cache, open_megnet_cache
    from materon.fixed_split.config import load_protocol
    from materon.fixed_split.explain_utils import flat
    from materon.fixed_split.splits import positions_for_ids
    from materon.megnet.config import load_protocol as megnet_protocol
    from materon.runtime import configure_tensorflow, set_seed

    protocol = load_protocol()
    configure_tensorflow(tf, protocol["runtime"]["tf_intra_op_threads"], protocol["runtime"]["tf_inter_op_threads"])
    model_id = f"materon_explain_{a.backbone}"
    split = json.loads((Path(a.output_root) / "splits" / f"{a.dataset}_split_manifest.json").read_text())
    run_dir = Path(a.output_root) / "runs" / a.dataset / model_id / "seed_42"
    result = {"dataset": a.dataset, "backbone": a.backbone, "mode": a.mode, "feature_names": list(RAW_FEATURE_COLUMNS),
              "runs": []}

    if a.backbone == "cgcnn":
        from materon.cgcnn.models import build_paired_model
        cache = open_cgcnn_cache(a.dataset, model_id, a.cgcnn_cache_root)
        pos = positions_for_ids(cache.indices, split)
        x, target = cache.subset_positions(pos["test"])
        source_cfg, recipe = cgcnn_source_config(model_id)
        if a.mode == "untrained":
            sample = list(cache.subset_positions(pos["train"][:4])[0])  # build sample of train_job.py
        else:
            sample = [item[:min(4, len(target))] for item in x]  # build sample of explain_job.py
    else:
        from materon.megnet.models import build_model, find_physical_megnet_backbone
        cache = open_megnet_cache(a.dataset, a.megnet_cache_root)
        pos = positions_for_ids(cache.sample_ids, split)
        x = cache.inputs(model_id, pos["test"])
        source_protocol = megnet_protocol()
        if a.mode == "untrained":
            x_tr4 = cache.inputs(model_id, pos["train"][:4])
    xin = np.asarray(flat(x[0]).numpy(), np.float64)
    result["n_test_nodes"] = int(xin.shape[0])
    result["input_mean_abs_x"] = np.mean(np.abs(xin), axis=0).tolist()

    for seed in a.seeds:
        tf.keras.backend.clear_session(); gc.collect()
        if a.backbone == "cgcnn":
            set_seed(seed)
            model, _, _ = build_paired_model(source_cfg, recipe, sample, seed)
            if a.mode == "trained_check":
                model.load_weights(str(run_dir / "best.weights.h5"))
            _, analysis = model(x, training=False, return_analysis=True)
            ga = analysis["input_gate"]
        else:
            if a.mode == "untrained":
                # Same pairing as train_job.py: gated backbone takes the no-gate model's initial weights.
                set_seed(seed)
                reference, _, _ = build_model("materon_nogate_megnet", source_protocol, seed)
                _ = reference([item[:4] for item in x_tr4[:6]], training=False)
                set_seed(seed)
                model, analysis_model, names = build_model(model_id, source_protocol, seed)
                _ = model([item[:4] for item in x_tr4], training=False)
                find_physical_megnet_backbone(model).set_weights(reference.get_weights())
                del reference
            else:
                tf.keras.utils.set_random_seed(seed)
                model, analysis_model, names = build_model(model_id, source_protocol, seed)
                model.load_weights(str(run_dir / "best.weights.h5"))
            ga = dict(zip(names, analysis_model(x, training=False)))
        delta = np.asarray(flat(ga["actual_total_delta"]).numpy(), np.float64)
        dim = np.asarray(ga["dimension_gates"].numpy(), np.float64).reshape(-1)
        syn = float(np.asarray(ga["synergy"].numpy()))
        promo = np.asarray(flat(ga["promotion_features"]).numpy(), np.float64)
        inhib = np.asarray(flat(ga["inhibition_features"]).numpy(), np.float64)
        w = np.asarray(flat(ga["gate_weights"]).numpy(), np.float64)
        dim_term = (dim[None, :] - 1.0) * xin
        ei_term = syn * (w[:, 0:1] * promo - w[:, 1:2] * inhib)
        result["runs"].append({
            "seed": int(seed),
            "mean_abs_actual_delta": np.mean(np.abs(delta), axis=0).tolist(),
            "mean_signed_actual_delta": np.mean(delta, axis=0).tolist(),
            "dimension_gate": dim.tolist(), "synergy": syn, "n_nodes": int(delta.shape[0]),
            "identity_max_abs_err": float(np.max(np.abs(dim_term + ei_term - delta))),
            "mean_abs_dimension_term": np.mean(np.abs(dim_term), axis=0).tolist(),
            "mean_abs_ei_term": np.mean(np.abs(ei_term), axis=0).tolist()})
        print(f"[DONE] {a.dataset} {a.backbone} {a.mode} seed={seed}", flush=True)
        del model
        gc.collect()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
