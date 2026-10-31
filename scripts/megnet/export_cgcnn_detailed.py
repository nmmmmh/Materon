#!/usr/bin/env python3
"""Per-atom and per-attribute readouts of the trained Materon-CGCNN transfer models (inference only).

Same exports as scripts/megnet/interpret.py (node_scores, feature_scores,
feature_fidelity, node_fidelity), for CGCNN-MEGNet comparison in cross_backbone.py.
Node fidelity disables the gate correction at selected atoms while keeping the
raw attributes.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cgcnn-output-root", required=True, help="output root of scripts/cgcnn/run_transfer.py")
    p.add_argument("--cgcnn-cache-root", required=True)
    p.add_argument("--tasks", default="matbench_dielectric,matbench_jdft2d")
    p.add_argument("--export-root", required=True)
    a = p.parse_args()

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from pymatgen.core import Element
    from materon.attributes import RAW_FEATURE_COLUMNS, load_feature_bundle
    from materon.cgcnn.cache import official_positions_from_serialized, open_existing_cache
    from materon.cgcnn.config import get_config, get_recipe, transfer_protocol
    from materon.cgcnn.models import build_paired_model
    from materon.megnet.interpret import prediction_metrics, top_count, unscale

    protocol = transfer_protocol()
    table = load_feature_bundle("full20")["table"][1:]
    zvals = np.arange(1, 101)
    fn = list(RAW_FEATURE_COLUMNS)
    spec = protocol["models"]["fixed_materon_explain"]
    for task in [x for x in a.tasks.split(",") if x]:
        cfg = get_config(spec["source_config"])
        cfg["id"] = "fixed_materon_explain"
        cache = open_existing_cache(cfg, a.cgcnn_cache_root, task)
        for fold in range(5):
            run = Path(a.cgcnn_output_root) / task / "fixed_materon_explain" / f"outer_fold_{fold}" / "seed_42"
            _, _, test, target, _, test_ids, _ = official_positions_from_serialized(a.cgcnn_cache_root, task, cache, fold)
            x, y = cache.subset_positions(test)
            opt = json.loads((run / "selected_optimization.json").read_text())
            recipe = dict(get_recipe(spec["base_recipe"]))
            recipe["learning_rate"] = opt["learning_rate"]
            tf.keras.backend.clear_session(); gc.collect()
            model, _, _ = build_paired_model(cfg, recipe, [v[:min(4, len(y))] for v in x], 42)
            model.load_weights(str(run / "model.weights.h5"))
            pred_scaled, analysis = model(x, training=False, return_analysis=True)
            scaler = json.loads((run / "target_scaler.json").read_text())
            full = unscale(pred_scaled.numpy(), scaler)
            target = np.asarray(target).ravel()
            gate = analysis["input_gate"]

            def rows(rt):
                return [np.asarray(v) for v in rt]

            raw, att, delta = rows(analysis["raw_physical_features"]), rows(gate["attention_scores"]), rows(gate["actual_total_delta"])
            promo, inhib, gw = rows(gate["promotion_features"]), rows(gate["inhibition_features"]), rows(gate["gate_weights"])
            out = Path(a.export_root) / task / f"outer_fold_{fold}"
            out.mkdir(parents=True, exist_ok=True)
            node_rows, scores = [], []
            for sid, r, at, de, pr, ih, w in zip(test_ids, raw, att, delta, promo, inhib, gw):
                # Recover atomic numbers from the full20 rows.
                z = zvals[np.argmin(((r[:, None, :] - table[None, :, :]) ** 2).sum(axis=-1), axis=1)]
                scores.append(np.linalg.norm(de, axis=-1))
                for k in range(len(r)):
                    node_rows.append({"sample_id": sid, "site_index": k, "atomic_number": int(z[k]),
                                      "element": Element.from_Z(int(z[k])).symbol,
                                      "attention": float(np.asarray(at[k]).reshape(-1)[0]),
                                      "actual_delta_norm": float(np.linalg.norm(de[k])),
                                      "promotion_norm": float(np.linalg.norm(pr[k])),
                                      "inhibition_norm": float(np.linalg.norm(ih[k])),
                                      "promotion_weight": float(w[k, 0]), "inhibition_weight": float(w[k, 1])})
            pd.DataFrame(node_rows).to_csv(out / "node_scores.csv.gz", index=False, compression="gzip")
            D, P, I = (np.concatenate(v, axis=0) for v in (delta, promo, inhib))
            feat = pd.DataFrame({"feature": fn, "mean_abs_actual_delta": np.mean(np.abs(D), axis=0),
                                 "mean_signed_actual_delta": np.mean(D, axis=0),
                                 "mean_abs_promotion": np.mean(np.abs(P), axis=0),
                                 "mean_abs_inhibition": np.mean(np.abs(I), axis=0)})
            feat["rank"] = feat.mean_abs_actual_delta.rank(method="min", ascending=False).astype(int)
            feat.to_csv(out / "feature_scores.csv", index=False)

            frows = [{"intervention": "full", **prediction_metrics(target, full, full)}]
            order = np.argsort(-feat.mean_abs_actual_delta.to_numpy())
            rng = np.random.RandomState(42 + 100 * fold)
            interventions = [("top", order[:5]), ("bottom", order[-5:])] + \
                [(f"random_{r}", rng.choice(20, 5, replace=False)) for r in range(5)]
            for label, idx in interventions:
                mask = np.ones(20, np.float32)
                mask[idx] = 0
                model.input_gate.set_feature_mask(mask)
                frows.append({"intervention": label, "features": ";".join(fn[i] for i in idx),
                              **prediction_metrics(target, unscale(model(x, training=False).numpy(), scaler), full)})
            model.input_gate.reset_feature_mask()
            pd.DataFrame(frows).to_csv(out / "feature_fidelity.csv", index=False)

            node, edge_idx, edge_feat = x[0], x[1], x[2]

            def masked_predict(mask_rows):
                h = model.input_gate(node, training=False)
                mask_flat = tf.convert_to_tensor(np.concatenate(mask_rows, axis=0), dtype=tf.float32)
                if mask_flat.shape.rank == 1:
                    mask_flat = tf.expand_dims(mask_flat, axis=-1)
                hmod = node.with_flat_values(node.flat_values + (h.flat_values - node.flat_values) * mask_flat)
                conv = model.core.convolve(hmod, edge_feat, edge_idx, False)
                return unscale(model.core.readout(conv, False).numpy(), scaler)

            def make_masks(kind, repeat=0):
                rr = np.random.RandomState(42000 + fold * 100 + repeat)
                ans = []
                for sc in scores:
                    n = len(sc)
                    kk = top_count(n, 0.1)
                    idx = np.argsort(-sc)[:kk] if kind == "top" else np.argsort(sc)[:kk] if kind == "bottom" else rr.choice(n, kk, replace=False)
                    m = np.ones((n, 1), np.float32)
                    m[idx] = 0
                    ans.append(m)
                return ans

            nrows = [{"intervention": "full", **prediction_metrics(target, full, full)}]
            for kind in ["top", "bottom"]:
                nrows.append({"intervention": kind, **prediction_metrics(target, masked_predict(make_masks(kind)), full)})
            for r in range(5):
                nrows.append({"intervention": f"random_{r}",
                              **prediction_metrics(target, masked_predict(make_masks("random", r)), full)})
            pd.DataFrame(nrows).to_csv(out / "node_fidelity.csv", index=False)
            print(f"[CGCNN DETAIL] {task} fold {fold}", flush=True)


if __name__ == "__main__":
    main()
