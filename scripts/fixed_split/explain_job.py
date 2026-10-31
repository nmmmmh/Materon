#!/usr/bin/env python3
"""Interventions on a trained fixed-split Materon model (CGCNN or MEGNet) on the locked test set.

Readouts: per-attribute mean |net modulation| (feature_scores.csv) and
per-atom node scores. Interventions (each with structure-bootstrap intervals):
  gate_modulation_mask      disable the gate correction of Top-5 / Bottom-5 / 10 random attributes
  raw_feature_zero          set the raw Top-5 / Bottom-5 attributes to zero
  single/physical-group     disable one attribute or one physical group
  node_gate_mask            disable the gate at the top / bottom / random 10% of atoms
  chemical_role_gate_mask   disable the gate at all atoms of one structural role
  pathway                   remove the promotion, inhibition, dimension-gate or synergy term
CGCNN pathways are fixed-activation counterfactuals; MEGNet pathways use the gate's pathway masks.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", choices=["cgcnn", "megnet"], required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--cgcnn-cache-root", required=True)
    p.add_argument("--megnet-cache-root", required=True)
    a = p.parse_args()

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from materon.attributes import RAW_FEATURE_COLUMNS
    from materon.fixed_split.cache_access import cgcnn_source_config, open_cgcnn_cache, open_megnet_cache
    from materon.fixed_split.config import load_protocol
    from materon.fixed_split.datasets import atomic_symbol, infer_roles
    from materon.fixed_split.explain_utils import (bootstrap_ci, feature_groups, flat, intervention_metrics,
                                                   row_splits, top_count, unscale)
    from materon.fixed_split.splits import positions_for_ids
    from materon.megnet.config import load_protocol as megnet_protocol
    from materon.runtime import configure_tensorflow

    protocol = load_protocol()
    cfg_i = protocol["interpretability"]
    configure_tensorflow(tf, protocol["runtime"]["tf_intra_op_threads"], protocol["runtime"]["tf_inter_op_threads"])
    model_id = f"materon_explain_{a.backbone}"
    run = Path(a.output_root) / "runs" / a.dataset / model_id / "seed_42"
    out = Path(a.output_root) / "interpretability" / a.dataset / a.backbone
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((Path(a.output_root) / "splits" / f"{a.dataset}_split_manifest.json").read_text())
    scaler = json.loads((run / "target_scaler.json").read_text())
    seed = int(protocol["seed"])
    rng = np.random.RandomState(seed)

    # Roles and coordination from the MEGNet cache (same atom order as the CGCNN cache).
    mg_cache = open_megnet_cache(a.dataset, a.megnet_cache_root)
    mg_pos = positions_for_ids(mg_cache.sample_ids, split)["test"]
    role_by_id, z_by_id, coord_by_id = {}, {}, {}
    frac_values, frac_splits = mg_cache.r["node_frac_coordinates"]
    edge_values, edge_splits = mg_cache.r["range_indices"]
    for local, (sid, z, pos) in enumerate(zip(mg_cache.ids(mg_pos), mg_cache.node_numbers(mg_pos), mg_pos)):
        frac = np.asarray(frac_values[frac_splits[pos]:frac_splits[pos + 1]], np.float32)
        edges = np.asarray(edge_values[edge_splits[pos]:edge_splits[pos + 1]], np.int64)
        role_by_id[sid] = infer_roles(a.dataset, z, frac)
        z_by_id[sid] = z
        coord_by_id[sid] = np.bincount(edges[:, 0] if len(edges) else np.asarray([], np.int64),
                                       minlength=len(z)).astype(np.float32)

    if a.backbone == "cgcnn":
        from materon.cgcnn.models import build_paired_model
        cache = open_cgcnn_cache(a.dataset, model_id, a.cgcnn_cache_root)
        positions = positions_for_ids(cache.indices, split)["test"]
        x, target = cache.subset_positions(positions)
        ids = [cache.indices[int(i)] for i in positions]
        source_cfg, recipe = cgcnn_source_config(model_id)
        tf.keras.backend.clear_session(); gc.collect(); tf.keras.utils.set_random_seed(seed)
        model, _, _ = build_paired_model(source_cfg, recipe, [item[:min(4, len(target))] for item in x], seed)
        model.load_weights(str(run / "best.weights.h5"))
        base_z, analysis = model(x, training=False, return_analysis=True)
        gate_analysis = analysis["input_gate"]
        base = unscale(base_z.numpy(), scaler)
        raw = gate_analysis["input_features"]
        gate = model.input_gate

        def predict_normal(inputs=x):
            return unscale(model(inputs, training=False).numpy(), scaler)

        def predict_node_masks(mask_rows):
            h = model.input_gate(x[0], training=False)
            mask = tf.ragged.constant([np.asarray(v, np.float32).reshape(-1, 1) for v in mask_rows], dtype=tf.float32)
            hmod = x[0].with_flat_values(x[0].flat_values + (h.flat_values - x[0].flat_values)
                                         * tf.expand_dims(mask.flat_values, axis=-1))
            return unscale(model.core.readout(model.core.convolve(hmod, x[2], x[1], False), False).numpy(), scaler)

        def predict_pathway(kind):
            xin = flat(gate_analysis["input_features"])
            xdim = xin + flat(gate_analysis["dimension_delta"])
            weights_gate = flat(gate_analysis["gate_weights"])
            synergy = float(gate_analysis["synergy"].numpy())
            pterm = synergy * weights_gate[:, 0:1] * flat(gate_analysis["promotion_features"])
            iterm = synergy * weights_gate[:, 1:2] * flat(gate_analysis["inhibition_features"])
            alt = {"promotion": lambda: xdim - iterm, "inhibition": lambda: xdim + pterm,
                   "dimension_gate": lambda: xin + pterm - iterm, "synergy": lambda: xdim, "all_ei": lambda: xdim}[kind]()
            conv = model.core.convolve(raw.with_flat_values(alt), x[2], x[1], False)
            return unscale(model.core.readout(conv, False).numpy(), scaler)

        pathway_semantics = "fixed-activation counterfactual decomposition for CGCNN"
    else:
        from materon.megnet.models import build_model
        cache, positions = mg_cache, mg_pos
        x = cache.inputs(model_id, positions)
        target = cache.targets(positions).reshape(-1)
        ids = cache.ids(positions)
        tf.keras.backend.clear_session(); gc.collect(); tf.keras.utils.set_random_seed(seed)
        model, analysis_model, names = build_model(model_id, megnet_protocol(), seed)
        model.load_weights(str(run / "best.weights.h5"))
        gate_analysis = dict(zip(names, analysis_model(x, training=False)))
        base = unscale(gate_analysis["prediction"].numpy(), scaler)
        raw = x[0]
        gate = model.get_layer("biomimetic_ei_gate")

        def predict_normal(inputs=x):
            return unscale(model(inputs, training=False).numpy(), scaler)

        def predict_node_masks(mask_rows):
            return predict_normal(cache.inputs(model_id, positions, node_masks=mask_rows))

        def predict_pathway(kind):
            states = {"promotion": dict(promotion=0.0), "inhibition": dict(inhibition=0.0),
                      "dimension_gate": dict(dimension=0.0), "synergy": dict(synergy=0.0),
                      "all_ei": dict(promotion=0.0, inhibition=0.0, synergy=0.0)}[kind]
            gate.set_pathway_state(**states)
            try:
                return predict_normal()
            finally:
                gate.reset_pathway_state()

        pathway_semantics = "inference-time gate pathway masks for MEGNet"

    rs = row_splits(raw)

    def predict_feature_mask(mask):
        gate.set_feature_mask(mask)
        try:
            return predict_normal()
        finally:
            gate.reset_feature_mask()

    def predict_raw_zero(indices):
        arr = x[0].flat_values.numpy().copy()
        arr[:, indices] = 0.0
        changed = list(x)
        changed[0] = x[0].with_flat_values(tf.convert_to_tensor(arr, tf.float32))
        return predict_normal(changed)

    target = np.asarray(target).reshape(-1)
    delta = np.asarray(flat(gate_analysis["actual_total_delta"]).numpy(), np.float32)
    attention = np.asarray(flat(gate_analysis["attention_scores"]).numpy(), np.float32).reshape(-1)
    rel = np.asarray(flat(gate_analysis["relative_actual_modulation"]).numpy(), np.float32)
    promotion = np.asarray(flat(gate_analysis["promotion_features"]).numpy(), np.float32)
    inhibition = np.asarray(flat(gate_analysis["inhibition_features"]).numpy(), np.float32)
    gate_weights = np.asarray(flat(gate_analysis["gate_weights"]).numpy(), np.float32)
    node_score = np.sum(np.abs(delta), axis=1) * (0.5 + attention)
    feature_score = np.mean(np.abs(delta), axis=0)
    feature_names = list(RAW_FEATURE_COLUMNS)

    feature_table = pd.DataFrame({
        "feature_index": np.arange(len(feature_names)), "feature": feature_names,
        "mean_abs_actual_delta": feature_score, "mean_signed_actual_delta": np.mean(delta, axis=0),
        "dimension_gate": np.asarray(gate_analysis["dimension_gates"].numpy(), np.float32).reshape(-1),
        "mean_abs_promotion": np.mean(np.abs(promotion), axis=0), "mean_abs_inhibition": np.mean(np.abs(inhibition), axis=0),
        "mean_signed_promotion": np.mean(promotion, axis=0), "mean_signed_inhibition": np.mean(inhibition, axis=0),
    }).sort_values("mean_abs_actual_delta", ascending=False)
    feature_table["rank"] = np.arange(1, len(feature_table) + 1)
    feature_table.to_csv(out / "feature_scores.csv", index=False)

    node_rows, mask_scores, sample_feature_rows = [], [], []
    for i, sid in enumerate(ids):
        start, stop = int(rs[i]), int(rs[i + 1])
        scores = node_score[start:stop]
        mask_scores.append(scores)
        zlist, roles, coords = np.asarray(z_by_id[sid]), role_by_id[sid], coord_by_id[sid]
        if len(zlist) != len(scores):
            raise RuntimeError(f"Node alignment mismatch for {sid}")
        for j, name in enumerate(feature_names):
            sample_feature_rows.append({"sample_id": sid, "feature_index": j, "feature": name,
                                        "mean_abs_actual_delta": float(np.mean(np.abs(delta[start:stop, j]))),
                                        "mean_signed_actual_delta": float(np.mean(delta[start:stop, j])),
                                        "mean_abs_promotion": float(np.mean(np.abs(promotion[start:stop, j]))),
                                        "mean_abs_inhibition": float(np.mean(np.abs(inhibition[start:stop, j])))})
        for site in range(len(scores)):
            node_rows.append({"sample_id": sid, "site_index": site, "atomic_number": int(zlist[site]),
                              "element": atomic_symbol(int(zlist[site])), "role": roles[site],
                              "coordination": float(coords[site]), "node_score": float(scores[site]),
                              "attention": float(attention[start + site]),
                              "promotion_weight": float(gate_weights[start + site, 0]),
                              "inhibition_weight": float(gate_weights[start + site, 1]),
                              "signed_total_delta": float(np.sum(delta[start + site])),
                              "relative_modulation_sum": float(np.sum(rel[start + site]))})
    pd.DataFrame(node_rows).to_csv(out / "node_scores.csv", index=False)
    pd.DataFrame(sample_feature_rows).to_csv(out / "sample_feature_scores.csv", index=False)

    repeats = int(cfg_i["bootstrap_repeats"])
    sample_effect_rows = []

    def record(kind, label, pred, extra=None):
        pred = np.asarray(pred).reshape(-1)
        values = np.abs(pred - base)
        error_delta = np.abs(target - pred) - np.abs(target - base)
        lo, hi = bootstrap_ci(values, repeats, seed)
        elo, ehi = bootstrap_ci(error_delta, repeats, seed + 1)
        for i, sid in enumerate(ids):
            sample_effect_rows.append({"sample_id": sid, "intervention_type": kind, "intervention": label,
                                       "target": float(target[i]), "base_prediction": float(base[i]),
                                       "intervention_prediction": float(pred[i]),
                                       "absolute_error_change": float(error_delta[i]),
                                       "prediction_change": float(pred[i] - base[i])})
        return {"intervention_type": kind, "intervention": label, **intervention_metrics(target, base, pred),
                "prediction_change_ci95_low": lo, "prediction_change_ci95_high": hi,
                "delta_mae_ci95_low": elo, "delta_mae_ci95_high": ehi, **(extra or {})}

    k = int(cfg_i["feature_top_k"])
    order = np.argsort(-feature_score)
    feature_rows = []
    for label, idx in [("top", order[:k]), ("bottom", order[-k:])]:
        mask = np.ones(20, np.float32)
        mask[idx] = 0
        info = {"feature_indices": ",".join(map(str, idx))}
        feature_rows.append(record("gate_modulation_mask", label, predict_feature_mask(mask), info))
        feature_rows.append(record("raw_feature_zero", label, predict_raw_zero(idx), info))
    for repeat in range(int(cfg_i["feature_random_repeats"])):
        idx = np.sort(rng.choice(20, size=k, replace=False))
        mask = np.ones(20, np.float32)
        mask[idx] = 0
        feature_rows.append(record("gate_modulation_mask", f"random_{repeat}", predict_feature_mask(mask),
                                   {"feature_indices": ",".join(map(str, idx))}))
    for i, name in enumerate(feature_names):
        mask = np.ones(20, np.float32)
        mask[i] = 0
        feature_rows.append(record("single_dimension_gate_mask", name, predict_feature_mask(mask), {"feature_index": i}))
    for group, idx in feature_groups(feature_names).items():
        mask = np.ones(20, np.float32)
        mask[idx] = 0
        feature_rows.append(record("physical_group_gate_mask", group, predict_feature_mask(mask),
                                   {"feature_indices": ",".join(map(str, idx))}))
    pd.DataFrame(feature_rows).to_csv(out / "feature_interventions.csv", index=False)

    fraction = float(cfg_i["node_fraction"])
    node_out = []
    for kind in ["top", "bottom"]:
        masks = []
        for scores in mask_scores:
            n = len(scores)
            kk = top_count(n, fraction)
            idx = np.argsort(-scores)[:kk] if kind == "top" else np.argsort(scores)[:kk]
            mask = np.ones(n, np.float32)
            mask[idx] = 0
            masks.append(mask)
        node_out.append(record("node_gate_mask", kind, predict_node_masks(masks), {"node_fraction": fraction}))
    for repeat in range(int(cfg_i["node_random_repeats"])):
        masks = []
        for scores in mask_scores:
            n = len(scores)
            idx = rng.choice(n, size=top_count(n, fraction), replace=False)
            mask = np.ones(n, np.float32)
            mask[idx] = 0
            masks.append(mask)
        node_out.append(record("node_gate_mask", f"random_{repeat}", predict_node_masks(masks), {"node_fraction": fraction}))
    pd.DataFrame(node_out).to_csv(out / "node_interventions.csv", index=False)

    role_rows = []
    for role in sorted({r for sid in ids for r in role_by_id[sid]}):
        masks = []
        for sid, scores in zip(ids, mask_scores):
            mask = np.ones(len(scores), np.float32)
            mask[[i for i, value in enumerate(role_by_id[sid]) if value == role]] = 0
            masks.append(mask)
        role_rows.append(record("chemical_role_gate_mask", role, predict_node_masks(masks), {"masked_role": role}))
    pd.DataFrame(role_rows).to_csv(out / "role_interventions.csv", index=False)

    pathway_rows = [record("pathway", kind, predict_pathway(kind), {"semantics": pathway_semantics})
                    for kind in cfg_i["pathways"]]
    pd.DataFrame(pathway_rows).to_csv(out / "pathway_interventions.csv", index=False)
    pd.DataFrame(sample_effect_rows).to_csv(out / "intervention_sample_effects.csv.gz", index=False, compression="gzip")
    print(f"[INTERPRET DONE] {a.dataset} {a.backbone} base_mae={float(np.mean(np.abs(target - base))):.6g}", flush=True)


if __name__ == "__main__":
    main()
