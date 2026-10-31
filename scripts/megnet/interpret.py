#!/usr/bin/env python3
"""Interpretation of a trained Materon-MEGNet model on its outer test fold.

Outputs (in <run>/interpretability):
  feature_scores.csv          mean |net modulation| per attribute (dimension ranking)
  node_scores.csv.gz          per-atom node confidence and modulation norms
  feature_fidelity.csv        disable Top-5 / Bottom-5 / 5 random attribute corrections
  node_fidelity.csv           disable the gate at the top / bottom / random 10% of atoms
  pathway_interventions.csv   shut down the promotion, inhibition, dimension-gate or synergy pathway
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from pymatgen.core import Element
    from materon.attributes import RAW_FEATURE_COLUMNS
    from materon.runtime import configure_tensorflow

    configure_tensorflow(tf, 2, 2)
    from materon.megnet.cache import MEGNetCache
    from materon.megnet.config import load_protocol
    from materon.megnet.interpret import flat_rows, prediction_metrics, top_count, unscale
    from materon.megnet.models import build_model

    q = load_protocol()
    cfg = q["interpretability"]
    model_id = "materon_explain_megnet"
    run = Path(a.outdir) / a.task / model_id / f"outer_fold_{a.fold}" / "seed_42"
    out = run / "interpretability"
    out.mkdir(parents=True, exist_ok=True)
    cache = MEGNetCache(a.cache_root, a.task)
    _, test = cache.split(a.fold)
    x = cache.inputs(model_id, test)
    target = cache.targets(test).ravel()
    scaler = json.loads((run / "target_scaler.json").read_text())
    tf.keras.backend.clear_session(); gc.collect()
    model, analysis_model, names = build_model(model_id, q, 42)
    model.load_weights(str(run / "model.weights.h5"))
    data = dict(zip(names, analysis_model(x, training=False)))
    full = unscale(data["prediction"].numpy(), scaler)
    attention = flat_rows(data["attention_scores"])
    delta = flat_rows(data["actual_total_delta"])
    promo = flat_rows(data["promotion_features"])
    inhib = flat_rows(data["inhibition_features"])
    weights_ei = flat_rows(data["gate_weights"])

    node_rows, sample_rows = [], []
    for sid, z, site, att, de, pr, ih, gw in zip(cache.ids(test), cache.node_numbers(test), cache.site_indices(test),
                                                 attention, delta, promo, inhib, weights_ei):
        score = np.linalg.norm(de, axis=-1)
        sample_rows.append({"sample_id": sid, "mean_node_delta_norm": float(score.mean()),
                            "max_node_delta_norm": float(score.max()), "mean_attention": float(np.asarray(att).mean())})
        for i in range(len(z)):
            node_rows.append({"sample_id": sid, "site_index": int(site[i]), "atomic_number": int(z[i]),
                              "element": Element.from_Z(int(z[i])).symbol,
                              "attention": float(np.asarray(att[i]).reshape(-1)[0]), "actual_delta_norm": float(score[i]),
                              "promotion_norm": float(np.linalg.norm(pr[i])), "inhibition_norm": float(np.linalg.norm(ih[i])),
                              "promotion_weight": float(gw[i, 0]), "inhibition_weight": float(gw[i, 1])})
    pd.DataFrame(node_rows).to_csv(out / "node_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(sample_rows).to_csv(out / "sample_gate_summary.csv", index=False)

    D, P, I = (np.concatenate(v, axis=0) for v in (delta, promo, inhib))
    names20 = list(RAW_FEATURE_COLUMNS)
    feat = pd.DataFrame({"feature": names20, "mean_abs_actual_delta": np.mean(np.abs(D), axis=0),
                         "mean_signed_actual_delta": np.mean(D, axis=0), "mean_abs_promotion": np.mean(np.abs(P), axis=0),
                         "mean_abs_inhibition": np.mean(np.abs(I), axis=0),
                         "dimension_gate": np.asarray(data["dimension_gates"].numpy()).reshape(-1)})
    feat["rank"] = feat["mean_abs_actual_delta"].rank(method="min", ascending=False).astype(int)
    feat.to_csv(out / "feature_scores.csv", index=False)

    gate = model.get_layer("biomimetic_ei_gate")
    base = {"intervention": "full", **prediction_metrics(target, full, full)}

    def predict(inputs):
        return unscale(model.predict(inputs, batch_size=32, verbose=0), scaler)

    # Attribute fidelity: disable the gate correction of selected attributes.
    topk = int(cfg["feature_top_k"])
    order = np.argsort(-feat["mean_abs_actual_delta"].to_numpy())
    interventions = [("top", order[:topk]), ("bottom", order[-topk:])]
    rng = np.random.RandomState(42 + 100 * a.fold)
    for r in range(cfg["feature_random_repeats"]):
        interventions.append((f"random_{r}", rng.choice(len(names20), topk, replace=False)))
    rows = [base]
    for label, idx in interventions:
        mask = np.ones(len(names20), np.float32)
        mask[idx] = 0
        gate.set_feature_mask(mask)
        rows.append({"intervention": label, "features": ";".join(names20[i] for i in idx),
                     **prediction_metrics(target, predict(x), full)})
    gate.reset_feature_mask()
    pd.DataFrame(rows).to_csv(out / "feature_fidelity.csv", index=False)

    # Node fidelity: disable the gate at a fraction of atoms per structure.
    scores = [np.linalg.norm(v, axis=-1) for v in delta]

    def masks(kind, repeat=0):
        rr = np.random.RandomState(42000 + a.fold * 100 + repeat)
        result = []
        for sc in scores:
            n = len(sc)
            k = top_count(n, cfg["node_fraction"])
            idx = np.argsort(-sc)[:k] if kind == "top" else np.argsort(sc)[:k] if kind == "bottom" else rr.choice(n, k, replace=False)
            m = np.ones((n, 1), np.float32)
            m[idx] = 0
            result.append(m)
        return result

    rows = [base]
    for kind in ["top", "bottom"]:
        rows.append({"intervention": kind, **prediction_metrics(target, predict(cache.inputs(model_id, test, masks(kind))), full)})
    for r in range(cfg["node_random_repeats"]):
        rows.append({"intervention": f"random_{r}",
                     **prediction_metrics(target, predict(cache.inputs(model_id, test, masks("random", r))), full)})
    pd.DataFrame(rows).to_csv(out / "node_fidelity.csv", index=False)

    # Pathway shutdown.
    rows = [base]
    for label, kw in {"promotion": {"promotion": 0}, "inhibition": {"inhibition": 0},
                      "dimension_gate": {"dimension": 0}, "synergy": {"synergy": 0}}.items():
        gate.reset_pathway_state()
        gate.set_pathway_state(**kw)
        rows.append({"intervention": label, **prediction_metrics(target, predict(x), full)})
    gate.reset_pathway_state()
    pd.DataFrame(rows).to_csv(out / "pathway_interventions.csv", index=False)
    print("[INTERPRET DONE]", a.task, a.fold, feat.sort_values("rank").head(5)["feature"].tolist(), flush=True)


if __name__ == "__main__":
    main()
