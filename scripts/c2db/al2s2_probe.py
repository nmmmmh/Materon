#!/usr/bin/env python3
"""Al2S2 substitution screen and single-atom interpretation (Sec. S10, S11; Tables S22-S29, Fig. S9).

Screen: the first Al replaced by B or Ga and the first S by O or Se at the
parent geometry; predictions with the gate on and bypassed, gate readouts of
the unchanged atoms, and the parent-to-candidate joint input IG over x and the
chemical context (16/32 intervals, completeness <= 0.002 eV).

Local analysis of the unchanged Al1 for B and Ga: per-site chemical-context
restoration, the Al1 context curve, IG along the local context path (Eq. S6)
and the local gate path (Eq. S7), single-channel gate restoration R_j (Eq. S8)
with the interaction remainder, and the node Set2Set attention weights.
Optional global/element checkpoints add their gate factors (Table S28).

Example:
  python scripts/c2db/al2s2_probe.py --prepared work/c2db/prepared \
      --checkpoint work/c2db/runs/short/physical20_environment_gate_s42/best.weights.h5 \
      --out work/c2db/al2s2_probe.json
"""
import argparse

import numpy as np

from materon.attributes import RAW_FEATURE_COLUMNS, config_dir, feature_table
from materon.c2db.dataset import Prepared, read_json, write_json
from materon.c2db.graph import copy_with_features, model_inputs
from materon.runtime import configure_tensorflow

ENV = "physical20_environment_gate"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True, help="environment-gate checkpoint")
    parser.add_argument("--global-checkpoint", default=None)
    parser.add_argument("--element-checkpoint", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import tensorflow as tf

    from materon.c2db import attribution as A
    from materon.c2db import restoration as R
    from materon.c2db.models import backbone_of, load_member

    configure_tensorflow(tf, intra=1, inter=1, tf32=False)
    cfg = read_json(config_dir() / "c2db" / "probes.json")["al2s2"]
    data = Prepared(args.prepared)
    mean, std = data.mean, data.std
    table = feature_table()
    original = data.graphs[cfg["anchor_id"]]
    parent = copy_with_features(original, table)
    z = np.asarray(parent["z"], dtype=np.int64)
    specs, candidates = [], []
    for cid, (old_z, new_z) in cfg["candidates"].items():
        site = int(np.flatnonzero(z == old_z)[0])
        specs.append({"id": cid, "old_Z": old_z, "new_Z": new_z, "site": site})
        candidates.append(copy_with_features(parent, table, site, new_z))
    model = load_member(args.checkpoint, ENV)

    # Screen of the four candidates.
    all_graphs = [parent] + candidates
    mask1 = A.predict(model, all_graphs, mean, std, mask=1.0, batch_size=len(all_graphs))
    mask0 = A.predict(model, all_graphs, mean, std, mask=0.0, batch_size=len(all_graphs))
    gate, correction, _ = A.gate_readouts(model, all_graphs)
    gate, correction = gate.reshape(len(all_graphs), -1, 20), correction.reshape(len(all_graphs), -1, 20)
    ig = R.joint_input_ig(tf, model, parent, candidates, std)
    direct32, neighbor32 = ig["direct32_node_eV"].sum(axis=1), ig["neighbor32_node_eV"].sum(axis=1)
    delta = mask1[1:] - mask1[0]
    completeness = direct32.sum(axis=1) + neighbor32.sum(axis=1) - delta
    if np.max(np.abs(completeness)) > 0.002:
        raise ValueError("joint 32-interval IG completeness exceeded 0.002 eV")
    screen = []
    for k, spec in enumerate(specs):
        screen.append({**spec, "parent_prediction_eV": float(mask1[0]),
                       "candidate_prediction_eV": float(mask1[k + 1]), "delta_eV": float(delta[k]),
                       "delta_bypass_eV": float(mask0[k + 1] - mask0[0]),
                       "completeness_error_ig32_eV": float(completeness[k]),
                       "ig32_direct_eV": dict(zip(RAW_FEATURE_COLUMNS, direct32[k].tolist())),
                       "ig32_neighbor_context_eV": dict(zip(RAW_FEATURE_COLUMNS, neighbor32[k].tolist())),
                       "gate_parent": gate[0].tolist(), "gate_candidate": gate[k + 1].tolist(),
                       "correction_candidate": correction[k + 1].tolist()})
    result = {"anchor_id": cfg["anchor_id"], "screen": screen,
              "ig16_direct_node_eV": ig["direct16_node_eV"].tolist(),
              "ig32_direct_node_eV": ig["direct32_node_eV"].tolist(),
              "ig16_neighbor_node_eV": ig["neighbor16_node_eV"].tolist(),
              "ig32_neighbor_node_eV": ig["neighbor32_node_eV"].tolist()}

    # Local analysis of the unchanged Al1 for B and Ga.
    site = int(cfg["observed_site"])
    local = [candidates[[s["id"] for s in specs].index(cid)] for cid in cfg["local_candidates"]]
    graphs3 = [parent] + local
    predictions = A.predict(model, graphs3, mean, std, mask=1.0, batch_size=3)
    gates3 = R.gate_factors(model, graphs3)
    cases = []
    for k, (cid, g) in enumerate(zip(cfg["local_candidates"], local)):
        def predict_context(chemical):
            args_ = model_inputs([g], arm=ENV, mask=1.)
            full = np.asarray(g["context"]).copy()
            full[:, :20] = chemical
            args_[1] = args_[1].with_flat_values(tf.constant(full, dtype=tf.float32))
            return float(model(args_, training=False).numpy().reshape(-1)[0] * std + mean), args_

        restored = []
        for s in range(len(z)):
            chemical = g["context"][:, :20].copy()
            chemical[s] = parent["context"][s, :20]
            restored.append(predict_context(chemical)[0])
        curve = []
        for lam in cfg["context_curve_lambda"]:
            chemical = g["context"][:, :20].copy()
            chemical[site] = parent["context"][site, :20] + lam * (g["context"][site, :20] - parent["context"][site, :20])
            value, args_ = predict_context(chemical)
            _, ro = model.get_layer("environment_gate")(args_[:3], return_readouts=True)
            curve.append({"lambda": lam, "prediction_eV": value, "g": ro["gate"].flat_values.numpy()[site].tolist()})
        full = float(predictions[k + 1])
        cases.append({"id": cid, "prediction_eV": full, "predicted_delta_eV": full - float(predictions[0]),
                      "site_environment_effect_eV": (full - np.asarray(restored)).tolist(),
                      "context_curve": curve})
    for kind, key in (("context", "local_context_ig"), ("gate", "gate_path_ig")):
        for k, value in enumerate(R.local_path_ig(tf, model, parent, local, gates3[0], gates3[1:], site,
                                                  mean, std, kind)):
            cases[k][key] = value
    for k, g in enumerate(local):
        restoration = R.single_site_restoration(tf, model, parent, g, site, mean, std)
        response = predictions[k + 1] - np.asarray(restoration["restored_eV"])
        cases[k]["delta_g"] = restoration["delta_g"]
        cases[k]["channel_restore_eV"] = response.tolist()
        cases[k]["interaction_remainder_eV"] = float(cases[k]["site_environment_effect_eV"][site] - response.sum())
    backbone = backbone_of(model)
    env_inputs = model_inputs(graphs3, arm=ENV, mask=1.)
    x = np.asarray([g["x"] for g in graphs3], dtype=np.float32)
    corrected = env_inputs[0].with_flat_values(tf.constant((x + x * (gates3 - 1.)).reshape(-1, 20)))
    attention, pooling_error = R.set2set_node_attention(tf, backbone, [corrected] + env_inputs[3:], len(z), 3)
    result["local"] = {"observed_site": site, "cases": cases, "environment_g": gates3.tolist(),
                       "set2set_node_attention": attention.tolist(), "set2set_replay_max_abs_error": pooling_error}
    if args.global_checkpoint:
        layer = load_member(args.global_checkpoint, "physical20_global_gate").get_layer("global_feature_gate")
        result["local"]["global_g"] = (1. + .5 * np.tanh(layer.b.numpy())).tolist()
    if args.element_checkpoint:
        element = load_member(args.element_checkpoint, "physical20_element_gate")
        a = model_inputs(graphs3, arm="physical20_element_gate", mask=1.)
        _, ro = element.get_layer("element_gate")([a[0], element.get_layer("zero_environment")(a[0]), a[1]],
                                                  return_readouts=True)
        result["local"]["element_g"] = ro["gate"].flat_values.numpy().reshape(3, len(z), 20).tolist()
    write_json(args.out, result)
    for case in cases:
        print(case["id"], "Al1 environment effect %.6f eV" % case["site_environment_effect_eV"][site])


if __name__ == "__main__":
    main()
