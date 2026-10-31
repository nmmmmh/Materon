#!/usr/bin/env python3
"""Environmental variation of the gate (Sec. S8, Figs. S6 and S7, Sec. S7.2 symmetry checks).

1. On all validation records: full prediction, gate bypass (mask 0) and
   replacement of each atom's context by the training-set mean context of its
   element (all 37 entries, chemistry 0-19 only, geometry 20-36 only).
2. Gate factors and corrections of every validation atom for the environment
   gate and, if given, the element-conditioned gate (saved as arrays).
3. Gate-path IG (16 and 32 intervals) of six fixed same-composition cases, and
   the invariance of predictions and atom IG under atom permutation and a
   90-degree in-plane rotation.

Example:
  python scripts/c2db/environment_probe.py --prepared work/c2db/prepared \
      --env-checkpoint work/c2db/runs/short/physical20_environment_gate_s42/best.weights.h5 \
      --element-checkpoint work/c2db/runs/short/physical20_element_gate_s42/best.weights.h5 \
      --out work/c2db/environment_probe
"""
import argparse
from pathlib import Path

import numpy as np

from materon.attributes import config_dir
from materon.c2db import attribution as A
from materon.c2db.dataset import Prepared, metrics, read_json, write_json
from materon.c2db.restoration import context_references, replace_context
from materon.runtime import configure_tensorflow

ENV, ELEMENT = "physical20_environment_gate", "physical20_element_gate"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--env-checkpoint", required=True)
    parser.add_argument("--element-checkpoint", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import tensorflow as tf

    from materon.c2db.models import load_member

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Prepared(args.prepared)
    mean, std, graphs = data.mean, data.std, data.graphs
    val_ids = data.ids("validation")
    val_graphs = [graphs[i] for i in val_ids]
    y_val, w_val = data.targets("validation"), data.weights("validation")
    models = {ENV: load_member(args.env_checkpoint, ENV)}
    if args.element_checkpoint:
        models[ELEMENT] = load_member(args.element_checkpoint, ELEMENT)
    env = models[ENV]

    # 1. Context replacement and bypass on the validation set.
    means, global_mean = context_references(graphs, data.ids("train"))
    sets = {"full": val_graphs}
    for name, section in (("element_context_mean", "all"), ("element_chemistry_mean", "chemistry"),
                          ("element_geometry_mean", "geometry")):
        sets[name] = [replace_context(g, means, global_mean, section) for g in val_graphs]
    predictions = {name: A.predict(env, batch, mean, std) for name, batch in sets.items()}
    predictions["mask0"] = A.predict(env, val_graphs, mean, std, mask=0.0)
    summary = {"validation_metrics": {name: metrics(y_val, p, w_val) for name, p in predictions.items()},
               "mean_abs_change_from_full_eV": {name: float(np.mean(np.abs(p - predictions["full"])))
                                                for name, p in predictions.items()}}

    # 2. Gate readouts of all validation atoms.
    arrays = {"validation_ids": np.asarray(val_ids),
              "node_z": np.concatenate([np.asarray(g["z"], dtype=np.int32) for g in val_graphs])}
    for arm, model in models.items():
        g, correction, splits = A.gate_readouts(model, val_graphs, arm)
        arrays[arm + "_gate"], arrays[arm + "_correction"], arrays["node_row_splits"] = g, correction, splits

    # 3. Six fixed cases: gate-path IG and symmetry checks.
    case_ids = read_json(config_dir() / "c2db" / "probes.json")["same_composition_cases"]["ids"]
    cases = [graphs[i] for i in case_ids]
    case_splits = np.concatenate(([0], np.cumsum([len(g["z"]) for g in cases])))
    summary["cases"] = {"ids": case_ids, "completeness_max_abs_eV": {}}
    for arm, model in models.items():
        full = A.predict(model, cases, mean, std, arm=arm, mask=1.0, batch_size=len(cases))
        bypass = A.predict(model, cases, mean, std, arm=arm, mask=0.0, batch_size=len(cases))
        for steps in (16, 32):
            atom_ig, _ = A.mask_path_ig(tf, model, cases, std, steps, arm=arm)
            feature_ig = A.per_graph_sum(atom_ig, case_splits)
            arrays["%s_case_atom_ig%d" % (arm, steps)] = atom_ig
            arrays["%s_case_feature_ig%d" % (arm, steps)] = feature_ig
            summary["cases"]["completeness_max_abs_eV"]["%s_%d" % (arm, steps)] = float(
                np.max(np.abs(feature_ig.sum(axis=1) - (full - bypass))))
        arrays[arm + "_case_full_eV"], arrays[arm + "_case_bypass_eV"] = full, bypass

    reference_ig = arrays[ENV + "_case_atom_ig32"]
    reference_full = arrays[ENV + "_case_full_eV"]
    permuted = [A.permute_graph(g) for g in cases]
    permuted_full = A.predict(env, [p[0] for p in permuted], mean, std, batch_size=len(cases))
    permuted_ig, _ = A.mask_path_ig(tf, env, [p[0] for p in permuted], std, 32)
    aligned = np.empty_like(permuted_ig)
    for index, (_, order) in enumerate(permuted):
        start, end = case_splits[index:index + 2]
        aligned[start:end][order] = permuted_ig[start:end]
    rotated = [A.rotate_graph(g) for g in cases]
    rotated_full = A.predict(env, rotated, mean, std, batch_size=len(cases))
    rotated_ig, _ = A.mask_path_ig(tf, env, rotated, std, 32)
    summary["symmetry"] = {
        "prediction_max_abs_difference_eV": max(float(np.max(np.abs(permuted_full - reference_full))),
                                                float(np.max(np.abs(rotated_full - reference_full)))),
        "permuted_atom_ig_max_abs_difference_eV": float(np.max(np.abs(aligned - reference_ig))),
        "rotated_atom_ig_max_abs_difference_eV": float(np.max(np.abs(rotated_ig - reference_ig))),
    }
    np.savez_compressed(out / "ARRAYS.npz", **arrays)
    write_json(out / "SUMMARY.json", summary)
    print(summary["validation_metrics"]["full"]["mae"], summary["validation_metrics"]["element_context_mean"]["mae"],
          summary["validation_metrics"]["mask0"]["mae"])


if __name__ == "__main__":
    main()
