#!/usr/bin/env python3
"""Ungated baselines for the neighbor readout over the metal-site substitutions (SI Section S3.6, Table S11c,d).

Input IG (ungated models, ``physical20_nogate``): integrated gradients of the predicted change F(q) - F(P) with
respect to the node inputs along x(t) = x_P + t (x_q - x_P) on the substituted graph, with midpoint steps. Only the
substituted atom changes its input, so the unsubstituted atoms receive zero attribution; the per-channel
attribution at the substituted atom and the completeness residual are written for every substitution.

Element-gate control (``physical20_element_gate``): largest change of the gated inputs x * g at the unsubstituted
atoms between parent and candidate (zero by construction, since that gate sees the element only).

The substitutions are the metal-site jobs of a channel-rules file written by ``channel_rules.py``.

Example:
  python scripts/c2db/ungated_baselines.py --prepared data/c2db_prepared --runs data/checkpoints \
      --rules data/results/GATE_RULES_L400.json --out-dir work/baselines
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Element

from materon.attributes import feature_table
from materon.c2db.dataset import Prepared
from materon.c2db.graph import copy_with_features, model_inputs
from materon.c2db.training import SEEDS, member_dir
from materon.runtime import configure_tensorflow


def substitution_pairs(rules_path, graphs, table):
    jobs = [j for j in json.loads(Path(rules_path).read_text())["jobs"] if j["kind"] == "M"]
    pairs = []
    for j in jobs:
        p = copy_with_features(graphs[j["parent_id"]], table)
        pairs.append((p, copy_with_features(p, table, j["site"], Element(j["substituent"]).Z)))
    return jobs, pairs


def predict(model, graphs, arm, batch=256):
    out = []
    for i in range(0, len(graphs), batch):
        out.append(model(model_inputs(graphs[i:i + batch], arm=arm), training=False).numpy().reshape(-1))
    return np.concatenate(out)


def input_ig(tf, model, jobs, pairs, std, steps, batch):
    ts = (np.arange(steps) + 0.5) / steps
    arm = "physical20_nogate"
    fp = predict(model, [p for p, _ in pairs], arm)
    fq = predict(model, [q for _, q in pairs], arm)
    native, attr, env_share, resid = [], [], [], []
    for i0 in range(0, len(pairs), batch):
        chunk = list(range(i0, min(i0 + batch, len(pairs))))
        gs, flats = [], []
        for i in chunk:
            p, q = pairs[i]
            xp, xq = np.asarray(p["x"], np.float32), np.asarray(q["x"], np.float32)
            gs += [q] * steps
            flats += [xp + t * (xq - xp) for t in ts]
        args = model_inputs(gs, arm=arm)
        flat = tf.constant(np.concatenate(flats))
        with tf.GradientTape() as tape:
            tape.watch(flat)
            total = tf.reduce_sum(model([args[0].with_flat_values(flat)] + list(args[1:]), training=False))
        grad = tape.gradient(total, flat).numpy()
        off = 0
        for i in chunk:
            p, q = pairs[i]
            n = len(q["x"])
            xp, xq = np.asarray(p["x"], np.float32), np.asarray(q["x"], np.float32)
            g = grad[off:off + steps * n].reshape(steps, n, -1).mean(0)
            off += steps * n
            a = (xq - xp) * g * std
            d_f = float(fq[i] - fp[i]) * std
            site = jobs[i]["site"]
            env = [k for k in range(n) if k != site]
            native.append(d_f)
            attr.append(a[site].tolist())
            env_share.append(float(np.abs(a[env]).sum() / max(np.abs(a).sum(), 1e-12)))
            resid.append(float(a.sum() - d_f))
    return dict(native_eV=native, site_attr_eV=attr, env_abs_share=env_share, completeness_resid_eV=resid)


def element_gate_change(tf, model, jobs, pairs, batch=128):
    arm = "physical20_element_gate"
    gated = tf.keras.Model(model.inputs, model.get_layer("element_gate").output)
    dmax = []
    for i0 in range(0, len(pairs), batch):
        chunk = list(range(i0, min(i0 + batch, len(pairs))))
        hp = gated(model_inputs([pairs[i][0] for i in chunk], arm=arm)).flat_values.numpy()
        hq = gated(model_inputs([pairs[i][1] for i in chunk], arm=arm)).flat_values.numpy()
        off = 0
        for i in chunk:
            n = len(pairs[i][0]["x"])
            env = [off + k for k in range(n) if k != jobs[i]["site"]]
            off += n
            dmax.append(float(np.abs(hq[env] - hp[env]).max()))
    return dmax


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--runs", required=True, help="checkpoint root: <runs>/<tag>/<arm>_s<seed>/best.weights.h5")
    parser.add_argument("--rules", required=True, help="channel-rules JSON (jobs list)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tags", nargs="+", default=["short", "L400"], help="schedules of the ungated models")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch", type=int, default=16, help="substitutions per gradient call")
    args = parser.parse_args()

    import tensorflow as tf

    from materon.c2db.models import load_member

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Prepared(args.prepared)
    table = feature_table()
    jobs, pairs = substitution_pairs(args.rules, data.graphs, table)
    print(len(jobs), "metal-site substitutions", flush=True)
    for tag in args.tags:
        for seed in args.seeds:
            f = out / ("nogate_%s_s%d.json" % (tag, seed))
            if f.exists():
                continue
            model = load_member(member_dir(args.runs, tag, seed, "physical20_nogate") / "best.weights.h5",
                                "physical20_nogate")
            r = input_ig(tf, model, jobs, pairs, data.std, args.steps, args.batch)
            f.write_text(json.dumps(dict(schedule=tag, seed=seed, **r)))
            print("ungated", tag, seed, "max neighbor share", max(r["env_abs_share"]),
                  "max |residual| %.4f eV" % max(abs(x) for x in r["completeness_resid_eV"]), flush=True)
    for seed in args.seeds:
        f = out / ("element_short_s%d.json" % seed)
        if f.exists():
            continue
        model = load_member(member_dir(args.runs, "short", seed, "physical20_element_gate") / "best.weights.h5",
                            "physical20_element_gate")
        d = element_gate_change(tf, model, jobs, pairs)
        f.write_text(json.dumps(dict(schedule="short", seed=seed, max_env_gated_input_change=d)))
        print("element gate", seed, "max neighbor gated-input change", max(d), flush=True)


if __name__ == "__main__":
    main()
