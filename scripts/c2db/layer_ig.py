#!/usr/bin/env python3
"""Layer integrated gradients of the ungated network over the metal-site substitutions (SI Section S3.6, Table S11d).

After residual block k (k = 1, 2, 3) of the MEGNet backbone, the hidden states of all nodes and edges and the
global state are interpolated jointly from their parent to their substituted values, with midpoint steps. The rest
of the network depends only on these quantities, so the attributions sum to the predicted change. The attribution
is split into the unsubstituted atoms (neighbors), the substituted atom, the edges and the global state. As a check,
replacing all three quantities by their parent values must reproduce the parent prediction.

Example:
  python scripts/c2db/layer_ig.py --prepared data/c2db_prepared --runs data/checkpoints \
      --rules data/results/GATE_RULES_L400.json --out work/baselines/HIDDEN_IG.json
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

ARM = "physical20_nogate"
STATE = {"mode": None, "k": None, "rec": {}, "rep": {}}


def patch(layer, key):
    """Record or replace the output of one residual add (key = (block, 'v' | 'e' | 'u'))."""
    original = layer.call

    def call(inputs, *a, **kw):
        out = original(inputs, *a, **kw)
        if STATE["k"] is None or key[0] != STATE["k"]:
            return out
        if STATE["mode"] == "record":
            STATE["rec"][key[1]] = out
        elif STATE["mode"] == "replace":
            value = STATE["rep"][key[1]]
            out = out.with_flat_values(value) if hasattr(out, "flat_values") else value
        return out

    layer.call = call


def flat(x):
    return x.flat_values if hasattr(x, "flat_values") else x


def run(model, graphs):
    return model(model_inputs(graphs, arm=ARM), training=False)


def layer_ig(tf, model, jobs, pairs, std, steps, batch):
    from materon.c2db.models import backbone_of

    backbone = backbone_of(model)
    node_edge_adds = [l for l in backbone.layers if type(l).__name__ == "LazyAdd"]
    state_adds = [l for l in backbone.layers if type(l).__name__ == "Add"]
    if len(node_edge_adds) != 6 or len(state_adds) != 3:
        raise RuntimeError("unexpected MEGNet residual layout: %d LazyAdd, %d Add" % (len(node_edge_adds), len(state_adds)))
    for k in range(3):
        patch(node_edge_adds[2 * k], (k, "v"))
        patch(node_edge_adds[2 * k + 1], (k, "e"))
        patch(state_adds[k], (k, "u"))
    ts = (np.arange(steps) + 0.5) / steps
    result = {}
    for k in range(3):
        nb, site_a, edge_a, state_a, resid = [], [], [], [], []
        for i0 in range(0, len(pairs), batch):
            chunk = list(range(i0, min(i0 + batch, len(pairs))))
            P = [pairs[i][0] for i in chunk for _ in ts]
            Q = [pairs[i][1] for i in chunk for _ in ts]
            STATE.update(k=k, mode="record", rec={})
            f_p = run(model, P).numpy().reshape(-1)
            rp = {a: flat(b) for a, b in STATE["rec"].items()}
            STATE["rec"] = {}
            f_q = run(model, Q).numpy().reshape(-1)
            rq = {a: flat(b) for a, b in STATE["rec"].items()}
            edge_rows = STATE["rec"]["e"].row_lengths().numpy()
            if k == 0 and i0 == 0:
                STATE.update(mode="replace", rep=rp)
                result["full_restore_check_eV"] = float(np.abs(run(model, Q).numpy().reshape(-1) - f_p).max() * std)
            node_rows = [len(g["x"]) for g in Q]
            tg = np.tile(ts, len(chunk)).astype(np.float32)
            tv = {"v": np.repeat(tg, node_rows)[:, None], "e": np.repeat(tg, edge_rows)[:, None], "u": tg[:, None]}
            interp = {a: tf.constant((rp[a].numpy() + tv[a] * (rq[a].numpy() - rp[a].numpy())).astype(np.float32))
                      for a in "veu"}
            with tf.GradientTape() as tape:
                for a in "veu":
                    tape.watch(interp[a])
                STATE.update(mode="replace", rep=interp)
                total = tf.reduce_sum(run(model, Q))
            grads = tape.gradient(total, [interp[a] for a in "veu"])
            A = {a: ((rq[a].numpy() - rp[a].numpy()) * g.numpy()).sum(1) for a, g in zip("veu", grads)}
            ov = oe = 0
            for b, i in enumerate(chunk):
                n, e = node_rows[b * steps], int(edge_rows[b * steps])
                site = jobs[i]["site"]
                av = A["v"][ov:ov + steps * n].reshape(steps, n).mean(0) * std
                ae = A["e"][oe:oe + steps * e].reshape(steps, e).mean(0).sum() * std
                au = A["u"][b * steps:(b + 1) * steps].mean() * std
                ov += steps * n
                oe += steps * e
                d_f = float(f_q[b * steps] - f_p[b * steps]) * std
                neighbors = float(np.delete(av, site).sum())
                nb.append(neighbors)
                site_a.append(float(av[site]))
                edge_a.append(float(ae))
                state_a.append(float(au))
                resid.append(float(neighbors + av[site] + ae + au - d_f))
            STATE.update(mode=None, k=None)
        result["block%d" % k] = dict(nb_eV=nb, site_eV=site_a, edge_eV=edge_a, state_eV=state_a, resid_eV=resid)
    return result


def shares(block):
    a = np.abs(np.array([block["nb_eV"], block["site_eV"], block["edge_eV"], block["state_eV"]])).mean(1)
    return a / a.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--rules", required=True, help="channel-rules JSON (jobs list)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tags", nargs="+", default=["short", "L400"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    import tensorflow as tf

    from materon.c2db.models import load_member

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    data = Prepared(args.prepared)
    table = feature_table()
    jobs = [j for j in json.loads(Path(args.rules).read_text())["jobs"] if j["kind"] == "M"]
    pairs = []
    for j in jobs:
        p = copy_with_features(data.graphs[j["parent_id"]], table)
        pairs.append((p, copy_with_features(p, table, j["site"], Element(j["substituent"]).Z)))
    out = {}
    for tag in args.tags:
        for seed in args.seeds:
            model = load_member(member_dir(args.runs, tag, seed, ARM) / "best.weights.h5", ARM)
            r = layer_ig(tf, model, jobs, pairs, data.std, args.steps, args.batch)
            r.update(schedule=tag, seed=seed)
            out["%s_s%d" % (tag, seed)] = r
            s = shares(r["block2"])
            print(tag, seed, "block 3 shares neighbors/site/edges/state", np.round(s, 3),
                  "full-restore check %.1e eV" % r["full_restore_check_eV"], flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
