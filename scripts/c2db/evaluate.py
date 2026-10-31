#!/usr/bin/env python3
"""One-time evaluation of trained C2DB predictors (Tables S5, S24, S26; Sec. S8, S11).

For every arm and seed under <runs>/<tag>: validation and test MAE/RMSE/R^2,
the within-composition centered test metrics (Sec. S8), and the member-mean
ensemble. For the environment-gate arm additionally: the Al2S2 four-candidate
screen (B, Ga on Al site 0; O, Se on S site 2) and the single-channel
restoration of the unchanged Al1 gate for B and Ga (Table S24). With
--contrast-lock, post hoc predicted changes for the locked contrast set.

Example:
  python scripts/c2db/evaluate.py --prepared work/c2db/prepared --runs work/c2db/runs --tag L400 \
      --arms physical20_environment_gate physical20_nogate --out work/c2db/EVAL_L400.json
"""
import argparse
import json

import numpy as np
from pymatgen.core import Element

from materon.attributes import feature_table
from materon.c2db.dataset import Prepared, metrics, read_json, within_composition_metrics, write_json
from materon.c2db.graph import copy_with_features, model_inputs
from materon.c2db.training import SEEDS, member_dir
from materon.runtime import configure_tensorflow

ANCHOR = "Al2S2-6a3995b58556"
AL2S2_CANDIDATES = {"B": (0, 5), "Ga": (0, 31), "O": (2, 8), "Se": (2, 34)}


def predict(model, arm, graphs, mean, std, bs=64):
    kw = {} if arm == "physical20_nogate" else {"mask": 1.}
    return np.concatenate([model(model_inputs(graphs[s:s + bs], arm=arm, **kw), training=False).numpy().reshape(-1)
                           for s in range(0, len(graphs), bs)]) * std + mean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--tag", required=True, help="short, L400, FT1 or FT2")
    parser.add_argument("--arms", nargs="+", default=["physical20_environment_gate", "physical20_nogate"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--contrast-lock", default=None, help="locked contrast-set JSON (post hoc predictions)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import tensorflow as tf

    from materon.c2db.models import load_member
    from materon.c2db.restoration import single_site_restoration

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    data = Prepared(args.prepared)
    mean, std, graphs = data.mean, data.std, data.graphs
    table = feature_table()
    parent = copy_with_features(graphs[ANCHOR], table)
    screen = {k: copy_with_features(parent, table, site, z) for k, (site, z) in AL2S2_CANDIDATES.items()}
    contrast = []
    if args.contrast_lock:
        for c in read_json(args.contrast_lock)["candidates"]:
            p = copy_with_features(graphs[c["parent_id"]], table)
            contrast.append((p, copy_with_features(p, table, c["substituted_site"], Element(c["substituent"]).Z)))
    composition = data.split["reduced_composition_by_id"]

    result = {"tag": args.tag, "runs": [], "ensemble": {}}
    for arm in args.arms:
        preds = {"validation": [], "test": []}
        for seed in args.seeds:
            run = member_dir(args.runs, args.tag, seed, arm)
            model = load_member(run / "best.weights.h5", arm)
            record = {"arm": arm, "seed": seed}
            if (run / "RESULT.json").exists():
                info = read_json(run / "RESULT.json")
                record.update(best_epoch=info.get("best_epoch"), epochs_run=info.get("epochs_run"))
            for part in ("validation", "test"):
                ids = data.ids(part)
                y = data.targets(part)
                p = predict(model, arm, [graphs[i] for i in ids], mean, std)
                record[part] = metrics(y, p)
                preds[part].append(p)
                if part == "test":
                    record["test_within_composition"] = within_composition_metrics(ids, y, p, composition)
            if arm == "physical20_environment_gate":
                gs = [parent] + list(screen.values())
                pr = predict(model, arm, gs, mean, std)
                record["al2s2_screen_delta_eV"] = {k: float(pr[n + 1] - pr[0]) for n, k in enumerate(screen)}
                record["Al1"] = {k: single_site_restoration(tf, model, parent, screen[k], 1, mean, std)
                                 for k in ("B", "Ga")}
            if contrast:
                cp = predict(model, arm, [g for pq in contrast for g in pq], mean, std)
                record["contrast_delta_eV"] = (cp[1::2] - cp[0::2]).tolist()
            result["runs"].append(record)
            print(arm, seed, round(record["validation"]["mae"], 4), round(record["test"]["mae"], 4), flush=True)
        result["ensemble"][arm] = {part: metrics(data.targets(part), np.mean(preds[part], axis=0))
                                   for part in ("validation", "test")}
    write_json(args.out, result, indent=1)


if __name__ == "__main__":
    main()
