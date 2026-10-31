#!/usr/bin/env python3
"""Predicted changes F(q) - F(P) of listed candidates by one or more ensembles (inference only).

Used to compare L400, FT1 and FT2 on the locked candidate sets (Sec. III D).
The candidate file is a lock JSON (key "candidates") or a list; each entry needs
parent_id, substituted_site and substituent.

Example:
  python scripts/c2db/predict_pairs.py --prepared P --runs runs --tags L400 FT1 FT2 \
      --candidates locks/round2/ROUND2_PREREGISTERED.json --out work/c2db/ROUND2_ALL_MODELS.json
"""
import argparse

import numpy as np
from pymatgen.core import Element

from materon.attributes import feature_table
from materon.c2db.dataset import Prepared, read_json, write_json
from materon.c2db.graph import copy_with_features, model_inputs
from materon.c2db.screening import ARM, sha256_file
from materon.c2db.training import SEEDS, member_dir
from materon.runtime import configure_tensorflow


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import tensorflow as tf

    from materon.c2db.models import load_member

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    data = Prepared(args.prepared)
    table = feature_table()
    candidates = read_json(args.candidates)
    candidates = candidates["candidates"] if isinstance(candidates, dict) else candidates
    graphs = []
    for c in candidates:
        p = copy_with_features(data.graphs[c["parent_id"]], table)
        graphs += [p, copy_with_features(p, table, c["substituted_site"], Element(c["substituent"]).Z)]
    result = {"candidates": [c.get("candidate", c.get("name")) for c in candidates], "tags": {}}
    for tag in args.tags:
        record = {"members_eV": {}, "checkpoint_sha256": {}}
        for seed in args.seeds:
            path = member_dir(args.runs, tag, seed) / "best.weights.h5"
            model = load_member(path)
            pr = np.concatenate([model(model_inputs(graphs[k:k + 64], arm=ARM, mask=1.), training=False)
                                 .numpy().reshape(-1) for k in range(0, len(graphs), 64)]) * data.std + data.mean
            record["members_eV"][str(seed)] = (pr[1::2] - pr[0::2]).tolist()
            record["checkpoint_sha256"][str(seed)] = sha256_file(path)
        record["mean_eV"] = np.mean(list(record["members_eV"].values()), axis=0).tolist()
        result["tags"][tag] = record
        print(tag, np.round(record["mean_eV"], 3), flush=True)
    write_json(args.out, result, indent=1)


if __name__ == "__main__":
    main()
