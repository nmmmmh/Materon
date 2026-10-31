#!/usr/bin/env python3
"""Channel rules over all single substitutions of the application parents (Sec. S3.2, Fig. 3c, Table S6).

For every parent, every usable metal on the first M site and S/Se/Te on the
first X site are substituted at the parent geometry. For each ensemble member:
native change F(q) - F(P); R_env[j], the change removed when channel j of the
gates of all unsubstituted atoms is restored to its parent value; env_joint,
the change removed when all their channels are restored; dx[j], the attribute
difference substituent minus host.

The summary reports, per member and over the metal-site substitutions, the
Spearman correlation between R_env[j] and the native change and the
environment share mean|env_joint| / mean|native| (definitions of Table S6).

Example:
  python scripts/c2db/channel_rules.py --prepared work/c2db/prepared --runs work/c2db/runs --tag L400 \
      --out work/c2db/RULES_L400.json
"""
import argparse

import numpy as np
from pymatgen.core import Element
from scipy.stats import spearmanr

from materon.attributes import RAW_FEATURE_COLUMNS, feature_table
from materon.c2db.dataset import Prepared, write_json
from materon.c2db.graph import copy_with_features
from materon.c2db.screening import screening_parents, usable_substituents
from materon.c2db.training import SEEDS, member_dir
from materon.runtime import configure_tensorflow


def summarize(jobs, n_members):
    metal = [j for j in jobs if j["kind"] == "M"]
    summary = []
    for m in range(n_members):
        native = np.array([j["native_eV"][m] for j in metal])
        R = np.array([j["R_env_eV"][m] for j in metal])
        joint = np.array([j["env_joint_eV"][m] for j in metal])
        summary.append({"member": m, "n_substitutions": len(metal),
                        "spearman_R_vs_native": {RAW_FEATURE_COLUMNS[c]: float(spearmanr(R[:, c], native).correlation)
                                                 for c in range(20)},
                        "environment_share": float(np.mean(np.abs(joint)) / np.mean(np.abs(native)))})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--tag", required=True, help="short or L400")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import tensorflow as tf

    from materon.c2db.models import load_member
    from materon.c2db.restoration import environment_restoration

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    data = Prepared(args.prepared)
    mean, std, graphs = data.mean, data.std, data.graphs
    table = feature_table()
    parents = screening_parents(data.split, graphs)
    # Metal substituents of the screens plus the parent cations themselves.
    subs = sorted({e.symbol for e in usable_substituents(table)} | {m for (_, _, m, _) in parents.values()},
                  key=lambda e: Element(e).Z)
    jobs = []
    for composition, (_, pid, cation, anion) in sorted(parents.items()):
        z = [int(k) for k in graphs[pid]["z"]]
        ms, xs = z.index(Element(cation).Z), z.index(Element(anion).Z)
        for e in subs:
            if e != cation:
                jobs.append(dict(parent_id=pid, parent=composition, site=ms, kind="M", host=cation, substituent=e))
        for e in ("S", "Se", "Te"):
            if e != anion:
                jobs.append(dict(parent_id=pid, parent=composition, site=xs, kind="X", host=anion, substituent=e))
    for j in jobs:
        j["dx"] = (table[Element(j["substituent"]).Z] - table[Element(j["host"]).Z]).astype(float).tolist()

    models = [load_member(member_dir(args.runs, args.tag, s) / "best.weights.h5") for s in args.seeds]
    for j in jobs:
        p = copy_with_features(graphs[j["parent_id"]], table)
        q = copy_with_features(p, table, j["site"], Element(j["substituent"]).Z)
        j.update(native_eV=[], R_env_eV=[], env_joint_eV=[], check_eV=[])
        for model in models:
            r = environment_restoration(tf, model, p, q, j["site"], mean, std)
            j["native_eV"].append(r["native_eV"])
            j["R_env_eV"].append(r["R_env_eV"])
            j["env_joint_eV"].append(r["env_joint_eV"])
            j["check_eV"].append(r["check_eV"])
    write_json(args.out, {"tag": args.tag, "seeds": args.seeds, "channels": RAW_FEATURE_COLUMNS,
                          "summary": summarize(jobs, len(models)), "jobs": jobs})
    print(len(jobs), "substitutions; max |backbone - model| =", max(max(j["check_eV"]) for j in jobs))


if __name__ == "__main__":
    main()
