#!/usr/bin/env python3
"""Locked substitution screens (Sec. S11.4, S13; Table S35). Inference only.

--kind contrast  : one adjacent-group substitution per parent (element properties
                   only), predicted by one checkpoint (seed-42 short model).
--kind contrast-seeds : predictions of an ensemble for an existing contrast lock.
--kind round1    : all usable metal substitutions minus earlier candidates;
                   confident = five members agree in sign and |mean| >= ensemble
                   validation MAE; one per parent by |mean|/SD; up to 16 (FT1 ensemble).
--kind round2    : as round 1 from the remaining pool: 8 confident candidates,
                   then 8 uncertain ones on the other parents by member SD (FT2 ensemble).
Each lock is written with the SHA-256 digest of its text before any DFT.

Examples:
  python scripts/c2db/screen.py --kind contrast --prepared P --checkpoint runs/short/physical20_environment_gate_s42/best.weights.h5 --out locks/contrast
  python scripts/c2db/screen.py --kind round1 --prepared P --runs runs --tag FT1 --previous locks/contrast/CONTRAST_PREREGISTERED.json --out locks/round1
  python scripts/c2db/screen.py --kind round2 --prepared P --runs runs --tag FT2 --previous locks/contrast/CONTRAST_PREREGISTERED.json locks/round1/ROUND1_PREREGISTERED.json --out locks/round2
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Element

from materon.attributes import feature_table
from materon.c2db import screening as S
from materon.c2db.dataset import Prepared, read_json, write_json
from materon.c2db.graph import copy_with_features, model_inputs
from materon.c2db.training import SEEDS, member_dir
from materon.runtime import configure_tensorflow

CONTRAST_RULE = {
    "parents": "C2DB validation/test structures with four atoms, M2X2, X in {S, Se, Te}, M not a chalcogen and "
               "not in {V, Cr, Mn, Fe, Co, Ni}; one structure per composition (first C2DB ID in sorted order); "
               "Al2S2 keeps its earlier screen",
    "substitution": "first M site replaced by the same-period element of an adjacent group (valence change of one "
                    "electron); of the two neighbours the one with the larger Pauling electronegativity difference; "
                    "metals only, not V-Ni, not radioactive; selection uses element properties only",
    "geometry_for_model": "parent C2DB geometry with the substituted species",
}
ROUND1_RULE = ("Every metal substituent on the first M site (rows 3-6, not V-Ni, not radioactive, not lanthanide, "
               "with feature row) of the contrast-set parents; contrast candidates and Al2S2 excluded. Candidates "
               "whose five members agree in sign and whose ensemble-mean |change| is at least the ensemble "
               "validation MAE; one per parent (largest |mean|/SD); at most 16 by |mean|/SD.")
ROUND2_RULE = ("As round 1, excluding contrast and round-1 candidates. Half confident: chosen as in round 1, top 8 "
               "by |mean|/SD. Half uncertain: on the remaining parents, the substitution with the largest member "
               "SD, one per parent, top 8 by SD.")


def contrast(args, data, table):
    from materon.c2db.models import load_member

    model = load_member(args.checkpoint)
    graphs = data.graphs
    rows = []
    for composition, (part, uid, m, x) in sorted(S.screening_parents(data.split, graphs).items()):
        sub = S.adjacent_group_substituent(m, table)
        if sub is None:
            continue
        parent = copy_with_features(graphs[uid], table)
        site = [int(k) for k in parent["z"]].index(Element(m).Z)
        cand = copy_with_features(parent, table, site, Element(sub).Z)
        p, c = model(model_inputs([parent, cand], arm=S.ARM, mask=1.), training=False).numpy().reshape(-1) \
            * data.std + data.mean
        rows.append({"parent_id": uid, "partition": part, "parent": composition, "cation": m, "anion": x,
                     "substituent": sub, "substituted_site": site, "candidate": f"{sub}{m}{x}2",
                     "model_parent_eV": float(p), "model_candidate_eV": float(c), "predicted_change_eV": float(c - p),
                     "electronegativity_baseline": float(Element(sub).X - Element(m).X),
                     **S.geometry_record(graphs[uid])})
    digest = S.write_lock(args.out, "CONTRAST_PREREGISTERED",
                          {"rule": CONTRAST_RULE, "checkpoint_sha256": S.sha256_file(args.checkpoint),
                           "candidates": rows})
    return digest, rows


def contrast_seeds(args, data, table):
    from materon.c2db.models import load_member

    lock_path = Path(args.previous[0])
    lock = read_json(lock_path)
    pairs = []
    for c in lock["candidates"]:
        parent = copy_with_features(data.graphs[c["parent_id"]], table)
        pairs.append((parent, copy_with_features(parent, table, c["substituted_site"], Element(c["substituent"]).Z)))
    predictions = {}
    for seed in args.seeds:
        path = member_dir(args.runs, args.tag, seed) / "best.weights.h5"
        model = load_member(path)
        values = S.predict_pair_deltas(model, pairs, data.mean, data.std, pairs_per_call=1)
        predictions[str(seed)] = {"checkpoint_sha256": S.sha256_file(path), "predicted_change_eV": values.tolist()}
    record = {"locked_file_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
              "candidates": [c["candidate"] + "__" + c["parent_id"] for c in lock["candidates"]],
              "seeds": predictions,
              "ensemble_mean_eV": np.mean([v["predicted_change_eV"] for v in predictions.values()], axis=0).tolist()}
    digest = S.write_lock(args.out, "CONTRAST_SEED_PREDICTIONS", record)
    return digest, record["candidates"]


def ensemble_round(args, data, table, kind):
    from materon.c2db.models import load_member

    previous = [read_json(p) for p in args.previous]
    done = {(c["parent_id"], c["substituent"]) for lock in previous for c in lock["candidates"]}
    runs = [member_dir(args.runs, args.tag, s) for s in args.seeds]
    threshold = S.ensemble_validation_mae(runs, data.ids("validation"), data.labels("validation"))
    parents = S.screening_parents(data.split, data.graphs)
    pairs, meta = S.candidate_pool(parents, data.graphs, table, done)
    P = np.zeros((len(runs), len(pairs)))
    shas = []
    for k, run in enumerate(runs):
        P[k] = S.predict_pair_deltas(load_member(run / "best.weights.h5"), pairs, data.mean, data.std)
        shas.append(S.sha256_file(run / "best.weights.h5"))
    m, s = P.mean(0), P.std(0, ddof=1)
    if kind == "round1":
        chosen, eligible, best = S.select_confident(P, meta, threshold, 16)
        chosen = [(score, i, "confident") for score, i in chosen]
    else:
        confident, eligible, best = S.select_confident(P, meta, threshold, 8)
        chosen = [(score, i, "confident") for score, i in confident]
        used = {meta[i]["parent_id"] for _, i, _ in chosen}
        chosen += [(score, i, "uncertain") for score, i in S.select_uncertain(P, meta, used, 8)]
    rows = [{**meta[i], "selection": label, "pred_members_eV": P[:, i].tolist(), "pred_mean_eV": float(m[i]),
             "pred_sd_eV": float(s[i]), "confidence": float(score), **S.geometry_record(data.graphs[meta[i]["parent_id"]])}
            for score, i, label in chosen]
    payload = {"rule": ROUND1_RULE if kind == "round1" else ROUND2_RULE, "tag": args.tag,
               "checkpoint_sha256": dict(zip(map(str, args.seeds), shas)),
               "member_validation_mae_eV": [read_json(r / "RESULT.json")["scores"]["validation"]["mae"] for r in runs],
               "ensemble_validation_mae_eV": threshold, "pool_size": len(pairs), "eligible": len(eligible),
               "eligible_parents": len(best),
               "previous_lock_sha256": [S.sha256_file(p) for p in args.previous], "candidates": rows}
    name = "ROUND1_PREREGISTERED" if kind == "round1" else "ROUND2_PREREGISTERED"
    digest = S.write_lock(args.out, name, payload)
    write_json(Path(args.out) / (name.split("_")[0] + "_POOL.json"),
               [{**meta[i], "pred_members_eV": P[:, i].tolist()} for i in range(len(pairs))], indent=None)
    return digest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=["contrast", "contrast-seeds", "round1", "round2"])
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", help="single checkpoint (contrast)")
    parser.add_argument("--runs", help="runs root (ensembles)")
    parser.add_argument("--tag", help="ensemble tag: short, L400, FT1, FT2")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--previous", nargs="*", default=[], help="earlier locks whose candidates are excluded")
    parser.add_argument("--out", required=True, help="new lock directory")
    args = parser.parse_args()

    import tensorflow as tf

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    data = Prepared(args.prepared)
    table = feature_table()
    if args.kind == "contrast":
        digest, rows = contrast(args, data, table)
    elif args.kind == "contrast-seeds":
        digest, rows = contrast_seeds(args, data, table)
    else:
        digest, rows = ensemble_round(args, data, table, args.kind)
    print(json.dumps({"sha256": digest, "candidates": len(rows)}))


if __name__ == "__main__":
    main()
