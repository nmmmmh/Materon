#!/usr/bin/env python3
"""Locked Ta2Te2 work-function designs (Sec. III E, S14.1, Table S34). Inference only.

Round 1 (mechanism after the contrast set and round 1): designs releasing two or
three nominal electrons per cell (Re or Os on one Ta site, W on both, W and Re)
and two controls (isoelectronic Nb, hole-doping Hf). Rigid-band prediction
Phi = Phi(Ta2Te2) - dE_F(dN) from the parent total DOS (DOSCAR of the parent
static calculation).
Round 2 (application criterion, Schottky-Mott n-type contact to MoS2): one
released electron per cell, Mo on one Ta site and I on one Te site; rigid-band
Phi = 5.009 - 1.23 eV.
Model predictions of every member of the given ensembles are recorded.

Examples:
  python scripts/c2db/lock_design.py --round 1 --prepared P --runs runs --tags L400 FT1 FT2 \
      --parent-doscar contrast_dft/Ta2Te2__14789ef267c9/static24/DOSCAR --out locks/design1
  python scripts/c2db/lock_design.py --round 2 --prepared P --runs runs --tags FT2 \
      --first-lock locks/design1/DESIGN_PREREGISTERED.json --out locks/design2
"""
import argparse
import hashlib
import json

import numpy as np
from pymatgen.core import Element

from materon.attributes import config_dir, feature_table
from materon.c2db.dataset import Prepared, read_json
from materon.c2db.graph import copy_with_features, model_inputs
from materon.c2db.screening import ARM, geometry_record, write_lock
from materon.c2db.training import SEEDS, member_dir
from materon.runtime import configure_tensorflow

ROUND1_RULE = (
    "Mechanism-guided low-work-function designs on Ta2Te2 (parent lattice), locked before DFT. The Ta2Te2 Fermi "
    "level lies at the lower edge of a pseudogap (parent DOS ~0.2-0.6 states/eV/cell from E_F+0.7 to E_F+2.3 eV), "
    "so electrons that a substituent releases into the host band raise E_F by ~1 eV each. Designs add more released "
    "electrons per cell; controls test the electron-count origin (isoelectronic Nb: no change; Hf: hole doping, "
    "small rise). Rigid-band prediction: Phi = Phi(Ta2Te2) - dE_F(dN) from the parent DOS with nominal dN.")
ROUND1_CRITERIA = {
    "better_material": "design DFT Phi below 3.511 eV (lowest of all 48 systems computed before this lock: Sc2Se2)",
    "stability": "formation energy per atom relative to the elemental ground states (PBE, same protocol) below zero; "
                 "relaxed structure keeps the parent connectivity (no atom displaced > 1 A out of its layer); "
                 "metallic (nonzero DOS at E_F)",
    "mechanism": "controls: |dPhi(NbTaTe2)| < 0.3 eV; dPhi(HfTaTe2) > 0. Designs: dN_eff (rigid-band inversion, "
                 "chalcogen-core reference) >= 1.5 for the two-electron designs"}
ROUND2_RULE = (
    "Stability-aware low-work-function designs on Ta2Te2 (parent lattice), locked before DFT. The first electron "
    "released into the Ta2Te2 host band crosses the pseudogap and lowers Phi by ~1.1-1.5 eV; further electrons are "
    "pinned by the substituent d band and raise the formation energy. Designs release one electron per cell: Mo on "
    "one Ta site and I on one Te site. Criteria: Phi below the MoS2 electron affinity, 4.253 eV, at the parent "
    "lattice and after in-plane relaxation; formation energy per atom relative to the elemental ground states "
    "below zero; metallic; parent connectivity kept. Rigid-band prediction for one released electron: "
    "Phi = 5.009 - 1.23 = 3.78 eV.")
RIGID_ONE_ELECTRON_EV = 1.23


def ensemble_deltas(runs, tags, seeds, graphs, mean, std):
    from materon.c2db.models import load_member

    preds = {}
    for tag in tags:
        members = []
        for seed in seeds:
            model = load_member(member_dir(runs, tag, seed) / "best.weights.h5")
            pr = model(model_inputs(graphs, arm=ARM, mask=1.), training=False).numpy().reshape(-1) * std + mean
            members.append((pr[1::2] - pr[0::2]).tolist())
        preds[tag] = np.array(members)
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, choices=[1, 2], required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--parent-doscar", help="round 1: DOSCAR of the Ta2Te2 parent static calculation")
    parser.add_argument("--first-lock", help="round 2: the round-1 design lock")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import tensorflow as tf

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    cfg = read_json(config_dir() / "c2db" / "probes.json")["design"]
    pid, phi_parent = cfg["parent_id"], float(cfg["parent_phi_eV"])
    data = Prepared(args.prepared)
    table = feature_table()
    z = [int(k) for k in data.graphs[pid]["z"]]
    ta = [k for k, v in enumerate(z) if v == 73]
    te = [k for k, v in enumerate(z) if v == 52]
    if args.round == 1:
        designs = [("ReTaTe2", {ta[0]: "Re"}, 2, "design"), ("OsTaTe2", {ta[0]: "Os"}, 3, "design"),
                   ("W2Te2", {ta[0]: "W", ta[1]: "W"}, 2, "design"), ("WReTe2", {ta[0]: "W", ta[1]: "Re"}, 3, "design"),
                   ("NbTaTe2", {ta[0]: "Nb"}, 0, "control"), ("HfTaTe2", {ta[0]: "Hf"}, -1, "control")]
    else:
        designs = [("MoTaTe2", {ta[0]: "Mo"}, 1, "design"), ("Ta2TeI", {te[0]: "I"}, 1, "design")]
    parent = copy_with_features(data.graphs[pid], table)
    graphs = []
    for _, subs, _, _ in designs:
        q = parent
        for site, element in subs.items():
            q = copy_with_features(q, table, site, Element(element).Z)
        graphs += [parent, q]
    preds = ensemble_deltas(args.runs, args.tags, args.seeds, graphs, data.mean, data.std)

    rows = []
    for k, (name, subs, dn, kind) in enumerate(designs):
        row = {"name": name, "kind": kind, "substitutions": {str(s): e for s, e in subs.items()}}
        if args.round == 1:
            from materon.dft.dos import fermi_shift

            shift = fermi_shift(args.parent_doscar, dn)
            row.update(nominal_dN=dn, rigid_band_dPhi_eV=-shift, rigid_band_Phi_eV=phi_parent - shift)
        else:
            row.update(released_electrons=dn, rigid_band_Phi_eV=phi_parent - RIGID_ONE_ELECTRON_EV)
        row["model_dPhi_eV"] = {t: {"members": preds[t][:, k].tolist(), "mean": float(preds[t][:, k].mean())}
                                for t in args.tags}
        row.update(geometry_record(data.graphs[pid]))
        rows.append(row)
    if args.round == 1:
        payload = {"rule": ROUND1_RULE, "parent": pid, "parent_phi_eV": phi_parent, "criteria": ROUND1_CRITERIA,
                   "designs": rows}
        digest = write_lock(args.out, "DESIGN_PREREGISTERED", payload)
    else:
        first = read_json(args.first_lock)
        payload = {"rule": ROUND2_RULE, "parent": pid, "parent_phi_eV": phi_parent,
                   "mos2_electron_affinity_eV": float(cfg["mos2_electron_affinity_eV"]),
                   "first_round_lock_sha256": hashlib.sha256(json.dumps(first, indent=1).encode()).hexdigest(),
                   "known_before_lock": "first design round (Phi, lattice relaxation, formation energies), WTaTe2 "
                                        "and IrTaTe2 DFT, MoS2 band edges",
                   "designs": rows}
        digest = write_lock(args.out, "DESIGN2_PREREGISTERED", payload)
    print(digest)
    for r in rows:
        print(r["name"], "rigid Phi %.2f" % r["rigid_band_Phi_eV"],
              " ".join("%s %+.2f" % (t, r["model_dPhi_eV"][t]["mean"]) for t in args.tags))


if __name__ == "__main__":
    main()
