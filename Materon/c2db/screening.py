"""Substitution screens of the application study (Sec. S13).

Parents: the four-atom M2X2 monolayers of the validation and test partitions
with X in {S, Se, Te} and M not a chalcogen and not V-Ni, one structure per
composition (first C2DB ID in sorted order). A candidate replaces the first
M site at the parent geometry (``copy_with_features``).

Every lock writes the selection rule, checkpoint SHA-256 digests, member
predictions and selected candidates to a JSON file, and the SHA-256 digest of
that text to a sidecar ``.sha256`` file, before any DFT calculation.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Element

from materon.c2db.graph import copy_with_features, model_inputs

EXCLUDED = {"V", "Cr", "Mn", "Fe", "Co", "Ni"}
CHALCOGENS = {"S", "Se", "Te"}
RADIOACTIVE = {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra"}
ARM = "physical20_environment_gate"


def screening_parents(split, graphs, exclude=("Al2S2",)):
    """{composition: (partition, id, M, X)} for the 16 application parents."""
    parents = {}
    for part in ("validation", "test"):
        for uid in sorted(split["ids"][part]):
            z = [int(k) for k in graphs[uid]["z"]]
            symbols = [Element.from_Z(k).symbol for k in z]
            elements = sorted(set(symbols))
            if len(z) != 4 or len(elements) != 2:
                continue
            x = [e for e in elements if e in CHALCOGENS]
            m = [e for e in elements if e not in CHALCOGENS]
            if len(x) != 1 or len(m) != 1 or symbols.count(m[0]) != 2 or m[0] in EXCLUDED:
                continue
            composition = m[0] + "2" + x[0] + "2"
            if composition not in exclude and composition not in parents:
                parents[composition] = (part, uid, m[0], x[0])
    return parents


def has_feature_row(table, z):
    return bool(np.all(np.isfinite(table[z])) and np.any(table[z] != 0))


def usable_substituents(table):
    """Metals of rows 3-6, not V-Ni, radioactive or lanthanide, with an attribute row."""
    def usable(e):
        return (e.is_metal and e.row in (3, 4, 5, 6) and e.symbol not in EXCLUDED | RADIOACTIVE
                and not e.is_lanthanoid and has_feature_row(table, e.Z))
    return [Element.from_Z(z) for z in range(1, 84) if usable(Element.from_Z(z))]


def adjacent_group_substituent(symbol, table):
    """Contrast-set rule: same-period neighbor (group +-1) with the larger electronegativity difference."""
    e = Element(symbol)
    best = None
    for dz in (-1, 1):
        c = Element.from_Z(e.Z + dz)
        if (c.row != e.row or c.symbol in RADIOACTIVE or c.symbol in EXCLUDED or not c.is_metal
                or c.is_lanthanoid or not has_feature_row(table, c.Z)):
            continue
        d = abs(c.X - e.X)
        if best is None or d > best[0] + 1e-9 or (abs(d - best[0]) <= 1e-9 and dz < 0):
            best = (d, c.symbol)
    return best[1] if best else None


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_members(checkpoints):
    from materon.c2db.models import load_member

    return [load_member(path, ARM) for path in checkpoints]


def predict_pair_deltas(model, pairs, mean, std, pairs_per_call=32):
    """F(q) - F(P) for a list of (parent, candidate) graphs, ``pairs_per_call`` pairs per forward pass."""
    out = np.zeros(len(pairs))
    for n in range(0, len(pairs), pairs_per_call):
        chunk = pairs[n:n + pairs_per_call]
        graphs = [g for pq in chunk for g in pq]
        pr = model(model_inputs(graphs, arm=ARM, mask=1.), training=False).numpy().reshape(-1) * std + mean
        out[n:n + len(chunk)] = pr[1::2] - pr[0::2]
    return out


def ensemble_validation_mae(member_dirs, val_ids, val_labels):
    """MAE of the member-mean validation prediction (record weights): the confidence threshold."""
    predictions = [json.loads((Path(d) / "VALIDATION_PREDICTIONS.json").read_text()) for d in member_dirs]
    for p in predictions:
        if list(p.keys()) != list(val_ids):
            raise ValueError("validation prediction order differs from the split")
    mean_prediction = np.mean([list(p.values()) for p in predictions], axis=0)
    return float(np.mean(np.abs(mean_prediction - np.array([val_labels[i] for i in val_ids]))))


def candidate_pool(parents, graphs, table, done=()):
    """All usable metal substituents on the first M site of every parent, minus ``done`` (parent_id, symbol)."""
    subs = usable_substituents(table)
    pairs, meta = [], []
    for composition, (part, uid, m, x) in sorted(parents.items()):
        parent = copy_with_features(graphs[uid], table)
        site = [int(k) for k in parent["z"]].index(Element(m).Z)
        for e in subs:
            if e.symbol == m or (uid, e.symbol) in done:
                continue
            pairs.append((parent, copy_with_features(parent, table, site, e.Z)))
            meta.append(dict(parent_id=uid, partition=part, parent=composition, cation=m, anion=x,
                             substituent=e.symbol, substituted_site=site, candidate=f"{e.symbol}{m}{x}2",
                             electronegativity_baseline=float(e.X - Element(m).X)))
    return pairs, meta


def select_confident(members_pred, meta, threshold, limit):
    """Members agree in sign and |mean| >= threshold; one per parent by |mean|/SD; top ``limit``."""
    m, s = members_pred.mean(0), members_pred.std(0, ddof=1)
    agree = (np.sign(members_pred) == np.sign(m)).all(0)
    eligible = [i for i in range(len(meta)) if agree[i] and abs(m[i]) >= threshold]
    best = {}
    for i in eligible:
        key = meta[i]["parent_id"]
        score = abs(m[i]) / max(s[i], 1e-6)
        if key not in best or score > best[key][0]:
            best[key] = (score, i)
    return sorted(best.values(), reverse=True)[:limit], eligible, best


def select_uncertain(members_pred, meta, used_parents, limit):
    """On the remaining parents, the substitution with the largest member SD; top ``limit`` by SD."""
    s = members_pred.std(0, ddof=1)
    unc = {}
    for i in range(len(meta)):
        key = meta[i]["parent_id"]
        if key in used_parents:
            continue
        if key not in unc or s[i] > unc[key][0]:
            unc[key] = (s[i], i)
    return sorted(unc.values(), reverse=True)[:limit]


def geometry_record(graph):
    return {"lattice": np.asarray(graph["lattice"]).tolist(), "coords": np.asarray(graph["coords"]).tolist(),
            "z": [int(k) for k in graph["z"]]}


def write_lock(out_dir, name, payload):
    """Write the locked JSON (indent 1) and the SHA-256 of its text; returns the digest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    payload = dict(payload, locked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    text = json.dumps(payload, indent=1)
    (out / (name + ".json")).write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (out / (name + ".sha256")).write_text(digest + "\n")
    return digest
