"""CGCNN configurations, training recipes and protocols (configs/cgcnn)."""
from __future__ import annotations

import json

from materon.attributes import config_dir

ALLOWED_MODEL_TYPES = {"cgcnn", "physics", "materon"}
ALLOWED_FEATURE_MODES = {"atomic_number", "full20", "grouped_pca", "shuffled_full20"}
ALLOWED_CONTEXT_MODES = {"none", "coord", "qeff", "emag", "all"}
ALLOWED_GATES = {"simple", "legacy_simple", "legacy", "refined"}


def read_json(name):
    return json.loads((config_dir() / "cgcnn" / name).read_text(encoding="utf-8"))


def validate_config(cid, cfg):
    if cfg.get("id") != cid:
        raise ValueError("Config key/id mismatch: %s" % cid)
    if cfg.get("model_type") not in ALLOWED_MODEL_TYPES:
        raise ValueError("Invalid model_type: %s" % cid)
    if cfg.get("feature_mode") not in ALLOWED_FEATURE_MODES:
        raise ValueError("Invalid feature_mode: %s" % cid)
    kind, gate, context = cfg["model_type"], cfg.get("input_gate"), cfg.get("context_mode", "none")
    if context not in ALLOWED_CONTEXT_MODES:
        raise ValueError("Invalid context_mode: %s" % cid)
    if kind == "cgcnn":
        if cfg["feature_mode"] != "atomic_number" or gate is not None or context != "none":
            raise ValueError("CGCNN contract failed: %s" % cid)
    elif kind == "physics":
        if cfg["feature_mode"] == "atomic_number" or gate is not None or context != "none":
            raise ValueError("Physics contract failed: %s" % cid)
    else:
        if gate is None or gate.get("type") not in ALLOWED_GATES:
            raise ValueError("Materon input gate invalid: %s" % cid)
        if context != "none" and gate.get("type") != "legacy":
            raise ValueError("Named context requires legacy input gate: %s" % cid)
        if gate.get("type") == "legacy":
            expected = {"none": (False, False, False), "coord": (True, False, False),
                        "qeff": (False, True, False), "emag": (False, False, True),
                        "all": (True, True, True)}[context]
            actual = (bool(gate.get("use_coord_proj", False)), bool(gate.get("use_qeff_in_gate", False)),
                      bool(gate.get("use_emag_in_gate", False)))
            if actual != expected:
                raise ValueError("Context flags mismatch for %s" % cid)
    return cfg


def all_configs():
    out = read_json("configurations.json")
    for key, value in out.items():
        validate_config(key, value)
    return out


def suites():
    cfg = all_configs()
    out = read_json("experiment_suites.json")
    for name, ids in out.items():
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate in suite %s" % name)
        bad = [x for x in ids if x not in cfg]
        if bad:
            raise ValueError("Unknown configs in suite %s: %s" % (name, bad))
    return out


def recipes():
    return read_json("recipes.json")


def protocol():
    """Phonons inner-pool selection protocol."""
    return read_json("phonons_protocol.json")


def transfer_protocol():
    """Calibrated transfer protocol for dielectric and JDFT2D."""
    return read_json("transfer_protocol.json")


def get_config(cid):
    return all_configs()[cid]


def get_recipe(rid):
    return recipes()[rid]
