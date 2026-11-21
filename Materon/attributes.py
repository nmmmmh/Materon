"""The 20 named elemental attributes (Table S2) and their reference tables.

``configs/attributes/physical20.csv`` holds the standardized attribute values
for Z = 1..100; ``grouped_pca12.csv`` holds the 12 frozen grouped principal
components of Materon-PCA (Table S3), whose loadings are listed in
``feature_metadata.json``. Values are dimensionless; no further scaling is
applied by any model.

Two readers are kept on purpose: ``load_feature_bundle`` (pandas) is the reader
used by the CGCNN/MEGNet benchmarks and ``feature_table`` (csv module, strict
validation) is the reader used by the C2DB study. They read the same file; each
stack keeps its own reader so that float parsing is identical to the runs
reported in the paper.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

RAW_FEATURE_COLUMNS = [
    "density", "atom_melting_point", "atom_specific_heat_capacity",
    "atomic_boiling_point", "atomic_fusion_heat", "atomic_thermal_cond",
    "atomic_ip", "atomic_ea", "atomic_hfomo", "atomic_lfumo", "X",
    "atomic_pol", "atomic_nvalences", "atomic_ndval", "atomic_npval",
    "atomic_nsval", "atomic_radius", "atomic_covalent Radius",
    "atomic_volume", "atomic_weight",
]

# Symbols used in the paper, in channel order (Table S2).
SYMBOLS = [
    "rho", "T_m", "c_p", "T_b", "dH_f", "kappa", "E_IP", "E_EA", "eps_H", "eps_L",
    "chi", "alpha", "N_v", "N_d", "N_p", "N_s", "r_a", "r_c", "V_a", "m_a",
]

SHUFFLE_SEED = 20260725  # attribute-shuffled control (Table S7)


def config_dir() -> Path:
    """Repository ``configs`` directory; override with MATERON_CONFIG_DIR."""
    env = os.environ.get("MATERON_CONFIG_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "configs"


def attribute_dir() -> Path:
    return config_dir() / "attributes"


def physical20_path() -> Path:
    return attribute_dir() / "physical20.csv"


def feature_source_sha256(mode: str) -> str:
    name = "grouped_pca12.csv" if mode == "grouped_pca" else "physical20.csv"
    return hashlib.sha256((attribute_dir() / name).read_bytes()).hexdigest()


def load_feature_bundle(mode: str = "full20", shuffle_seed: Optional[int] = None) -> Dict:
    """Benchmark reader: table indexed by Z for full20, grouped_pca or shuffled_full20."""
    import pandas as pd

    raw = pd.read_csv(attribute_dir() / "physical20.csv").sort_values("Z")
    if list(raw["Z"].astype(int)) != list(range(1, 101)):
        raise ValueError("Element table must contain Z=1..100 exactly once")
    if mode in {"full20", "shuffled_full20"}:
        frame = raw[["Z"] + RAW_FEATURE_COLUMNS].copy()
        names = list(RAW_FEATURE_COLUMNS)
    elif mode == "grouped_pca":
        frame = pd.read_csv(attribute_dir() / "grouped_pca12.csv").sort_values("Z")
        names = [c for c in frame.columns if c != "Z"]
    else:
        raise ValueError("Unknown feature mode: %s" % mode)
    values = frame[names].to_numpy(np.float32)
    if mode == "shuffled_full20":
        # Rows (elements) are permuted, so each element receives another element's attributes.
        rng = np.random.RandomState(SHUFFLE_SEED if shuffle_seed is None else int(shuffle_seed))
        values = values[rng.permutation(len(values))]
    if not np.all(np.isfinite(values)):
        raise ValueError("Feature table contains NaN or infinity")
    table = np.zeros((101, len(names)), dtype=np.float32)
    for z, row in zip(frame.Z.astype(int), values):
        table[z] = row
    metadata = json.loads((attribute_dir() / "feature_metadata.json").read_text(encoding="utf-8"))
    return {"mode": mode, "table": table, "names": names, "metadata": metadata}


def map_atomic_numbers(z_values, mode="full20", shuffle_seed=None):
    bundle = load_feature_bundle(mode, shuffle_seed)
    z = np.asarray(z_values, dtype=np.int64)
    if np.any(z < 1) or np.any(z > 100):
        raise ValueError("Atomic number outside supported range 1..100")
    return bundle["table"][z].astype(np.float32), bundle


def feature_table(path=None) -> np.ndarray:
    """C2DB reader: a (101, 20) float32 table indexed directly by Z (row 0 unused)."""
    path = Path(path) if path is not None else physical20_path()
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = ["Z"] + RAW_FEATURE_COLUMNS
        if any(fields.count(name) != 1 for name in required):
            raise ValueError("feature CSV must contain Z and each canonical feature column once")
        table = np.zeros((101, 20), dtype=np.float32)
        seen = set()
        for row in reader:
            z_value = float(row["Z"])
            if not np.isfinite(z_value) or not z_value.is_integer() or not 1 <= z_value <= 100:
                raise ValueError("feature CSV Z values must be integers from 1 through 100")
            z = int(z_value)
            if z in seen:
                raise ValueError("feature CSV contains duplicate Z=%d" % z)
            values = np.asarray([float(row[name]) for name in RAW_FEATURE_COLUMNS], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("feature row for Z=%d contains nonfinite values" % z)
            table[z] = values.astype(np.float32)
            seen.add(z)
    if seen != set(range(1, 101)):
        raise ValueError("feature CSV must contain Z=1..100 exactly once")
    return table
