"""Datasets of the fixed-split 2D study: CoRE 2D-HOIP band gaps and Matbench JDFT2D, plus atom roles."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    name: str
    ids: list[str]
    targets: np.ndarray
    structures: object
    atomic_numbers: list[np.ndarray]
    frac_coordinates: object
    unit: str
    source: str
    source_sha256: str


def _decode_structure(obj):
    from pymatgen.core import Structure
    if isinstance(obj, Structure):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"Expected pymatgen Structure dict, got {type(obj)!r}")
    return Structure.from_dict(obj)


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hoip(path, expected_n: int = 499) -> DatasetBundle:
    """Read the pandas split-orient JSON.GZ with columns 'structure' and 'band_gap_pbe'."""
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if set(payload) != {"columns", "index", "data"}:
        raise ValueError(f"Unexpected HOIP JSON split schema: {sorted(payload)}")
    columns = list(payload["columns"])
    if "structure" not in columns or "band_gap_pbe" not in columns:
        raise ValueError(f"HOIP columns missing required fields: {columns}")
    structure_col = columns.index("structure")
    target_col = columns.index("band_gap_pbe")
    ids = [f"hoip_{int(idx):04d}" if str(idx).isdigit() else f"hoip_{idx}" for idx in payload["index"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate HOIP sample IDs")
    structures = [_decode_structure(row[structure_col]) for row in payload["data"]]
    targets = np.asarray([float(row[target_col]) for row in payload["data"]], dtype=np.float32)
    if len(ids) != expected_n or len(structures) != expected_n or len(targets) != expected_n:
        raise RuntimeError(f"HOIP sample count mismatch: ids={len(ids)} structures={len(structures)} targets={len(targets)} expected={expected_n}")
    if not np.all(np.isfinite(targets)):
        raise ValueError("HOIP target contains NaN or infinity")
    atomic_numbers = [np.asarray(st.atomic_numbers, dtype=np.int64) for st in structures]
    frac = [np.asarray(st.frac_coords, dtype=np.float32) for st in structures]
    return DatasetBundle(
        name="hoip_bandgap", ids=ids, targets=targets, structures=structures,
        atomic_numbers=atomic_numbers, frac_coordinates=frac, unit="eV", source=str(path.resolve()),
        source_sha256=file_sha256(path)
    )


def load_from_megnet_cache(cache_root, task: str, name: str, unit: str) -> DatasetBundle:
    from materon.megnet.cache import MEGNetCache
    cache = MEGNetCache(cache_root, task)
    ids = list(cache.sample_ids)
    positions = np.arange(len(ids), dtype=np.int64)
    atomic_numbers = cache.node_numbers(positions)
    frac_values, frac_splits = cache.r["node_frac_coordinates"]
    frac = [np.asarray(frac_values[frac_splits[i]:frac_splits[i+1]], dtype=np.float32) for i in positions]
    return DatasetBundle(
        name=name,
        ids=ids,
        targets=np.asarray(cache.labels, dtype=np.float32).reshape(-1),
        structures=None,
        atomic_numbers=atomic_numbers,
        frac_coordinates=frac,
        unit=unit,
        source=str(cache.path),
        source_sha256=str(cache.metadata.get("content_sha256", "")),
    )


def composition_group(z: np.ndarray) -> str:
    z = np.asarray(z, dtype=np.int64).reshape(-1)
    values, counts = np.unique(z, return_counts=True)
    divisor = 0
    for count in counts:
        divisor = math.gcd(divisor, int(count))
    divisor = max(1, divisor)
    reduced = [(int(v), int(c // divisor)) for v, c in zip(values, counts)]
    return ";".join(f"Z{z}:{n}" for z, n in reduced)


def bundle_manifest(bundle: DatasetBundle) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": bundle.ids,
        "target": bundle.targets,
        "group": [composition_group(z) for z in bundle.atomic_numbers],
        "n_atoms": [len(z) for z in bundle.atomic_numbers],
    })


def atomic_symbol(z: int) -> str:
    from pymatgen.core import Element
    return Element.from_Z(int(z)).symbol


def infer_roles(dataset: str, z: np.ndarray, frac) -> list[str]:
    symbols = [atomic_symbol(int(v)) for v in np.asarray(z).reshape(-1)]
    if dataset == "hoip_bandgap":
        metals = {"Pb", "Sn", "Ge"}
        halogens = {"F", "Cl", "Br", "I"}
        donors = {"N", "O", "S", "P"}
        roles = []
        for symbol in symbols:
            if symbol in metals:
                roles.append("metal_center")
            elif symbol in halogens:
                roles.append("halogen_bridge")
            elif symbol in donors:
                roles.append("organic_inorganic_interface")
            elif symbol in {"C", "H"}:
                roles.append("organic_spacer")
            else:
                roles.append("other")
        return roles

    # JDFT2D: combine chemical and slab-position roles into a compact label.
    transition_metals = {
        "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
        "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
        "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"
    }
    anions = {"B","C","N","O","F","Si","P","S","Cl","Se","Br","Te","I"}
    if frac is None or len(frac) == 0:
        surface = np.zeros(len(symbols), dtype=bool)
    else:
        # Unwrap the slab along fractional c by cutting at the largest cyclic
        # vacuum gap. This avoids misclassifying a slab that straddles 0/1.
        zfrac = np.mod(np.asarray(frac, dtype=np.float32)[:, 2], 1.0)
        order = np.argsort(zfrac)
        sorted_z = zfrac[order]
        cyclic = np.diff(np.concatenate([sorted_z, sorted_z[:1] + 1.0]))
        cut_after = int(np.argmax(cyclic))
        origin = float(sorted_z[(cut_after + 1) % len(sorted_z)])
        unwrapped = np.mod(zfrac - origin, 1.0)
        zspan = float(np.max(unwrapped) - np.min(unwrapped))
        if zspan < 1e-8:
            surface = np.zeros(len(symbols), dtype=bool)
        else:
            lo = float(np.min(unwrapped) + 0.25 * zspan)
            hi = float(np.max(unwrapped) - 0.25 * zspan)
            surface = (unwrapped <= lo) | (unwrapped >= hi)
    roles = []
    for i, symbol in enumerate(symbols):
        chem = "transition_metal" if symbol in transition_metals else ("anion" if symbol in anions else "other")
        layer = "surface" if bool(surface[i]) else "interior"
        roles.append(f"{layer}_{chem}")
    return roles


class OneFoldAllTestTask:
    """Minimal Matbench-like adapter used only for deterministic graph caching."""
    def __init__(self, bundle: DatasetBundle):
        if bundle.structures is None:
            raise ValueError("Structure objects are required to build a new graph cache")
        self.bundle = bundle
        self.folds = [0]
        self.dataset_name = bundle.name
        self.task_name = bundle.name

    def get_test_data(self, fold, include_target=True):
        if int(fold) != 0:
            raise ValueError(fold)
        x = pd.Series(self.bundle.structures, index=self.bundle.ids, dtype=object)
        y = pd.Series(self.bundle.targets, index=self.bundle.ids, dtype=np.float32)
        return (x, y) if include_target else x

    def get_train_and_val_data(self, fold):
        if int(fold) != 0:
            raise ValueError(fold)
        return pd.Series([], dtype=object), pd.Series([], dtype=np.float32)
