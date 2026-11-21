from __future__ import absolute_import, division, print_function

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .. import CACHE_SCHEMA_VERSION
from ..utils.io import read_json, sha256_file


class CachedDataset(object):
    def __init__(self, root: Path, manifest: Dict[str, Any], frame: pd.DataFrame,
                 element_fraction: np.ndarray, physical_context: np.ndarray,
                 engineered_features=None, engineered_group_ids=None):
        self.root = root
        self.manifest = manifest
        self.frame = frame
        self.element_fraction = np.asarray(element_fraction, dtype=np.float32)
        self.physical_context = np.asarray(physical_context, dtype=np.float32)
        self.engineered_features = (
            None if engineered_features is None
            else np.asarray(engineered_features, dtype=np.float32)
        )
        self.engineered_group_ids = (
            None if engineered_group_ids is None
            else tuple(int(value) for value in engineered_group_ids)
        )
        self.compositions = frame["composition"].astype(str).to_numpy()
        self.targets = frame["target"].astype(float).to_numpy(dtype=np.float32)

    @classmethod
    def load(cls, cache_dir, verify_hashes=True):
        root = Path(cache_dir).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Missing cache manifest: %s" % manifest_path)
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise RuntimeError("Cache schema mismatch: expected %s, got %r" %
                               (CACHE_SCHEMA_VERSION, manifest.get("schema_version")))
        if verify_hashes:
            failures = []
            for relative, expected in manifest.get("file_sha256", {}).items():
                path = root / relative
                if not path.is_file() or sha256_file(path) != expected:
                    failures.append(relative)
            if failures:
                raise RuntimeError("Cache hash verification failed: %r" % failures)
        frame = pd.read_csv(root / "raw" / "dataset.csv")
        fractions = np.load(root / "features" / "element_fraction_float32.npz")["x"]
        context = np.load(root / "features" / "physical_context_float32.npz")["x"]
        engineered_path = root / "features" / "engineered_features_float32.npz"
        group_ids_path = root / "features" / "engineered_feature_group_ids.json"
        engineered = np.load(engineered_path)["x"] if engineered_path.is_file() else None
        group_ids = read_json(group_ids_path) if group_ids_path.is_file() else None
        arrays = [fractions, context] + ([] if engineered is None else [engineered])
        if any(len(frame) != len(array) for array in arrays):
            raise RuntimeError("Cache row counts are inconsistent")
        if engineered is not None and (group_ids is None or len(group_ids) != engineered.shape[1]):
            raise RuntimeError("Engineered feature group IDs are missing or inconsistent")
        return cls(root, manifest, frame, fractions, context, engineered, group_ids)

    def feature_view(self, name):
        if name == "elem_fraction":
            return self.element_fraction
        if name == "legacy_engineered":
            if self.engineered_features is None:
                raise FileNotFoundError(
                    "Engineered feature cache not built. Rebuild once with engineered features enabled."
                )
            return self.engineered_features
        raise KeyError("Unknown feature view %r" % name)

    def feature_group_ids(self, name):
        if name == "legacy_engineered" and self.engineered_group_ids is not None:
            return self.engineered_group_ids
        return tuple(0 for _ in range(self.feature_view(name).shape[1]))
