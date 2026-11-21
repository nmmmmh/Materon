"""Memory-mapped CGCNN graph caches and serialized Matbench folds.

Layout: <cache_root>/<task>/<model_name>/<feature_mode>...mmcache, each holding
ragged arrays as ``<key>.values.npy`` + ``<key>.row_splits.npy``. Graphs are
built once from the Matbench structures; training reads only the caches.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tensorflow as tf

from materon.attributes import feature_source_sha256
from materon.cgcnn.graphs import (build_materon_cgcnn_graph, build_physics_cgcnn_graph,
                                  build_standard_cgcnn_graph)

try:
    import fcntl
except ImportError:  # non-POSIX: locking becomes a no-op
    fcntl = None

SCHEMA_VERSION = 7
SPLIT_SCHEMA_VERSION = 1


def _parallel_graphs(builder, structures):
    workers = max(1, int(os.environ.get("MATERON_CACHE_WORKERS", "1")))
    if workers == 1 or len(structures) < 2:
        return [builder(structure) for structure in structures]
    with ThreadPoolExecutor(max_workers=min(workers, len(structures))) as executor:
        return list(executor.map(builder, structures))


def collect_complete_task(task):
    """All structures of a Matbench task (fold-0 train+val followed by fold-0 test)."""
    fold = list(task.folds)[0]
    x_train, y_train = task.get_train_and_val_data(fold)
    x_test, y_test = task.get_test_data(fold, include_target=True)
    structures = list(x_train) + list(x_test)
    labels = np.concatenate([np.asarray(y_train, np.float32), np.asarray(y_test, np.float32)])
    indices = [str(value) for value in list(x_train.index) + list(x_test.index)]
    if len(indices) != len(set(indices)):
        raise RuntimeError("Duplicate Matbench indices")
    return structures, labels, indices


def _safe_unicode(values: Sequence[object]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max([1] + [len(value) for value in strings])
    return np.asarray(strings, dtype=f"<U{width}")


def _file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _array_sequence_sha256(fields: Mapping[str, Sequence[np.ndarray]]) -> str:
    """Content hash of graph arrays, used to check that physics and Materon graphs are identical."""
    digest = hashlib.sha256()
    for key in sorted(fields):
        digest.update(key.encode("utf-8"))
        for array in fields[key]:
            value = np.ascontiguousarray(array)
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(json.dumps(value.shape).encode("utf-8"))
            digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _save_ragged(directory: Path, key: str, arrays: Sequence[np.ndarray], dtype) -> None:
    normalized = [np.asarray(value, dtype=dtype) for value in arrays]
    tail_shape = normalized[0].shape[1:] if normalized else ()
    if any(value.shape[1:] != tail_shape for value in normalized):
        raise RuntimeError(f"Inconsistent ragged tail shape for {key}")
    lengths = np.asarray([len(value) for value in normalized], dtype=np.int64)
    row_splits = np.concatenate([np.asarray([0], np.int64), np.cumsum(lengths, dtype=np.int64)])
    values = np.concatenate(normalized, axis=0) if normalized else np.empty((0, *tail_shape), dtype=dtype)
    np.save(directory / f"{key}.values.npy", values, allow_pickle=False)
    np.save(directory / f"{key}.row_splits.npy", row_splits, allow_pickle=False)


def _load_ragged(directory: Path, key: str):
    return (np.load(directory / f"{key}.values.npy", mmap_mode="r", allow_pickle=False),
            np.load(directory / f"{key}.row_splits.npy", mmap_mode="r", allow_pickle=False))


def _ragged_subset(field, positions: Sequence[int], dtype):
    values, row_splits = field
    positions = [int(value) for value in positions]
    arrays = [values[row_splits[index]:row_splits[index + 1]] for index in positions]
    tf_dtype = tf.as_dtype(dtype)
    np_dtype = tf_dtype.as_numpy_dtype
    lengths = np.asarray([len(value) for value in arrays], dtype=np.int64)
    if arrays:
        flat = np.concatenate([np.asarray(value, dtype=np_dtype) for value in arrays], axis=0)
    else:
        flat = np.empty((0, *values.shape[1:]), dtype=np_dtype)
    return tf.RaggedTensor.from_row_lengths(tf.convert_to_tensor(flat, dtype=tf_dtype), lengths)


def _atomic_directory_write(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=path.name + ".", dir=str(path.parent)))
    try:
        writer(temporary)
        if path.exists():
            shutil.rmtree(path)
        os.replace(temporary, path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@contextmanager
def cache_lock(path):
    if fcntl is None:
        yield
        return
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class BaseCache:
    model_name = "base"
    ragged_fields: Mapping[str, object] = {}
    dense_fields: Mapping[str, object] = {}

    def __init__(self, task=None, cache_root="cache", feature_mode="full20", rebuild=False,
                 task_name=None, cache_only=False):
        self.feature_mode = str(feature_mode)
        self.task_name = str(task_name or getattr(task, "dataset_name", None)
                             or getattr(task, "task_name", None) or "unknown_task")
        self.path = Path(cache_root) / self.task_name / self.model_name / self.cache_dirname()
        with cache_lock(self.path):
            if rebuild or not self.path.exists():
                if task is None or (cache_only and not rebuild):
                    raise FileNotFoundError(f"Graph cache is missing: {self.path}; build it with scripts/*/build_caches.py")
                self.build(task)
            self.load()

    def cache_dirname(self):
        return self.feature_mode + ".mmcache"

    def expected_metadata(self):
        return {"schema": SCHEMA_VERSION, "storage": "npy_mmap_ragged_v1", "task": self.task_name,
                "model_name": self.model_name, "feature_mode": self.feature_mode}

    def _write_cache(self, payload, metadata):
        def writer(directory: Path):
            for key, dtype in self.ragged_fields.items():
                _save_ragged(directory, key, payload[key], dtype)
            for key, dtype in self.dense_fields.items():
                np.save(directory / f"{key}.npy", np.asarray(payload[key], dtype=dtype), allow_pickle=False)
            np.save(directory / "labels.npy", np.asarray(payload["labels"], np.float32), allow_pickle=False)
            np.save(directory / "indices.npy", _safe_unicode(payload["indices"]), allow_pickle=False)
            files = {f.name: {"size_bytes": f.stat().st_size, "sha256": _file_sha256(f)}
                     for f in sorted(directory.glob("*.npy"))}
            out = dict(metadata, files=files, content_sha256=_canonical_hash(files))
            (directory / "metadata.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

        _atomic_directory_write(self.path, writer)

    def load(self):
        metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        for key, expected in self.expected_metadata().items():
            if metadata.get(key) != expected:
                raise RuntimeError(f"Cache metadata mismatch {key}: {metadata.get(key)!r} != {expected!r}")
        for name, record in metadata.get("files", {}).items():
            file = self.path / name
            if not file.exists() or file.stat().st_size != int(record["size_bytes"]):
                raise RuntimeError(f"Missing or truncated cache array: {file}")
        self.metadata = metadata
        self._ragged = {key: _load_ragged(self.path, key) for key in self.ragged_fields}
        self._dense = {key: np.load(self.path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                       for key in self.dense_fields}
        self.y = np.load(self.path / "labels.npy", mmap_mode="r", allow_pickle=False)
        self.indices = [str(v) for v in np.load(self.path / "indices.npy", mmap_mode="r", allow_pickle=False)]
        self.position = {value: index for index, value in enumerate(self.indices)}
        if len(self.indices) != len(self.position):
            raise RuntimeError("Duplicate indices in cache")

    def subset_positions(self, positions: Sequence[int]):
        raise NotImplementedError

    def subset_indices(self, indices):
        return self.subset_positions([self.position[str(value)] for value in indices])

    def _finish_metadata(self, metadata, indices, common=None, feature_names=None):
        if feature_names is not None:
            metadata["feature_names"] = list(feature_names)
        if common is not None:
            metadata["graph_payload_sha256"] = _array_sequence_sha256(common)
        metadata["n_samples"] = len(indices)
        metadata["indices_sha256"] = _canonical_hash(indices)
        return metadata


class StandardCGCNNCache(BaseCache):
    model_name = "cgcnn_baseline"
    ragged_fields = {"node_number": np.int64, "node_frac_coordinates": np.float32,
                     "range_indices": np.int64, "range_image": np.float32}
    dense_fields = {"graph_lattice": np.float32}

    def cache_dirname(self):
        return "atomic_number_periodic_radius8.mmcache"

    def expected_metadata(self):
        metadata = super().expected_metadata()
        metadata.update({"graph": "periodic_radius", "radius": 8.0, "node_schema": "atomic_number",
                         "edge_schema": "range_indices+range_image", "context_schema": "none"})
        return metadata

    def build(self, task):
        structures, labels, indices = collect_complete_task(task)
        graphs = _parallel_graphs(lambda s: build_standard_cgcnn_graph(s, 8.0), structures)
        if any(np.max(g["node_number"], initial=0) >= 95 for g in graphs):
            raise RuntimeError("CGCNN embedding input_dim=95 cannot represent Z>=95")
        payload = {key: [g[key] for g in graphs]
                   for key in ["node_number", "node_frac_coordinates", "range_indices", "range_image"]}
        payload.update({"graph_lattice": np.asarray([g["graph_lattice"] for g in graphs], np.float32),
                        "labels": labels, "indices": indices})
        self._write_cache(payload, self._finish_metadata(self.expected_metadata(), indices))

    def subset_positions(self, positions):
        positions = list(positions)
        r = self._ragged
        return [
            _ragged_subset(r["node_number"], positions, tf.int64),
            _ragged_subset(r["node_frac_coordinates"], positions, tf.float32),
            _ragged_subset(r["range_indices"], positions, tf.int64),
            tf.convert_to_tensor(np.asarray(self._dense["graph_lattice"][positions]), tf.float32),
            _ragged_subset(r["range_image"], positions, tf.float32),
        ], np.asarray(self.y[positions], np.float32)


class PhysicsCGCNNCache(BaseCache):
    model_name = "physics_cgcnn"
    ragged_fields = {"node_features": np.float32, "edge_indices": np.int64, "edge_features": np.float32}

    def cache_dirname(self):
        return f"{self.feature_mode}_knn12_edge21.mmcache"

    def expected_metadata(self):
        metadata = super().expected_metadata()
        metadata.update({"graph": "periodic_knn", "radius": 8.0, "max_neighbors": 12,
                         "node_schema": self.feature_mode, "edge_schema": "distance+rbf16+cartesian_l01=21",
                         "context_schema": "none", "feature_source_sha256": feature_source_sha256(self.feature_mode)})
        return metadata

    def build(self, task):
        structures, labels, indices = collect_complete_task(task)
        graphs = _parallel_graphs(lambda s: build_physics_cgcnn_graph(s, self.feature_mode, None), structures)
        common = {key: [g[key] for g in graphs] for key in ["node_features", "edge_indices", "edge_features"]}
        payload = {**common, "labels": labels, "indices": indices}
        names = graphs[0]["feature_names"] if graphs else []
        self._write_cache(payload, self._finish_metadata(self.expected_metadata(), indices, common, names))

    def subset_positions(self, positions):
        positions = list(positions)
        r = self._ragged
        return [
            _ragged_subset(r["node_features"], positions, tf.float32),
            _ragged_subset(r["edge_indices"], positions, tf.int64),
            _ragged_subset(r["edge_features"], positions, tf.float32),
        ], np.asarray(self.y[positions], np.float32)


class MateronCGCNNCache(BaseCache):
    model_name = "materon_cgcnn"
    ragged_fields = {"node_features": np.float32, "edge_indices": np.int64, "edge_features": np.float32,
                     "coordination": np.float32, "q_eff": np.float32, "electrostatic_magnitude": np.float32}

    def cache_dirname(self):
        return f"{self.feature_mode}_knn12_edge21_context.mmcache"

    def expected_metadata(self):
        metadata = super().expected_metadata()
        metadata.update({"graph": "periodic_knn", "radius": 8.0, "max_neighbors": 12,
                         "node_schema": self.feature_mode, "edge_schema": "distance+rbf16+cartesian_l01=21",
                         "context_schema": "pre_dedup_coord_qeff_emag",
                         "feature_source_sha256": feature_source_sha256(self.feature_mode)})
        return metadata

    def build(self, task):
        structures, labels, indices = collect_complete_task(task)
        graphs = _parallel_graphs(lambda s: build_materon_cgcnn_graph(s, self.feature_mode, None), structures)
        keys = list(self.ragged_fields)
        payload = {key: [g[key] for g in graphs] for key in keys}
        payload.update({"labels": labels, "indices": indices})
        common = {key: payload[key] for key in keys[:3]}
        names = graphs[0]["feature_names"] if graphs else []
        self._write_cache(payload, self._finish_metadata(self.expected_metadata(), indices, common, names))

    def subset_positions(self, positions):
        positions = list(positions)
        r = self._ragged
        return [_ragged_subset(r[key], positions, tf.int64 if key == "edge_indices" else tf.float32)
                for key in self.ragged_fields], np.asarray(self.y[positions], np.float32)


def cache_class_for_config(config):
    return {"cgcnn": StandardCGCNNCache, "physics": PhysicsCGCNNCache,
            "materon": MateronCGCNNCache}[config["model_type"]]


def open_existing_cache(config, cache_root, task_name):
    return cache_class_for_config(config)(task=None, cache_root=cache_root,
                                          feature_mode=config.get("feature_mode", "full20"),
                                          rebuild=False, task_name=task_name, cache_only=True)


def assert_physics_materon_graphs_equal(physics, materon):
    left = physics.metadata.get("graph_payload_sha256")
    right = materon.metadata.get("graph_payload_sha256")
    if not left or left != right:
        raise RuntimeError(f"Physics/Materon common graph mismatch: {left} vs {right}")
    return left


def assert_cache_indices_equal(caches):
    hashes = {cache.metadata.get("indices_sha256") for cache in caches}
    if len(hashes) != 1:
        raise RuntimeError(f"Cache index mismatch: {sorted(hashes)}")
    return next(iter(hashes))


def split_cache_path(cache_root, task_name):
    return Path(cache_root) / str(task_name) / "_task" / "official_matbench_folds.splitcache"


def build_official_split_cache(task, reference_cache, cache_root, task_name, rebuild=False):
    """Serialize the official Matbench folds as cache positions (used by the transfer jobs)."""
    path = split_cache_path(cache_root, task_name)
    expected = {"schema": SPLIT_SCHEMA_VERSION, "storage": "npy_mmap_official_folds_v1", "task": str(task_name),
                "n_samples": len(reference_cache.indices),
                "indices_sha256": reference_cache.metadata["indices_sha256"]}
    with cache_lock(path):
        if path.exists() and not rebuild:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise RuntimeError(f"Split cache mismatch {key}: {metadata.get(key)!r} != {value!r}")
            return metadata

        def writer(directory: Path):
            folds, all_test = [], []
            for fold in task.folds:
                fold = int(fold)
                x_train, y_train = task.get_train_and_val_data(fold)
                x_test, y_test = task.get_test_data(fold, include_target=True)
                train_pos = np.asarray([reference_cache.position[str(v)] for v in x_train.index], np.int64)
                test_pos = np.asarray([reference_cache.position[str(v)] for v in x_test.index], np.int64)
                for pos, y, label in ((train_pos, y_train, "Training"), (test_pos, y_test, "Test")):
                    if not np.allclose(np.asarray(reference_cache.y[pos], np.float32),
                                       np.asarray(y, np.float32), rtol=1e-6, atol=1e-6):
                        raise RuntimeError(f"{label} targets mismatch for fold {fold}")
                np.save(directory / f"fold_{fold}_train_positions.npy", train_pos, allow_pickle=False)
                np.save(directory / f"fold_{fold}_test_positions.npy", test_pos, allow_pickle=False)
                folds.append(fold)
                all_test.extend(test_pos.tolist())
            if sorted(all_test) != list(range(len(reference_cache.indices))):
                raise RuntimeError("Official test folds do not cover each sample exactly once")
            files = {f.name: {"size_bytes": f.stat().st_size, "sha256": _file_sha256(f)}
                     for f in sorted(directory.glob("*.npy"))}
            metadata = {**expected, "folds": folds, "files": files, "content_sha256": _canonical_hash(files)}
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        _atomic_directory_write(path, writer)
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def official_positions_from_serialized(cache_root, task_name, cache, fold):
    """Outer-train/test positions, targets and IDs of one serialized official fold."""
    path = split_cache_path(cache_root, task_name)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("indices_sha256") != cache.metadata.get("indices_sha256"):
        raise RuntimeError("Serialized split/cache index mismatch")
    fold = int(fold)
    train = np.asarray(np.load(path / f"fold_{fold}_train_positions.npy", allow_pickle=False), np.int64)
    test = np.asarray(np.load(path / f"fold_{fold}_test_positions.npy", allow_pickle=False), np.int64)
    return (train, np.asarray(cache.y[train], np.float32), test, np.asarray(cache.y[test], np.float32),
            [cache.indices[i] for i in train], [cache.indices[i] for i in test], metadata)
