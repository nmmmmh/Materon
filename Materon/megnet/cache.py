"""MEGNet graph caches: official kgcnn 2.1 periodic radius-5 graphs plus full20/PCA node tables.

Samples are stored in Matbench test-fold order and each fold's official
train/test positions are saved alongside, preserving the order returned by
Matbench. Layout: <cache_root>/<task>/official_radius5_full_features_v2.mmcache
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from materon.attributes import feature_source_sha256, map_atomic_numbers

try:
    import fcntl
except ImportError:  # non-POSIX: locking becomes a no-op
    fcntl = None

SCHEMA = "materon_megnet_mmcache_v2_official_order"
CACHE_DIRNAME = "official_radius5_full_features_v2.mmcache"
RAGGED = {"node_number": np.int64, "node_coordinates": np.float32, "node_frac_coordinates": np.float32,
          "range_indices": np.int64, "range_image": np.int64, "node_full20": np.float32,
          "node_grouped_pca": np.float32, "site_index": np.int64}
DENSE = {"graph_lattice": np.float32, "charge": np.float32}


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canon(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def _lock(path):
    if fcntl is None:
        yield
        return
    p = Path(str(path) + ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+") as h:
        fcntl.flock(h.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(h.fileno(), fcntl.LOCK_UN)


def collect_complete_task(task):
    """Structures and labels in official test-fold order, with each fold's train/test positions."""
    records, fold_ids = {}, {}
    for fold in task.folds:
        x, y = task.get_test_data(int(fold), include_target=True)
        ids = [str(v) for v in x.index]
        fold_ids[int(fold)] = ids
        for sid, st, target in zip(ids, list(x), np.asarray(y, dtype=np.float32)):
            if sid in records:
                raise RuntimeError("sample appears in multiple official test folds: " + sid)
            records[sid] = (st, float(target))
    ordered = []
    for fold in sorted(fold_ids):
        ordered.extend(fold_ids[fold])
    structures = [records[i][0] for i in ordered]
    labels = np.asarray([records[i][1] for i in ordered], np.float32)
    pos = {sid: i for i, sid in enumerate(ordered)}
    splits = {}
    for fold, ids in fold_ids.items():
        # Keep the exact Matbench order of both partitions.
        train_inputs, _ = task.get_train_and_val_data(int(fold))
        train = np.asarray([pos[str(v)] for v in train_inputs.index], np.int64)
        test = np.asarray([pos[str(v)] for v in ids], np.int64)
        if set(train.tolist()) & set(test.tolist()) or len(train) + len(test) != len(ordered):
            raise RuntimeError(f"official fold {fold} split is inconsistent")
        splits[fold] = (train, test)
    return structures, labels, ordered, splits


def _save_ragged(root, name, seq, dtype):
    seq = [np.asarray(x, dtype=dtype) for x in seq]
    lengths = np.asarray([len(x) for x in seq], np.int64)
    splits = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tail = seq[0].shape[1:] if seq else ()
    flat = np.concatenate(seq, axis=0) if seq else np.empty((0, *tail), dtype=dtype)
    np.save(root / f"{name}_values.npy", flat, allow_pickle=False)
    np.save(root / f"{name}_row_splits.npy", splits, allow_pickle=False)


def _load_ragged(root, name):
    return (np.load(root / f"{name}_values.npy", mmap_mode="r", allow_pickle=False),
            np.load(root / f"{name}_row_splits.npy", mmap_mode="r", allow_pickle=False))


def _subset(field, positions, dtype):
    import tensorflow as tf
    values, splits = field
    np_dtype = tf.as_dtype(dtype).as_numpy_dtype
    arrays = [np.asarray(values[splits[i]:splits[i + 1]], dtype=np_dtype) for i in positions]
    lengths = np.asarray([len(a) for a in arrays], np.int64)
    flat = np.concatenate(arrays, axis=0) if arrays else np.empty((0, *values.shape[1:]), dtype=np_dtype)
    return tf.RaggedTensor.from_row_lengths(tf.convert_to_tensor(flat, dtype=dtype), lengths)


def _periodic_graph_worker(index_and_structure):
    """One official kgcnn radius-5 periodic graph, built in a single-threaded CPU process."""
    index, structure = index_and_structure
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from kgcnn.data.crystal import CrystalDataset

    callbacks = {
        "graph_labels": lambda st, ds: np.expand_dims(ds, axis=-1),
        "node_coordinates": lambda st, ds: np.asarray(st.cart_coords, dtype="float"),
        "node_frac_coordinates": lambda st, ds: np.asarray(st.frac_coords, dtype="float"),
        "graph_lattice": lambda st, ds: np.ascontiguousarray(np.asarray(st.lattice.matrix), dtype="float"),
        "charge": lambda st, ds: np.asarray([st.charge], dtype="float"),
        "node_number": lambda st, ds: np.asarray(st.atomic_numbers, dtype="int"),
    }
    labels = pd.Series(np.asarray([0.0], dtype=np.float32)).reset_index(drop=True)
    ds = CrystalDataset()
    ds._map_callbacks([structure], labels, callbacks)
    ds.set_methods([{"map_list": {"method": "set_range_periodic", "max_distance": 5.0}}])
    required = [
        {"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
        {"shape": [1], "name": "charge", "dtype": "float32", "ragged": False},
        {"shape": (None, 3), "name": "range_image", "dtype": "int64", "ragged": True},
        {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32", "ragged": False},
    ]
    removed = ds.clean(required)
    if removed:
        raise RuntimeError(f"Official graph construction removed sample index {index}: {removed}")
    graph = {
        "node_number": np.asarray(ds.get("node_number")[0], np.int64),
        "node_coordinates": np.asarray(ds.get("node_coordinates")[0], np.float32),
        "node_frac_coordinates": np.asarray(ds.get("node_frac_coordinates")[0], np.float32),
        "range_indices": np.asarray(ds.get("range_indices")[0], np.int64),
        "range_image": np.asarray(ds.get("range_image")[0], np.int64),
        "graph_lattice": np.asarray(ds.get("graph_lattice")[0], np.float32),
        "charge": np.asarray(ds.get("charge")[0], np.float32),
    }
    return index, graph


def _parallel_periodic_graphs(structures, workers):
    total = len(structures)
    workers = min(max(1, int(workers)), max(1, total))
    if workers == 1:
        return [_periodic_graph_worker((i, s))[1] for i, s in enumerate(structures)]
    # Spawned processes avoid inheriting TensorFlow/kgcnn runtime state.
    results = [None] * total
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_periodic_graph_worker, (i, s)): i for i, s in enumerate(structures)}
        for done, future in enumerate(as_completed(futures), 1):
            index, graph = future.result()
            results[index] = graph
            if done % 200 == 0 or done == total:
                print(f"[MEGNET GRAPHS] {done}/{total} {done / max(time.time() - started, 1e-9):.1f}/s", flush=True)
    return results


def build_cache(task, cache_root, task_name, rebuild=False, workers=None):
    path = Path(cache_root) / task_name / CACHE_DIRNAME
    with _lock(path):
        if path.exists() and not rebuild:
            return path
        structures, labels, ids, splits = collect_complete_task(task)
        workers = int(workers if workers is not None else os.environ.get("MATERON_CACHE_WORKERS", "1"))
        graphs = _parallel_periodic_graphs(structures, workers)
        node_number = [g["node_number"] for g in graphs]
        payload = {key: [g[key] for g in graphs] for key in
                   ["node_number", "node_coordinates", "node_frac_coordinates", "range_indices", "range_image"]}
        payload["graph_lattice"] = np.asarray([g["graph_lattice"] for g in graphs], np.float32)
        payload["charge"] = np.asarray([g["charge"] for g in graphs], np.float32)
        payload["node_full20"], payload["node_grouped_pca"], payload["site_index"] = [], [], []
        for z in node_number:
            payload["node_full20"].append(map_atomic_numbers(z, "full20")[0])
            payload["node_grouped_pca"].append(map_atomic_numbers(z, "grouped_pca")[0])
            payload["site_index"].append(np.arange(len(z), dtype=np.int64))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=path.name + ".", dir=str(path.parent)))
        try:
            for k, d in RAGGED.items():
                _save_ragged(tmp, k, payload[k], d)
            for k, d in DENSE.items():
                np.save(tmp / f"{k}.npy", np.asarray(payload[k], d), allow_pickle=False)
            np.save(tmp / "labels.npy", labels, allow_pickle=False)
            np.save(tmp / "sample_ids.npy", np.asarray(ids, dtype=f"<U{max(map(len, ids))}"), allow_pickle=False)
            for fold, (train, test) in splits.items():
                np.save(tmp / f"fold_{fold}_train.npy", train, allow_pickle=False)
                np.save(tmp / f"fold_{fold}_test.npy", test, allow_pickle=False)
            files = {p.name: {"size": p.stat().st_size, "sha256": _sha256(p)} for p in sorted(tmp.glob("*.npy"))}
            meta = {"schema": SCHEMA, "task": task_name, "n_samples": len(ids), "radius": 5.0,
                    "kgcnn_graph_method": "set_range_periodic",
                    "feature_full20_sha256": feature_source_sha256("full20"),
                    "feature_pca_sha256": feature_source_sha256("grouped_pca"),
                    "files": files, "content_sha256": _canon(files)}
            (tmp / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            if path.exists():
                shutil.rmtree(path)
            os.replace(tmp, path)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return path


class MEGNetCache:
    """Model inputs: [node, coordinates, range_indices, charge, range_image, lattice(, node_mask)]."""

    def __init__(self, cache_root, task):
        self.path = Path(cache_root) / task / CACHE_DIRNAME
        if not self.path.exists():
            raise FileNotFoundError(f"Missing MEGNet cache {self.path}; run scripts/megnet/build_caches.py")
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        if self.metadata.get("schema") != SCHEMA:
            raise RuntimeError("cache schema mismatch")
        self.r = {k: _load_ragged(self.path, k) for k in RAGGED}
        self.d = {k: np.load(self.path / f"{k}.npy", mmap_mode="r") for k in DENSE}
        self.labels = np.load(self.path / "labels.npy", mmap_mode="r")
        self.sample_ids = [str(x) for x in np.load(self.path / "sample_ids.npy", mmap_mode="r")]

    def split(self, fold):
        return (np.asarray(np.load(self.path / f"fold_{fold}_train.npy", mmap_mode="r"), np.int64),
                np.asarray(np.load(self.path / f"fold_{fold}_test.npy", mmap_mode="r"), np.int64))

    def inputs(self, model_id, positions, node_masks=None):
        """``node_masks`` (one array per structure) scales the gate correction per atom."""
        import tensorflow as tf
        positions = [int(x) for x in positions]
        common = [_subset(self.r["node_coordinates"], positions, tf.float32),
                  _subset(self.r["range_indices"], positions, tf.int64),
                  tf.convert_to_tensor(np.asarray(self.d["charge"][positions]), tf.float32),
                  _subset(self.r["range_image"], positions, tf.int64),
                  tf.convert_to_tensor(np.asarray(self.d["graph_lattice"][positions]), tf.float32)]
        node_field = {"megnet_baseline": "node_number", "materon_nogate_megnet": "node_full20",
                      "materon_pca_megnet": "node_grouped_pca", "materon_explain_megnet": "node_full20"}
        if model_id not in node_field:
            raise KeyError(model_id)
        node = _subset(self.r[node_field[model_id]], positions, tf.float32)
        if model_id != "materon_explain_megnet":
            return [node, *common]
        if node_masks is None:
            splits = self.r["node_full20"][1]
            arrays = [np.ones((int(splits[p + 1] - splits[p]), 1), np.float32) for p in positions]
        else:
            arrays = [np.asarray(a, np.float32).reshape(-1, 1) for a in node_masks]
        lengths = np.asarray([len(a) for a in arrays], np.int64)
        flat = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 1), np.float32)
        mask = tf.RaggedTensor.from_row_lengths(tf.convert_to_tensor(flat), lengths)
        return [node, *common, mask]

    def targets(self, positions):
        return np.asarray(self.labels[list(positions)], np.float32).reshape(-1, 1)

    def ids(self, positions):
        return [self.sample_ids[int(i)] for i in positions]

    def node_numbers(self, positions):
        v, s = self.r["node_number"]
        return [np.asarray(v[s[i]:s[i + 1]], np.int64) for i in positions]

    def site_indices(self, positions):
        v, s = self.r["site_index"]
        return [np.asarray(v[s[i]:s[i + 1]], np.int64) for i in positions]
