# -*- coding: utf-8 -*-
"""Persistent Matbench representation cache with Linux file locking."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
import kgcnn

from kgcnn.data.crystal import CrystalDataset

from materon_tf29_engine.data.composition import composition_matrix
from materon_tf29_engine.data.crystal import (
    CGCNN_INPUTS,
    CRYSTAL_CALLBACKS,
    MEGNET_INPUTS,
    range_method,
)

CACHE_SCHEMA_VERSION = 3


def _collect_complete_task(task):
    first_fold = list(task.folds)[0]
    train_x, train_y = task.get_train_and_val_data(first_fold)
    test_x, test_y = task.get_test_data(first_fold, include_target=True)
    all_x = pd.concat([train_x, test_x])
    all_y = pd.concat([train_y, test_y])
    if all_x.index.duplicated().any():
        raise RuntimeError("Duplicate Matbench sample indices detected.")
    all_y = all_y.loc[all_x.index]
    return all_x, all_y


def _positions(index_to_position: Dict, indices: Sequence) -> np.ndarray:
    return np.asarray(
        [index_to_position[x] for x in indices],
        dtype=np.int64,
    )


def _gather(value, positions):
    return tf.gather(value, positions, axis=0)


def _task_name(task):
    for name in ("dataset_name", "task_name"):
        value = getattr(task, name, None)
        if value:
            return str(value)
    metadata = getattr(task, "metadata", {})
    return str(metadata.get("dataset_name", metadata.get("name", "task")))


def _digest_payload(payload):
    raw = pickle.dumps(payload, protocol=4)
    return hashlib.sha256(raw).hexdigest()


def _directory_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _save_tensor(root, index, tensor):
    root = Path(root)
    item = root / ("tensor_%02d" % index)
    item.mkdir(parents=True, exist_ok=False)

    if isinstance(tensor, tf.RaggedTensor):
        np.save(item / "flat_values.npy", tensor.flat_values.numpy())
        for level, row_splits in enumerate(tensor.nested_row_splits):
            np.save(
                item / ("row_splits_%02d.npy" % level),
                row_splits.numpy(),
            )
        return {
            "kind": "ragged",
            "dtype": tensor.dtype.name,
            "ragged_rank": int(tensor.ragged_rank),
        }

    array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    np.save(item / "dense.npy", array)
    return {
        "kind": "dense",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _load_tensor(root, index, metadata):
    item = Path(root) / ("tensor_%02d" % index)
    if metadata["kind"] == "dense":
        return tf.convert_to_tensor(
            np.load(item / "dense.npy", mmap_mode="r")
        )

    values = tf.convert_to_tensor(
        np.load(item / "flat_values.npy", mmap_mode="r")
    )
    splits = []
    for level in range(int(metadata["ragged_rank"])):
        splits.append(
            tf.convert_to_tensor(
                np.load(
                    item / ("row_splits_%02d.npy" % level),
                    mmap_mode="r",
                ),
                dtype=tf.int64,
            )
        )
    return tf.RaggedTensor.from_nested_row_splits(
        values,
        splits,
        validate=False,
    )


class _FileLock:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _default_cache_dir():
    return Path(
        os.environ.get(
            "MATERON_CACHE_DIR",
            "cache/matbench_v0_1",
        )
    ).resolve()


class GraphTaskCache:
    def __init__(
        self,
        task,
        backbone: str,
        cache_dir=None,
        rebuild_cache=False,
    ):
        self.task = task
        self.backbone = backbone
        all_inputs, all_outputs = _collect_complete_task(task)
        self.all_indices = list(all_inputs.index)
        self.index_to_position = {
            index: position
            for position, index in enumerate(self.all_indices)
        }

        if backbone == "megnet":
            self.input_spec = MEGNET_INPUTS
            cutoff = 5.0
        elif backbone == "cgcnn":
            self.input_spec = CGCNN_INPUTS
            cutoff = 8.0
        else:
            raise ValueError("GraphTaskCache supports megnet or cgcnn")

        task_name = _task_name(task)
        fingerprint_payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "task": task_name,
            "backbone": backbone,
            "cutoff": cutoff,
            "input_spec": self.input_spec,
            "indices": self.all_indices,
            "labels": np.asarray(all_outputs, dtype=np.float32),
            "kgcnn": str(getattr(kgcnn, "__version__", "unknown")),
        }
        fingerprint = _digest_payload(fingerprint_payload)[:20]
        base = Path(cache_dir).resolve() if cache_dir else _default_cache_dir()
        parent = base / task_name / backbone
        cache_path = parent / fingerprint
        lock_path = parent / (fingerprint + ".lock")
        complete_path = cache_path / "COMPLETE"

        wait_start = time.perf_counter()
        with _FileLock(lock_path):
            lock_wait = time.perf_counter() - wait_start
            if rebuild_cache and cache_path.exists():
                shutil.rmtree(cache_path)

            if complete_path.exists():
                load_start = time.perf_counter()
                metadata = json.loads(
                    (cache_path / "metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.x_all = [
                    _load_tensor(cache_path, index, item)
                    for index, item in enumerate(metadata["tensors"])
                ]
                self.y_all = np.load(
                    cache_path / "labels.npy",
                    mmap_mode="r",
                )
                cache_status = "hit"
                cache_seconds = time.perf_counter() - load_start
            else:
                build_start = time.perf_counter()
                print(
                    "[GRAPH CACHE BUILD] backbone=%s task=%s n=%d path=%s"
                    % (
                        backbone,
                        task_name,
                        len(all_inputs),
                        cache_path,
                    ),
                    flush=True,
                )
                dataset = CrystalDataset()
                dataset._map_callbacks(
                    all_inputs,
                    pd.Series(all_outputs.values),
                    CRYSTAL_CALLBACKS,
                )
                dataset.set_methods(range_method(cutoff))
                removed = dataset.clean(self.input_spec)
                if np.size(removed) > 0:
                    raise RuntimeError(
                        "Unexpected invalid graphs: %s" % removed
                    )
                x_all = list(dataset.tensor(self.input_spec))
                if backbone == "cgcnn":
                    x_all[4] = tf.cast(x_all[4], tf.float32)
                    if (
                        tf.as_dtype(x_all[1].dtype) != tf.float32
                        or tf.as_dtype(x_all[4].dtype) != tf.float32
                    ):
                        raise RuntimeError(
                            "CGCNN fractional coordinates and range_image "
                            "must both be float32."
                        )
                labels = np.asarray(
                    dataset.get("graph_labels"),
                    dtype=np.float32,
                ).reshape(-1)

                parent.mkdir(parents=True, exist_ok=True)
                temp_path = Path(
                    tempfile.mkdtemp(
                        prefix=".%s.tmp." % fingerprint,
                        dir=str(parent),
                    )
                )
                try:
                    tensor_metadata = [
                        _save_tensor(temp_path, index, tensor)
                        for index, tensor in enumerate(x_all)
                    ]
                    np.save(temp_path / "labels.npy", labels)
                    metadata = {
                        "schema": CACHE_SCHEMA_VERSION,
                        "task": task_name,
                        "backbone": backbone,
                        "fingerprint": fingerprint,
                        "n_samples": len(self.all_indices),
                        "cutoff": cutoff,
                        "tensors": tensor_metadata,
                    }
                    (temp_path / "metadata.json").write_text(
                        json.dumps(metadata, indent=2),
                        encoding="utf-8",
                    )
                    (temp_path / "COMPLETE").write_text(
                        "complete\n",
                        encoding="utf-8",
                    )
                    if cache_path.exists():
                        shutil.rmtree(cache_path)
                    os.replace(str(temp_path), str(cache_path))
                except Exception:
                    shutil.rmtree(temp_path, ignore_errors=True)
                    raise

                self.x_all = x_all
                self.y_all = labels
                cache_status = "built"
                cache_seconds = time.perf_counter() - build_start

        self.cache_info = {
            "kind": "graph",
            "status": cache_status,
            "path": str(cache_path),
            "fingerprint": fingerprint,
            "seconds": float(cache_seconds),
            "lock_wait_seconds": float(lock_wait),
            "size_bytes": int(_directory_size(cache_path)),
        }
        print(
            "[GRAPH CACHE %s] task=%s backbone=%s load=%.2fs "
            "wait=%.2fs size=%.3fGB"
            % (
                cache_status.upper(),
                task_name,
                backbone,
                cache_seconds,
                lock_wait,
                self.cache_info["size_bytes"] / 1.0e9,
            ),
            flush=True,
        )

    def fold(self, fold_id):
        train_x, _ = self.task.get_train_and_val_data(fold_id)
        test_x, test_y = self.task.get_test_data(
            fold_id,
            include_target=True,
        )
        train_positions = _positions(
            self.index_to_position,
            train_x.index,
        )
        test_positions = _positions(
            self.index_to_position,
            test_x.index,
        )
        x_train = [
            _gather(x, train_positions)
            for x in self.x_all
        ]
        x_test = [
            _gather(x, test_positions)
            for x in self.x_all
        ]
        y_train = np.asarray(self.y_all[train_positions])
        y_test = np.asarray(
            test_y,
            dtype=np.float32,
        ).reshape(-1)
        return x_train, y_train, x_test, y_test


class CompositionTaskCache:
    def __init__(
        self,
        task,
        max_atomic_number: int = 100,
        cache_dir=None,
        rebuild_cache=False,
    ):
        self.task = task
        self.max_atomic_number = int(max_atomic_number)
        all_inputs, all_outputs = _collect_complete_task(task)
        self.all_indices = list(all_inputs.index)
        self.index_to_position = {
            index: position
            for position, index in enumerate(self.all_indices)
        }

        task_name = _task_name(task)
        fingerprint = _digest_payload({
            "schema": CACHE_SCHEMA_VERSION,
            "task": task_name,
            "representation": "composition",
            "max_atomic_number": self.max_atomic_number,
            "indices": self.all_indices,
            "labels": np.asarray(all_outputs, dtype=np.float32),
        })[:20]
        base = Path(cache_dir).resolve() if cache_dir else _default_cache_dir()
        cache_path = (
            base
            / task_name
            / ("composition_z%d" % self.max_atomic_number)
            / fingerprint
        )
        lock_path = cache_path.parent / (fingerprint + ".lock")

        wait_start = time.perf_counter()
        with _FileLock(lock_path):
            lock_wait = time.perf_counter() - wait_start
            if rebuild_cache and cache_path.exists():
                shutil.rmtree(cache_path)
            if (cache_path / "COMPLETE").exists():
                start = time.perf_counter()
                self.x_all = np.load(
                    cache_path / "composition.npy",
                    mmap_mode="r",
                )
                self.y_all = np.load(
                    cache_path / "labels.npy",
                    mmap_mode="r",
                )
                status = "hit"
                seconds = time.perf_counter() - start
            else:
                start = time.perf_counter()
                print(
                    "[COMPOSITION CACHE BUILD] task=%s n=%d"
                    % (task_name, len(all_inputs)),
                    flush=True,
                )
                x_all = composition_matrix(
                    list(all_inputs.values),
                    max_atomic_number=self.max_atomic_number,
                )
                y_all = np.asarray(
                    all_outputs,
                    dtype=np.float32,
                ).reshape(-1)
                cache_path.mkdir(parents=True, exist_ok=True)
                np.save(cache_path / "composition.npy", x_all)
                np.save(cache_path / "labels.npy", y_all)
                (cache_path / "COMPLETE").write_text(
                    "complete\n",
                    encoding="utf-8",
                )
                self.x_all = x_all
                self.y_all = y_all
                status = "built"
                seconds = time.perf_counter() - start

        self.cache_info = {
            "kind": "composition",
            "status": status,
            "path": str(cache_path),
            "fingerprint": fingerprint,
            "seconds": float(seconds),
            "lock_wait_seconds": float(lock_wait),
            "size_bytes": int(_directory_size(cache_path)),
        }
        print(
            "[COMPOSITION CACHE %s] task=%s load=%.2fs size=%.3fGB"
            % (
                status.upper(),
                task_name,
                seconds,
                self.cache_info["size_bytes"] / 1.0e9,
            ),
            flush=True,
        )

    def fold(self, fold_id):
        train_x, _ = self.task.get_train_and_val_data(fold_id)
        test_x, test_y = self.task.get_test_data(
            fold_id,
            include_target=True,
        )
        train_positions = _positions(
            self.index_to_position,
            train_x.index,
        )
        test_positions = _positions(
            self.index_to_position,
            test_x.index,
        )
        return (
            np.asarray(self.x_all[train_positions]),
            np.asarray(self.y_all[train_positions]),
            np.asarray(self.x_all[test_positions]),
            np.asarray(test_y, dtype=np.float32).reshape(-1),
        )
