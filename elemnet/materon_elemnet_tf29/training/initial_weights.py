from __future__ import absolute_import, division, print_function

import os
import time
from pathlib import Path

import numpy as np

from ..models.elemnet_performance import ModelConfig, build_model
from ..utils.io import atomic_write_json


class InitialWeightStore(object):
    """Backbone-only random-initialization library.

    This standalone package stores one backbone NPZ per seed and backbone shape. Gate weights are initialized by
    explicit per-layer seeds when the candidate model is built. This preserves
    paired backbone starts while avoiding hundreds of disposable model files.
    """

    def __init__(self, root, lock_timeout_seconds=180):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._cache = {}

    @staticmethod
    def _shape_key(config, seed):
        return (
            int(config.input_dim),
            tuple(int(v) for v in config.hidden_units),
            bool(config.batch_norm),
            int(seed),
        )

    def paths(self, config, seed):
        hidden = "-".join(str(int(v)) for v in config.hidden_units)
        token = "in%d_h%s_bn%d_seed%d" % (
            int(config.input_dim), hidden, int(bool(config.batch_norm)), int(seed)
        )
        return (
            self.root / (token + ".backbone.npz"),
            self.root / (token + ".json"),
            self.root / (token + ".lock"),
        )

    @staticmethod
    def _baseline_config(config):
        return ModelConfig(
            name="initial_backbone_template",
            input_dim=int(config.input_dim),
            hidden_units=tuple(int(v) for v in config.hidden_units),
            dropout=float(config.dropout),
            l2_regularization=float(config.l2_regularization),
            batch_norm=bool(config.batch_norm),
            input_gate=None,
            post_gate=None,
        )

    @staticmethod
    def _save_arrays(path, arrays):
        payload = {"arr_%04d" % i: np.asarray(value) for i, value in enumerate(arrays)}
        temp = path.with_name(path.name + ".tmp.%d.npz" % os.getpid())
        np.savez_compressed(str(temp), **payload)
        os.replace(str(temp), str(path))

    @staticmethod
    def _load_arrays(path):
        with np.load(str(path), allow_pickle=False) as handle:
            keys = sorted(handle.files)
            return [np.asarray(handle[key]).copy() for key in keys]

    def _create_file_safely(self, config, seed, npz_path, meta_path, lock_path):
        started = time.time()
        acquired = False
        while not acquired:
            try:
                lock_path.mkdir(parents=False, exist_ok=False)
                acquired = True
            except FileExistsError:
                if npz_path.is_file():
                    return
                if time.time() - started > self.lock_timeout_seconds:
                    raise RuntimeError("Timed out waiting for backbone lock: %s" % lock_path)
                time.sleep(0.2)
        try:
            if npz_path.is_file():
                return
            template = build_model(self._baseline_config(config), int(seed))
            arrays = [value.numpy().copy() for value in template.backbone_weights()]
            self._save_arrays(npz_path, arrays)
            atomic_write_json(meta_path, {
                "seed": int(seed),
                "untrained_random_initialization": True,
                "backbone_only": True,
                "backbone_sha256": template.backbone_sha256(),
                "input_dim": int(config.input_dim),
                "hidden_units": [int(v) for v in config.hidden_units],
                "array_count": len(arrays),
            })
        finally:
            try:
                lock_path.rmdir()
            except OSError:
                pass

    def backbone_arrays(self, config, seed):
        key = self._shape_key(config, seed)
        if key in self._cache:
            return [value.copy() for value in self._cache[key]]
        npz_path, meta_path, lock_path = self.paths(config, seed)
        if not npz_path.is_file():
            self._create_file_safely(config, seed, npz_path, meta_path, lock_path)
        arrays = self._load_arrays(npz_path)
        self._cache[key] = [value.copy() for value in arrays]
        return arrays

    def build_model(self, config, seed):
        # Resolve/create the backbone cache before building the candidate.
        # Creating a cache template clears the Keras session, so doing it after
        # candidate construction would risk invalidating graph state.
        arrays = self.backbone_arrays(config, seed)
        model = build_model(config, int(seed))
        variables = model.backbone_weights()
        if len(arrays) != len(variables):
            raise RuntimeError(
                "Backbone array count mismatch: %d != %d" % (len(arrays), len(variables))
            )
        for variable, value in zip(variables, arrays):
            if tuple(variable.shape) != tuple(value.shape):
                raise RuntimeError(
                    "Backbone shape mismatch: %r != %r" % (tuple(variable.shape), tuple(value.shape))
                )
            variable.assign(value)
        return model
