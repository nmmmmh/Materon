"""Open the CGCNN and MEGNet caches used by the fixed-split models.

JDFT2D reuses the Matbench ``matbench_jdft2d`` caches; HOIP has its own caches
(task name ``hoip_bandgap``) built by scripts/fixed_split/build_caches.py.
"""
from __future__ import annotations

import copy

import numpy as np

from materon.cgcnn.config import all_configs, recipes
from materon.fixed_split.config import load_protocol


def cgcnn_source_config(model_id: str) -> tuple[dict, dict]:
    spec = load_protocol()["models"][model_id]
    return copy.deepcopy(all_configs()[spec["source_config"]]), copy.deepcopy(recipes()[spec["recipe"]])


def cgcnn_cache_task(dataset: str) -> str:
    return str(load_protocol()["datasets"][dataset].get("reuse_cgcnn_task", dataset))


def megnet_cache_task(dataset: str) -> str:
    return str(load_protocol()["datasets"][dataset].get("reuse_megnet_task", dataset))


def open_cgcnn_cache(dataset: str, model_id: str, cache_root: str):
    from materon.cgcnn.cache import MateronCGCNNCache, open_existing_cache
    cfg, _ = cgcnn_source_config(model_id)
    task_name = cgcnn_cache_task(dataset)
    if model_id == "materon_nogate_cgcnn":
        # The no-gate model reads the first three arrays of the full20 Materon cache
        # (identical graph, attributes and edges; the context arrays are not passed).
        base = MateronCGCNNCache(task=None, cache_root=cache_root, feature_mode="full20",
                                 task_name=task_name, cache_only=True)

        class PhysicsView:
            path = base.path
            metadata = dict(base.metadata)
            indices = base.indices
            y = base.y
            position = base.position

            def subset_positions(self, positions):
                x, y = base.subset_positions(positions)
                return x[:3], y

            def subset_indices(self, indices):
                return self.subset_positions([self.position[str(v)] for v in indices])

        return PhysicsView()
    return open_existing_cache(cfg, cache_root, task_name)


def open_megnet_cache(dataset: str, cache_root: str):
    from materon.megnet.cache import MEGNetCache
    return MEGNetCache(cache_root, megnet_cache_task(dataset))


def assert_backend_alignment(dataset: str, cgcnn_cache_root: str, megnet_cache_root: str) -> dict:
    """Same IDs, targets, atom order and fractional coordinates in the CGCNN and MEGNet caches."""
    cg = open_cgcnn_cache(dataset, "cgcnn_baseline", cgcnn_cache_root)
    mg = open_megnet_cache(dataset, megnet_cache_root)
    if set(cg.indices) != set(mg.sample_ids):
        raise RuntimeError(f"CGCNN/MEGNet sample ID mismatch for {dataset}")
    mg_position = {sid: i for i, sid in enumerate(mg.sample_ids)}
    delta = max(abs(float(cg.y[i]) - float(mg.labels[mg_position[sid]])) for i, sid in enumerate(cg.indices))
    if delta > 1e-5:
        raise RuntimeError(f"CGCNN/MEGNet target mismatch for {dataset}: max_abs={delta}")
    cg_zv, cg_zs = cg._ragged["node_number"]
    cg_fv, cg_fs = cg._ragged["node_frac_coordinates"]
    mg_zv, mg_zs = mg.r["node_number"]
    mg_fv, mg_fs = mg.r["node_frac_coordinates"]
    max_frac = 0.0
    for i, sid in enumerate(cg.indices):
        j = mg_position[sid]
        if not np.array_equal(np.asarray(cg_zv[cg_zs[i]:cg_zs[i + 1]], np.int64),
                              np.asarray(mg_zv[mg_zs[j]:mg_zs[j + 1]], np.int64)):
            raise RuntimeError(f"CGCNN/MEGNet atom-order mismatch for {dataset}/{sid}")
        d = np.abs(np.asarray(cg_fv[cg_fs[i]:cg_fs[i + 1]], np.float64) - np.asarray(mg_fv[mg_fs[j]:mg_fs[j + 1]], np.float64))
        d = np.minimum(d, np.abs(d - 1.0))
        max_frac = max(max_frac, float(np.max(d)) if d.size else 0.0)
    if max_frac > 1e-5:
        raise RuntimeError(f"CGCNN/MEGNet fractional-coordinate mismatch for {dataset}: max_abs={max_frac}")
    return {"dataset": dataset, "n_samples": len(cg.indices), "maximum_target_difference": float(delta),
            "maximum_periodic_fractional_coordinate_difference": max_frac}
