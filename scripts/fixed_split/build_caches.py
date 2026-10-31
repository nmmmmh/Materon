#!/usr/bin/env python3
"""Build the HOIP CGCNN and MEGNet caches and check CGCNN/MEGNet alignment for HOIP and JDFT2D.

JDFT2D reuses the Matbench caches (scripts/cgcnn/build_caches.py and
scripts/megnet/build_caches.py must have been run for matbench_jdft2d).
"""
from __future__ import annotations

import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hoip-path", required=True)
    p.add_argument("--cgcnn-cache-root", required=True)
    p.add_argument("--megnet-cache-root", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rebuild-hoip", action="store_true")
    a = p.parse_args()
    os.environ["MATERON_CACHE_WORKERS"] = str(max(1, a.workers))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import numpy as np
    from materon.cgcnn.cache import (MateronCGCNNCache, PhysicsCGCNNCache, StandardCGCNNCache,
                                     assert_cache_indices_equal, assert_physics_materon_graphs_equal)
    from materon.fixed_split.cache_access import assert_backend_alignment
    from materon.fixed_split.config import load_protocol
    from materon.fixed_split.datasets import OneFoldAllTestTask, load_hoip
    from materon.megnet.cache import MEGNetCache, build_cache

    protocol = load_protocol()
    bundle = load_hoip(a.hoip_path, int(protocol["datasets"]["hoip_bandgap"]["expected_n"]))
    task = OneFoldAllTestTask(bundle)
    name, root, rebuild = "hoip_bandgap", a.cgcnn_cache_root, a.rebuild_hoip
    caches = [StandardCGCNNCache(task, root, "atomic_number", rebuild, task_name=name),
              PhysicsCGCNNCache(task, root, "full20", rebuild, task_name=name),
              PhysicsCGCNNCache(task, root, "grouped_pca", rebuild, task_name=name),
              MateronCGCNNCache(task, root, "full20", rebuild, task_name=name)]
    assert_cache_indices_equal(caches)
    assert_physics_materon_graphs_equal(caches[1], caches[3])
    build_cache(task, a.megnet_cache_root, name, rebuild=rebuild, workers=a.workers)
    mg = MEGNetCache(a.megnet_cache_root, name)

    # A cache built from another HOIP file must not be reused silently.
    cg = caches[0]
    if list(cg.indices) != list(bundle.ids) or list(mg.sample_ids) != list(bundle.ids):
        raise RuntimeError("HOIP cache IDs/order differ from the input file; rerun with --rebuild-hoip")
    for labels in (cg.y, mg.labels):
        if not np.allclose(np.asarray(labels, float).reshape(-1), np.asarray(bundle.targets, float), rtol=0.0, atol=1e-6):
            raise RuntimeError("HOIP cache targets differ from the input file; rerun with --rebuild-hoip")
    for dataset in ["hoip_bandgap", "jdft2d_exfoliation"]:
        print("[ALIGNED]", assert_backend_alignment(dataset, a.cgcnn_cache_root, a.megnet_cache_root), flush=True)


if __name__ == "__main__":
    main()
