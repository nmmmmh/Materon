#!/usr/bin/env python3
"""Build the CGCNN graph caches for the Matbench tasks (run once, CPU only).

matbench_phonons: conventional graph plus physics and Materon graphs for the
feature modes full20, grouped_pca and shuffled_full20 (inner-pool selection).
matbench_dielectric / matbench_jdft2d: conventional, grouped-PCA physics and
full20 Materon graphs plus the serialized official folds (transfer protocol).
"""
from __future__ import annotations

import argparse
import os

PHONON_MODES = ["full20", "grouped_pca", "shuffled_full20"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True)
    p.add_argument("--tasks", default="matbench_phonons,matbench_dielectric,matbench_jdft2d")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rebuild", action="store_true")
    a = p.parse_args()
    os.environ["MATERON_CACHE_WORKERS"] = str(max(1, a.workers))

    from matbench.bench import MatbenchBenchmark
    from materon.cgcnn.cache import (MateronCGCNNCache, PhysicsCGCNNCache, StandardCGCNNCache,
                                     assert_cache_indices_equal, assert_physics_materon_graphs_equal,
                                     build_official_split_cache)

    for name in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        task = next(iter(MatbenchBenchmark(subset=[name], autoload=False).tasks))
        task.load()
        standard = StandardCGCNNCache(task, a.cache_root, "atomic_number", a.rebuild, task_name=name)
        caches = [standard]
        if name == "matbench_phonons":
            for mode in PHONON_MODES:
                physics = PhysicsCGCNNCache(task, a.cache_root, mode, a.rebuild, task_name=name)
                materon = MateronCGCNNCache(task, a.cache_root, mode, a.rebuild, task_name=name)
                assert_physics_materon_graphs_equal(physics, materon)
                caches += [physics, materon]
        else:
            caches += [PhysicsCGCNNCache(task, a.cache_root, "grouped_pca", a.rebuild, task_name=name),
                       MateronCGCNNCache(task, a.cache_root, "full20", a.rebuild, task_name=name)]
            build_official_split_cache(task, standard, a.cache_root, name, rebuild=a.rebuild)
        assert_cache_indices_equal(caches)
        for cache in caches:
            print("[CACHE READY]", cache.path, flush=True)


if __name__ == "__main__":
    main()
