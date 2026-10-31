#!/usr/bin/env python3
"""Build the MEGNet graph caches of the Matbench tasks (official kgcnn radius-5 graphs; CPU only)."""
from __future__ import annotations

import argparse

from materon.megnet.config import load_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True)
    p.add_argument("--tasks", default="")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--rebuild", action="store_true")
    a = p.parse_args()

    from matbench.bench import MatbenchBenchmark
    from materon.megnet.cache import build_cache

    protocol = load_protocol()
    tasks = [x.strip() for x in a.tasks.split(",") if x.strip()] or list(protocol["tasks"])
    for name in tasks:
        task = next(iter(MatbenchBenchmark(subset=[name], autoload=False).tasks))
        task.load()
        print("[CACHE READY]", name, build_cache(task, a.cache_root, name, a.rebuild, workers=a.workers), flush=True)


if __name__ == "__main__":
    main()
