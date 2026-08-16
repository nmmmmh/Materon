#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line workflow for build graph cache in the TF2.9 multi-backbone engine."""

from __future__ import annotations

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))

import argparse
from pathlib import Path

from matbench.bench import MatbenchBenchmark
from materon_tf29_engine.training.cache import GraphTaskCache
from materon_tf29_engine.training.task_registry import STRUCTURE_SMALL_TASKS


def parse_csv(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default=",".join(STRUCTURE_SMALL_TASKS),
        help="Comma-separated structure tasks.",
    )
    parser.add_argument(
        "--backbones",
        default="megnet,cgcnn",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/matbench_v0_1"),
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    tasks = parse_csv(args.tasks)
    backbones = parse_csv(args.backbones)
    for backbone in backbones:
        if backbone not in {"megnet", "cgcnn"}:
            raise ValueError("Graph cache backbone must be megnet or cgcnn")

    for task_name in tasks:
        if task_name not in STRUCTURE_SMALL_TASKS:
            raise ValueError(
                "%s is not a registered structure task under 5000 samples."
                % task_name
            )
        benchmark = MatbenchBenchmark(
            subset=[task_name],
            autoload=False,
        )
        task = next(iter(benchmark.tasks))
        task.load()
        for backbone in backbones:
            cache = GraphTaskCache(
                task,
                backbone,
                cache_dir=args.cache_dir,
                rebuild_cache=args.rebuild,
            )
            print(
                "[CACHE READY] task=%s backbone=%s %s"
                % (task_name, backbone, cache.cache_info),
                flush=True,
            )


if __name__ == "__main__":
    main()
