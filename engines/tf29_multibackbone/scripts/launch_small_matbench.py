#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequentially launch every valid <5000 Matbench v0.1 experiment."""
from __future__ import annotations

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))

import argparse
import subprocess
import sys
from pathlib import Path

from materon_tf29_engine.training.task_registry import (
    ALL_SMALL_TASKS,
    STRUCTURE_SMALL_TASKS,
)


def parse_csv(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def stream_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[EXPERIMENT COMMAND] %s" % " ".join(command), flush=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbones",
        default="megnet,cgcnn,elemnet",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(ALL_SMALL_TASKS),
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs_small_matbench_v36"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/matbench_v0_1"),
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--feature-mode",
        default="grouped_pca",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    backbones = parse_csv(args.backbones)
    tasks = parse_csv(args.tasks)
    launch = Path(__file__).resolve().parent / "launch_4gpu.py"

    experiments = []
    for backbone in backbones:
        if backbone not in {"megnet", "cgcnn", "elemnet"}:
            raise ValueError("Unknown backbone: %s" % backbone)
        for task in tasks:
            if task not in ALL_SMALL_TASKS:
                raise ValueError("Task is not in the frozen small registry: %s" % task)
            if backbone in {"megnet", "cgcnn"} and task not in STRUCTURE_SMALL_TASKS:
                continue
            experiments.append((backbone, task))

    print(
        "[SMALL MATBENCH] %d valid backbone-task combinations"
        % len(experiments),
        flush=True,
    )
    for index, (backbone, task) in enumerate(experiments, start=1):
        print(
            "[EXPERIMENT %02d/%02d] backbone=%s task=%s"
            % (index, len(experiments), backbone, task),
            flush=True,
        )
        command = [
            sys.executable,
            str(launch),
            "--backbone", backbone,
            "--task", task,
            "--variants", "baseline,materon",
            "--seeds", args.seeds,
            "--folds", args.folds,
            "--gpus", args.gpus,
            "--outdir", str(args.outdir),
            "--cache-dir", str(args.cache_dir),
            "--log-every", str(args.log_every),
            "--feature-mode", args.feature_mode,
        ]
        if args.force:
            command.append("--force")
        log_path = (
            args.outdir
            / "launcher_logs"
            / ("%s__%s.log" % (backbone, task))
        )
        return_code = stream_command(command, log_path)
        if return_code != 0:
            raise SystemExit(
                "Experiment failed: backbone=%s task=%s log=%s"
                % (backbone, task, log_path)
            )

    print("ALL SMALL MATBENCH EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
