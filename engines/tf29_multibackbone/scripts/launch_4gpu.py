#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed-bound four-GPU scheduler for paired baseline/Materon experiments.

One base seed is assigned to one GPU.  All requested variants for that seed run
sequentially on the same physical GPU, and each variant process runs its five
folds sequentially.  Therefore no GPU is shared concurrently and paired models
also share the same GPU hardware.
"""

from __future__ import annotations

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def parse_list(text, cast=str):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["megnet", "cgcnn", "elemnet"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variants", default="baseline,materon")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--threads-per-process", type=int, default=2)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--max-atomic-number", type=int, default=100)
    parser.add_argument("--feature-mode", choices=["grouped_pca","full20","none","z_only","shuffled_grouped_pca"], default="grouped_pca")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/matbench_v0_1"))
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--strict-cross-graph-identity", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.processes = {}
        self.completed = []
        self.failures = []
        self.stop = False


def preexec_nice(value):
    def inner():
        if value:
            os.nice(value)
    return inner


def summary_path(args, variant, seed):
    return (
        args.outdir
        / args.backbone
        / args.task
        / variant
        / ("seed_%d" % seed)
        / "seed_summary.json"
    )


def build_command(args, run_script, variant, seed):
    command = [
        sys.executable,
        str(run_script),
        "--backbone", args.backbone,
        "--variant", variant,
        "--task", args.task,
        "--seed", str(seed),
        "--folds", args.folds,
        "--outdir", str(args.outdir),
        "--max-atomic-number", str(args.max_atomic_number),
        "--cache-dir", str(args.cache_dir),
        "--log-every", str(args.log_every),
    ]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate is not None:
        command.extend(["--learning-rate", str(args.learning_rate)])
    command.extend(["--feature-mode", args.feature_mode])
    if args.strict_cross_graph_identity:
        command.append("--strict-cross-graph-identity")
    if args.force:
        command.append("--force")
    return command


def worker(gpu, seed_jobs, variants, args, state, run_script):
    while not state.stop:
        try:
            seed = seed_jobs.get_nowait()
        except queue.Empty:
            return

        print(
            "[SEED START] GPU=%d backbone=%s seed=%d variants=%s"
            % (gpu, args.backbone, seed, variants),
            flush=True,
        )
        seed_failed = False
        for variant in variants:
            if state.stop:
                seed_failed = True
                break
            summary = summary_path(args, variant, seed)
            if summary.exists() and not args.force:
                print(
                    "[SKIP] GPU=%d variant=%s seed=%d complete"
                    % (gpu, variant, seed),
                    flush=True,
                )
                with state.lock:
                    state.completed.append((variant, seed, gpu))
                continue

            log_dir = summary.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "seed_train.log"
            command = build_command(args, run_script, variant, seed)

            env = os.environ.copy()
            thread_count = str(args.threads_per_process)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "PYTHONHASHSEED": str(seed),
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": thread_count,
                    "MKL_NUM_THREADS": thread_count,
                    "OPENBLAS_NUM_THREADS": thread_count,
                    "NUMEXPR_NUM_THREADS": thread_count,
                    "TF_NUM_INTRAOP_THREADS": thread_count,
                    "TF_NUM_INTEROP_THREADS": str(args.inter_threads),
                    "OMP_WAIT_POLICY": "PASSIVE",
                    "KMP_BLOCKTIME": "0",
                    "TF_GPU_THREAD_MODE": "gpu_private",
                    "TF_GPU_THREAD_COUNT": "2",
                    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                    "MALLOC_ARENA_MAX": "2",
                }
            )

            print(
                "[START] GPU=%d variant=%s seed=%d"
                % (gpu, variant, seed),
                flush=True,
            )
            start = time.time()
            with open(log_path, "a", encoding="utf-8", buffering=1) as log:
                log.write("\n" + "=" * 100 + "\n")
                log.write("[RESUME/START] %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
                log.write("[COMMAND] %s\n" % " ".join(command))
                log.flush()
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=preexec_nice(args.nice),
                    text=True,
                )
                with state.lock:
                    state.processes[gpu] = process
                return_code = process.wait()
                with state.lock:
                    state.processes.pop(gpu, None)
            elapsed = time.time() - start

            if return_code == 0:
                with state.lock:
                    state.completed.append((variant, seed, gpu))
                print(
                    "[DONE ] GPU=%d variant=%s seed=%d elapsed=%.2f h"
                    % (gpu, variant, seed, elapsed / 3600.0),
                    flush=True,
                )
            else:
                failure = {
                    "gpu": gpu,
                    "variant": variant,
                    "seed": seed,
                    "return_code": return_code,
                    "log": str(log_path),
                }
                with state.lock:
                    state.failures.append(failure)
                print("[FAIL ] %s" % failure, flush=True)
                seed_failed = True
                break

        if not seed_failed:
            print("[SEED DONE] GPU=%d seed=%d" % (gpu, seed), flush=True)
        seed_jobs.task_done()


def terminate_all(state):
    with state.lock:
        state.stop = True
        processes = list(state.processes.values())
    for process in processes:
        if process.poll() is None:
            process.terminate()
    time.sleep(2)
    for process in processes:
        if process.poll() is None:
            process.kill()


def main():
    args = parse_args()
    gpus = parse_list(args.gpus, int)
    seeds = parse_list(args.seeds, int)
    variants = parse_list(args.variants, str)
    if len(set(gpus)) != len(gpus):
        raise SystemExit("GPU list contains duplicates.")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("Seed list contains duplicates.")
    if any(x not in {"baseline", "materon"} for x in variants):
        raise SystemExit("variants must contain baseline and/or materon")

    script = Path(__file__).resolve().parent / "run_seed.py"
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["outdir"] = str(args.outdir)
    config["cache_dir"] = str(args.cache_dir)
    config["policy"] = "one_seed_bound_to_one_gpu_variants_and_folds_sequential"
    with open(args.outdir / "launcher_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    seed_jobs = queue.Queue()
    for seed in seeds:
        seed_jobs.put(seed)

    print("=" * 100, flush=True)
    print("ONE SEED = ONE GPU; VARIANTS AND FOLDS RUN SEQUENTIALLY", flush=True)
    print("backbone=%s task=%s" % (args.backbone, args.task), flush=True)
    print("GPUs=%s seeds=%s variants=%s" % (gpus, seeds, variants), flush=True)
    print(
        "Main CPU threads <= %d; nice=%d; VASP-friendly mode"
        % (len(gpus) * args.threads_per_process, args.nice),
        flush=True,
    )
    print("The fifth seed waits for the first free GPU.", flush=True)
    print("=" * 100, flush=True)

    state = State()

    def signal_handler(signum, frame):
        del frame
        print("Received signal %d; terminating workers." % signum, flush=True)
        terminate_all(state)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    threads = []
    for gpu in gpus:
        thread = threading.Thread(
            target=worker,
            args=(gpu, seed_jobs, variants, args, state, script),
            daemon=False,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    status = {"completed": state.completed, "failures": state.failures}
    with open(args.outdir / "launcher_status.json", "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
    if state.failures:
        raise SystemExit("Some workers failed. See launcher_status.json and seed logs.")
    print("All requested seeds and variants completed successfully.", flush=True)


if __name__ == "__main__":
    main()
