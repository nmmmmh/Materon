from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .. import DEFAULT_SEEDS
from ..data.cache import CachedDataset
from ..models.registry import get_registry
from ..utils.io import read_json


def acceptance_guard(path):
    value = read_json(path)
    if not value.get("all_pass", False):
        raise RuntimeError("Runtime acceptance did not fully pass: %r" % value)


def parse_csv(value):
    return [token.strip() for token in str(value).split(",") if token.strip()]


def load_plan(path):
    value = read_json(path)
    configs = value.get("configs", {})
    if not configs:
        raise RuntimeError("Plan has no configs: %s" % path)
    return value, configs


def build_tasks(args, cache):
    tasks = []
    common = {
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "outdir": str(Path(args.outdir).resolve()),
        "hidden_units": [int(v) for v in args.hidden_units],
        "overwrite": bool(args.overwrite),
        "verbose": int(args.verbose),
        "save_predictions": True,
        "save_analysis": bool(args.save_screen_analysis),
        "save_history": bool(args.save_screen_history),
    }
    if not args.protocol_json:
        raise ValueError("--protocol-json is required")
    common["protocol_json"] = str(Path(args.protocol_json).resolve())

    if args.mode in ("config_screen", "formal"):
        input_dim = cache.feature_view("legacy_engineered").shape[1]
        registry = get_registry(args.stage, input_dim, args.hidden_units)
        names = parse_csv(args.configs)
        if names == ["all"]:
            names = sorted(registry)
        formal_save = set(parse_csv(args.save_formal_configs))
        for name in names:
            if name not in registry:
                raise KeyError("Unknown config=%r stage=%r" % (name, args.stage))
            seeds = (args.screen_seeds or [args.screen_seed]) if args.mode == "config_screen" else args.seeds
            for fold in args.folds:
                for seed in seeds:
                    save_model = bool(args.mode == "formal" and name in formal_save)
                    task = dict(common)
                    task.update({
                        "mode": args.mode,
                        "stage": args.stage,
                        "config": name,
                        "fold": int(fold),
                        "seed": int(seed),
                        "purpose": args.purpose,
                        "label": "%s fold=%d seed=%d" % (name, fold, seed),
                        "save_model": save_model,
                        "save_model_role": "baseline" if save_model else None,
                        "selected_models_dir": (
                            str(Path(args.selected_models_dir).resolve())
                            if save_model and args.selected_models_dir else None
                        ),
                    })
                    if save_model and not task["selected_models_dir"]:
                        raise ValueError("--selected-models-dir is required for saved formal configs")
                    tasks.append(task)
        return tasks

    if args.mode in ("plan_screen", "plan_formal"):
        if not args.plan_json:
            raise ValueError("--plan-json is required")
        plan, configs = load_plan(args.plan_json)
        seeds = (args.screen_seeds or [args.screen_seed]) if args.mode == "plan_screen" else args.seeds
        saved_roles = plan.get("save_model_roles", {})
        save_names = set(plan.get("save_model_configs", []))
        selected_models_dir = plan.get("selected_models_dir", args.selected_models_dir)
        for name, spec in configs.items():
            for fold in args.folds:
                for seed in seeds:
                    save_model = bool(args.mode == "plan_formal" and name in save_names)
                    task = dict(common)
                    task.update({
                        "mode": args.mode,
                        "stage": "plan",
                        "config": name,
                        "config_spec": spec,
                        "fold": int(fold),
                        "seed": int(seed),
                        "purpose": args.purpose,
                        "label": "%s fold=%d seed=%d" % (name, fold, seed),
                        "save_model": save_model,
                        "save_model_role": saved_roles.get(name),
                        "selected_models_dir": (
                            str(Path(selected_models_dir).resolve())
                            if selected_models_dir else None
                        ),
                    })
                    if save_model and not task["selected_models_dir"]:
                        raise ValueError("Selected model directory is required for saved configs")
                    tasks.append(task)
        return tasks
    raise KeyError(args.mode)


def main():
    parser = argparse.ArgumentParser(
        description="Fast resident GPU workers; model.fit only; no replay and no disposable weights."
    )
    parser.add_argument(
        "--mode",
        choices=["config_screen", "formal", "plan_screen", "plan_formal"],
        required=True,
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--initial-weights-dir", required=True)
    parser.add_argument("--acceptance-report", required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--plan-json")
    parser.add_argument("--stage", default="baseline")
    parser.add_argument("--configs", default="all")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--screen-seeds", type=int, nargs="+")
    parser.add_argument("--hidden-units", type=int, nargs="+", default=[1024, 512, 256, 128])
    parser.add_argument("--purpose", default="V14 baseline-anchored EIGate experiment")
    parser.add_argument("--selected-models-dir")
    parser.add_argument("--save-formal-configs", default="")
    parser.add_argument("--save-screen-analysis", action="store_true")
    parser.add_argument("--save-screen-history", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    acceptance_guard(args.acceptance_report)
    cache = CachedDataset.load(args.cache_dir, verify_hashes=True)
    tasks = build_tasks(args, cache)
    gpu_ids = parse_csv(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("No GPU IDs")
    buckets = [[] for _ in gpu_ids]
    for index, task in enumerate(tasks):
        buckets[index % len(gpu_ids)].append(task)

    work_root = Path(args.outdir) / "_resident_worker_state"
    work_root.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu_id, bucket in zip(gpu_ids, buckets):
        task_file = work_root / ("gpu_%s_tasks.jsonl" % gpu_id)
        status_file = work_root / ("gpu_%s_status.json" % gpu_id)
        with task_file.open("w", encoding="utf-8") as handle:
            for task in bucket:
                handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
        command = [
            sys.executable, "-m", "materon_elemnet_tf29.training.resident_worker",
            "--gpu-id", str(gpu_id),
            "--task-file", str(task_file),
            "--cache-dir", str(Path(args.cache_dir).resolve()),
            "--initial-weights-dir", str(Path(args.initial_weights_dir).resolve()),
            "--status-file", str(status_file),
        ]
        process = subprocess.Popen(command, env=dict(os.environ))
        processes.append((gpu_id, process, status_file))
        print("[LAUNCH RESIDENT GPU%s] tasks=%d" % (gpu_id, len(bucket)), flush=True)

    failures = []
    for gpu_id, process, status_file in processes:
        code = process.wait()
        if code != 0:
            failures.append((gpu_id, code, str(status_file)))
    if failures:
        raise RuntimeError("Resident workers failed: %r" % failures)
    print("[ALL RESIDENT WORKERS PASS] tasks=%d" % len(tasks), flush=True)


if __name__ == "__main__":
    main()
