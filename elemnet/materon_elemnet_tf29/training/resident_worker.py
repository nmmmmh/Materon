from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--initial-weights-dir", required=True)
    parser.add_argument("--status-file", required=True)
    args = parser.parse_args()

    # Must be set before importing TensorFlow-backed modules.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    from ..compat import configure_tensorflow
    from ..data.cache import CachedDataset
    from ..utils.io import atomic_write_json
    from .initial_weights import InitialWeightStore
    from .run_tasks import cleanup_tensorflow, run_config_screen, run_formal

    configure_tensorflow(42, deterministic=True)
    cache = CachedDataset.load(args.cache_dir, verify_hashes=True)
    initial_store = InitialWeightStore(args.initial_weights_dir)
    tasks = []
    with Path(args.task_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))

    completed = []
    failures = []
    print("[RESIDENT GPU%s] loaded cache once; tasks=%d" % (args.gpu_id, len(tasks)), flush=True)
    for index, task in enumerate(tasks):
        try:
            mode = task["mode"]
            label = task.get("label", task)
            print(
                "[RESIDENT GPU%s START %d/%d] %s" %
                (args.gpu_id, index + 1, len(tasks), label),
                flush=True,
            )
            if mode in ("config_screen", "plan_screen"):
                result = run_config_screen(task, cache, initial_store)
            elif mode in ("formal", "plan_formal"):
                result = run_formal(task, cache, initial_store)
            else:
                raise KeyError("Unknown task mode=%r" % mode)
            completed.append({"task": task, "result": result})
            print("[RESIDENT GPU%s PASS] %s" % (args.gpu_id, label), flush=True)
        except Exception as exc:
            failures.append({
                "task": task,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            print(
                "[RESIDENT GPU%s FAIL] %s\n%s" %
                (args.gpu_id, repr(exc), traceback.format_exc()),
                file=sys.stderr,
                flush=True,
            )
        finally:
            cleanup_tensorflow()

    payload = {
        "execution_status": "PASS" if not failures else "FAIL",
        "gpu_id": str(args.gpu_id),
        "tasks": len(tasks),
        "completed": len(completed),
        "failures": failures,
    }
    atomic_write_json(args.status_file, payload)
    if failures:
        raise SystemExit(2)
    print("[RESIDENT GPU%s ALL PASS]" % args.gpu_id, flush=True)


if __name__ == "__main__":
    main()
