#!/usr/bin/env python3
"""Run the jobs of a plan file: ``{"jobs": [{"job_id": ..., "args": [...]}, ...]}``.

Each job runs ``python <script> <args>``. Without --gpus jobs run one after
another; with --gpus 0,1,... one process runs per listed GPU. The runner stops
at the first failing job.

    python scripts/run_plan.py --plan plan.json --script scripts/cgcnn/run_trial.py --gpus 0,1 --log-dir logs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--script", required=True, help="entry point executed for every job")
    p.add_argument("--gpus", default="", help="comma-separated GPU ids; empty runs sequentially")
    p.add_argument("--log-dir", required=True)
    a = p.parse_args()

    jobs = list(json.loads(Path(a.plan).read_text(encoding="utf-8"))["jobs"])
    logs = Path(a.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    slots = [g.strip() for g in a.gpus.split(",") if g.strip()] or [None]
    free, active = list(slots), []
    print(f"[PLAN] {len(jobs)} jobs, slots={slots}", flush=True)
    while jobs or active:
        while jobs and free:
            job, slot = jobs.pop(0), free.pop(0)
            env = os.environ.copy()
            if slot is not None:
                env["CUDA_VISIBLE_DEVICES"] = slot
            handle = open(logs / f"{job['job_id']}.log", "w")
            proc = subprocess.Popen([sys.executable, a.script, *map(str, job["args"])],
                                    stdout=handle, stderr=subprocess.STDOUT, env=env)
            active.append((proc, slot, job["job_id"], handle))
            print(f"[START] {job['job_id']} gpu={slot}", flush=True)
        time.sleep(1)
        for item in list(active):
            proc, slot, job_id, handle = item
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            active.remove(item)
            free.append(slot)
            if code != 0:
                for other in active:
                    other[0].terminate()
                raise SystemExit(f"[FAIL] {job_id} exit={code}; see {logs / (job_id + '.log')}")
            print(f"[DONE] {job_id}", flush=True)
    print("ALL JOBS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
