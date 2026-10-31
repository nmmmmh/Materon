#!/usr/bin/env python3
"""Plan of the CGCNN transfer runs (dielectric, JDFT2D; configs/cgcnn/transfer_protocol.json).

Run with scripts/run_plan.py --script scripts/cgcnn/run_transfer.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from materon.cgcnn.config import transfer_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tasks", default="")
    p.add_argument("--models", default="")
    a = p.parse_args()
    protocol = transfer_protocol()
    tasks = [x for x in a.tasks.split(",") if x] or list(protocol["tasks"])
    models = [x for x in a.models.split(",") if x] or list(protocol["models"])
    unknown = sorted((set(tasks) - set(protocol["tasks"])) | (set(models) - set(protocol["models"])))
    if unknown:
        raise ValueError(f"Not in the transfer protocol: {unknown}")
    tasks.sort(key=lambda name: (-int(protocol["tasks"][name].get("priority", 0)), name))
    jobs = []
    for task in tasks:
        for fold in protocol["outer_folds"]:
            for model in models:
                for seed in protocol["seeds"]:
                    jobs.append({"job_id": f"{task}__{model}__f{fold}__s{seed}",
                                 "args": ["--task", task, "--model", model, "--outer-fold", str(fold), "--seed", str(seed),
                                          "--cache-root", a.cache_root, "--outdir", a.outdir]})
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol_id": protocol["protocol_id"], "jobs": jobs}, indent=2), encoding="utf-8")
    print(f"[PLAN] {out} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
