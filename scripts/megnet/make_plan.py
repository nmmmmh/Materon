#!/usr/bin/env python3
"""Plan of the 45 MEGNet Matbench runs (3 tasks x 3 models x 5 folds, seed 42).

Run with scripts/run_plan.py --script scripts/megnet/train_job.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from materon.megnet.config import load_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    q = load_protocol()
    jobs = [{"job_id": f"{task}__{model}__f{fold}__s42",
             "args": ["--task", task, "--model", model, "--fold", str(fold), "--cache-root", a.cache_root,
                      "--outdir", a.outdir]}
            for task in q["task_order"] for fold in q["outer_folds"] for model in q["models"]]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol_id": q["protocol_id"], "jobs": jobs}, indent=2), encoding="utf-8")
    print(f"[PLAN] {out} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
