#!/usr/bin/env python3
"""Plan of the 16 fixed-split runs (2 datasets x 8 models, seed 42).

Run with scripts/run_plan.py --script scripts/fixed_split/train_job.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from materon.fixed_split.config import load_protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--cgcnn-cache-root", required=True)
    p.add_argument("--megnet-cache-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    protocol = load_protocol()
    jobs = [{"job_id": f"{dataset}__{model}__s42",
             "args": ["--dataset", dataset, "--model", model, "--output-root", a.output_root,
                      "--cgcnn-cache-root", a.cgcnn_cache_root, "--megnet-cache-root", a.megnet_cache_root]}
            for dataset in protocol["datasets"] for model in protocol["models"]]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol_id": protocol["protocol_id"], "jobs": jobs}, indent=2), encoding="utf-8")
    print(f"[PLAN] {out} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
