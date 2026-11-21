from __future__ import absolute_import, division, print_function

import argparse
import json
from pathlib import Path

import numpy as np

from ..utils.io import atomic_write_json, read_json


def check_baseline(results_dir, out_json, baseline_config="H0_exact_baseline"):
    rows = []
    strategies = []
    for path in sorted(Path(results_dir).rglob("result.json")):
        value = read_json(path)
        if value.get("config") != baseline_config:
            continue
        rows.append({
            "fold": int(value["fold"]),
            "seed": int(value["seed"]),
            "mae": float(value["metrics"]["mae"]),
        })
        strategies.append(str(value.get("formal_strategy", "unknown")))
    folds = sorted(set(row["fold"] for row in rows))
    seeds = sorted(set(row["seed"] for row in rows))
    finite = bool(rows) and all(np.isfinite(row["mae"]) for row in rows)
    strategy_ok = set(strategies) == {"discovery_checkpoint"}
    passed = bool(
        len(rows) == 15
        and folds == [0, 1, 2, 3, 4]
        and seeds == [42, 123, 2024]
        and finite
        and strategy_ok
    )
    payload = {
        "execution_status": "PASS",
        "all_pass": passed,
        "scientific_gate_pass": None,
        "decision": "RUN_SELECTED_GATE" if passed else "STOP_INCOMPLETE_BASELINE",
        "baseline_config": baseline_config,
        "formal_strategy": "discovery_checkpoint",
        "runs": len(rows),
        "folds": folds,
        "seeds": seeds,
        "finite_mae": finite,
        "formal_outer_mean_mae": None if not rows else float(np.mean([row["mae"] for row in rows])),
        "hard_absolute_mae_threshold": None,
        "note": "No arbitrary absolute-MAE gate; completeness, finite metrics and the registered formal strategy are required.",
    }
    atomic_write_json(out_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--baseline-config", default="H0_exact_baseline")
    args = parser.parse_args()
    check_baseline(args.results_dir, args.out_json, args.baseline_config)


if __name__ == "__main__":
    main()
