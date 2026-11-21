from __future__ import absolute_import, division, print_function

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.cache import CachedDataset
from ..models.config_serialization import model_config_to_dict
from ..models.registry import v14_candidate_registry
from ..models.tuning_factory import plain_additive_control
from ..utils.io import atomic_write_json
from ..utils.metrics import regression_metrics


LAMBDA_GRID = (0.00, 0.05, 0.10, 0.20, 0.33, 0.50)


def collect_screen_results(*roots):
    rows = []
    for root in roots:
        root = Path(root)
        for path in sorted(root.rglob("screen_result.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("execution_status") != "PASS":
                continue
            prediction_path = value.get("prediction_file")
            if not prediction_path:
                candidate = path.parent / "validation_predictions.csv"
                prediction_path = str(candidate)
            if not Path(prediction_path).is_file():
                raise FileNotFoundError("Missing validation predictions: %s" % prediction_path)
            rows.append({
                "config": str(value["config"]),
                "fold": int(value["fold"]),
                "seed": int(value["seed"]),
                "mae": float(value["validation_metrics_mpa"]["mae"]),
                "best_epoch": int(value["best_epoch"]),
                "parameter_count": int(value.get("parameter_count", 0)),
                "gate_parameter_count": int(value.get("gate_parameter_count", 0)),
                "prediction_path": str(prediction_path),
                "result_path": str(path),
            })
    return pd.DataFrame(rows)


def prediction_frame(results, config):
    parts = []
    subset = results[results["config"] == config]
    for row in subset.itertuples(index=False):
        pred = pd.read_csv(row.prediction_path)
        required = {"sample_id", "y_true", "y_pred"}
        if not required.issubset(pred.columns):
            raise RuntimeError("Prediction file missing columns: %s" % row.prediction_path)
        pred = pred[["sample_id", "y_true", "y_pred"]].copy()
        pred["fold"] = int(row.fold)
        pred["seed"] = int(row.seed)
        pred["config"] = str(config)
        parts.append(pred)
    if not parts:
        raise RuntimeError("No prediction rows for %s" % config)
    return pd.concat(parts, ignore_index=True)


def run_metrics(predictions, label):
    rows = []
    for (fold, seed), group in predictions.groupby(["fold", "seed"]):
        values = regression_metrics(group["y_true"].values, group["y_pred"].values)
        rows.append({
            "config": str(label),
            "fold": int(fold),
            "seed": int(seed),
            "mae": float(values["mae"]),
        })
    return pd.DataFrame(rows)


def summarize_pair(baseline_runs, candidate_runs, config, blend_lambda,
                   gate_parameter_count, median_best_epoch):
    base = baseline_runs[["fold", "seed", "mae"]].rename(columns={"mae": "baseline_mae"})
    cand = candidate_runs[["fold", "seed", "mae"]].rename(columns={"mae": "candidate_mae"})
    paired = base.merge(cand, on=["fold", "seed"], how="inner")
    if len(paired) != 15:
        raise RuntimeError("Expected 15 paired runs for %s lambda=%s, got %d" % (
            config, blend_lambda, len(paired)
        ))
    paired["improvement"] = paired["baseline_mae"] - paired["candidate_mae"]
    paired["relative_improvement_percent"] = (
        100.0 * paired["improvement"] / paired["baseline_mae"]
    )
    seed_mean = paired.groupby("seed")["improvement"].mean()
    fold_mean = paired.groupby("fold")["improvement"].mean()
    return {
        "config": str(config),
        "blend_lambda": float(blend_lambda),
        "candidate_mean_mae": float(paired["candidate_mae"].mean()),
        "baseline_mean_mae": float(paired["baseline_mae"].mean()),
        "mean_improvement_mpa": float(paired["improvement"].mean()),
        "median_improvement_mpa": float(paired["improvement"].median()),
        "improvement_std_mpa": float(paired["improvement"].std(ddof=1)),
        "aggregate_relative_improvement_percent": float(
            100.0 * paired["improvement"].mean() / paired["baseline_mae"].mean()
        ),
        "mean_of_runwise_relative_improvement_percent": float(
            paired["relative_improvement_percent"].mean()
        ),
        "run_wins": int((paired["improvement"] > 0).sum()),
        "seed_wins": int((seed_mean > 0).sum()),
        "fold_wins": int((fold_mean > 0).sum()),
        "worst_seed_improvement_mpa": float(seed_mean.min()),
        "worst_fold_improvement_mpa": float(fold_mean.min()),
        "gate_parameter_count": int(gate_parameter_count),
        "median_best_epoch": float(median_best_epoch),
    }


def strict_constraints(row):
    return bool(
        float(row["blend_lambda"]) > 0.0
        and float(row["mean_improvement_mpa"]) > 0.0
        and float(row["median_improvement_mpa"]) > 0.0
        and int(row["run_wins"]) >= 9
        and int(row["seed_wins"]) == 3
        and int(row["fold_wins"]) >= 4
        and float(row["worst_seed_improvement_mpa"]) >= -1.0
        and float(row["worst_fold_improvement_mpa"]) >= -1.0
    )


def relaxed_constraints(row):
    return bool(
        float(row["blend_lambda"]) > 0.0
        and float(row["mean_improvement_mpa"]) > 0.0
        and float(row["median_improvement_mpa"]) >= 0.0
        and int(row["run_wins"]) >= 8
        and int(row["seed_wins"]) >= 2
        and int(row["fold_wins"]) >= 3
        and float(row["worst_seed_improvement_mpa"]) >= -2.0
        and float(row["worst_fold_improvement_mpa"]) >= -2.0
    )


def select_row(grid):
    frame = grid.copy()
    frame["strict_pass"] = frame.apply(strict_constraints, axis=1)
    frame["relaxed_pass"] = frame.apply(relaxed_constraints, axis=1)
    frame["robust_score"] = (
        frame["mean_improvement_mpa"]
        - 0.20 * frame["improvement_std_mpa"]
        + 0.10 * frame["worst_fold_improvement_mpa"]
        + 0.10 * frame["worst_seed_improvement_mpa"]
    )

    strict = frame[frame["strict_pass"]].copy()
    if not strict.empty:
        best_mean = float(strict["mean_improvement_mpa"].max())
        near_best = strict[strict["mean_improvement_mpa"] >= 0.80 * best_mean].copy()
        selected = near_best.sort_values(
            ["blend_lambda", "worst_fold_improvement_mpa", "mean_improvement_mpa", "config"],
            ascending=[True, False, False, True],
        ).iloc[0]
        return selected, True, "STRICT_ROBUST_BLEND_PASS", frame

    relaxed = frame[frame["relaxed_pass"]].copy()
    if not relaxed.empty:
        selected = relaxed.sort_values(
            ["robust_score", "blend_lambda", "mean_improvement_mpa", "config"],
            ascending=[False, True, False, True],
        ).iloc[0]
        return selected, False, "RELAXED_RECOVERY_BLEND", frame

    positive = frame[(frame["blend_lambda"] > 0) & (frame["mean_improvement_mpa"] > 0)].copy()
    if not positive.empty:
        selected = positive.sort_values(
            ["robust_score", "blend_lambda", "mean_improvement_mpa", "config"],
            ascending=[False, True, False, True],
        ).iloc[0]
        return selected, False, "POSITIVE_INNER_FALLBACK", frame

    # Guaranteed baseline fallback. The first candidate is retained only for
    # raw-gate diagnostics; deployment lambda=0 means the formal predictor is
    # exactly the Baseline.
    fallback = frame[frame["blend_lambda"] == 0.0].sort_values("config").iloc[0]
    return fallback, False, "BASELINE_ONLY_FALLBACK", frame


def launch_all_candidates(args, plan_json, outdir):
    command = [
        sys.executable, "-m", "materon_elemnet_tf29.training.launch_resident",
        "--mode", "plan_screen",
        "--cache-dir", str(Path(args.cache_dir).resolve()),
        "--outdir", str(Path(outdir).resolve()),
        "--initial-weights-dir", str(Path(args.initial_weights_dir).resolve()),
        "--acceptance-report", str(Path(args.acceptance_report).resolve()),
        "--protocol-json", str(Path(args.protocol_json).resolve()),
        "--plan-json", str(Path(plan_json).resolve()),
        "--gpu-ids", args.gpu_ids,
        "--folds", "0", "1", "2", "3", "4",
        "--screen-seeds", "42", "123", "2024",
        "--purpose", "V14 three-seed inner blend selection",
        "--verbose", str(args.verbose),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.check_call(command)


def main():
    parser = argparse.ArgumentParser(description="V14 Baseline-anchored blend selection")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--acceptance-report", required=True)
    parser.add_argument("--initial-weights-dir", required=True)
    parser.add_argument("--baseline-results-dir", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--selected-models-dir", required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache = CachedDataset.load(args.cache_dir, verify_hashes=True)
    input_dim = cache.feature_view("legacy_engineered").shape[1]
    configs = v14_candidate_registry(input_dim, [1024, 512, 256, 128])
    specs = {name: model_config_to_dict(config) for name, config in configs.items()}

    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "execution_status": "READY",
        "selection_basis": "all five candidates, five folds, seeds 42/123/2024; outer test untouched",
        "configs": specs,
    }
    plan_json = root / "CONFIG_PLAN.json"
    atomic_write_json(plan_json, plan)
    runs_dir = root / "candidate_runs"
    launch_all_candidates(args, plan_json, runs_dir)

    baseline_results = collect_screen_results(args.baseline_results_dir)
    candidate_results = collect_screen_results(runs_dir)
    baseline_runs = baseline_results[
        baseline_results["config"] == "H0_exact_baseline"
    ][["fold", "seed", "mae"]].copy()
    if len(baseline_runs) != 15:
        raise RuntimeError("Expected 15 baseline inner runs, got %d" % len(baseline_runs))
    baseline_predictions = prediction_frame(baseline_results, "H0_exact_baseline").rename(
        columns={"y_pred": "baseline_pred"}
    )

    grid_rows = []
    for name, config in configs.items():
        gate_predictions = prediction_frame(candidate_results, name).rename(
            columns={"y_pred": "gate_pred"}
        )
        merged = baseline_predictions.merge(
            gate_predictions[["fold", "seed", "sample_id", "y_true", "gate_pred"]],
            on=["fold", "seed", "sample_id", "y_true"],
            how="inner",
        )
        if merged.groupby(["fold", "seed"]).ngroups != 15:
            raise RuntimeError("Incomplete merged predictions for %s" % name)
        config_rows = candidate_results[candidate_results["config"] == name]
        gate_count = int(config_rows["gate_parameter_count"].max())
        median_epoch = float(config_rows["best_epoch"].median())
        for blend_lambda in LAMBDA_GRID:
            blended = merged[["fold", "seed", "sample_id", "y_true"]].copy()
            blended["y_pred"] = (
                (1.0 - float(blend_lambda)) * merged["baseline_pred"]
                + float(blend_lambda) * merged["gate_pred"]
            )
            candidate_runs = run_metrics(blended, "%s__blend_%s" % (name, blend_lambda))
            grid_rows.append(summarize_pair(
                baseline_runs, candidate_runs, name, blend_lambda,
                gate_count, median_epoch,
            ))

    grid = pd.DataFrame(grid_rows)
    selected, scientific_pass, decision, grid = select_row(grid)
    selected_name = str(selected["config"])
    selected_lambda = float(selected["blend_lambda"])
    selected_config = configs[selected_name]
    control = plain_additive_control(selected_config)

    grid.sort_values(
        ["strict_pass", "relaxed_pass", "robust_score", "mean_improvement_mpa"],
        ascending=[False, False, False, False],
    ).to_csv(root / "INNER_BLEND_GRID.csv", index=False)

    payload = {
        "execution_status": "PASS",
        "all_pass": True,
        "scientific_gate_pass": bool(scientific_pass),
        "selection_decision": decision,
        "selection_basis": "inner validation predictions only; five folds and three seeds; official outer test untouched",
        "baseline_config": "H0_exact_baseline",
        "primary_gate_config": selected_name,
        "blend_lambda": selected_lambda,
        "lambda_grid": [float(value) for value in LAMBDA_GRID],
        "selected_inner_metrics": {
            key: (bool(value) if isinstance(value, (np.bool_, bool)) else
                  int(value) if isinstance(value, (np.integer,)) else
                  float(value) if isinstance(value, (np.floating,)) else value)
            for key, value in selected.to_dict().items()
        },
        "mechanism_control_config": control.name,
        "configs": {
            selected_name: model_config_to_dict(selected_config),
            control.name: model_config_to_dict(control),
        },
        "save_model_configs": [selected_name],
        "save_model_roles": {selected_name: "selected_gate"},
        "selected_models_dir": str(Path(args.selected_models_dir).resolve()),
        "formal_reports": [
            "raw_gate",
            "baseline_anchored_gate_blend",
            "baseline_anchored_plain_control_blend",
            "three_seed_ensemble",
        ],
        "retention_policy": {
            "screen_weights": "NONE",
            "screen_predictions": "COMPACT_CSV_ONLY",
            "formal_initial_save": ["baseline", "selected_gate"],
            "formal_control_weights": "NOT_SAVED",
            "post_formal_pruning": "keep baseline only on blend failure; otherwise baseline plus selected gate",
        },
    }
    atomic_write_json(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
