from __future__ import absolute_import, division, print_function

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import atomic_write_json, read_json
from ..utils.metrics import regression_metrics


def collect_results(results_dir):
    rows = []
    for path in sorted(Path(results_dir).rglob("result.json")):
        value = read_json(path)
        if value.get("execution_status") != "PASS":
            continue
        metrics = value["metrics"]
        prediction_path = value.get("prediction_file") or str(path.parent / "predictions.csv")
        if not Path(prediction_path).is_file():
            raise FileNotFoundError("Missing formal prediction file: %s" % prediction_path)
        rows.append({
            "config": str(value["config"]),
            "fold": int(value["fold"]),
            "seed": int(value["seed"]),
            "formal_outer_mae": float(metrics["mae"]),
            "rmse": float(metrics.get("rmse", np.nan)),
            "r2": float(metrics.get("r2", np.nan)),
            "best_epoch": int(metrics.get("discovery_best_epoch", 0)),
            "epochs_ran": int(metrics.get("discovery_epochs_ran", 0)),
            "parameter_count": int(metrics.get("parameter_count", 0)),
            "gate_parameter_count": int(metrics.get("gate_parameter_count", 0)),
            "formal_strategy": str(value.get("formal_strategy", "unknown")),
            "prediction_path": str(prediction_path),
            "saved_model": bool(value.get("save_model", False)),
            "selected_model_weight_file": value.get("selected_model_weight_file"),
            "result_path": str(path),
        })
    return pd.DataFrame(rows)


def prediction_frame(frame, config):
    parts = []
    for row in frame[frame["config"] == config].itertuples(index=False):
        pred = pd.read_csv(row.prediction_path)
        pred = pred[["sample_id", "y_true", "y_pred"]].copy()
        pred["fold"] = int(row.fold)
        pred["seed"] = int(row.seed)
        pred["config"] = str(config)
        parts.append(pred)
    if not parts:
        raise RuntimeError("No predictions for config=%s" % config)
    return pd.concat(parts, ignore_index=True)


def run_metrics_from_predictions(predictions, label):
    rows = []
    for (fold, seed), group in predictions.groupby(["fold", "seed"]):
        metrics = regression_metrics(group["y_true"].values, group["y_pred"].values)
        rows.append({
            "config": str(label),
            "fold": int(fold),
            "seed": int(seed),
            "mae": float(metrics["mae"]),
        })
    return pd.DataFrame(rows)


def bootstrap_ci(values, seed=1401, repeats=20000):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return [None, None]
    rng = np.random.RandomState(int(seed))
    sample = rng.choice(array, size=(int(repeats), array.size), replace=True).mean(axis=1)
    return [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]


def paired_metrics(baseline_runs, candidate_runs, label):
    base = baseline_runs[["fold", "seed", "mae"]].rename(columns={"mae": "baseline_mae"})
    cand = candidate_runs[["fold", "seed", "mae"]].rename(columns={"mae": "candidate_mae"})
    paired = base.merge(cand, on=["fold", "seed"], how="inner")
    if len(paired) != 15:
        raise RuntimeError("Expected 15 paired rows for %s, got %d" % (label, len(paired)))
    paired["improvement"] = paired["baseline_mae"] - paired["candidate_mae"]
    paired["relative_improvement_percent"] = (
        100.0 * paired["improvement"] / paired["baseline_mae"]
    )
    seed_mean = paired.groupby("seed")["improvement"].mean()
    fold_mean = paired.groupby("fold")["improvement"].mean()
    return {
        "config": str(label),
        "baseline_mean_mae": float(paired["baseline_mae"].mean()),
        "candidate_mean_mae": float(paired["candidate_mae"].mean()),
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
        "paired_bootstrap_95ci_mpa": bootstrap_ci(paired["improvement"].values),
    }


def merge_predictions(frame, baseline_config, candidate_config, blend_lambda, label):
    base = prediction_frame(frame, baseline_config).rename(columns={"y_pred": "baseline_pred"})
    cand = prediction_frame(frame, candidate_config).rename(columns={"y_pred": "candidate_pred"})
    merged = base.merge(
        cand[["fold", "seed", "sample_id", "y_true", "candidate_pred"]],
        on=["fold", "seed", "sample_id", "y_true"],
        how="inner",
    )
    merged["y_pred"] = (
        (1.0 - float(blend_lambda)) * merged["baseline_pred"]
        + float(blend_lambda) * merged["candidate_pred"]
    )
    merged["config"] = str(label)
    return merged[[
        "fold", "seed", "sample_id", "y_true", "baseline_pred",
        "candidate_pred", "y_pred", "config"
    ]]


def seed_ensemble_predictions(predictions, label):
    parts = []
    for fold, group in predictions.groupby("fold"):
        pivot = group.pivot_table(
            index=["sample_id", "y_true"], columns="seed", values="y_pred"
        )
        if pivot.shape[1] != 3:
            raise RuntimeError("Expected three seeds for %s fold=%s" % (label, fold))
        part = pd.DataFrame({
            "sample_id": pivot.index.get_level_values("sample_id").values,
            "y_true": pivot.index.get_level_values("y_true").values,
            "y_pred": pivot.mean(axis=1).values,
        })
        part["fold"] = int(fold)
        part["config"] = str(label)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def seed_ensemble_metrics(baseline_predictions, candidate_predictions, label):
    base = seed_ensemble_predictions(baseline_predictions, "BASELINE_SEED_ENSEMBLE")
    cand = seed_ensemble_predictions(candidate_predictions, label)
    rows = []
    for fold in sorted(base["fold"].unique()):
        b = base[base["fold"] == fold]
        c = cand[cand["fold"] == fold]
        merged = b.merge(c[["sample_id", "y_true", "y_pred"]],
                         on=["sample_id", "y_true"], suffixes=("_base", "_cand"))
        bmae = regression_metrics(merged["y_true"], merged["y_pred_base"])["mae"]
        cmae = regression_metrics(merged["y_true"], merged["y_pred_cand"])["mae"]
        rows.append({
            "config": str(label),
            "fold": int(fold),
            "baseline_mae": float(bmae),
            "candidate_mae": float(cmae),
            "improvement": float(bmae - cmae),
        })
    table = pd.DataFrame(rows)
    return table, {
        "config": str(label),
        "folds": int(len(table)),
        "baseline_seed_ensemble_mean_mae": float(table["baseline_mae"].mean()),
        "candidate_seed_ensemble_mean_mae": float(table["candidate_mae"].mean()),
        "mean_improvement_mpa": float(table["improvement"].mean()),
        "aggregate_relative_improvement_percent": float(
            100.0 * table["improvement"].mean() / table["baseline_mae"].mean()
        ),
        "fold_wins": int((table["improvement"] > 0).sum()),
        "worst_fold_improvement_mpa": float(table["improvement"].min()),
    }


def summarize(results_dir, outdir, selection_json, baseline_config="H0_exact_baseline"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selection = read_json(selection_json)
    frame = collect_results(results_dir)
    gate = str(selection["primary_gate_config"])
    control = str(selection["mechanism_control_config"])
    blend_lambda = float(selection["blend_lambda"])
    expected = [baseline_config, gate, control]
    counts = frame.groupby("config").size().to_dict()
    incomplete = {name: int(counts.get(name, 0)) for name in expected if int(counts.get(name, 0)) != 15}
    if incomplete:
        raise RuntimeError("Incomplete formal configs: %r" % incomplete)
    if set(frame["formal_strategy"].unique()) != {"discovery_checkpoint"}:
        raise RuntimeError("Unexpected formal strategy: %r" % sorted(frame["formal_strategy"].unique()))
    frame.to_csv(outdir / "FORMAL_ALL_RUNS.csv", index=False)

    baseline_predictions = prediction_frame(frame, baseline_config)
    gate_predictions = prediction_frame(frame, gate)
    control_predictions = prediction_frame(frame, control)
    baseline_runs = run_metrics_from_predictions(baseline_predictions, baseline_config)
    raw_gate_runs = run_metrics_from_predictions(gate_predictions, gate)
    raw_control_runs = run_metrics_from_predictions(control_predictions, control)

    gate_blend_label = "BASELINE_ANCHORED_EIGATE_BLEND"
    control_blend_label = "BASELINE_ANCHORED_PLAIN_CONTROL_BLEND"
    gate_blend_predictions = merge_predictions(
        frame, baseline_config, gate, blend_lambda, gate_blend_label
    )
    control_blend_predictions = merge_predictions(
        frame, baseline_config, control, blend_lambda, control_blend_label
    )
    gate_blend_runs = run_metrics_from_predictions(gate_blend_predictions, gate_blend_label)
    control_blend_runs = run_metrics_from_predictions(control_blend_predictions, control_blend_label)

    metrics = {
        "raw_gate": paired_metrics(baseline_runs, raw_gate_runs, gate),
        "raw_control": paired_metrics(baseline_runs, raw_control_runs, control),
        "gate_blend": paired_metrics(baseline_runs, gate_blend_runs, gate_blend_label),
        "control_blend": paired_metrics(baseline_runs, control_blend_runs, control_blend_label),
    }

    # Direct paired comparison between the two blended mechanisms.
    gc = gate_blend_runs.rename(columns={"mae": "gate_blend_mae"}).merge(
        control_blend_runs.rename(columns={"mae": "control_blend_mae"}),
        on=["fold", "seed"],
    )
    gc["eigate_vs_control_improvement"] = gc["control_blend_mae"] - gc["gate_blend_mae"]
    gate_vs_control = {
        "mean_improvement_mpa": float(gc["eigate_vs_control_improvement"].mean()),
        "median_improvement_mpa": float(gc["eigate_vs_control_improvement"].median()),
        "run_wins": int((gc["eigate_vs_control_improvement"] > 0).sum()),
        "bootstrap_95ci_mpa": bootstrap_ci(gc["eigate_vs_control_improvement"].values, seed=1402),
    }

    gate_blend_predictions.to_csv(outdir / "FORMAL_BASELINE_ANCHORED_EIGATE_BLEND_PREDICTIONS.csv", index=False)
    control_blend_predictions.to_csv(outdir / "FORMAL_BASELINE_ANCHORED_CONTROL_BLEND_PREDICTIONS.csv", index=False)
    pd.concat([baseline_runs, raw_gate_runs, raw_control_runs, gate_blend_runs, control_blend_runs],
              ignore_index=True).to_csv(outdir / "FORMAL_MODEL_AND_BLEND_RUNS.csv", index=False)

    seed_tables = []
    seed_metrics = {}
    for label, predictions in [
        (gate, gate_predictions),
        (control, control_predictions),
        (gate_blend_label, gate_blend_predictions),
        (control_blend_label, control_blend_predictions),
    ]:
        table, values = seed_ensemble_metrics(baseline_predictions, predictions, label)
        seed_tables.append(table)
        seed_metrics[label] = values
    pd.concat(seed_tables, ignore_index=True).to_csv(
        outdir / "FORMAL_SEED_ENSEMBLE_FOLDS.csv", index=False
    )

    blend_values = metrics["gate_blend"]
    ensemble_values = seed_metrics[gate_blend_label]
    scientific_pass = bool(
        float(blend_values["aggregate_relative_improvement_percent"]) >= 3.0
        and int(blend_values["run_wins"]) >= 9
        and int(blend_values["seed_wins"]) >= 2
        and int(blend_values["fold_wins"]) >= 3
        and int(ensemble_values["fold_wins"]) >= 3
    )
    deployment_gate_pass = bool(
        float(blend_values["mean_improvement_mpa"]) > 0.0
        and int(blend_values["run_wins"]) >= 9
        and int(blend_values["seed_wins"]) >= 2
        and int(blend_values["fold_wins"]) >= 3
        and float(ensemble_values["mean_improvement_mpa"]) > 0.0
        and int(ensemble_values["fold_wins"]) >= 3
    )

    saved = frame[frame["saved_model"]]
    saved_configs = sorted(saved["config"].unique().tolist())
    expected_saved = sorted([baseline_config, gate])
    payload = {
        "execution_status": "PASS",
        "all_pass": saved_configs == expected_saved and int(len(saved)) == 30,
        "scientific_gate_pass": scientific_pass,
        "deployment_gate_pass": deployment_gate_pass,
        "deployment_decision": (
            "KEEP_BASELINE_AND_GATE_BLEND" if deployment_gate_pass else "FALL_BACK_TO_BASELINE_ONLY"
        ),
        "formal_strategy": "discovery_checkpoint",
        "full_outer_train_refit": False,
        "outer_test_used_for_model_or_lambda_selection": False,
        "baseline_config": baseline_config,
        "baseline_formal_outer_mean_mae": float(baseline_runs["mae"].mean()),
        "selected_gate_config": gate,
        "mechanism_control_config": control,
        "blend_lambda": blend_lambda,
        "inner_selection_decision": selection.get("selection_decision"),
        "metrics": metrics,
        "eigate_blend_vs_control_blend": gate_vs_control,
        "seed_ensemble_metrics": seed_metrics,
        "saved_model_configs_before_pruning": saved_configs,
        "expected_saved_model_configs_before_pruning": expected_saved,
        "saved_model_runs_before_pruning": int(len(saved)),
        "post_formal_retention": (
            [baseline_config, gate] if deployment_gate_pass else [baseline_config]
        ),
        "runs": int(len(frame)),
    }
    atomic_write_json(outdir / "FORMAL_SUMMARY.json", payload)

    lines = [
        "# Formal outer-fold results", "",
        "- Formal runs use the discovery checkpoint restored by early stopping; no re-initialized refit.",
        "- The blend weight is chosen on inner-validation predictions only; outer test data are not used for selection.",
        "- If the deployment check fails, the model falls back to the baseline and the gate weights are removed.", "",
        "| Result | MAE | Relative improvement | Run/seed/fold wins |",
        "|---|---:|---:|---:|",
    ]
    for label, values in [
        ("Raw EIGate", metrics["raw_gate"]),
        ("Baseline-anchored EIGate", metrics["gate_blend"]),
        ("Baseline-anchored Plain Control", metrics["control_blend"]),
    ]:
        lines.append("| %s | %.3f | %.3f%% | %d/%d/%d |" % (
            label,
            float(values["candidate_mean_mae"]),
            float(values["aggregate_relative_improvement_percent"]),
            int(values["run_wins"]),
            int(values["seed_wins"]),
            int(values["fold_wins"]),
        ))
    lines.extend([
        "", "- Selected global blend lambda: `%.3f`." % blend_lambda,
        "- Deployment decision: `%s`." % payload["deployment_decision"],
    ])
    (outdir / "FORMAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--baseline-config", default="H0_exact_baseline")
    args = parser.parse_args()
    summarize(args.results_dir, args.outdir, args.selection_json, args.baseline_config)


if __name__ == "__main__":
    main()
