#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one predefined seed over all requested Matbench folds.

The process owns exactly one visible GPU.  A baseline or Materon model is built
fresh for each fold.  Materon is built through the installed official kgcnn builder with one
identity-initialized node-embedding adapter, then receives the exact baseline
backbone weights before training.
"""

from __future__ import annotations

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))

import os

# Conservative defaults must be applied before NumPy/TensorFlow imports.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
os.environ.setdefault("TF_GPU_THREAD_COUNT", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import gc
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from matbench.bench import MatbenchBenchmark

from materon_tf29_engine.training.identity import assert_identity, assert_structural_identity
from materon_tf29_engine.training.cache import CompositionTaskCache, GraphTaskCache
from materon_tf29_engine.training.metrics import evaluate, make_training_metrics
from materon_tf29_engine.training.model_factory import build_model
from materon_tf29_engine.training.runtime import (
    configure_tensorflow_from_environment,
    derived_seed,
    set_random_streams,
)
from materon_tf29_engine.training.schedule import make_linear_callback
from materon_tf29_engine.training.callbacks import EpochPerformanceLogger
from materon_tf29_engine.training.task_registry import (
    task_spec,
    validate_task_size_and_type,
    validate_backbone_modality,
)
from materon_tf29_engine.core.small_data_ei import AdapterActivationCallback
from materon_tf29_engine.explain.export import save_adapter_state


DEFAULTS = {
    "megnet": {"epochs": 1000, "batch_size": 32, "learning_rate": 5.0e-4},
    "cgcnn": {"epochs": 800, "batch_size": 32, "learning_rate": 1.0e-3},
    "elemnet": {"epochs": 500, "batch_size": 64, "learning_rate": 1.0e-3},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["megnet", "cgcnn", "elemnet"], required=True)
    parser.add_argument("--variant", choices=["baseline", "materon"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-stop", type=float, default=5.0e-6)
    parser.add_argument("--decay-start-epoch", type=int, default=100)
    parser.add_argument("--max-atomic-number", type=int, default=100)
    parser.add_argument("--feature-mode", choices=["grouped_pca","full20","none","z_only","shuffled_grouped_pca"], default="grouped_pca")
    parser.add_argument("--identity-atol", type=float, default=1.0e-6)
    parser.add_argument("--identity-rtol", type=float, default=5.0e-7)
    parser.add_argument("--identity-max-ulp", type=int, default=4)
    parser.add_argument("--identity-tolerance", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/matbench_v0_1"))
    parser.add_argument("--rebuild-cache", action="store_true", help="Single-process debug only; use build_graph_cache.py --rebuild before a 4-GPU launch.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--strict-cross-graph-identity", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_folds(text):
    folds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not folds or any(x not in range(5) for x in folds):
        raise ValueError("Invalid folds: %s" % text)
    return folds


def slice_sample(inputs, count=4):
    if isinstance(inputs, (tuple, list)):
        return [x[:count] for x in inputs]
    return inputs[:count]


def fit_target_scaler(task_type, y_train):
    if task_type == "classification":
        return None, np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    scaled = scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).astype(np.float32)
    return scaler, scaled


def inverse_predictions(task_type, scaler, predictions):
    pred = np.asarray(predictions).reshape(-1, 1)
    if task_type == "classification":
        return pred.reshape(-1)
    return scaler.inverse_transform(pred).reshape(-1)


def main():
    args = parse_args()
    folds = parse_folds(args.folds)
    defaults = DEFAULTS[args.backbone]
    epochs = int(args.epochs or defaults["epochs"])
    batch_size = int(args.batch_size or defaults["batch_size"])
    learning_rate = float(args.learning_rate or defaults["learning_rate"])

    runtime = configure_tensorflow_from_environment(require_one_gpu=True)
    print("=" * 100, flush=True)
    print("host=%s pid=%d" % (socket.gethostname(), os.getpid()), flush=True)
    print(
        "backbone=%s variant=%s task=%s base_seed=%d"
        % (args.backbone, args.variant, args.task, args.seed),
        flush=True,
    )
    print("runtime=%s" % runtime, flush=True)
    print("deterministic_ops=OFF", flush=True)
    print("=" * 100, flush=True)

    benchmark = MatbenchBenchmark(subset=[args.task], autoload=False)
    task = next(iter(benchmark.tasks))
    task.load()
    fold_ids = list(task.folds)
    task_type = str(task.metadata["task_type"])
    registry_spec = validate_task_size_and_type(args.task, task)
    modality = validate_backbone_modality(args.backbone, args.task)
    print("[MODALITY] %s" % modality["message"], flush=True)

    if args.backbone in {"megnet", "cgcnn"}:
        cache = GraphTaskCache(
            task,
            args.backbone,
            cache_dir=args.cache_dir,
            rebuild_cache=args.rebuild_cache,
        )
    else:
        cache = CompositionTaskCache(
            task,
            max_atomic_number=args.max_atomic_number,
            cache_dir=args.cache_dir,
            rebuild_cache=args.rebuild_cache,
        )

    seed_dir = (
        args.outdir
        / args.backbone
        / args.task
        / args.variant
        / ("seed_%d" % args.seed)
    )
    seed_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold_index in folds:
        fold_dir = seed_dir / ("fold_%d" % fold_index)
        metrics_path = fold_dir / "metrics.json"
        prediction_path = fold_dir / "predictions.npy"
        if metrics_path.exists() and prediction_path.exists() and not args.force:
            with open(metrics_path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            predictions = np.load(prediction_path)
            task.record(fold_ids[fold_index], predictions, params=metrics)
            rows.append(metrics)
            print(
                "[SKIP] seed=%d fold=%d primary=%s"
                % (args.seed, fold_index, metrics.get("primary_metric")),
                flush=True,
            )
            continue

        fold_dir.mkdir(parents=True, exist_ok=True)
        tf.keras.backend.clear_session()
        gc.collect()
        run_seed = derived_seed(args.seed, fold_index)
        set_random_streams(run_seed)

        x_train, y_train, x_test, y_test = cache.fold(fold_ids[fold_index])
        y_scaler, y_fit = fit_target_scaler(task_type, y_train)

        model, baseline_reference, inserted_layer = build_model(
            backbone=args.backbone,
            variant=args.variant,
            task_type=task_type,
            max_atomic_number=args.max_atomic_number,
            feature_mode=args.feature_mode,
            shuffle_seed=run_seed if args.feature_mode=="shuffled_grouped_pca" else None,
        )

        identity_difference = None
        identity_report = None
        if baseline_reference is not None:
            identity_atol = (
                float(args.identity_tolerance)
                if args.identity_tolerance is not None
                else float(args.identity_atol)
            )
            identity_function = (
                assert_identity
                if args.strict_cross_graph_identity
                else assert_structural_identity
            )
            identity_difference, backbone_parity = identity_function(
                baseline_reference,
                model,
                slice_sample(x_train, count=4),
                **(
                    {
                        "atol": identity_atol,
                        "rtol": float(args.identity_rtol),
                        "max_ulp": int(args.identity_max_ulp),
                    }
                    if args.strict_cross_graph_identity
                    else {
                        "diagnostic_atol": identity_atol,
                        "diagnostic_rtol": float(args.identity_rtol),
                        "diagnostic_max_ulp": int(args.identity_max_ulp),
                    }
                )
            )
            full_numerics = backbone_parity["full_model_numerics"]
            identity_report = {
                "backbone_max_abs": float(
                    backbone_parity["max_abs_difference"]
                ),
                "adapter_max_abs": float(
                    backbone_parity["adapter_identity"][
                        "max_abs_difference"
                    ]
                ),
                "full_model_max_abs": float(
                    full_numerics["max_abs_difference"]
                ),
                "full_model_max_ulp": int(
                    full_numerics["max_ulp_distance"]
                ),
                "cross_graph_diagnostic_only": not args.strict_cross_graph_identity,
                "full_model_max_scale_normalized": float(
                    full_numerics["max_scale_normalized_difference"]
                ),
                "passed": True,
                "structural_identity_passed": True,
                "cross_graph_diagnostic_passed": bool(full_numerics["passed"]),
            }
            print(
                "[IDENTITY] seed=%d fold=%d layer=%s "
                "backbone=%.1e adapter=%.1e output=%.3e ulp=%d diagnostic=%s"
                % (
                    args.seed,
                    fold_index,
                    inserted_layer,
                    identity_report["backbone_max_abs"],
                    identity_report["adapter_max_abs"],
                    identity_report["full_model_max_abs"],
                    identity_report["full_model_max_ulp"],
                    ("PASS" if identity_report["cross_graph_diagnostic_passed"] else "OUTSIDE"),
                ),
                flush=True,
            )

        loss = (
            "binary_crossentropy"
            if task_type == "classification"
            else "mean_absolute_error"
        )
        target_scale = (
            1.0
            if y_scaler is None
            else float(y_scaler.scale_[0])
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss=loss,
            metrics=make_training_metrics(
                task_type,
                target_scale=target_scale,
            ),
        )
        callback = make_linear_callback(
            start=learning_rate,
            stop=args.learning_rate_stop,
            epochs=epochs,
            decay_start=args.decay_start_epoch,
        )
        unit = (
            registry_spec["unit"]
            if registry_spec is not None
            else ""
        )
        performance_callback = EpochPerformanceLogger(
            epochs=epochs,
            train_size=len(y_train),
            task_type=task_type,
            unit=unit,
            log_every=args.log_every,
        )

        # Reset both NumPy and TensorFlow streams immediately before fit so that
        # baseline and Materon receive the same shuffle stream for the same pair.
        fit_seed = run_seed + 7919
        set_random_streams(fit_seed)
        print(
            "[TRAIN] seed=%d run_seed=%d fit_seed=%d fold=%d train=%d test=%d"
            % (args.seed, run_seed, fit_seed, fold_index, len(y_train), len(y_test)),
            flush=True,
        )
        start = time.time()
        history = model.fit(
            x_train,
            y_fit,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=(
                [callback, AdapterActivationCallback(raw_alpha=0.01), performance_callback]
                if args.variant == "materon"
                else [callback, performance_callback]
            ),
            shuffle=True,
        )
        train_seconds = float(time.time() - start)

        raw_prediction = model.predict(x_test, verbose=1)
        predictions = inverse_predictions(task_type, y_scaler, raw_prediction)
        metric_values = evaluate(task_type, y_test, predictions)
        primary_name = "roc_auc" if task_type == "classification" else "mae"
        primary_metric = float(metric_values[primary_name])

        np.save(prediction_path, predictions)
        np.save(fold_dir / "targets.npy", np.asarray(y_test))
        model.save_weights(str(fold_dir / "weights.h5"))
        if args.variant == "materon": save_adapter_state(model, fold_dir / "adapter_state.json")
        history_frame = pd.DataFrame(history.history)
        history_frame.index.name = "epoch_zero_based"
        history_frame.to_csv(fold_dir / "history.csv")

        metrics = {
            "backbone": args.backbone,
            "variant": args.variant,
            "task": args.task,
            "task_type": task_type,
            "base_seed": int(args.seed),
            "run_seed": int(run_seed),
            "fit_seed": int(fit_seed),
            "fold": int(fold_index),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_stop": float(args.learning_rate_stop),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "train_seconds": train_seconds,
            "inserted_layer": inserted_layer,
            "identity_max_abs_difference": identity_difference,
            "identity_report": identity_report,
            "feature_mode": args.feature_mode,
            "primary_metric_name": primary_name,
            "primary_metric": primary_metric,
            "metrics": metric_values,
            "runtime": runtime,
            "cache": cache.cache_info,
            "input_modality": modality,
            "task_registry": registry_spec,
        }
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)

        task.record(fold_ids[fold_index], predictions, params=metrics)
        rows.append(metrics)
        print(
            "[DONE] seed=%d fold=%d %s=%.8f time=%.1f min"
            % (
                args.seed,
                fold_index,
                primary_name,
                primary_metric,
                train_seconds / 60.0,
            ),
            flush=True,
        )

    result_path = seed_dir / "results.json.gz"
    if set(folds) == {0, 1, 2, 3, 4}:
        benchmark.to_file(str(result_path))
    else:
        with open(seed_dir / "PARTIAL_FOLDS.txt", "w", encoding="utf-8") as handle:
            handle.write(
                "Only folds %s were requested; Matbench results.json.gz was not written.\n"
                % folds
            )
    frame = pd.DataFrame(rows).sort_values("fold")
    frame.to_json(seed_dir / "fold_metrics.json", orient="records", indent=2)

    primary_values = frame["primary_metric"].astype(float)
    summary = {
        "backbone": args.backbone,
        "variant": args.variant,
        "task": args.task,
        "base_seed": int(args.seed),
        "primary_metric_name": str(frame.iloc[0]["primary_metric_name"]),
        "mean_primary_metric": float(primary_values.mean()),
        "std_primary_metric_population": float(primary_values.std(ddof=0)),
        "std_primary_metric_sample": float(primary_values.std(ddof=1)),
        "num_folds": int(len(frame)),
    }
    with open(seed_dir / "seed_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("=" * 100, flush=True)
    print("SEED COMPLETE %s" % summary, flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
