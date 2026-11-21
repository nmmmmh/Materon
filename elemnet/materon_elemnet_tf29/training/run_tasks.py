from __future__ import absolute_import, division, print_function

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.config_serialization import model_config_from_dict, model_config_to_dict
from ..models.registry import get_registry
from ..utils.io import atomic_write_json, read_json
from .preprocess import PerformancePreprocessor
from .protocols import get_protocol
from .train_core import analysis_summary, fit_discovery, predict_original, regression_metrics


def resolve_config(cache, task):
    if task.get("config_spec") is not None:
        return model_config_from_dict(task["config_spec"])
    x_all = cache.feature_view("legacy_engineered")
    registry = get_registry(task["stage"], x_all.shape[1], task.get("hidden_units"))
    name = task["config"]
    if name not in registry:
        raise KeyError("Unknown config=%r stage=%r" % (name, task["stage"]))
    return registry[name]


def _protocol_from_selection(path):
    selection = read_json(path)
    if not selection.get("all_pass", False):
        raise RuntimeError("Protocol selection did not pass: %s" % path)
    return get_protocol(selection["selected_protocol"]), selection


def _run_dir(task, config):
    return (
        Path(task["outdir"]) / config.name /
        ("fold_%d" % int(task["fold"])) /
        ("seed_%d" % int(task["seed"]))
    )


def run_config_screen(task, cache, initial_weight_store):
    protocol, _ = _protocol_from_selection(task["protocol_json"])
    x_all = cache.feature_view("legacy_engineered")
    config = resolve_config(cache, task)
    split = read_json(cache.root / "splits" / "inner_splits.json")[
        "fold_%d/seed_%d" % (int(task["fold"]), int(task["seed"]))
    ]
    train_idx = np.asarray(split["train"], dtype=int)
    val_idx = np.asarray(split["val"], dtype=int)
    run_dir = _run_dir(task, config)
    result_path = run_dir / "screen_result.json"
    if result_path.is_file() and not task.get("overwrite", False):
        return {"execution_status": "SKIP", "path": str(result_path)}

    processor = PerformancePreprocessor.fit(
        x_all[train_idx], cache.targets[train_idx],
        input_scaler=protocol.input_scaler,
        target_scaling=protocol.target_scaling,
    )
    save_history = bool(task.get("save_history", False))
    output = fit_discovery(
        config, protocol, task["seed"],
        processor.transform_x(x_all[train_idx]),
        processor.transform_y(cache.targets[train_idx]),
        processor.transform_x(x_all[val_idx]),
        processor.transform_y(cache.targets[val_idx]),
        run_dir=run_dir if save_history else None,
        verbose=task.get("verbose", 0),
        initial_weight_store=initial_weight_store,
        save_history=save_history,
    )
    val_x = processor.transform_x(x_all[val_idx])
    val_pred = predict_original(
        output["model"], val_x, processor.inverse_y,
        batch_size=protocol.batch_size,
    )
    metrics = regression_metrics(cache.targets[val_idx], val_pred)
    run_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = run_dir / "validation_predictions.csv"
    # V14 deliberately stores only compact validation predictions during the
    # search. They are required to select a global Baseline/Gate blend weight.
    pd.DataFrame({
        "sample_id": val_idx,
        "composition": cache.compositions[val_idx],
        "y_true": cache.targets[val_idx],
        "y_pred": val_pred,
        "residual": val_pred - cache.targets[val_idx],
    }).to_csv(prediction_path, index=False)

    if bool(task.get("save_analysis", False)):
        atomic_write_json(
            run_dir / "analysis_summary.json",
            analysis_summary(output["model"], val_x),
        )

    result = {
        "execution_status": "PASS",
        "scientific_gate_pass": None,
        "purpose": task.get("purpose", "V14 inner prediction screen"),
        "official_test_accessed": False,
        "stage": task.get("stage", "plan"),
        "config": config.name,
        "config_spec": task.get("config_spec"),
        "protocol": protocol.to_dict(),
        "fold": int(task["fold"]),
        "seed": int(task["seed"]),
        "validation_metrics_mpa": metrics,
        "best_epoch": int(output["best_epoch"]),
        "epochs_ran": int(output["epochs_ran"]),
        "elapsed_seconds": float(output["elapsed_seconds"]),
        "parameter_count": int(output["model"].count_params()),
        "gate_parameter_count": int(output["model"].gate_parameter_count()),
        "initial_backbone_sha256": output["initial_backbone_sha256"],
        "initial_gate_sha256": output["initial_gate_sha256"],
        "weight_file_saved": False,
        "prediction_file": str(prediction_path),
        "predictions_saved": True,
        "history_saved": save_history,
    }
    atomic_write_json(result_path, result)
    return result


def _save_selected_model(task, config, model, processor, fold, seed, metrics):
    selected_root = Path(task["selected_models_dir"])
    model_dir = selected_root / config.name / ("fold_%d" % fold) / ("seed_%d" % seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_path = model_dir / "model.weights.h5"
    model.save_weights(str(weight_path))
    processor_path = model_dir / "discovery_preprocessor.joblib"
    processor.save(processor_path)
    atomic_write_json(model_dir / "model_manifest.json", {
        "execution_status": "PASS",
        "role": task.get("save_model_role"),
        "config": config.name,
        "config_spec": model_config_to_dict(config),
        "fold": int(fold),
        "seed": int(seed),
        "metrics": metrics,
        "weight_file": str(weight_path),
        "preprocessor_file": str(processor_path),
        "training_strategy": "early-stopping discovery checkpoint; no restart/refit",
        "retention_policy": "baseline and selected gate only",
    })
    return str(weight_path)


def run_formal(task, cache, initial_weight_store):
    """Official outer evaluation using the EarlyStopping-restored checkpoint.

    The model is trained once on the inner-train subset, monitored on the
    inner-validation subset, and the restored best checkpoint predicts the
    official outer test directly. This avoids the refit restart that reversed
    the inner signal in the previous experiment.
    """
    protocol, selection = _protocol_from_selection(task["protocol_json"])
    if str(protocol.formal_strategy) != "discovery_checkpoint":
        raise RuntimeError("V14 requires discovery_checkpoint formal strategy")

    x_all = cache.feature_view("legacy_engineered")
    config = resolve_config(cache, task)
    run_dir = _run_dir(task, config)
    result_path = run_dir / "result.json"
    if result_path.is_file() and not task.get("overwrite", False):
        return {"execution_status": "SKIP", "path": str(result_path)}

    split = read_json(cache.root / "splits" / "inner_splits.json")[
        "fold_%d/seed_%d" % (int(task["fold"]), int(task["seed"]))
    ]
    train_idx = np.asarray(split["train"], dtype=int)
    val_idx = np.asarray(split["val"], dtype=int)
    test_idx = np.asarray(split["test"], dtype=int)
    if (
        set(train_idx).intersection(val_idx)
        or set(train_idx).intersection(test_idx)
        or set(val_idx).intersection(test_idx)
    ):
        raise AssertionError("train/validation/test leakage")

    processor = PerformancePreprocessor.fit(
        x_all[train_idx], cache.targets[train_idx],
        input_scaler=protocol.input_scaler,
        target_scaling=protocol.target_scaling,
    )
    discovery = fit_discovery(
        config, protocol, task["seed"],
        processor.transform_x(x_all[train_idx]),
        processor.transform_y(cache.targets[train_idx]),
        processor.transform_x(x_all[val_idx]),
        processor.transform_y(cache.targets[val_idx]),
        run_dir=None,
        verbose=task.get("verbose", 0),
        initial_weight_store=initial_weight_store,
        save_history=False,
    )
    x_test = processor.transform_x(x_all[test_idx])
    predictions = predict_original(
        discovery["model"], x_test, processor.inverse_y, protocol.batch_size
    )
    metrics = regression_metrics(cache.targets[test_idx], predictions)
    metrics.update({
        "discovery_best_epoch": int(discovery["best_epoch"]),
        "discovery_epochs_ran": int(discovery["epochs_ran"]),
        "parameter_count": int(discovery["model"].count_params()),
        "gate_parameter_count": int(discovery["model"].gate_parameter_count()),
        "discovery_seconds": float(discovery["elapsed_seconds"]),
        "initial_backbone_sha256": discovery["initial_backbone_sha256"],
        "initial_gate_sha256": discovery["initial_gate_sha256"],
    })

    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "predictions.csv"
    pd.DataFrame({
        "sample_id": test_idx,
        "composition": cache.compositions[test_idx],
        "y_true": cache.targets[test_idx],
        "y_pred": predictions,
        "residual": predictions - cache.targets[test_idx],
    }).to_csv(prediction_path, index=False)

    save_model = bool(task.get("save_model", False))
    weight_file = None
    if save_model:
        atomic_write_json(
            run_dir / "analysis_summary.json",
            analysis_summary(discovery["model"], x_test),
        )
        weight_file = _save_selected_model(
            task, config, discovery["model"], processor,
            int(task["fold"]), int(task["seed"]), metrics,
        )

    result = {
        "execution_status": "PASS",
        "scientific_gate_pass": None,
        "task": cache.manifest.get("task"),
        "formal_official_folds": True,
        "official_test_accessed_during_selection": False,
        "stage": task.get("stage", "plan"),
        "config": config.name,
        "config_spec": task.get("config_spec"),
        "fold": int(task["fold"]),
        "seed": int(task["seed"]),
        "protocol": protocol.to_dict(),
        "protocol_selection_file": str(Path(task["protocol_json"]).resolve()),
        "formal_strategy": "discovery_checkpoint",
        "split_counts": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "full_outer_train_refit": False,
        "save_model": save_model,
        "save_model_role": task.get("save_model_role"),
        "selected_model_weight_file": weight_file,
        "prediction_file": str(prediction_path),
        "metrics": metrics,
        "selection_metadata": {
            "selected_protocol": selection.get("selected_protocol"),
            "outer_test_accessed": selection.get("outer_test_accessed", False),
        },
    }
    atomic_write_json(result_path, result)
    return result


def cleanup_tensorflow():
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()
