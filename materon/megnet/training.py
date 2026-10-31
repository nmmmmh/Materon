"""Official kgcnn 2.1 Matbench MEGNet training settings (Adam, MAE loss, linear LR decay)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def official_optimizer(training_cfg: dict):
    return tf.keras.optimizers.get({"class_name": training_cfg["optimizer"],
                                    "config": dict(training_cfg["optimizer_config"])})


def official_callbacks(training_cfg: dict):
    """kgcnn LinearLearningRateScheduler from learning_rate_start to learning_rate_stop."""
    import kgcnn.training.callbacks  # noqa: F401  (registers kgcnn objects)
    from kgcnn.training.scheduler import LinearLearningRateScheduler  # noqa: F401

    payload = {"class_name": training_cfg["learning_rate_scheduler"],
               "config": {"learning_rate_start": float(training_cfg["learning_rate_start"]),
                          "learning_rate_stop": float(training_cfg["learning_rate_stop"]),
                          "epo_min": int(training_cfg["lr_decay_start_epoch"]),
                          "epo": int(training_cfg["lr_decay_end_epoch"]), "verbose": 0}}
    return [tf.keras.utils.deserialize_keras_object(payload)]


def compile_official(model, training_cfg: dict) -> None:
    model.compile(loss=tf.keras.losses.get(training_cfg["loss"]), optimizer=official_optimizer(training_cfg))


def metrics(target, pred) -> dict:
    target = np.asarray(target).ravel()
    pred = np.asarray(pred).ravel()
    return {"mae": float(mean_absolute_error(target, pred)),
            "rmse": float(mean_squared_error(target, pred, squared=False)),
            "r2": float(r2_score(target, pred)),
            "target_mean": float(np.mean(target)), "target_std": float(np.std(target)),
            "prediction_mean": float(np.mean(pred)), "prediction_std": float(np.std(pred))}


def save_history(history, path: Path) -> None:
    frame = pd.DataFrame(history.history)
    frame.index = np.arange(1, len(frame) + 1)
    frame.index.name = "epoch"
    frame.to_csv(path)
