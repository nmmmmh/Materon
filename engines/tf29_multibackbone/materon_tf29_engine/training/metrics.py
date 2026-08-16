# -*- coding: utf-8 -*-
"""Metrics utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def evaluate(task_type, y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    if task_type == "classification":
        label = (p >= 0.5).astype(int)
        out = {
            "accuracy": float(accuracy_score(y, label)),
            "balanced_accuracy": float(balanced_accuracy_score(y, label)),
            "f1": float(f1_score(y, label)),
        }
        try:
            out["roc_auc"] = float(roc_auc_score(y, p))
        except ValueError:
            out["roc_auc"] = float("nan")
        return out
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p, squared=False)),
    }


def make_training_metrics(task_type, target_scale=1.0):
    """Metrics printed during fit without touching the outer test fold."""
    if task_type == "classification":
        return [
            tf.keras.metrics.BinaryAccuracy(name="train_accuracy"),
            tf.keras.metrics.AUC(name="train_roc_auc"),
        ]

    scale = tf.constant(float(target_scale), dtype=tf.float32)

    def train_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred)) * scale

    def train_rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred))) * scale

    train_mae.__name__ = "train_mae"
    train_rmse.__name__ = "train_rmse"
    return [train_mae, train_rmse]
