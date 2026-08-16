# -*- coding: utf-8 -*-
"""Compact performance logging for long formal runs."""
from __future__ import annotations

import time
import tensorflow as tf


def _current_learning_rate(optimizer):
    lr = optimizer.learning_rate
    if callable(lr):
        lr = lr(optimizer.iterations)
    return float(tf.keras.backend.get_value(lr))


def _adapter_alpha(model):
    values = []
    for layer in model.submodules:
        if hasattr(layer, "alpha") and hasattr(layer, "raw_alpha"):
            try:
                values.append(float(tf.keras.backend.get_value(layer.alpha)))
            except Exception:
                pass
    if not values:
        return None
    return max(values)


class EpochPerformanceLogger(tf.keras.callbacks.Callback):
    """Print loss, train metrics, alpha, epoch time and samples/s."""

    def __init__(
        self,
        epochs,
        train_size,
        task_type,
        unit="",
        log_every=1,
        prefix="[EPOCH]",
    ):
        super().__init__()
        self.epochs = int(epochs)
        self.train_size = int(train_size)
        self.task_type = str(task_type)
        self.unit = str(unit)
        self.log_every = max(1, int(log_every))
        self.prefix = str(prefix)
        self._epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        del epoch, logs
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        elapsed = max(time.perf_counter() - self._epoch_start, 1.0e-12)
        throughput = self.train_size / elapsed
        lr = _current_learning_rate(self.model.optimizer)
        alpha = _adapter_alpha(self.model)

        should_print = (
            epoch == 0
            or (epoch + 1) % self.log_every == 0
            or (epoch + 1) == self.epochs
        )
        if not should_print:
            return

        fields = [
            "%s %04d/%04d" % (self.prefix, epoch + 1, self.epochs),
            "loss=%.6g" % float(logs.get("loss", float("nan"))),
        ]
        if self.task_type == "classification":
            if "train_accuracy" in logs:
                fields.append(
                    "train_acc=%.5f" % float(logs["train_accuracy"])
                )
            if "train_roc_auc" in logs:
                fields.append(
                    "train_auc=%.5f" % float(logs["train_roc_auc"])
                )
        else:
            suffix = (" " + self.unit) if self.unit else ""
            if "train_mae" in logs:
                fields.append(
                    "train_mae=%.6g%s"
                    % (float(logs["train_mae"]), suffix)
                )
            if "train_rmse" in logs:
                fields.append(
                    "train_rmse=%.6g%s"
                    % (float(logs["train_rmse"]), suffix)
                )

        fields.append("lr=%.6g" % lr)
        if alpha is not None:
            fields.append("alpha=%.6g" % alpha)
        fields.extend([
            "epoch=%.3fs" % elapsed,
            "throughput=%.1f samples/s" % throughput,
        ])
        print(" ".join(fields), flush=True)
