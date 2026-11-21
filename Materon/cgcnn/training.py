"""CGCNN training: inner splits, recipe schedules, milestone selection and final refit.

Search runs train once per recipe and evaluate the inner-validation MAE at
fixed milestone epochs; the best milestone gives the selected epoch. Final runs
reinitialize, train on the whole outer training pool for the selected epoch
(no validation, no early stopping) and evaluate the outer test fold once.
Targets are standardized on the training data; the loss is MAE.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def quantile_bins(y, max_bins=10):
    y = np.asarray(y, float).reshape(-1)
    bins = min(int(max_bins), max(2, len(y) // 20))
    if bins < 2:
        return np.zeros(len(y), dtype=int)
    edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:
        return np.zeros(len(y), dtype=int)
    return np.digitize(y, edges[1:-1], right=True)


def inner_split(outer_positions, outer_targets, outer_fold, repeat, seed, val_fraction=0.12, bins=10):
    """Target-quantile-stratified holdout drawn from the outer training pool only."""
    positions = np.asarray(outer_positions, int)
    labels = quantile_bins(outer_targets, bins)
    random_state = int(seed + 1009 * int(outer_fold) + 104729 * int(repeat))
    try:
        tr, va = train_test_split(np.arange(len(positions)), test_size=float(val_fraction),
                                  random_state=random_state, shuffle=True, stratify=labels)
    except ValueError:
        tr, va = train_test_split(np.arange(len(positions)), test_size=float(val_fraction),
                                  random_state=random_state, shuffle=True)
    return positions[tr], positions[va], random_state


def official_positions(task, cache, fold):
    """Cache positions of the official Matbench outer fold, read from the loaded task."""
    xtr, ytr = task.get_train_and_val_data(int(fold))
    xte, yte = task.get_test_data(int(fold), include_target=True)
    train = np.asarray([cache.position[str(x)] for x in xtr.index], int)
    test = np.asarray([cache.position[str(x)] for x in xte.index], int)
    return train, np.asarray(ytr, np.float32), test, np.asarray(yte, np.float32), list(xtr.index), list(xte.index)


def set_global_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(int(seed))


def make_lr_schedule(recipe, steps_per_epoch, max_epochs):
    total = max(1, int(steps_per_epoch) * int(max_epochs))
    initial = float(recipe["learning_rate"])
    kind = recipe.get("schedule", "exponential")
    if kind == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial, decay_steps=total,
                                                         alpha=float(recipe.get("alpha", 0.01)))
    if kind == "exponential":
        decay_steps = max(1, int(total * float(recipe.get("decay_fraction", 0.60))))
        return tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial, decay_steps=decay_steps,
                                                              decay_rate=float(recipe.get("decay_rate", 0.5)),
                                                              staircase=False)
    raise ValueError("Unknown schedule: %s" % kind)


def current_lr(optimizer):
    try:
        lr = optimizer.learning_rate
        if callable(lr):
            lr = lr(optimizer.iterations)
        return float(tf.keras.backend.get_value(lr))
    except Exception:
        return float("nan")


class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, max_epochs, n_samples, target_scale, log_every=10):
        super().__init__()
        self.max_epochs = int(max_epochs)
        self.n = int(n_samples)
        self.scale = float(target_scale)
        self.log_every = max(1, int(log_every))
        self.started = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every and epoch + 1 != self.max_epochs:
            return
        logs = logs or {}
        train = float(logs.get("mae", logs.get("loss", np.nan))) * self.scale
        print("[EPOCH] %04d/%04d train_mae=%.6f lr=%.7g" % (epoch + 1, self.max_epochs, train,
                                                          current_lr(self.model.optimizer)), flush=True)


class MilestoneValidation(tf.keras.callbacks.Callback):
    """Validation MAE (original units) and a weight snapshot at each milestone epoch."""

    def __init__(self, epochs, x_val, y_val_original, scaler, batch_size, outdir):
        super().__init__()
        self.epochs = {int(x) for x in epochs}
        self.x_val = x_val
        self.truth = np.asarray(y_val_original).reshape(-1)
        self.scaler = scaler
        self.batch = int(batch_size)
        self.out = Path(outdir)
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        number = epoch + 1
        if number not in self.epochs:
            return
        pred_z = self.model.predict(self.x_val, batch_size=self.batch, verbose=0)
        pred = self.scaler.inverse_transform(np.asarray(pred_z).reshape(-1, 1)).reshape(-1)
        mae = float(mean_absolute_error(self.truth, pred))
        rmse = float(mean_squared_error(self.truth, pred, squared=False))
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / ("checkpoint_epoch_%03d.weights.h5" % number)
        self.model.save_weights(str(path))
        self.records.append({"epoch": number, "validation_mae": mae, "validation_rmse": rmse,
                             "learning_rate": current_lr(self.model.optimizer), "checkpoint": path.name})
        print("[CHECKPOINT] epoch=%d validation_mae=%.8f" % (number, mae), flush=True)


def compile_model(model, recipe, n_samples, max_epochs):
    """Adam with the recipe schedule over a fixed horizon (320 epochs), MAE loss."""
    steps = max(1, int(math.ceil(float(n_samples) / int(recipe["batch_size"]))))
    horizon = int(recipe.get("schedule_horizon_epochs", 320))
    if int(max_epochs) > horizon:
        raise ValueError("max_epochs exceeds the fixed learning-rate schedule horizon")
    lr = make_lr_schedule(recipe, steps, horizon)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mae",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")], jit_compile=False)


def save_common(out, history, model, scaler, metrics, extra, pred, truth, indices, save_weights=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(out / "history.csv", index=False)
    if save_weights:
        model.save_weights(str(out / "model.weights.h5"))
    truth = np.asarray(truth).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    pd.DataFrame({"index": [str(x) for x in indices], "target": truth, "prediction": pred,
                  "absolute_error": np.abs(truth - pred)}).to_csv(out / "predictions.csv", index=False)
    (out / "target_scaler.json").write_text(
        json.dumps({"mean": float(scaler.mean_[0]), "scale": float(scaler.scale_[0])}, indent=2), encoding="utf-8")
    payload = dict(extra)
    payload["metrics"] = metrics
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_search(model, x_train, y_train, x_val, y_val, recipe, outdir, max_epochs, checkpoint_epochs, log_every=10):
    scaler = StandardScaler()
    yt = scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).astype(np.float32)
    compile_model(model, recipe, len(y_train), max_epochs)
    monitor = MilestoneValidation(checkpoint_epochs, x_val, y_val, scaler, recipe["batch_size"], outdir)
    callbacks = [monitor, EpochLogger(max_epochs, len(y_train), scaler.scale_[0], log_every)]
    started = time.time()
    history = model.fit(x_train, yt, batch_size=int(recipe["batch_size"]), epochs=int(max_epochs),
                        callbacks=callbacks, verbose=0, shuffle=True)
    seconds = time.time() - started
    if not monitor.records:
        raise RuntimeError("No milestone checkpoint was evaluated")
    table = pd.DataFrame(monitor.records).sort_values(["validation_mae", "epoch"])
    best = table.iloc[0]
    selected = int(best.epoch)
    model.load_weights(str(Path(outdir) / best.checkpoint))
    pred = scaler.inverse_transform(model.predict(x_val, batch_size=int(recipe["batch_size"]), verbose=0)).reshape(-1)
    truth = np.asarray(y_val).reshape(-1)
    metrics = {"mae": float(mean_absolute_error(truth, pred)),
               "rmse": float(mean_squared_error(truth, pred, squared=False))}
    table.sort_values("epoch").to_csv(Path(outdir) / "checkpoint_metrics.csv", index=False)
    for record in monitor.records:  # milestone snapshots are temporary
        temporary = Path(outdir) / record["checkpoint"]
        if temporary.exists():
            temporary.unlink()
    details = {"selected_epoch": selected, "best_epoch": selected, "epochs_ran": int(max_epochs),
               "checkpoint_epochs": [int(x) for x in checkpoint_epochs],
               "boundary_hit": bool(selected == max(checkpoint_epochs)), "train_seconds": float(seconds),
               "schedule_horizon_epochs": int(recipe.get("schedule_horizon_epochs", 320))}
    return history, scaler, pred, metrics, details


def run_final(model, x_train, y_train, x_test, y_test, recipe, selected_epoch, log_every=10):
    scaler = StandardScaler()
    yt = scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).astype(np.float32)
    compile_model(model, recipe, len(y_train), selected_epoch)
    callbacks = [EpochLogger(selected_epoch, len(y_train), scaler.scale_[0], log_every)]
    started = time.time()
    history = model.fit(x_train, yt, batch_size=int(recipe["batch_size"]), epochs=int(selected_epoch),
                        callbacks=callbacks, verbose=0, shuffle=True)
    seconds = time.time() - started
    pred = scaler.inverse_transform(model.predict(x_test, batch_size=int(recipe["batch_size"]), verbose=0)).reshape(-1)
    truth = np.asarray(y_test).reshape(-1)
    metrics = {"mae": float(mean_absolute_error(truth, pred)),
               "rmse": float(mean_squared_error(truth, pred, squared=False))}
    return history, scaler, pred, metrics, {"selected_epoch": int(selected_epoch), "epochs_ran": int(selected_epoch),
                                            "train_seconds": float(seconds), "test_evaluations": 1,
                                            "validation_used": False, "early_stopping_used": False}
