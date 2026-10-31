"""Keras callbacks of the fixed-split study: original-unit validation and persistent early stopping."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error


class PersistentHistory(tf.keras.callbacks.Callback):
    """Epoch journal that remains complete across BackupAndRestore restarts."""
    def __init__(self, output_csv):
        super().__init__()
        self.output_csv = Path(output_csv)
        self.rows = []
        if self.output_csv.exists():
            try:
                self.rows = pd.read_csv(self.output_csv).to_dict("records")
            except Exception:
                self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": int(epoch + 1)}
        for key, value in (logs or {}).items():
            try:
                row[key] = float(value)
            except Exception:
                pass
        self.rows.append(row)
        frame = pd.DataFrame(self.rows)
        frame = frame.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
        self.rows = frame.to_dict("records")
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.output_csv, index=False)


class OriginalUnitValidation(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, scaler, batch_size, output_csv, log_every=10):
        super().__init__()
        self.x_val = x_val
        self.y_val = np.asarray(y_val).reshape(-1)
        self.scaler = scaler
        self.batch_size = int(batch_size)
        self.output_csv = Path(output_csv)
        self.log_every = max(1, int(log_every))
        self.rows = []
        if self.output_csv.exists():
            try:
                self.rows = pd.read_csv(self.output_csv).to_dict("records")
            except Exception:
                self.rows = []
        self.started = None

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        pred_z = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
        pred = self.scaler.inverse_transform(np.asarray(pred_z).reshape(-1, 1)).reshape(-1)
        mae = float(mean_absolute_error(self.y_val, pred))
        # Keras passes the same mutable logs mapping to callbacks in order. Add
        # the original-unit metric here so PersistentHistory and the formal
        # early-stopping callback can monitor it without touching the test set.
        if logs is not None:
            logs["val_mae_original"] = mae
        row = {
            "epoch": int(epoch + 1),
            "validation_mae_original": mae,
            "epoch_seconds": float(time.perf_counter() - self.started),
        }
        for key, value in (logs or {}).items():
            try:
                row[key] = float(value)
            except Exception:
                pass
        self.rows.append(row)
        frame = pd.DataFrame(self.rows)
        frame = frame.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
        self.rows = frame.to_dict("records")
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.output_csv, index=False)
        if (epoch + 1) % self.log_every == 0:
            print(f"[VAL] epoch={epoch+1} val_mae_original={mae:.8f}", flush=True)


class PersistentMinimumEpochEarlyStopping(tf.keras.callbacks.Callback):
    """Validation early stopping with state and best weights persisted on disk.

    This avoids resetting patience or losing a pre-interruption best checkpoint
    when Keras BackupAndRestore resumes a training process.
    """
    def __init__(self, minimum_epochs, monitor, patience, state_path, best_weights_path,
                 mode="min", min_delta=0.0, restore_best_weights=True, verbose=1):
        super().__init__()
        if mode != "min":
            raise ValueError("This release supports only mode='min'")
        self.minimum_epochs = int(minimum_epochs)
        self.monitor = str(monitor)
        self.patience = int(patience)
        self.state_path = Path(state_path)
        self.best_weights_path = Path(best_weights_path)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.verbose = int(verbose)
        self.best = float("inf")
        self.wait = 0
        self.observed_best_epoch = None
        self.stopped_epoch = 0

    @property
    def best_weights(self):
        # Compatibility with the prior acceptance guard.
        return self.best_weights_path if self.best_weights_path.exists() else None

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "monitor": self.monitor,
            "minimum_epochs": self.minimum_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": None if not np.isfinite(self.best) else self.best,
            "wait": self.wait,
            "best_epoch": self.observed_best_epoch,
            "stopped_epoch": self.stopped_epoch,
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def on_train_begin(self, logs=None):
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("monitor") != self.monitor or int(state.get("minimum_epochs", -1)) != self.minimum_epochs or int(state.get("patience", -1)) != self.patience:
                raise RuntimeError("Persistent early-stopping state does not match the current protocol")
            saved_best = state.get("best")
            self.best = float("inf") if saved_best is None else float(saved_best)
            self.wait = int(state.get("wait", 0))
            value = state.get("best_epoch")
            self.observed_best_epoch = None if value is None else int(value)
            if np.isfinite(self.best) and not self.best_weights_path.exists():
                raise RuntimeError("Early-stopping state exists but its best checkpoint is missing")
            if self.verbose:
                print(f"[EARLY STOP RESUME] best={self.best:.8g} best_epoch={self.observed_best_epoch} wait={self.wait}", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            raise RuntimeError(f"Early stopping monitor {self.monitor!r} is absent from epoch logs")
        current = float(current)
        if not np.isfinite(current):
            self.model.stop_training = True
            raise RuntimeError(f"Non-finite validation monitor at epoch {epoch+1}: {current}")
        improved = current < self.best - self.min_delta
        if improved:
            # Track and persist the true validation-best checkpoint from the
            # first epoch. minimum_epochs constrains when training may stop; it
            # must not discard an earlier, genuinely better validation model.
            self.best = current
            self.wait = 0
            self.observed_best_epoch = int(epoch + 1)
            self.best_weights_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_weights(str(self.best_weights_path))
            if self.verbose:
                print(f"[EARLY STOP BEST] epoch={epoch+1} {self.monitor}={current:.8g}", flush=True)
        elif epoch + 1 >= self.minimum_epochs:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = int(epoch + 1)
                self.model.stop_training = True
                if self.verbose:
                    print(f"[EARLY STOP] epoch={epoch+1} best_epoch={self.observed_best_epoch} best={self.best:.8g}", flush=True)
        else:
            # Before minimum_epochs, do not consume patience.
            self.wait = 0
        self._save_state()

    def on_train_end(self, logs=None):
        if self.restore_best_weights:
            if not self.best_weights_path.exists():
                raise RuntimeError("No validation-best checkpoint was produced")
            self.model.load_weights(str(self.best_weights_path))
            if self.verbose:
                print(f"[RESTORE BEST] epoch={self.observed_best_epoch} from={self.best_weights_path}", flush=True)
        self._save_state()


class PeriodicWeights(tf.keras.callbacks.Callback):
    def __init__(self, directory, every=25):
        super().__init__()
        self.directory = Path(directory)
        self.every = max(1, int(every))

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "latest.weights.h5"
        self.model.save_weights(str(path))
        (self.directory / "latest_epoch.txt").write_text(str(epoch + 1), encoding="utf-8")
