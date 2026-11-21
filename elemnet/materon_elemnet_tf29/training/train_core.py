from __future__ import absolute_import, division, print_function

import time

import numpy as np

from ..compat import configure_tensorflow
from ..models.elemnet_performance import build_model
from ..utils.reproducibility import set_global_seed


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / denominator if denominator > 0 else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def make_loss(tf, protocol):
    if protocol.loss_name == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    if protocol.loss_name == "huber":
        return tf.keras.losses.Huber(delta=float(protocol.huber_delta))
    raise KeyError("Unknown loss=%r" % protocol.loss_name)


def make_optimizer(tf, protocol):
    kwargs = {"learning_rate": float(protocol.learning_rate)}
    if float(protocol.clipnorm) > 0:
        kwargs["clipnorm"] = float(protocol.clipnorm)
    return tf.keras.optimizers.Adam(**kwargs)


class LearningRateRecorder(object):
    def __init__(self, tf_module):
        self.tf = tf_module
        self.values = []
        self.callback = self._build()

    def _build(self):
        outer = self
        tf = self.tf

        class Callback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                del epoch, logs
                lr = self.model.optimizer.learning_rate
                outer.values.append(float(tf.keras.backend.get_value(lr)))

        return Callback()


def make_dataset(tf, x, y, batch_size, seed, training):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if training:
        dataset = dataset.shuffle(
            buffer_size=max(1, len(x)), seed=int(seed),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(min(int(batch_size), max(1, len(x))), drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def _new_model(model_config, seed, initial_weight_store=None):
    if initial_weight_store is not None:
        return initial_weight_store.build_model(model_config, int(seed))
    return build_model(model_config, int(seed))


def fit_discovery(model_config, protocol, seed, x_train, y_train, x_val, y_val,
                  run_dir=None, verbose=0, initial_weight_store=None,
                  save_history=False):
    # Fixed seeds and common initial backbones are retained without
    # duplicate replay or Python-level per-batch loops.
    tf = configure_tensorflow(seed, deterministic=True)
    set_global_seed(seed, tf)
    model = _new_model(model_config, seed, initial_weight_store)
    initial_backbone_sha256 = model.backbone_sha256()
    initial_gate_sha256 = model.gate_sha256()
    model.compile(
        optimizer=make_optimizer(tf, protocol),
        loss=make_loss(tf, protocol), metrics=["mae"]
    )
    lr_recorder = LearningRateRecorder(tf)
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        lr_recorder.callback,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=int(protocol.patience),
            restore_best_weights=True, min_delta=1.0e-5,
            verbose=1 if verbose else 0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5,
            patience=int(protocol.lr_patience),
            min_lr=float(protocol.min_learning_rate),
            verbose=1 if verbose else 0,
        ),
    ]
    if save_history and run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(
            str(run_dir / "discovery_history.csv"), append=False
        ))
    train_dataset = make_dataset(tf, x_train, y_train, protocol.batch_size, seed, True)
    val_dataset = make_dataset(tf, x_val, y_val, protocol.batch_size, seed, False)
    started = time.time()
    history_obj = model.fit(
        train_dataset, validation_data=val_dataset,
        epochs=int(protocol.epochs), callbacks=callbacks, verbose=int(verbose),
    )
    elapsed = time.time() - started
    history = dict(history_obj.history)
    history["lr"] = list(lr_recorder.values[:len(history.get("loss", []))])
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": float(np.min(history["val_loss"])),
        "epochs_ran": int(len(history["loss"])),
        "elapsed_seconds": float(elapsed),
        "initial_backbone_sha256": initial_backbone_sha256,
        "initial_gate_sha256": initial_gate_sha256,
    }


def fit_refit(model_config, protocol, seed, x_full, y_full, epochs,
              run_dir=None, verbose=0, learning_rate_history=None,
              initial_weight_store=None, save_history=False):
    tf = configure_tensorflow(seed, deterministic=True)
    set_global_seed(seed, tf)
    model = _new_model(model_config, seed, initial_weight_store)
    initial_backbone_sha256 = model.backbone_sha256()
    initial_gate_sha256 = model.gate_sha256()
    model.compile(
        optimizer=make_optimizer(tf, protocol),
        loss=make_loss(tf, protocol), metrics=["mae"]
    )
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    if learning_rate_history:
        schedule_values = [float(value) for value in learning_rate_history]

        def replay_schedule(epoch, current_lr):
            del current_lr
            return schedule_values[min(int(epoch), len(schedule_values) - 1)]

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(
            replay_schedule, verbose=0
        ))
    if save_history and run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(
            str(run_dir / "refit_history.csv"), append=False
        ))
    full_dataset = make_dataset(tf, x_full, y_full, protocol.batch_size, seed, True)
    started = time.time()
    history = model.fit(
        full_dataset, epochs=int(epochs), callbacks=callbacks, verbose=int(verbose)
    )
    return {
        "model": model,
        "history": history.history,
        "elapsed_seconds": float(time.time() - started),
        "initial_backbone_sha256": initial_backbone_sha256,
        "initial_gate_sha256": initial_gate_sha256,
    }


def predict_original(model, x, inverse_target, batch_size=32):
    del batch_size  # Small Matbench folds fit comfortably in one direct call.
    tensor = np.asarray(x, dtype=np.float32)
    scaled = model(tensor, training=False).numpy().reshape(-1)
    return inverse_target(scaled)


def analysis_summary(model, x):
    try:
        _, analysis = model(np.asarray(x, dtype=np.float32), training=False, return_analysis=True)
    except Exception:
        return {}

    def summarize(value):
        if isinstance(value, dict):
            return {str(k): summarize(v) for k, v in value.items()}
        try:
            array = value.numpy()
        except Exception:
            array = np.asarray(value)
        array = np.asarray(array, dtype=float)
        if array.size == 0:
            return {"size": 0}
        return {
            "shape": list(array.shape),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean_abs": float(np.mean(np.abs(array))),
        }

    return summarize(analysis)
