"""Training of the C2DB work-function predictors (Sec. VI B, S3.1, S8).

Adam, batch 64, MSE on standardized targets, checkpoint = minimum validation
MAE (earliest tie). The seed fixes the paired initialization and the minibatch
permutations (one NumPy RandomState permutation per epoch, drawn up front).

Schedules:
  short : at most 80 epochs, patience 20, learning rate 1e-3, no schedule.
  L400  : at most 400 epochs, patience 80, learning rate 1e-3, halved after
          every 25 consecutive epochs without improvement (floor 1e-5).
"""
import json
import time
from pathlib import Path

import numpy as np

from materon.c2db.dataset import Prepared, metrics, write_json
from materon.c2db.graph import model_inputs

SCHEDULES = {
    "short": dict(maximum_epochs=80, early_stopping_patience=20, learning_rate=1e-3,
                  plateau_halving_after=None, learning_rate_floor=None),
    "L400": dict(maximum_epochs=400, early_stopping_patience=80, learning_rate=1e-3,
                 plateau_halving_after=25, learning_rate_floor=1e-5),
}
BATCH_SIZE = 64
SEEDS = (42, 7, 123, 2024, 31415)


def member_dir(runs_root, tag, seed, arm="physical20_environment_gate"):
    """Run directory convention: <runs>/<tag>/<arm>_s<seed> (tag: short, L400, FT1, FT2)."""
    return Path(runs_root) / tag / ("%s_s%d" % (arm, seed))


def batched_predict(infer, inputs, n, mean, std, batch_size=BATCH_SIZE):
    """Predict pre-batched ragged inputs in chunks; ``infer`` is a (tf.function) forward pass."""
    import tensorflow as tf

    chunks = []
    for start in range(0, n, batch_size):
        index = np.arange(start, min(start + batch_size, n))
        chunks.append(infer([tf.gather(v, index) for v in inputs]).numpy().reshape(-1))
    result = np.concatenate(chunks).astype(np.float64) * std + mean
    if not np.isfinite(result).all():
        raise ValueError("nonfinite prediction")
    return result


def train(prepared_dir, arm, seed, out_dir, schedule="short"):
    """Train one predictor; writes best.weights.h5, HISTORY.json, RESULT.json and predictions."""
    import tensorflow as tf

    from materon.c2db.models import build_paired_models, validate_initial_parity

    config = dict(SCHEDULES[schedule])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    data = Prepared(prepared_dir)
    split = data.split
    train_ids, val_ids = split["ids"]["train"], split["ids"]["validation"]
    graphs = data.graphs
    y_train, y_val = data.targets("train"), data.targets("validation")
    w_train, w_val = data.weights("train"), data.weights("validation")
    mean = float(np.average(y_train, weights=w_train))
    std = float(np.sqrt(np.average((y_train - mean) ** 2, weights=w_train)))
    if not (np.isclose(mean, data.mean) and np.isclose(std, data.std)):
        raise ValueError("training scaling differs from PREPARED.json")

    models = build_paired_models(seed=seed)
    validate_initial_parity(models, [graphs[i] for i in train_ids[:8]], model_inputs)
    model = models[arm]
    train_inputs = model_inputs([graphs[i] for i in train_ids], arm)
    val_inputs = model_inputs([graphs[i] for i in val_ids], arm)
    target = tf.constant(((y_train - mean) / std)[:, None], dtype=tf.float32)
    weight = tf.constant(w_train[:, None], dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

    @tf.function(reduce_retracing=True)
    def train_step(inputs, labels, weights):
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            loss = tf.reduce_mean(weights * tf.square(prediction - labels))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True)
    def infer(inputs):
        return model(inputs, training=False)

    rng = np.random.RandomState(seed)
    orders = [rng.permutation(len(train_ids)) for _ in range(config["maximum_epochs"])]
    best_mae, best_epoch, stale, updates, history = float("inf"), None, 0, 0, []
    started = time.monotonic()
    for epoch, order in enumerate(orders, 1):
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            loss = train_step([tf.gather(v, indices) for v in train_inputs],
                              tf.gather(target, indices), tf.gather(weight, indices))
            updates += 1
            losses.append((float(loss), len(indices)))
        val_scores = metrics(y_val, batched_predict(infer, val_inputs, len(val_ids), mean, std), w_val)
        history.append(dict(epoch=epoch, updates=updates, validation=val_scores,
                            learning_rate=float(optimizer.learning_rate.numpy()),
                            training_weighted_mse=float(sum(v * n for v, n in losses) / len(train_ids))))
        if val_scores["mae"] < best_mae:
            best_mae, best_epoch, stale = val_scores["mae"], epoch, 0
            model.save_weights(str(out / "best.weights.h5"))
        else:
            stale += 1
            halving = config["plateau_halving_after"]
            if halving and stale % halving == 0:
                optimizer.learning_rate.assign(
                    max(float(optimizer.learning_rate.numpy()) * 0.5, config["learning_rate_floor"]))
        print(json.dumps(dict(epoch=epoch, validation_mae=val_scores["mae"], best_epoch=best_epoch)), flush=True)
        if stale >= config["early_stopping_patience"]:
            break
    write_json(out / "HISTORY.json", history)
    model.load_weights(str(out / "best.weights.h5"))
    scores = {}
    for name, ids, inputs, y in (("train", train_ids, train_inputs, y_train),
                                 ("validation", val_ids, val_inputs, y_val)):
        prediction = batched_predict(infer, inputs, len(ids), mean, std)
        scores[name] = metrics(y, prediction)
        write_json(out / (name.upper() + "_PREDICTIONS.json"), dict(zip(ids, prediction.tolist())))
    write_json(out / "RESULT.json", dict(arm=arm, seed=seed, schedule=schedule, schedule_config=config,
                                         best_epoch=best_epoch, epochs_run=len(history), updates=updates,
                                         scores=scores, target_scaling=dict(mean=mean, std=std),
                                         seconds=time.monotonic() - started))
    return scores
