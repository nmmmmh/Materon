#!/usr/bin/env python3
"""Train one fixed-split 2D model (HOIP or JDFT2D; CGCNN or MEGNet family; seed 42).

Validation MAE in original units drives early stopping with a minimum epoch
count and restoration of the best weights (CGCNN: <=600 epochs, >=150,
patience 80; MEGNet: <=1000, >=200, patience 120). The locked test set is
evaluated once after training. Materon models start from the same backbone
weights as the matching no-gate model. --resume continues an interrupted run.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

STATE_FILES = ["early_stopping_state.json", "early_best.weights.h5", "history_live.csv", "validation_original_units.csv"]


def cgcnn_schedule(tf, recipe, max_epochs, steps_per_epoch):
    initial = float(recipe["learning_rate"])
    total = max(1, int(max_epochs) * int(steps_per_epoch))
    if recipe.get("schedule") == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(initial, total, alpha=float(recipe.get("alpha", 0.01)))
    decay_steps = max(1, int(total * float(recipe.get("decay_fraction", 0.60))))
    return tf.keras.optimizers.schedules.ExponentialDecay(initial, decay_steps, float(recipe.get("decay_rate", 0.5)),
                                                          staircase=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cgcnn-cache-root", required=True)
    p.add_argument("--megnet-cache-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from materon.fixed_split.cache_access import cgcnn_source_config, open_cgcnn_cache, open_megnet_cache
    from materon.fixed_split.callbacks import (OriginalUnitValidation, PeriodicWeights, PersistentHistory,
                                               PersistentMinimumEpochEarlyStopping)
    from materon.fixed_split.config import load_protocol
    from materon.fixed_split.splits import positions_for_ids
    from materon.megnet.config import load_protocol as megnet_protocol
    from materon.megnet.training import metrics
    from materon.runtime import configure_tensorflow, set_seed

    protocol = load_protocol()
    seed = int(protocol["seed"])
    spec = protocol["models"][a.model]
    backend = spec["backbone"]
    configure_tensorflow(tf, protocol["runtime"]["tf_intra_op_threads"], protocol["runtime"]["tf_inter_op_threads"])
    split = json.loads((Path(a.output_root) / "splits" / f"{a.dataset}_split_manifest.json").read_text(encoding="utf-8"))
    run = Path(a.output_root) / "runs" / a.dataset / a.model / "seed_42"
    run.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        for item in ["backup", "checkpoints"]:
            shutil.rmtree(run / item, ignore_errors=True)
        for item in STATE_FILES:
            (run / item).unlink(missing_ok=True)

    if backend == "cgcnn":
        cache = open_cgcnn_cache(a.dataset, a.model, a.cgcnn_cache_root)
        cache_ids = list(cache.indices)
    else:
        cache = open_megnet_cache(a.dataset, a.megnet_cache_root)
        cache_ids = list(cache.sample_ids)
    pos = positions_for_ids(cache_ids, split)
    training_cfg = protocol["training"][backend]

    tf.keras.backend.clear_session(); gc.collect(); set_seed(seed)
    if backend == "cgcnn":
        from materon.cgcnn.models import build_model, build_paired_model
        cfg, recipe = cgcnn_source_config(a.model)
        x_train, y_train = cache.subset_positions(pos["train"])
        x_val, y_val = cache.subset_positions(pos["validation"])
        sample = [x[:min(4, len(y_train))] for x in x_train]
        if cfg["model_type"] == "materon":
            model, _, _ = build_paired_model(cfg, recipe, sample, seed)
        else:
            model = build_model(cfg, recipe, seed)
            _ = model(sample, training=False)
        batch_size = int(recipe["batch_size"])
        max_epochs = int(training_cfg["max_epochs"])
        steps = max(1, int(math.ceil(len(y_train) / batch_size)))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cgcnn_schedule(tf, recipe, max_epochs, steps)),
                      loss="mae", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    else:
        from materon.megnet.models import build_model, find_physical_megnet_backbone
        x_train = cache.inputs(a.model, pos["train"])
        y_train = cache.targets(pos["train"]).reshape(-1)
        x_val = cache.inputs(a.model, pos["validation"])
        y_val = cache.targets(pos["validation"]).reshape(-1)
        source_protocol = megnet_protocol()
        if a.model == "materon_explain_megnet":
            # The gated model starts from the backbone weights of the full20 no-gate model.
            set_seed(seed)
            reference, _, _ = build_model("materon_nogate_megnet", source_protocol, seed)
            _ = reference([item[:min(4, len(y_train))] for item in x_train[:6]], training=False)
            set_seed(seed)
            model, _, _ = build_model(a.model, source_protocol, seed)
            _ = model([item[:min(4, len(y_train))] for item in x_train], training=False)
            backbone = find_physical_megnet_backbone(model)
            reference_weights = reference.get_weights()
            if [np.shape(w) for w in reference_weights] != [np.shape(w) for w in backbone.get_weights()]:
                raise RuntimeError("MEGNet pairing weight-shape mismatch")
            backbone.set_weights(reference_weights)
            del reference
        else:
            model, _, _ = build_model(a.model, source_protocol, seed)
        batch_size = int(training_cfg["batch_size"])
        max_epochs = int(training_cfg["max_epochs"])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(training_cfg["learning_rate_start"])),
                      loss="mae", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])

    scaler = StandardScaler().fit(np.asarray(y_train).reshape(-1, 1))
    y_train_z = scaler.transform(np.asarray(y_train).reshape(-1, 1)).astype(np.float32)
    y_val_z = scaler.transform(np.asarray(y_val).reshape(-1, 1)).astype(np.float32)
    early = PersistentMinimumEpochEarlyStopping(
        minimum_epochs=int(training_cfg["minimum_epochs"]), monitor="val_mae_original", mode="min",
        patience=int(training_cfg["patience"]), min_delta=float(training_cfg.get("min_delta_original_units", 0.0)),
        restore_best_weights=True, verbose=1, state_path=run / "early_stopping_state.json",
        best_weights_path=run / "early_best.weights.h5")
    callbacks = [
        OriginalUnitValidation(x_val, y_val, scaler, batch_size, run / "validation_original_units.csv", log_every=10),
        PersistentHistory(run / "history_live.csv"),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(run / "backup")),
        early,
        PeriodicWeights(run / "checkpoints", int(training_cfg["checkpoint_every"])),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if backend == "megnet":
        from kgcnn.training.scheduler import LinearLearningRateScheduler
        callbacks.insert(0, LinearLearningRateScheduler(
            learning_rate_start=float(training_cfg["learning_rate_start"]),
            learning_rate_stop=float(training_cfg["learning_rate_stop"]),
            epo_min=int(training_cfg["lr_decay_start_epoch"]), epo=int(training_cfg["lr_decay_end_epoch"]), verbose=0))
    np.savez_compressed(run / "split_positions.npz", train=pos["train"], validation=pos["validation"], test=pos["test"])
    model.fit(x_train, y_train_z, validation_data=(x_val, y_val_z), batch_size=batch_size, epochs=max_epochs,
              verbose=2, callbacks=callbacks, shuffle=True)
    history = pd.read_csv(run / "history_live.csv").drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    history.to_csv(run / "history.csv", index=False)
    if early.observed_best_epoch is None:
        raise RuntimeError("Early stopping did not record a validation checkpoint")
    model.save_weights(str(run / "best.weights.h5"))  # best validation weights were restored at train end

    # The test set is read only after training and best-checkpoint restoration.
    if backend == "megnet":
        x_test = cache.inputs(a.model, pos["test"])
        y_test = cache.targets(pos["test"]).reshape(-1)
    else:
        x_test, y_test = cache.subset_positions(pos["test"])
    pred = scaler.inverse_transform(np.asarray(model.predict(x_test, batch_size=batch_size, verbose=0)).reshape(-1, 1))
    pred = pred.reshape(-1).astype(np.float64, copy=False)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    score = metrics(y_test, pred)
    pd.DataFrame({"sample_id": [cache_ids[int(i)] for i in pos["test"]], "target": y_test, "prediction": pred,
                  "absolute_error": np.abs(y_test - pred)}).to_csv(run / "predictions.csv", index=False, float_format="%.17g")
    (run / "target_scaler.json").write_text(json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                                                        "var": scaler.var_.tolist()}, indent=2), encoding="utf-8")
    payload = {"status": "complete", "protocol_id": protocol["protocol_id"], "dataset": a.dataset, "model": a.model,
               "backend": backend, "role": spec["role"], "seed": seed, "parameter_count": int(model.count_params()),
               "epochs_ran": int(history["epoch"].max()), "best_epoch": int(early.observed_best_epoch),
               "train_size": len(pos["train"]), "validation_size": len(pos["validation"]), "test_size": len(pos["test"]),
               **score}
    (run / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DONE] dataset={a.dataset} model={a.model} best_epoch={payload['best_epoch']} mae={score['mae']:.8f}", flush=True)


if __name__ == "__main__":
    main()
