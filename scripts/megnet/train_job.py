#!/usr/bin/env python3
"""Train one MEGNet Matbench model on an official outer fold and test it once.

Fixed kgcnn 2.1.0 Matbench settings for all models: Adam, lr 5e-4 with linear
decay to 5e-6 between epochs 100 and 1000, batch 32, 1000 epochs, MAE loss,
StandardScaler targets fitted on the outer training fold, seed 42; no
validation, early stopping or hyper-parameter scan.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from materon.runtime import configure_tensorflow, set_seed

    configure_tensorflow(tf, 2, 2)
    from materon.megnet.cache import MEGNetCache
    from materon.megnet.config import load_protocol
    from materon.megnet.models import build_model
    from materon.megnet.training import compile_official, metrics, official_callbacks, save_history

    protocol = load_protocol()
    if a.task not in protocol["tasks"] or a.model not in protocol["models"] or a.fold not in protocol["outer_folds"]:
        raise ValueError("task, model or fold not in the protocol")
    cache = MEGNetCache(a.cache_root, a.task)
    outer_train, outer_test = cache.split(a.fold)
    run = Path(a.outdir) / a.task / a.model / f"outer_fold_{a.fold}" / f"seed_{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    training = protocol["official_training"]

    tf.keras.backend.clear_session(); gc.collect(); set_seed(a.seed)
    model, _, _ = build_model(a.model, protocol, a.seed)
    compile_official(model, training)
    x_train = cache.inputs(a.model, outer_train)
    y_train = cache.targets(outer_train)
    scaler = StandardScaler(**training["target_scaler_config"]).fit(y_train)
    history = model.fit(x_train, scaler.transform(y_train), batch_size=int(training["batch_size"]),
                        epochs=int(training["epochs"]), verbose=2, callbacks=official_callbacks(training))
    save_history(history, run / "history.csv")
    model.save_weights(str(run / "model.weights.h5"))

    # The outer test fold is read only after training.
    x_test = cache.inputs(a.model, outer_test)
    pred_scaled = model.predict(x_test, batch_size=int(training["batch_size"]), verbose=0)
    prediction = scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()
    target = cache.targets(outer_test).ravel()
    score = metrics(target, prediction)
    pd.DataFrame({"sample_id": cache.ids(outer_test), "target": target, "prediction": prediction,
                  "absolute_error": np.abs(target - prediction)}).to_csv(run / "predictions.csv", index=False)
    (run / "target_scaler.json").write_text(json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                                                        "var": scaler.var_.tolist()}, indent=2), encoding="utf-8")
    payload = {"status": "complete", "protocol_id": protocol["protocol_id"], "task": a.task, "model": a.model,
               "paper_name": protocol["models"][a.model]["paper_name"], "fold": a.fold, "seed": a.seed,
               "epochs": int(training["epochs"]), "batch_size": int(training["batch_size"]),
               "parameter_count": int(model.count_params()), "train_size": int(len(outer_train)),
               "test_size": int(len(outer_test)), **score}
    (run / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DONE] {json.dumps(payload, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
