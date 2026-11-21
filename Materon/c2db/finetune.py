"""Fine-tuning on DFT substitution increments (Eq. S3, Sec. S3.3).

Each L400 member continues training on the C2DB training set with

    L = MSE(C2DB minibatch) + lam * MSE[(F(q) - F(P) - dPhi_DFT) / sigma]

where F(q) and F(P) are evaluated at the parent geometry and the pair term runs
over all training pairs at every step. Absolute DFT work functions never enter.
Paper setting: lam = 1, 30 epochs, learning rate 2e-4, batch 64.

Pair file: JSON list of {"name", "group", "parent_id", "site", "z_sub",
"dft_delta_eV"}; ``group`` is the parent used for leave-one-parent-out
cross-validation (pass ``holdout=None`` to train on all pairs).
"""
import time
from pathlib import Path

import numpy as np

from materon.c2db.dataset import Prepared, write_json
from materon.c2db.graph import copy_with_features, model_inputs

ARM = "physical20_environment_gate"


def finetune(prepared_dir, base_checkpoint, pairs, table, seed, out_dir,
             holdout=None, lam=1.0, epochs=30, lr=2e-4, save=True, batch_size=64):
    import tensorflow as tf

    from materon.c2db.models import build_megnet

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(seed)
    data = Prepared(prepared_dir)
    mean, std = data.mean, data.std
    graphs = data.graphs
    tr_ids, va_ids = data.ids("train"), data.ids("validation")
    ytr, yva = data.targets("train"), data.targets("validation")

    def pair_graphs(rows):
        parents, candidates = [], []
        for r in rows:
            p = copy_with_features(graphs[r["parent_id"]], table)
            parents.append(p)
            candidates.append(copy_with_features(p, table, r["site"], r["z_sub"]))
        return parents, candidates

    train_rows = [r for r in pairs if r["group"] != holdout]
    hold_rows = [r for r in pairs if r["group"] == holdout]
    ptr, qtr = pair_graphs(train_rows)
    dtr = tf.constant(np.array([r["dft_delta_eV"] for r in train_rows]) / std, dtype=tf.float32)
    pair_inputs = model_inputs(ptr + qtr, ARM)
    npair = len(train_rows)

    model = build_megnet(ARM)
    model.load_weights(str(base_checkpoint))
    tr_inputs = model_inputs([graphs[i] for i in tr_ids], ARM)
    va_inputs = model_inputs([graphs[i] for i in va_ids], ARM)
    target = tf.constant(((ytr - mean) / std)[:, None], dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function(reduce_retracing=True)
    def step(inputs, labels):
        with tf.GradientTape() as tape:
            l1 = tf.reduce_mean(tf.square(model(inputs, training=True) - labels))
            pp = tf.reshape(model(pair_inputs, training=True), [-1])
            l2 = tf.reduce_mean(tf.square(pp[npair:] - pp[:npair] - dtr))
            loss = l1 + lam * l2
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return l1, l2

    def predict_graphs(gs, bs=64):
        return np.concatenate([model(model_inputs(gs[s:s + bs], ARM), training=False).numpy().reshape(-1)
                               for s in range(0, len(gs), bs)]) * std + mean

    def predict_inputs(inputs, n, bs=64):
        return np.concatenate([model([tf.gather(v, np.arange(s, min(s + bs, n))) for v in inputs],
                                     training=False).numpy().reshape(-1)
                               for s in range(0, n, bs)]) * std + mean

    def deltas(rows):
        if not rows:
            return []
        parents, candidates = pair_graphs(rows)
        pr = predict_graphs(parents + candidates)
        return (pr[len(rows):] - pr[:len(rows)]).tolist()

    t0 = time.monotonic()
    before = {"val_mae": float(np.mean(np.abs(predict_inputs(va_inputs, len(va_ids)) - yva))),
              "hold_delta": deltas(hold_rows), "train_delta": deltas(train_rows)}
    rng = np.random.RandomState(seed + 1000)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(tr_ids))
        total1 = total2 = 0.0
        nb = 0
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            l1, l2 = step([tf.gather(v, idx) for v in tr_inputs], tf.gather(target, idx))
            total1 += float(l1)
            total2 += float(l2)
            nb += 1
        history.append({"epoch": epoch + 1, "c2db_mse": total1 / nb, "pair_mse": total2 / nb})
    vp = predict_inputs(va_inputs, len(va_ids))
    after = {"val_mae": float(np.mean(np.abs(vp - yva))),
             "hold_delta": deltas(hold_rows), "train_delta": deltas(train_rows)}
    record = {"seed": seed, "holdout": holdout, "lam": lam, "epochs": epochs, "lr": lr,
              "n_train_pairs": npair, "hold": [r["name"] for r in hold_rows],
              "hold_dft": [r["dft_delta_eV"] for r in hold_rows], "before": before, "after": after,
              "history": history, "seconds": time.monotonic() - t0}
    if save:
        model.save_weights(str(out / "best.weights.h5"))
        write_json(out / "VALIDATION_PREDICTIONS.json", dict(zip(va_ids, vp.tolist())))
        write_json(out / "RESULT.json", {"fine_tuned_from": str(base_checkpoint), "seed": seed, "lam": lam,
                                         "epochs": epochs, "lr": lr,
                                         "scores": {"validation": {"mae": after["val_mae"]}}})
    write_json(out / ("FT_%s_lam%s_ep%d_s%d.json" % (holdout, lam, epochs, seed)), record, indent=1)
    return record
