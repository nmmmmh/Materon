#!/usr/bin/env python3
"""One CGCNN transfer job (dielectric or JDFT2D, one model, one outer fold).

Architectures are frozen from the phonons study. Inside the outer training
pool, two learning-rate multipliers of the base recipe are trained on a
quantile-stratified inner split and the best (validation MAE, then epoch, then
learning rate) milestone of {160, 240, 320} is selected; the model is then
reinitialized, refit on the whole outer training pool and tested once.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def fit_seed(seed: int, fold: int) -> int:
    return int(seed) + 100_003 * int(fold) + 17


def calibration_seed(seed: int, fold: int) -> int:
    return int(seed) + 700_001 * int(fold) + 101


def lr_label(multiplier: float) -> str:
    return (f"x{multiplier:.6g}").replace(".", "p").replace("-", "m")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--outer-fold", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--log-every", type=int, default=20)
    a = p.parse_args()

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from materon.cgcnn.cache import official_positions_from_serialized, open_existing_cache
    from materon.cgcnn.config import get_config, get_recipe, transfer_protocol
    from materon.cgcnn.explain import export_analysis_summary
    from materon.cgcnn.models import build_paired_model
    from materon.cgcnn.training import inner_split, run_final, run_search, save_common, set_global_seed

    protocol = transfer_protocol()
    spec = protocol["models"][a.model]
    config = copy.deepcopy(get_config(spec["source_config"]))
    config["id"] = a.model
    base_recipe_id = spec["base_recipe"]
    base_recipe = copy.deepcopy(get_recipe(base_recipe_id))
    calibration = protocol["optimization_calibration"]
    checkpoint_epochs = [int(v) for v in calibration["checkpoint_epochs"]]

    cache = open_existing_cache(config, a.cache_root, a.task)
    outer_train_pos, outer_targets, outer_test_pos, _, outer_train_ids, outer_test_ids, _ = \
        official_positions_from_serialized(a.cache_root, a.task, cache, a.outer_fold)
    if len(outer_train_pos) + len(outer_test_pos) != int(protocol["tasks"][a.task]["n_samples"]):
        raise RuntimeError(f"Dataset-size mismatch for {a.task}")
    run_dir = Path(a.outdir) / a.task / a.model / f"outer_fold_{a.outer_fold}" / f"seed_{a.seed}"

    inner_train_pos, inner_val_pos, inner_random_state = inner_split(
        outer_train_pos, outer_targets, a.outer_fold, repeat=0, seed=int(calibration["inner_split_seed"]),
        val_fraction=float(calibration["inner_validation_fraction"]))
    x_inner_train, y_inner_train = cache.subset_positions(inner_train_pos)
    x_inner_val, y_inner_val = cache.subset_positions(inner_val_pos)
    sample = [v[:min(4, len(y_inner_train))] for v in x_inner_train]

    candidates = []
    for multiplier in [float(v) for v in calibration["learning_rate_multipliers"]]:
        recipe = copy.deepcopy(base_recipe)
        recipe["learning_rate"] = float(base_recipe["learning_rate"]) * multiplier
        recipe["id"] = f"{base_recipe_id}_{lr_label(multiplier)}"
        candidate_dir = run_dir / "calibration" / lr_label(multiplier)
        tf.keras.backend.clear_session(); gc.collect(); set_global_seed(a.seed)
        model, _, _ = build_paired_model(config, recipe, sample, a.seed)
        set_global_seed(calibration_seed(a.seed, a.outer_fold))
        history, _, _, inner_metrics, details = run_search(
            model, x_inner_train, y_inner_train, x_inner_val, y_inner_val, recipe, candidate_dir,
            max_epochs=int(calibration["max_epochs"]), checkpoint_epochs=checkpoint_epochs, log_every=a.log_every)
        pd.DataFrame(history.history).to_csv(candidate_dir / "history.csv", index=False)
        candidates.append({"base_recipe": base_recipe_id, "learning_rate_multiplier": multiplier,
                           "learning_rate": float(recipe["learning_rate"]), "selected_epoch": int(details["selected_epoch"]),
                           "validation_mae": float(inner_metrics["mae"]), "validation_rmse": float(inner_metrics["rmse"]),
                           "boundary_hit": bool(details["boundary_hit"])})
        del model
    table = pd.DataFrame(candidates).sort_values(["validation_mae", "selected_epoch", "learning_rate"],
                                                 ascending=[True, True, True])
    selected = table.iloc[0].to_dict()
    run_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(run_dir / "calibration_candidates.csv", index=False)
    (run_dir / "selected_optimization.json").write_text(
        json.dumps({**selected, "inner_split_random_state": int(inner_random_state)}, indent=2), encoding="utf-8")

    final_recipe = copy.deepcopy(base_recipe)
    final_recipe["learning_rate"] = float(selected["learning_rate"])
    final_recipe["id"] = f"{base_recipe_id}_selected"
    selected_epoch = int(selected["selected_epoch"])
    x_train, y_train = cache.subset_positions(outer_train_pos)
    x_test, y_test = cache.subset_positions(outer_test_pos)
    tf.keras.backend.clear_session(); gc.collect(); set_global_seed(a.seed)
    model, _, _ = build_paired_model(config, final_recipe, [v[:min(4, len(y_train))] for v in x_train], a.seed)
    actual_fit_seed = fit_seed(a.seed, a.outer_fold)
    set_global_seed(actual_fit_seed)
    history, scaler, predictions, metrics, details = run_final(model, x_train, y_train, x_test, y_test, final_recipe,
                                                               selected_epoch, a.log_every)
    np.savez_compressed(run_dir / "split_indices.npz", outer_train_positions=outer_train_pos,
                        outer_test_positions=outer_test_pos, inner_train_positions=inner_train_pos,
                        inner_validation_positions=inner_val_pos)
    extra = {"status": "complete", "protocol_id": protocol["protocol_id"], "backend": protocol["backend"],
             "task": a.task, "config": a.model, "paper_name": spec["paper_name"],
             "resolved_source_config": spec["source_config"], "base_recipe": base_recipe_id,
             "selected_learning_rate": float(selected["learning_rate"]),
             "selected_learning_rate_multiplier": float(selected["learning_rate_multiplier"]),
             "outer_fold": a.outer_fold, "seed": a.seed, "fit_seed": actual_fit_seed,
             "parameter_count": int(model.count_params()), **details, "selected_epoch": selected_epoch}
    save_common(run_dir, history, model, scaler, metrics, extra, predictions, y_test, outer_test_ids, save_weights=True)
    if config["model_type"] == "materon":
        export_analysis_summary(model, [v[:min(64, len(y_test))] for v in x_test], cache.metadata.get("feature_names", []),
                                run_dir / "analysis_summary.json")
    print(f"[DONE] task={a.task} model={a.model} fold={a.outer_fold} lr={selected['learning_rate']:.7g} "
          f"epoch={selected_epoch} mae={metrics['mae']:.8f}", flush=True)


if __name__ == "__main__":
    main()
