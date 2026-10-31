#!/usr/bin/env python3
"""One CGCNN phonons run: an inner-pool search trial or a final outer-fold refit.

inner: train on the inner split of an outer training pool and select the
       milestone epoch by inner-validation MAE (no test data).
final: retrain on the whole outer training pool for the selected epoch and
       evaluate the official outer test fold once.

Special --config values of the final phase:
  physics_parameter_matched   no-gate model enlarged to the parameter count of --match-to
  materon_feature_grouped     --source-config gate on grouped-PCA attributes
  materon_feature_shuffled    --source-config gate on attribute-shuffled full20
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

TASK = "matbench_phonons"


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["inner", "final"], required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--max-epochs", type=int, required=True)
    p.add_argument("--checkpoint-epochs", default="")
    p.add_argument("--selected-epoch", type=int)
    p.add_argument("--match-to")
    p.add_argument("--source-config")
    p.add_argument("--result-config")
    p.add_argument("--log-every", type=int, default=10)
    return p


def _sample(x, n):
    return [v[:n] for v in x]


def _fit_seed(seed, fold, repeat=0):
    return int(seed) + 100003 * int(fold) + 1009 * int(repeat) + 17


def _parameter_matched(task, cfg, target_cfg, recipe, sample, outer_ids, cache_root, seed):
    """No-gate model with an extra MLP sized to within 2% of the gated model's parameter count."""
    import numpy as np
    import tensorflow as tf
    from materon.cgcnn.cache import cache_class_for_config
    from materon.cgcnn.models import (build_model, build_paired_model, common_fingerprint,
                                      estimate_parameter_match_units)
    from materon.cgcnn.training import set_global_seed

    target_cache = cache_class_for_config(target_cfg)(task, cache_root, target_cfg.get("feature_mode", "full20"), False)
    ids = list(outer_ids[:min(4, len(outer_ids))])
    target_sample, _ = target_cache.subset_positions(np.asarray([target_cache.position[str(x)] for x in ids], int))
    tf.keras.backend.clear_session(); gc.collect(); set_global_seed(seed)
    target, _, _ = build_paired_model(target_cfg, recipe, target_sample, seed)
    target_params = int(target.count_params())
    del target
    tf.keras.backend.clear_session(); gc.collect(); set_global_seed(seed)
    base = build_model(cfg, recipe, seed)
    _ = base(sample, training=False)
    units = estimate_parameter_match_units(int(base.count_params()), target_params, recipe["hidden_dim"])
    del base
    tf.keras.backend.clear_session(); gc.collect(); set_global_seed(seed)
    model = build_model(cfg, recipe, seed, extra_capacity_units=units)
    _ = model(sample, training=False)
    gap = abs(int(model.count_params()) - target_params) / max(1, target_params)
    if gap > 0.02:
        raise RuntimeError("Parameter-match gap exceeds 2%")
    return model, target_params, units, gap, common_fingerprint(model)


def main():
    a = parser().parse_args()
    import numpy as np
    import tensorflow as tf
    from matbench.bench import MatbenchBenchmark
    from materon.cgcnn.cache import PhysicsCGCNNCache, cache_class_for_config
    from materon.cgcnn.config import get_config, get_recipe, protocol
    from materon.cgcnn.explain import export_analysis_summary
    from materon.cgcnn.models import build_paired_model
    from materon.cgcnn.training import (inner_split, official_positions, run_final, run_search, save_common,
                                        set_global_seed)

    proto = protocol()
    recipe = get_recipe(a.recipe)
    parameter_match = a.config == "physics_parameter_matched"
    if parameter_match:
        cfg = get_config("physics_full20")
    elif a.config in {"materon_feature_grouped", "materon_feature_shuffled"}:
        if not a.source_config:
            raise ValueError("--source-config is required for dynamic Materon feature controls")
        cfg = copy.deepcopy(get_config(a.source_config))
        cfg["id"] = a.config
        cfg["feature_mode"] = "grouped_pca" if a.config == "materon_feature_grouped" else "shuffled_full20"
    else:
        cfg = get_config(a.config)
    result = a.result_config or a.config
    run = (Path(a.outdir) / a.stage / result / ("recipe_" + a.recipe) / ("outer_fold_%d" % a.outer_fold)
           / ("seed_%d" % a.seed) / ("repeat_%d" % a.repeat if a.phase == "inner" else "final"))
    checkpoints = [int(x) for x in a.checkpoint_epochs.split(",") if x.strip()]

    task = next(iter(MatbenchBenchmark(subset=[TASK], autoload=False).tasks))
    task.load()
    cache_cls = PhysicsCGCNNCache if parameter_match else cache_class_for_config(cfg)
    cache = cache_cls(task, a.cache_root, cfg.get("feature_mode", "full20"), False)
    outer_train, outer_y, outer_test, _, outer_ids, test_ids = official_positions(task, cache, a.outer_fold)
    common = {"phase": a.phase, "stage": a.stage, "config": result, "resolved_config": cfg["id"],
              "recipe": a.recipe, "outer_fold": a.outer_fold, "seed": a.seed, "status": "complete"}

    if a.phase == "inner":
        if not checkpoints or max(checkpoints) != a.max_epochs:
            raise ValueError("Search checkpoints must include max_epochs")
        train_pos, val_pos, split_seed = inner_split(outer_train, outer_y, a.outer_fold, a.repeat,
                                                     proto["inner_split_base_seed"], proto["inner_validation_fraction"],
                                                     proto["inner_quantile_bins"])
        xtr, ytr = cache.subset_positions(train_pos)
        xva, yva = cache.subset_positions(val_pos)
        sample = _sample(xtr, min(4, len(ytr)))
        tf.keras.backend.clear_session(); gc.collect(); set_global_seed(a.seed)
        model, _, _ = build_paired_model(cfg, recipe, sample, a.seed)
        fit_seed = _fit_seed(a.seed, a.outer_fold, a.repeat)
        set_global_seed(fit_seed)
        history, scaler, pred, metrics, details = run_search(model, xtr, ytr, xva, yva, recipe, run, a.max_epochs,
                                                             checkpoints, a.log_every)
        run.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run / "split_indices.npz", train_positions=train_pos, validation_positions=val_pos)
        extra = {**common, "repeat": a.repeat, "split_seed": split_seed, "fit_seed": fit_seed,
                 "train_count": len(train_pos), "validation_count": len(val_pos), "max_epochs": a.max_epochs,
                 "parameter_count": int(model.count_params()), **details}
        save_common(run, history, model, scaler, metrics, extra, pred, yva, [cache.indices[i] for i in val_pos],
                    save_weights=False)
    else:
        if a.selected_epoch is None:
            raise ValueError("--selected-epoch is required for the final phase")
        refit = proto["final_refit"]
        if not refit["selected_epoch_min"] <= a.selected_epoch <= refit["selected_epoch_max"]:
            raise ValueError("Selected epoch outside approved range")
        xtr, ytr = cache.subset_positions(outer_train)
        xte, yte = cache.subset_positions(outer_test)
        sample = _sample(xtr, min(4, len(ytr)))
        tf.keras.backend.clear_session(); gc.collect(); set_global_seed(a.seed)
        target_params, units, gap = None, 0, None
        if parameter_match:
            if not a.match_to:
                raise ValueError("--match-to required")
            model, target_params, units, gap, _ = _parameter_matched(task, cfg, get_config(a.match_to), recipe,
                                                                     sample, outer_ids, a.cache_root, a.seed)
        else:
            model, _, _ = build_paired_model(cfg, recipe, sample, a.seed)
        fit_seed = _fit_seed(a.seed, a.outer_fold)
        set_global_seed(fit_seed)
        history, scaler, pred, metrics, details = run_final(model, xtr, ytr, xte, yte, recipe, a.selected_epoch,
                                                            a.log_every)
        run.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run / "split_indices.npz", outer_train_positions=outer_train, outer_test_positions=outer_test)
        extra = {**common, "match_to": a.match_to, "source_config": a.source_config, "fit_seed": fit_seed,
                 "outer_train_count": len(outer_train), "outer_test_count": len(outer_test),
                 "parameter_count": int(model.count_params()), "target_parameter_count": target_params,
                 "parameter_match_units": units, "parameter_match_relative_gap": gap, "max_epochs": a.selected_epoch,
                 **details}
        save_common(run, history, model, scaler, metrics, extra, pred, yte, test_ids, save_weights=True)
        if cfg["model_type"] == "materon":
            export_analysis_summary(model, _sample(xte, min(64, len(yte))), cache.metadata.get("feature_names", []),
                                    run / "analysis_summary.json")
    print("[DONE] phase=%s stage=%s config=%s recipe=%s fold=%d seed=%d mae=%.8f"
          % (a.phase, a.stage, result, a.recipe, a.outer_fold, a.seed, metrics["mae"]), flush=True)


if __name__ == "__main__":
    main()
