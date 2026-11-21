from __future__ import absolute_import, division, print_function

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .compat import configure_tensorflow, validate_target_environment
from .data.cache import CachedDataset
from .data.cache_builder import build_cache
from .models.config_serialization import model_config_from_dict, model_config_to_dict
from .models.registry import baseline_registry, v14_candidate_registry
from .models.tuning_factory import plain_additive_control
from .training.blend_tuning_controller import select_row
from .training.initial_weights import InitialWeightStore
from .training.preprocess import PerformancePreprocessor
from .training.protocols import get_protocol
from .training.train_core import fit_discovery, predict_original, regression_metrics
from .utils.io import atomic_write_json


class Report(object):
    def __init__(self):
        self.items = []

    def check(self, name, passed, evidence):
        self.items.append({"name": str(name), "passed": bool(passed), "evidence": str(evidence)})
        if not passed:
            raise AssertionError("%s: %s" % (name, evidence))


def make_debug_csv(path):
    formulas = ["Fe0.95Cr0.05", "Fe0.90Ni0.10", "Fe0.92Mn0.08", "Fe0.97C0.03"]
    pd.DataFrame([
        {"composition": formulas[i % 4], "target": 450.0 + 7.0 * i}
        for i in range(60)
    ]).to_csv(path, index=False)


def synthetic_blend_grid():
    return pd.DataFrame([
        {
            "config": "gate_a", "blend_lambda": 0.05,
            "candidate_mean_mae": 9.6, "baseline_mean_mae": 10.0,
            "mean_improvement_mpa": 0.4, "median_improvement_mpa": 0.3,
            "improvement_std_mpa": 0.5, "run_wins": 11,
            "seed_wins": 3, "fold_wins": 4,
            "worst_seed_improvement_mpa": -0.2,
            "worst_fold_improvement_mpa": -0.4,
        },
        {
            "config": "gate_b", "blend_lambda": 0.20,
            "candidate_mean_mae": 9.5, "baseline_mean_mae": 10.0,
            "mean_improvement_mpa": 0.5, "median_improvement_mpa": 0.2,
            "improvement_std_mpa": 0.8, "run_wins": 10,
            "seed_wins": 3, "fold_wins": 4,
            "worst_seed_improvement_mpa": -0.3,
            "worst_fold_improvement_mpa": -0.8,
        },
        {
            "config": "gate_a", "blend_lambda": 0.0,
            "candidate_mean_mae": 10.0, "baseline_mean_mae": 10.0,
            "mean_improvement_mpa": 0.0, "median_improvement_mpa": 0.0,
            "improvement_std_mpa": 0.0, "run_wins": 0,
            "seed_wins": 0, "fold_wins": 0,
            "worst_seed_improvement_mpa": 0.0,
            "worst_fold_improvement_mpa": 0.0,
        },
    ])


def main():
    parser = argparse.ArgumentParser(description="V14 target-runtime acceptance")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = Report()

    env = validate_target_environment(strict=True)
    for name, status in env["checks"].items():
        report.check("environment_%s" % name, status, env["environment"])
    tf = configure_tensorflow(42, deterministic=True)
    report.check("tensorflow_eager", tf.executing_eagerly(), tf.__version__)

    with tempfile.TemporaryDirectory(prefix="elemnet_v14_acceptance_") as temp:
        root = Path(temp)
        csv_path = root / "debug.csv"
        cache_dir = root / "cache"
        make_debug_csv(csv_path)
        build_cache(argparse.Namespace(
            source="csv", cache_dir=str(cache_dir), csv=str(csv_path),
            composition_column="composition", target_column="target",
            include_engineered=True, max_atomic_number=118,
            seeds=[42, 123, 2024], val_fraction=0.15,
            fallback_fold_seed=18012019, allow_fallback_folds=True,
            rebuild_cache=False,
        ))
        cache = CachedDataset.load(cache_dir, verify_hashes=True)
        x = cache.feature_view("legacy_engineered")
        report.check("legacy_engineered_258", x.shape == (60, 258), x.shape)

        candidates = v14_candidate_registry(258, [16, 8])
        report.check("five_candidates", len(candidates) == 5, sorted(candidates))
        trust = candidates["V14_G3_I64P64_PD0p025_TR0p10"]
        input_only = candidates["V14_G4_INPUT_ONLY_FULL_R32_D0p10"]
        report.check(
            "post_trust_only",
            trust.input_gate.modulation_trust_ratio == 0.0
            and trust.post_gate.modulation_trust_ratio == 0.10,
            trust,
        )
        report.check("input_only_fallback", input_only.post_gate is None, input_only)
        restored = model_config_from_dict(model_config_to_dict(trust))
        report.check("config_roundtrip", restored == trust, restored)

        store = InitialWeightStore(root / "initial_backbones")
        baseline = baseline_registry(258, [16, 8])["H0_exact_baseline"]
        bmodel = store.build_model(baseline, 42)
        gmodel = store.build_model(trust, 42)
        report.check(
            "same_backbone",
            bmodel.backbone_sha256() == gmodel.backbone_sha256(),
            bmodel.backbone_sha256(),
        )
        control = plain_additive_control(trust)
        cmodel = store.build_model(control, 42)
        report.check(
            "control_parameter_match",
            gmodel.count_params() == cmodel.count_params(),
            (gmodel.count_params(), cmodel.count_params()),
        )

        protocol = replace(
            get_protocol("V14_P1_stdX_rawY_mae_b32_checkpoint"),
            epochs=3, patience=2, lr_patience=1,
        )
        train_idx = np.arange(40)
        val_idx = np.arange(40, 50)
        test_idx = np.arange(50, 60)
        processor = PerformancePreprocessor.fit(
            x[train_idx], cache.targets[train_idx],
            protocol.input_scaler, protocol.target_scaling,
        )
        discovery = fit_discovery(
            trust, protocol, 42,
            processor.transform_x(x[train_idx]),
            processor.transform_y(cache.targets[train_idx]),
            processor.transform_x(x[val_idx]),
            processor.transform_y(cache.targets[val_idx]),
            initial_weight_store=store,
        )
        pred = predict_original(
            discovery["model"], processor.transform_x(x[test_idx]), processor.inverse_y
        )
        report.check("checkpoint_formal_path_finite", np.isfinite(pred).all(), pred.shape)
        report.check(
            "formal_strategy_checkpoint",
            protocol.formal_strategy == "discovery_checkpoint"
            and not protocol.refit_full_outer_train,
            protocol.to_dict(),
        )
        metrics = regression_metrics(cache.targets[test_idx], pred)
        report.check("checkpoint_metrics_finite", np.isfinite(metrics["mae"]), metrics)

        selected, scientific, decision, _ = select_row(synthetic_blend_grid())
        report.check("blend_selector_small_lambda", float(selected["blend_lambda"]) == 0.05, selected.to_dict())
        report.check("blend_selector_strict_pass", scientific and decision == "STRICT_ROBUST_BLEND_PASS", decision)

    payload = {
        "execution_status": "PASS",
        "all_pass": all(item["passed"] for item in report.items),
        "scientific_gate_pass": None,
        "scope": "V14 standalone baseline-anchored runtime acceptance",
        "environment": env,
        "items": report.items,
        "passed": sum(item["passed"] for item in report.items),
        "total": len(report.items),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print("TARGET_RUNTIME_ACCEPTANCE_PASS %d/%d" % (payload["passed"], payload["total"]))


if __name__ == "__main__":
    main()
