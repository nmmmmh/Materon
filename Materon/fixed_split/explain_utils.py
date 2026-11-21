"""Intervention statistics of the fixed-split interpretation study."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from materon.megnet.interpret import flat, top_count, unscale  # noqa: F401  (re-exported)


def row_splits(value):
    if hasattr(value, "row_splits"):
        return np.asarray(value.row_splits.numpy(), dtype=np.int64)
    raise TypeError("Expected RaggedTensor")


def intervention_metrics(target, base, changed) -> dict:
    target = np.asarray(target).reshape(-1)
    base = np.asarray(base).reshape(-1)
    changed = np.asarray(changed).reshape(-1)
    base_abs = np.abs(target - base)
    changed_abs = np.abs(target - changed)
    pred_change = np.abs(changed - base)
    return {
        "base_mae": float(np.mean(base_abs)),
        "intervention_mae": float(np.mean(changed_abs)),
        "delta_mae": float(np.mean(changed_abs - base_abs)),
        "base_rmse": float(mean_squared_error(target, base, squared=False)),
        "intervention_rmse": float(mean_squared_error(target, changed, squared=False)),
        "mean_abs_prediction_change": float(np.mean(pred_change)),
        "median_abs_prediction_change": float(np.median(pred_change)),
        "max_abs_prediction_change": float(np.max(pred_change)),
        "signed_mean_prediction_change": float(np.mean(changed - base)),
    }


def bootstrap_ci(values, repeats=1000, seed=42, alpha=0.05):
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    stats = np.empty(int(repeats), dtype=float)
    for i in range(int(repeats)):
        stats[i] = np.mean(values[rng.randint(0, len(values), len(values))])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def feature_groups(feature_names):
    groups = {
        "thermophysical": ["density", "atom_melting_point", "atom_specific_heat_capacity", "atomic_boiling_point",
                           "atomic_fusion_heat", "atomic_thermal_cond"],
        "electronic": ["atomic_ip", "atomic_ea", "atomic_hfomo", "atomic_lfumo", "X", "atomic_pol"],
        "valence": ["atomic_nvalences", "atomic_ndval", "atomic_npval", "atomic_nsval"],
        "size_mass": ["atomic_radius", "atomic_covalent Radius", "atomic_volume", "atomic_weight"],
    }
    lookup = {name: i for i, name in enumerate(feature_names)}
    return {group: [lookup[x] for x in names if x in lookup] for group, names in groups.items()}
