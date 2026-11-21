"""Small helpers shared by the MEGNet, CGCNN-detail and fixed-split intervention scripts."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error


def flat(value):
    return value.flat_values if hasattr(value, "flat_values") else value


def flat_rows(ragged):
    return [np.asarray(x) for x in ragged]


def unscale(pred, scaler: dict) -> np.ndarray:
    """Undo target standardization; ``scaler`` holds scalar or one-element 'mean' and 'scale'."""
    mean = float(np.asarray(scaler["mean"], dtype=float).reshape(-1)[0])
    scale = float(np.asarray(scaler["scale"], dtype=float).reshape(-1)[0])
    return np.asarray(pred).reshape(-1) * scale + mean


def prediction_metrics(target, pred, full) -> dict:
    return {"mae": float(mean_absolute_error(target, pred)),
            "mean_abs_prediction_change": float(np.mean(np.abs(pred - full))),
            "max_abs_prediction_change": float(np.max(np.abs(pred - full)))}


def top_count(n, fraction):
    return max(1, min(int(n), int(math.ceil(float(n) * float(fraction)))))
