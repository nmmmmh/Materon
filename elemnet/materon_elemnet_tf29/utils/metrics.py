from __future__ import absolute_import, division, print_function

import numpy as np


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
