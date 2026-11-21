from __future__ import absolute_import, division, print_function

import math
from typing import List, Tuple

import numpy as np

from .formula import ELEMENT_SYMBOLS

PRIOR_PROPERTY_NAMES = [
    "atomic_number", "period", "group", "atomic_mass", "atomic_radius",
    "electronegativity", "mendeleev_number", "average_ionic_radius",
    "block_s", "block_p", "block_d", "block_f",
]


def _finite_float(value, default=np.nan):
    try:
        converted = float(value)
        if np.isfinite(converted):
            return converted
    except Exception:
        pass
    return float(default)


def _safe_attr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _element_raw_properties(atomic_number: int) -> List[float]:
    from pymatgen.core.periodic_table import Element

    element = Element.from_Z(int(atomic_number))
    radius = _safe_attr(element, "atomic_radius", None)
    if radius is None:
        radius = _safe_attr(element, "atomic_radius_calculated", None)
    average_ionic_radius = _safe_attr(element, "average_ionic_radius", None)
    block = str(_safe_attr(element, "block", ""))
    return [
        float(atomic_number),
        _finite_float(_safe_attr(element, "row", np.nan)),
        _finite_float(_safe_attr(element, "group", np.nan)),
        _finite_float(_safe_attr(element, "atomic_mass", np.nan)),
        _finite_float(radius),
        _finite_float(_safe_attr(element, "X", np.nan)),
        _finite_float(_safe_attr(element, "mendeleev_no", np.nan)),
        _finite_float(average_ionic_radius),
        1.0 if block == "s" else 0.0,
        1.0 if block == "p" else 0.0,
        1.0 if block == "d" else 0.0,
        1.0 if block == "f" else 0.0,
    ]


def build_element_prior(max_atomic_number: int = 118) -> Tuple[np.ndarray, List[str]]:
    rows = np.asarray([_element_raw_properties(z) for z in range(1, max_atomic_number + 1)], dtype=np.float64)
    # Fill unavailable old-pymatgen values column-wise, then robustly scale only
    # continuous properties. This table is independent of the supervised task.
    for column in range(rows.shape[1]):
        finite = np.isfinite(rows[:, column])
        if not finite.any():
            rows[:, column] = 0.0
        else:
            median = float(np.median(rows[finite, column]))
            rows[~finite, column] = median
    continuous = 8
    for column in range(continuous):
        values = rows[:, column]
        median = float(np.median(values))
        q1 = float(np.percentile(values, 25.0))
        q3 = float(np.percentile(values, 75.0))
        scale = q3 - q1
        if not np.isfinite(scale) or scale < 1.0e-12:
            scale = float(np.std(values))
        if not np.isfinite(scale) or scale < 1.0e-12:
            scale = 1.0
        rows[:, column] = (values - median) / scale
    return rows.astype(np.float32), list(PRIOR_PROPERTY_NAMES)


def contexts_from_fraction_matrix(fractions: np.ndarray, prior: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    fractions = np.asarray(fractions, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    if fractions.ndim != 2 or prior.ndim != 2 or fractions.shape[1] != prior.shape[0]:
        raise ValueError("Fraction/prior shape mismatch: %r vs %r" % (fractions.shape, prior.shape))
    means = np.matmul(fractions, prior)
    second = np.matmul(fractions, prior * prior)
    variances = np.maximum(second - means * means, 0.0)
    stds = np.sqrt(variances).astype(np.float32)
    nonzero = fractions > 0.0
    n_elements = nonzero.sum(axis=1, keepdims=True).astype(np.float32) / np.float32(fractions.shape[1])
    max_fraction = fractions.max(axis=1, keepdims=True).astype(np.float32)
    safe = np.where(nonzero, fractions, 1.0)
    entropy = (-np.where(nonzero, fractions * np.log(safe), 0.0).sum(axis=1, keepdims=True)).astype(np.float32)
    context = np.concatenate([means, stds, n_elements, max_fraction, entropy], axis=1).astype(np.float32)
    names = (["mean_" + name for name in PRIOR_PROPERTY_NAMES] +
             ["std_" + name for name in PRIOR_PROPERTY_NAMES] +
             ["normalized_element_count", "max_element_fraction", "mixing_entropy"])
    return context, names
