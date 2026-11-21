from __future__ import absolute_import, division, print_function

from typing import Dict, List, Sequence, Tuple

import numpy as np


def quantile_labels(targets: np.ndarray, max_bins: int = 10) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    if values.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    order = np.argsort(values, kind="mergesort")
    labels = np.zeros(len(values), dtype=int)
    bins = max(2, min(int(max_bins), max(2, len(values) // 8)))
    for rank, index in enumerate(order):
        labels[index] = min(bins - 1, int(rank * bins / max(1, len(values))))
    return labels


def make_inner_split(outer_train: Sequence[int], all_targets: np.ndarray, seed: int, val_fraction: float) -> Tuple[List[int], List[int]]:
    from sklearn.model_selection import StratifiedShuffleSplit

    indices = np.asarray(list(outer_train), dtype=int)
    labels = quantile_labels(np.asarray(all_targets)[indices])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(val_fraction), random_state=int(seed))
    local_train, local_val = next(splitter.split(np.zeros(len(indices)), labels))
    return indices[local_train].astype(int).tolist(), indices[local_val].astype(int).tolist()


def validate_outer_folds(folds: Dict[int, Dict[str, object]], n_samples: int) -> None:
    expected = set(range(int(n_samples)))
    if set(folds.keys()) != set(range(5)):
        raise AssertionError("Expected official folds 0..4, got %r" % sorted(folds.keys()))
    test_seen = []
    for fold in range(5):
        split = folds[fold]
        train = set(int(v) for v in split["train"])
        test = set(int(v) for v in split["test"])
        if train.intersection(test):
            raise AssertionError("Fold %d has train/test leakage" % fold)
        if train.union(test) != expected:
            raise AssertionError("Fold %d does not cover all samples" % fold)
        test_seen.extend(sorted(test))
    if sorted(test_seen) != sorted(expected):
        raise AssertionError("Each sample must appear in exactly one test fold")


def nested_fraction_subset(indices: Sequence[int], targets: np.ndarray, fraction: float, seed: int) -> List[int]:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    indices_array = np.asarray(list(indices), dtype=int)
    if float(fraction) >= 1.0:
        return indices_array.astype(int).tolist()
    ordered = indices_array[np.argsort(np.asarray(targets)[indices_array], kind="mergesort")]
    bucket_count = min(10, max(2, len(ordered) // 6))
    buckets = [np.asarray(bucket, dtype=int) for bucket in np.array_split(ordered, bucket_count)]
    rng = np.random.RandomState(int(seed))
    for bucket in buckets:
        rng.shuffle(bucket)
    interleaved = []
    longest = max(len(bucket) for bucket in buckets)
    for row in range(longest):
        for bucket in buckets:
            if row < len(bucket):
                interleaved.append(int(bucket[row]))
    count = max(8, int(round(len(indices_array) * float(fraction))))
    return interleaved[:min(count, len(interleaved))]
