"""Composition-group-aware 70/15/15 split searched over candidate random states (seed 42)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _hist_distance(target: np.ndarray, positions: np.ndarray, edges: np.ndarray) -> float:
    if len(positions) == 0:
        return 10.0
    hist, _ = np.histogram(target[positions], bins=edges)
    hist = hist.astype(float) / max(1.0, float(hist.sum()))
    all_hist, _ = np.histogram(target, bins=edges)
    all_hist = all_hist.astype(float) / max(1.0, float(all_hist.sum()))
    return float(np.abs(hist - all_hist).sum())


def make_group_aware_split(frame: pd.DataFrame, cfg: dict) -> dict:
    ids = frame["sample_id"].astype(str).tolist()
    target = frame["target"].to_numpy(float)
    groups = frame["group"].astype(str).to_numpy()
    n = len(frame)
    seed = int(cfg["seed"])
    train_fraction = float(cfg["train_fraction"])
    val_fraction = float(cfg["validation_fraction"])
    test_fraction = float(cfg["test_fraction"])
    if len(set(groups)) < 3:
        raise RuntimeError("Need at least three independent composition groups for train/validation/test")
    quantiles = np.unique(np.quantile(target, np.linspace(0, 1, int(cfg.get("target_quantile_bins", 10)) + 1)))
    if len(quantiles) < 3:
        quantiles = np.linspace(float(np.min(target)) - 1e-6, float(np.max(target)) + 1e-6, 3)
    best = None
    candidate_searches = int(cfg.get("candidate_searches", 500))
    for attempt in range(candidate_searches):
        rs = seed + 1009 * attempt
        outer = GroupShuffleSplit(n_splits=1, test_size=val_fraction + test_fraction, random_state=rs)
        train_pos, temp_pos = next(outer.split(np.zeros(n), target, groups))
        temp_groups = groups[temp_pos]
        if len(set(temp_groups)) < 2:
            continue
        relative_test = test_fraction / (val_fraction + test_fraction)
        inner = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=rs + 17)
        val_local, test_local = next(inner.split(np.zeros(len(temp_pos)), target[temp_pos], temp_groups))
        val_pos = temp_pos[val_local]
        test_pos = temp_pos[test_local]
        if min(len(train_pos), len(val_pos), len(test_pos)) < 5:
            continue
        frac_penalty = (
            abs(len(train_pos) / n - train_fraction)
            + abs(len(val_pos) / n - val_fraction)
            + abs(len(test_pos) / n - test_fraction)
        )
        dist = sum(_hist_distance(target, pos, quantiles) for pos in (train_pos, val_pos, test_pos))
        score = 5.0 * frac_penalty + dist
        record = (score, rs, train_pos, val_pos, test_pos)
        if best is None or record[0] < best[0]:
            best = record
    if best is None:
        raise RuntimeError("Could not construct a valid group-aware 70/15/15 split")
    score, rs, train_pos, val_pos, test_pos = best
    sets = [set(train_pos.tolist()), set(val_pos.tolist()), set(test_pos.tolist())]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Split overlap detected")
    if set(groups[train_pos]) & set(groups[val_pos]) or set(groups[train_pos]) & set(groups[test_pos]) or set(groups[val_pos]) & set(groups[test_pos]):
        raise RuntimeError("Composition-group leakage detected")
    payload = {
        "schema": "materon_2d_group_split_v1",
        "seed": seed,
        "selected_candidate_random_state": int(rs),
        "selection_score": float(score),
        "train_ids": [ids[i] for i in train_pos],
        "validation_ids": [ids[i] for i in val_pos],
        "test_ids": [ids[i] for i in test_pos],
        "counts": {"train": len(train_pos), "validation": len(val_pos), "test": len(test_pos), "total": n},
        "fractions": {"train": len(train_pos)/n, "validation": len(val_pos)/n, "test": len(test_pos)/n},
        "group_counts": {
            "train": len(set(groups[train_pos])),
            "validation": len(set(groups[val_pos])),
            "test": len(set(groups[test_pos])),
        },
        "target_summary": {
            label: {"mean": float(np.mean(target[pos])), "std": float(np.std(target[pos])), "min": float(np.min(target[pos])), "max": float(np.max(target[pos]))}
            for label, pos in [("train", train_pos), ("validation", val_pos), ("test", test_pos)]
        },
        "test_locked": True,
    }
    payload["content_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def positions_for_ids(cache_ids: list[str], split: dict) -> dict[str, np.ndarray]:
    mapping = {str(sid): i for i, sid in enumerate(cache_ids)}
    if len(mapping) != len(cache_ids):
        raise RuntimeError("Duplicate IDs in cache")
    out = {}
    for key, split_key in [("train", "train_ids"), ("validation", "validation_ids"), ("test", "test_ids")]:
        missing = [sid for sid in split[split_key] if sid not in mapping]
        if missing:
            raise RuntimeError(f"Split IDs missing from cache ({key}): {missing[:5]}")
        out[key] = np.asarray([mapping[sid] for sid in split[split_key]], dtype=np.int64)
    return out


def write_split(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
