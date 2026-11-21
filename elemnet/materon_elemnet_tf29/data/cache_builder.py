from __future__ import absolute_import, division, print_function

import argparse
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .. import CACHE_SCHEMA_VERSION, DEFAULT_SEEDS
from ..compat import collect_environment
from ..utils.io import atomic_write_json, build_file_hashes
from .element_context import build_element_prior, contexts_from_fraction_matrix
from .formula import ELEMENT_SYMBOLS, build_fraction_matrix
from .splits import make_inner_split, validate_outer_folds


def load_matbench_steels() -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    try:
        from matbench.bench import MatbenchBenchmark
    except ImportError as exc:
        raise RuntimeError("matbench==0.6 is required to build the official cache") from exc

    benchmark = MatbenchBenchmark(autoload=False, subset=["matbench_steels"])
    task = getattr(benchmark, "matbench_steels", None)
    if task is None:
        tasks = list(getattr(benchmark, "tasks", []))
        if not tasks:
            raise RuntimeError("Could not locate matbench_steels task in matbench 0.6")
        task = tasks[0]
    task.load()
    if task.df is None:
        raise RuntimeError("matbench_steels loaded without a dataframe")

    metadata = task.metadata
    input_column = str(getattr(metadata, "input_type", "composition"))
    target_column = str(getattr(metadata, "target", "yield strength"))
    if input_column not in task.df.columns:
        # matbench 0.6 metadata sometimes describes the input type rather than
        # the literal dataframe column. The composition column is the first one.
        input_column = str(task.df.columns[0])
    if target_column not in task.df.columns:
        target_column = str(task.df.columns[-1])

    original_indices = list(task.df.index)
    position_by_index = {index: position for position, index in enumerate(original_indices)}
    if len(position_by_index) != len(original_indices):
        raise RuntimeError("Matbench dataframe has duplicate row indices")
    frame = task.df[[input_column, target_column]].copy()
    frame.columns = ["composition", "target"]
    frame["composition"] = frame["composition"].astype(str)
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(float)
    frame = frame.reset_index(drop=True)

    folds = {}
    for raw_fold in task.folds:
        fold = int(raw_fold)
        train_df = task.get_train_and_val_data(raw_fold, as_type="df")
        test_df = task.get_test_data(raw_fold, as_type="df", include_target=True)
        folds[fold] = {
            "train": [int(position_by_index[index]) for index in train_df.index],
            "test": [int(position_by_index[index]) for index in test_df.index],
            "source": "matbench_v0.1_official",
            "matbench_fold": fold,
        }
    return frame, folds


def load_local_csv(path: Path, composition_column: str, target_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [name for name in [composition_column, target_column] if name not in frame.columns]
    if missing:
        raise KeyError("Missing CSV columns: %r" % missing)
    output = frame[[composition_column, target_column]].copy()
    output.columns = ["composition", "target"]
    output["composition"] = output["composition"].astype(str)
    output["target"] = pd.to_numeric(output["target"], errors="raise").astype(float)
    return output.reset_index(drop=True)


def fallback_outer_folds(n_samples: int, seed: int) -> Dict[int, Dict[str, Any]]:
    from sklearn.model_selection import KFold

    indices = np.arange(int(n_samples), dtype=int)
    splitter = KFold(n_splits=5, shuffle=True, random_state=int(seed))
    folds = {}
    for fold, (train, test) in enumerate(splitter.split(indices)):
        folds[fold] = {
            "train": indices[train].astype(int).tolist(),
            "test": indices[test].astype(int).tolist(),
            "source": "local_debug_kfold",
            "matbench_fold": None,
        }
    return folds


def build_engineered_features(compositions: List[str]):
    """Build the historical Magpie + Meredig + Stoichiometry feature view once.

    The group metadata is saved with the matrix so EIGate-Pro can apply a
    hierarchical group gate without guessing feature-family boundaries.
    """
    try:
        from matminer.featurizers.base import MultipleFeaturizer
        from matminer.featurizers.composition import ElementProperty, Meredig, Stoichiometry
        from pymatgen.core import Composition
    except ImportError as exc:
        raise RuntimeError("Engineered features require matminer and pymatgen") from exc
    family_featurizers = [
        ("magpie", ElementProperty.from_preset("magpie")),
        ("meredig", Meredig()),
        ("stoichiometry", Stoichiometry()),
    ]
    featurizer = MultipleFeaturizer([value for _, value in family_featurizers])
    objects = [Composition(value) for value in compositions]
    values = featurizer.featurize_many(objects, ignore_errors=False)
    matrix = np.asarray(values, dtype=np.float32)
    labels = list(featurizer.feature_labels())
    groups = []
    group_ids = []
    start = 0
    for group_index, (name, family) in enumerate(family_featurizers):
        family_labels = list(family.feature_labels())
        end = start + len(family_labels)
        groups.append({
            "name": name,
            "group_id": int(group_index),
            "start": int(start),
            "end": int(end),
            "size": int(end - start),
        })
        group_ids.extend([int(group_index)] * (end - start))
        start = end
    if start != matrix.shape[1] or len(labels) != matrix.shape[1]:
        raise RuntimeError("Engineered feature group boundaries are inconsistent")
    return matrix, labels, groups, group_ids


def verify_existing_cache(cache_dir: Path) -> Dict[str, Any]:
    from .cache import CachedDataset
    cached = CachedDataset.load(cache_dir, verify_hashes=True)
    return cached.manifest


def build_cache(args):
    output = Path(args.cache_dir).expanduser().resolve()
    if output.exists() and not args.rebuild_cache:
        manifest = verify_existing_cache(output)
        print("[CACHE REUSE] verified immutable cache: %s" % output)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return output
    if output.exists():
        shutil.rmtree(str(output))
    for relative in ["raw", "features", "splits", "preprocessing"]:
        (output / relative).mkdir(parents=True, exist_ok=True)

    if args.source == "matbench":
        frame, folds = load_matbench_steels()
        dataset_source = "matbench_v0.1_matbench_steels"
        official = True
    else:
        if not args.csv:
            raise ValueError("--csv is required for --source csv")
        frame = load_local_csv(Path(args.csv), args.composition_column, args.target_column)
        folds = fallback_outer_folds(len(frame), args.fallback_fold_seed)
        dataset_source = "local_csv:%s" % Path(args.csv).expanduser().resolve()
        official = False
        if not args.allow_fallback_folds:
            raise RuntimeError("Local CSV folds are debugging-only. Add --allow-fallback-folds explicitly.")

    if frame["composition"].isna().any() or frame["target"].isna().any():
        raise ValueError("Dataset contains missing composition or target")
    validate_outer_folds(folds, len(frame))
    frame.insert(0, "sample_id", np.arange(len(frame), dtype=int))
    frame.to_csv(output / "raw" / "dataset.csv", index=False)

    fractions = build_fraction_matrix(frame["composition"].astype(str).tolist(), args.max_atomic_number)
    np.savez_compressed(output / "features" / "element_fraction_float32.npz", x=fractions)
    atomic_write_json(output / "features" / "element_names.json", ELEMENT_SYMBOLS[:args.max_atomic_number])

    prior, prior_names = build_element_prior(args.max_atomic_number)
    contexts, context_names = contexts_from_fraction_matrix(fractions, prior)
    np.savez_compressed(output / "features" / "element_prior_float32.npz", x=prior)
    np.savez_compressed(output / "features" / "physical_context_float32.npz", x=contexts)
    atomic_write_json(output / "features" / "element_prior_names.json", prior_names)
    atomic_write_json(output / "features" / "physical_context_names.json", context_names)

    feature_views = ["elem_fraction"]
    engineered_dim = None
    engineered_group_count = None
    if args.include_engineered:
        engineered, engineered_names, engineered_groups, engineered_group_ids = build_engineered_features(
            frame["composition"].astype(str).tolist()
        )
        np.savez_compressed(output / "features" / "engineered_features_float32.npz", x=engineered)
        atomic_write_json(output / "features" / "engineered_feature_names.json", engineered_names)
        atomic_write_json(output / "features" / "engineered_feature_groups.json", engineered_groups)
        atomic_write_json(output / "features" / "engineered_feature_group_ids.json", engineered_group_ids)
        feature_views.append("legacy_engineered")
        engineered_dim = int(engineered.shape[1])
        engineered_group_count = int(len(engineered_groups))

    atomic_write_json(output / "splits" / "outer_folds.json",
                      {str(key): value for key, value in folds.items()})
    target_values = frame["target"].to_numpy(dtype=float)
    inner = {}
    for fold in range(5):
        for seed in args.seeds:
            train, val = make_inner_split(folds[fold]["train"], target_values, int(seed), args.val_fraction)
            inner["fold_%d/seed_%d" % (fold, int(seed))] = {
                "train": train,
                "val": val,
                "test": folds[fold]["test"],
            }
    atomic_write_json(output / "splits" / "inner_splits.json", inner)

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "dataset_source": dataset_source,
        "task": "matbench_steels" if official else "local_composition_regression",
        "n_samples": int(len(frame)),
        "target_name": "yield strength (MPa)" if official else args.target_column,
        "max_atomic_number": int(args.max_atomic_number),
        "seeds": [int(value) for value in args.seeds],
        "outer_folds": 5,
        "val_fraction": float(args.val_fraction),
        "feature_views": feature_views,
        "physical_context_dim": int(contexts.shape[1]),
        "engineered_feature_dim": engineered_dim,
        "engineered_feature_group_count": engineered_group_count,
        "formal_default_feature_view": "legacy_engineered" if args.include_engineered else "elem_fraction",
        "formal_official_folds": bool(official),
        "cache_policy": "immutable reuse; only --rebuild-cache may replace",
        "software_versions": collect_environment(),
    }
    manifest["file_sha256"] = build_file_hashes(output, exclude_manifest=True)
    atomic_write_json(output / "manifest.json", manifest)
    verify_existing_cache(output)
    print("[CACHE BUILD PASS] %s" % output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return output


def make_parser():
    parser = argparse.ArgumentParser(description="Build or verify the immutable local composition cache once.")
    parser.add_argument("--source", choices=["matbench", "csv"], default="matbench")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--csv")
    parser.add_argument("--composition-column", default="composition")
    parser.add_argument("--target-column", default="yield strength")
    parser.add_argument("--include-engineered", action="store_true", default=True)
    parser.add_argument("--skip-engineered", action="store_false", dest="include_engineered")
    parser.add_argument("--max-atomic-number", type=int, default=118)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--fallback-fold-seed", type=int, default=18012019)
    parser.add_argument("--allow-fallback-folds", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


def main():
    args = make_parser().parse_args()
    if tuple(args.seeds) != DEFAULT_SEEDS:
        print("WARNING: formal seeds are %r; received %r" % (DEFAULT_SEEDS, tuple(args.seeds)))
    build_cache(args)


if __name__ == "__main__":
    main()
