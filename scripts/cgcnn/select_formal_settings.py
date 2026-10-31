#!/usr/bin/env python3
"""Fold-specific formal settings of every reported CGCNN phonons model, from inner results only.

materon_best_nested is the reported Materon model (Table I); materon_best_consensus
is the fixed shared configuration (Consensus, Table S7).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from materon.cgcnn.config import protocol, suites


def best_inner_setting(df: pd.DataFrame, config: str, fold: int, std_weight: float) -> dict:
    group = df[(df.config == config) & (df.outer_fold == fold)].copy()
    if group.empty:
        raise RuntimeError(f"No inner result for formal config {config} fold {fold}")
    agg = group.groupby("recipe", as_index=False).agg(
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        median_epoch=("selected_epoch", "median"),
        runs=("mae", "size"),
    )
    agg["score"] = agg.mean_mae + std_weight * agg.std_mae.fillna(0)
    row = agg.sort_values(["score", "mean_mae", "recipe"]).iloc[0]
    return {
        "recipe": str(row.recipe),
        "selected_epoch": int(round(row.median_epoch)),
        "inner_selection_score": float(row.score),
        "inner_runs": int(row.runs),
    }


def best_capacity(df: pd.DataFrame, fold: int, prefix: str, std_weight: float) -> dict | None:
    group = df[(df.outer_fold == fold) & df.config.str.startswith(prefix)].copy()
    if group.empty:
        return None
    agg = group.groupby(["config", "recipe"], as_index=False).agg(
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        median_epoch=("selected_epoch", "median"),
        runs=("mae", "size"),
    )
    agg["score"] = agg.mean_mae + std_weight * agg.std_mae.fillna(0)
    row = agg.sort_values(["score", "mean_mae", "config", "recipe"]).iloc[0]
    return {
        "resolved_config": str(row.config),
        "recipe": str(row.recipe),
        "selected_epoch": int(round(row.median_epoch)),
        "inner_selection_score": float(row.score),
        "inner_runs": int(row.runs),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inner-csv", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--winner", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--std-weight", type=float, default=0.10)
    a = p.parse_args()

    df = pd.read_csv(a.inner_csv)
    backbone = json.loads(Path(a.backbone).read_text(encoding="utf-8"))
    winner = json.loads(Path(a.winner).read_text(encoding="utf-8"))
    folds = protocol()["outer_folds"]
    settings: dict[str, dict] = {str(fold): {} for fold in folds}

    # First materialize every fixed article configuration. Physics feature controls
    # reuse the selected Physics recipe/epoch so only the feature representation changes.
    for fold in folds:
        for config in suites()["formal_fixed"]:
            if config in {"cgcnn_baseline", "physics_full20"}:
                item = backbone["per_fold"][config][str(fold)][0]
                settings[str(fold)][config] = {
                    "recipe": item["recipe"],
                    "selected_epoch": int(item["selected_epoch"]),
                    "selection_source": "backbone_search",
                }
            elif config in {"physics_grouped_pca", "physics_shuffled_full20"}:
                item = backbone["per_fold"]["physics_full20"][str(fold)][0]
                settings[str(fold)][config] = {
                    "recipe": item["recipe"],
                    "selected_epoch": int(item["selected_epoch"]),
                    "selection_source": "physics_matched_feature_control",
                }
            else:
                selected = best_inner_setting(df, config, fold, a.std_weight)
                selected["selection_source"] = "inner_validation"
                settings[str(fold)][config] = selected

    nested = {str(fold): winner["per_fold"][str(fold)][0] for fold in folds}
    fold_winners = [nested[str(fold)]["config"] for fold in folds]
    consensus = Counter(fold_winners).most_common(1)[0][0]

    for fold in folds:
        consensus_item = best_inner_setting(df, consensus, fold, a.std_weight)
        consensus_item.update({
            "resolved_config": consensus,
            "selection_source": "cross_fold_consensus_architecture",
        })
        settings[str(fold)]["materon_best_consensus"] = consensus_item

        nested_item = nested[str(fold)]
        settings[str(fold)]["materon_best_nested"] = {
            "resolved_config": nested_item["config"],
            "recipe": nested_item["recipe"],
            "selected_epoch": int(nested_item["selected_epoch"]),
            "inner_selection_score": float(nested_item["selection_score"]),
            "selection_source": "fold_specific_nested_winner",
        }

        legacy_capacity = best_capacity(df, fold, "legacy_capacity_", a.std_weight)
        refined_capacity = best_capacity(df, fold, "refined_capacity_", a.std_weight)
        if legacy_capacity is not None:
            legacy_capacity["selection_source"] = "legacy_capacity_search"
            settings[str(fold)]["materon_best_legacy_capacity"] = legacy_capacity
        if refined_capacity is not None:
            refined_capacity["selection_source"] = "refined_capacity_search"
            settings[str(fold)]["materon_best_refined_capacity"] = refined_capacity

        # Dynamic formal controls use the exact consensus input-gate architecture.
        source_config = settings[str(fold)]["materon_best_consensus"]["resolved_config"]
        recipe = settings[str(fold)]["materon_best_consensus"]["recipe"]
        epoch = settings[str(fold)]["materon_best_consensus"]["selected_epoch"]
        settings[str(fold)]["physics_parameter_matched"] = {
            "resolved_config": "physics_parameter_matched",
            "match_to": source_config,
            "recipe": recipe,
            "selected_epoch": epoch,
            "selection_source": "consensus_matched_parameter_control",
        }
        settings[str(fold)]["materon_best_grouped_pca"] = {
            "resolved_config": "materon_feature_grouped",
            "source_config": source_config,
            "recipe": recipe,
            "selected_epoch": epoch,
            "selection_source": "consensus_matched_feature_control",
        }
        settings[str(fold)]["materon_best_shuffled_full20"] = {
            "resolved_config": "materon_feature_shuffled",
            "source_config": source_config,
            "recipe": recipe,
            "selected_epoch": epoch,
            "selection_source": "consensus_matched_feature_control",
        }

    payload = {
        "rule": "all settings chosen from outer-training-pool inner results only",
        "consensus_config": consensus,
        "consensus_fold_votes": dict(Counter(fold_winners)),
        "nested_winner_by_fold": nested,
        "per_fold": settings,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
