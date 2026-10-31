#!/usr/bin/env python3
"""Write a CGCNN phonons plan (backbone rounds, gate searches, confirmation, final refits).

Run the plan with scripts/run_plan.py --script scripts/cgcnn/run_trial.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from materon.cgcnn.config import protocol, recipes, suites


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kind",
        choices=[
            "backbone_r1", "backbone_r2", "backbone_r3",
            "gate_core", "gate_context", "gate_capacity", "gate_confirm", "final",
        ],
        required=True,
    )
    p.add_argument("--output", required=True)
    p.add_argument("--root", required=True, help="output root of the runs")
    p.add_argument("--cache-root", required=True)
    p.add_argument("--selection")
    p.add_argument("--formal-settings")
    a = p.parse_args()
    proto = protocol()
    jobs = []

    def add(job_id: str, args: list) -> None:
        jobs.append({"job_id": job_id, "args": [str(x) for x in args]})

    def common(phase, stage, config, recipe, fold, seed, repeat, max_epochs, checkpoints):
        return [
            "--phase", phase,
            "--stage", stage,
            "--config", config,
            "--recipe", recipe,
            "--outer-fold", fold,
            "--seed", seed,
            "--repeat", repeat,
            "--cache-root", a.cache_root,
            "--outdir", a.root,
            "--max-epochs", max_epochs,
            "--checkpoint-epochs", ",".join(map(str, checkpoints)),
        ]

    if a.kind == "backbone_r1":
        stage = proto["epoch_search"]["round1"]
        for config in suites()["backbone_families"]:
            for recipe in recipes():
                for fold in proto["outer_folds"]:
                    for seed in stage["seeds"]:
                        for repeat in stage["repeats"]:
                            add(
                                f"br1_{config}_{recipe}_f{fold}_s{seed}_r{repeat}",
                                common(
                                    "inner", "backbone_round1", config, recipe, fold, seed, repeat,
                                    stage["max_epochs"], stage["checkpoint_epochs"],
                                ),
                            )

    elif a.kind in {"backbone_r2", "backbone_r3"}:
        if not a.selection:
            raise ValueError("--selection required")
        selection = json.loads(Path(a.selection).read_text(encoding="utf-8"))
        round_name = "round2" if a.kind == "backbone_r2" else "round3"
        stage = proto["epoch_search"][round_name]
        for config, folds in selection["per_fold"].items():
            for fold, items in folds.items():
                for item in items:
                    for seed in stage["seeds"]:
                        for repeat in stage["repeats"]:
                            add(
                                f"{round_name}_{config}_{item['recipe']}_f{fold}_s{seed}_r{repeat}",
                                common(
                                    "inner", f"backbone_{round_name}", config, item["recipe"],
                                    int(fold), seed, repeat, stage["max_epochs"], stage["checkpoint_epochs"],
                                ),
                            )

    elif a.kind in {"gate_core", "gate_context", "gate_capacity"}:
        if not a.selection:
            raise ValueError("--selection required")
        backbone = json.loads(Path(a.selection).read_text(encoding="utf-8"))
        mapping = {
            "gate_core": ("core", suites()["gate_core"]),
            "gate_context": ("context", suites()["context"]),
            "gate_capacity": ("capacity", suites()["legacy_capacity"] + suites()["refined_capacity"]),
        }
        stage_key, configs = mapping[a.kind]
        stage = proto["gate_search"][stage_key]
        for fold in proto["outer_folds"]:
            selected = backbone["per_fold"]["physics_full20"][str(fold)]
            recipe_items = selected[:2] if a.kind == "gate_core" else selected[:1]
            for config in configs:
                for item in recipe_items:
                    for seed in stage["seeds"]:
                        for repeat in stage["repeats"]:
                            add(
                                f"{stage_key}_{config}_{item['recipe']}_f{fold}_s{seed}_r{repeat}",
                                common(
                                    "inner", f"gate_{stage_key}", config, item["recipe"], fold,
                                    seed, repeat, stage["max_epochs"], stage["checkpoint_epochs"],
                                ),
                            )

    elif a.kind == "gate_confirm":
        if not a.selection:
            raise ValueError("--selection required")
        selection = json.loads(Path(a.selection).read_text(encoding="utf-8"))
        stage = proto["gate_search"]["confirm"]
        for fold, items in selection["per_fold"].items():
            for item in items:
                for seed in stage["seeds"]:
                    for repeat in stage["repeats"]:
                        add(
                            f"confirm_{item['config']}_{item['recipe']}_f{fold}_s{seed}_r{repeat}",
                            common(
                                "inner", "gate_confirm", item["config"], item["recipe"], int(fold),
                                seed, repeat, stage["max_epochs"], stage["checkpoint_epochs"],
                            ),
                        )

    else:
        if not a.formal_settings:
            raise ValueError("--formal-settings required")
        settings = json.loads(Path(a.formal_settings).read_text(encoding="utf-8"))
        seeds = proto["final_refit"]["seeds"]
        for fold in proto["outer_folds"]:
            entries = dict(settings["per_fold"][str(fold)])
            for label, item in entries.items():
                resolved = item.get("resolved_config", label)
                config_arg = resolved
                args = common(
                    "final", "official_final", config_arg, item["recipe"], fold, 0, 0,
                    item["selected_epoch"], [],
                )
                # Replace placeholder seed below, preserving one job per requested seed.
                for seed in seeds:
                    seeded = list(args)
                    seeded[seeded.index("--seed") + 1] = str(seed)
                    seeded += [
                        "--selected-epoch", item["selected_epoch"],
                        "--result-config", label,
                    ]
                    if item.get("match_to"):
                        seeded += ["--match-to", item["match_to"]]
                    if item.get("source_config"):
                        seeded += ["--source-config", item["source_config"]]
                    add(
                        f"final_{label}_{item['recipe']}_e{item['selected_epoch']}_f{fold}_s{seed}",
                        seeded,
                    )

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"kind": a.kind, "job_count": len(jobs), "jobs": jobs}, indent=2), encoding="utf-8")
    print("WROTE", out, "jobs", len(jobs))


if __name__ == "__main__":
    main()
