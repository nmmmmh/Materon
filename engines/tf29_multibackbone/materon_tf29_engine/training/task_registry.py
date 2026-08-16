# -*- coding: utf-8 -*-
"""Frozen Matbench v0.1 task registry used by the small-data protocol."""
from __future__ import annotations

from collections import OrderedDict

SMALL_MATBENCH_TASKS = OrderedDict([
    ("matbench_steels", {
        "n_samples": 312,
        "input_type": "composition",
        "task_type": "regression",
        "unit": "MPa",
        "target": "yield strength",
    }),
    ("matbench_jdft2d", {
        "n_samples": 636,
        "input_type": "structure",
        "task_type": "regression",
        "unit": "meV/atom",
        "target": "exfoliation energy",
    }),
    ("matbench_phonons", {
        "n_samples": 1265,
        "input_type": "structure",
        "task_type": "regression",
        "unit": "cm^-1",
        "target": "last phonon DOS peak",
    }),
    ("matbench_expt_gap", {
        "n_samples": 4604,
        "input_type": "composition",
        "task_type": "regression",
        "unit": "eV",
        "target": "experimental band gap",
    }),
    ("matbench_dielectric", {
        "n_samples": 4764,
        "input_type": "structure",
        "task_type": "regression",
        "unit": "unitless",
        "target": "refractive index",
    }),
    ("matbench_expt_is_metal", {
        "n_samples": 4921,
        "input_type": "composition",
        "task_type": "classification",
        "unit": "probability",
        "target": "experimental metal label",
    }),
])

STRUCTURE_SMALL_TASKS = tuple(
    name for name, spec in SMALL_MATBENCH_TASKS.items()
    if spec["input_type"] == "structure"
)
COMPOSITION_SMALL_TASKS = tuple(
    name for name, spec in SMALL_MATBENCH_TASKS.items()
    if spec["input_type"] == "composition"
)
ALL_SMALL_TASKS = tuple(SMALL_MATBENCH_TASKS)


def task_spec(task_name):
    return SMALL_MATBENCH_TASKS.get(task_name)


def validate_task_size_and_type(task_name, task):
    """Validate the loaded official task against the frozen registry."""
    spec = task_spec(task_name)
    if spec is None:
        return None
    actual_type = str(task.metadata["task_type"])
    if actual_type != spec["task_type"]:
        raise RuntimeError(
            "Task type mismatch for %s: registry=%s loaded=%s"
            % (task_name, spec["task_type"], actual_type)
        )
    return spec


def validate_backbone_modality(backbone, task_name):
    spec = task_spec(task_name)
    if spec is None:
        return {
            "known_task": False,
            "input_type": "unknown",
            "message": "Task is outside the frozen <5000 registry.",
        }

    if backbone in {"megnet", "cgcnn"} and spec["input_type"] != "structure":
        raise ValueError(
            "%s is a composition-only Matbench task and cannot be run with "
            "%s, which requires crystal structures."
            % (task_name, backbone)
        )

    if backbone == "elemnet" and spec["input_type"] == "structure":
        return {
            "known_task": True,
            "input_type": "structure",
            "message": (
                "ElemNet is composition-only. The task supplies structures, "
                "but only structure.composition is used; coordinates, lattice "
                "and neighbour topology are discarded."
            ),
        }

    return {
        "known_task": True,
        "input_type": spec["input_type"],
        "message": "Backbone and task input modality are compatible.",
    }
