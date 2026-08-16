#!/usr/bin/env python3
"""Command-line workflow for list small matbench tasks in the TF2.9 multi-backbone engine."""

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))

from materon_tf29_engine.training.task_registry import SMALL_MATBENCH_TASKS

print("%-28s %8s %-12s %-14s %s" % (
    "task", "samples", "input", "type", "unit"
))
print("-" * 82)
for name, spec in SMALL_MATBENCH_TASKS.items():
    print("%-28s %8d %-12s %-14s %s" % (
        name,
        spec["n_samples"],
        spec["input_type"],
        spec["task_type"],
        spec["unit"],
    ))
