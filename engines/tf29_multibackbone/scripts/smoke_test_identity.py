#!/usr/bin/env python3
"""Command-line workflow for smoke test identity in the TF2.9 multi-backbone engine."""

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))

import argparse
import gc
import json
import numpy as np
import tensorflow as tf
from matbench.bench import MatbenchBenchmark
from materon_tf29_engine.training.cache import GraphTaskCache, CompositionTaskCache
from materon_tf29_engine.training.model_factory import build_model
from materon_tf29_engine.training.identity import assert_identity

p = argparse.ArgumentParser()
p.add_argument("--backbone", choices=["megnet", "cgcnn", "elemnet"], required=True)
p.add_argument("--task", required=True)
p.add_argument("--atol", type=float, default=1e-6)
p.add_argument("--rtol", type=float, default=5e-7)
p.add_argument("--max-ulp", type=int, default=4)
# Backward-compatible alias for old commands.
p.add_argument("--tolerance", type=float, default=None, help=argparse.SUPPRESS)
a = p.parse_args()

mb = MatbenchBenchmark(subset=[a.task], autoload=False)
task = next(iter(mb.tasks))
task.load()
fold = list(task.folds)[0]
cache = (
    CompositionTaskCache(task)
    if a.backbone == "elemnet"
    else GraphTaskCache(task, a.backbone)
)
x, *_ = cache.fold(fold)
xs = [v[:4] for v in x] if isinstance(x, (list, tuple)) else x[:4]

tf.keras.backend.clear_session()
gc.collect()
tf.random.set_seed(0)
np.random.seed(0)
model, base, layer = build_model(
    a.backbone, "materon", str(task.metadata["task_type"])
)
atol = float(a.tolerance) if a.tolerance is not None else float(a.atol)
diff, parity = assert_identity(
    base,
    model,
    xs,
    atol=atol,
    rtol=float(a.rtol),
    max_ulp=int(a.max_ulp),
)
print(
    json.dumps(
        {
            "backbone": a.backbone,
            "baseline_params": base.count_params(),
            "materon_params": model.count_params(),
            "added": model.count_params() - base.count_params(),
            "inserted_layer": layer,
            "identity_max_abs": diff,
            "parity": parity,
        },
        indent=2,
    )
)
print("IDENTITY SMOKE TEST: PASS")
