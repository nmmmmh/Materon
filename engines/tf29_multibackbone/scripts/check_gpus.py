#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify each physical GPU in a fresh TensorFlow process."""

from __future__ import annotations

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    code = (
        "import tensorflow as tf; "
        "g=tf.config.list_physical_devices('GPU'); "
        "print(g); assert len(g)==1, g"
    )
    failed = False
    for gpu in gpus:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            text=True,
            capture_output=True,
        )
        print("GPU %d rc=%d" % (gpu, result.returncode))
        print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.returncode != 0:
            failed = True
    if failed:
        raise SystemExit("GPU PREFLIGHT FAILED")
    print("GPU PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
