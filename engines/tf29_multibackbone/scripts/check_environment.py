#!/usr/bin/env python3
"""Command-line workflow for check environment in the TF2.9 multi-backbone engine."""

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))

from importlib.metadata import version
import sys
EXPECTED={"numpy":"1.20.1","scikit-learn":"0.24.1","tensorflow":"2.9.0","kgcnn":"2.1.0","pymatgen":"2022.9.8","pyxtal":"0.5.2","pandas":"1.3.5"}
failed=False
print("Python:",sys.version)
for p,e in EXPECTED.items():
    try: x=version(p); ok=x==e
    except Exception as err: x=str(err); ok=False
    print("%-16s installed=%-14s expected=%-12s %s"%(p,x,e,"PASS" if ok else "MISMATCH")); failed|=not ok
# Matbench package 0.6 is accepted if it loads the v0.1 benchmark contract.
try:
    print("matbench package:",version("matbench"),"(recorded, not hard-failed)")
    from matbench.bench import MatbenchBenchmark
    mb=MatbenchBenchmark(subset=["matbench_phonons"],autoload=False); task=next(iter(mb.tasks)); task.load()
    contract=(len(task.folds)==5 and len(task.df)==1265)
    print("matbench_v0.1 contract: folds=%d samples=%d %s"%(len(task.folds),len(task.df),"PASS" if contract else "FAIL")); failed|=not contract
except Exception as e: print("matbench contract FAIL",repr(e)); failed=True
import tensorflow as tf
print("GPUs:",tf.config.list_physical_devices("GPU")); failed|=not bool(tf.config.list_physical_devices("GPU"))
if failed:
    raise SystemExit("ENVIRONMENT CHECK FAILED")
print("ENVIRONMENT CHECK: PASS")
