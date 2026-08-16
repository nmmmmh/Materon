#!/usr/bin/env python3
"""Command-line workflow for inspect kgcnn contract in the TF2.9 multi-backbone engine."""

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))

import json
from materon_tf29_engine.integration.official_builder_patch import inspect_builder_contract
from kgcnn.literature.Megnet import make_crystal_model as m
from kgcnn.literature.CGCNN import make_crystal_model as c
print(json.dumps({"megnet":inspect_builder_contract(m),"cgcnn":inspect_builder_contract(c)},indent=2))
