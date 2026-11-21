"""Materon ElemNet V14: standalone Baseline-anchored EIGate.

Target runtime: Python 3.9, TensorFlow 2.9, kgcnn 2.1.
"""

import os

# Fast model.fit path with deterministic kernels, but no repeated replay and no
# Python-level train_on_batch loop.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

__version__ = "14.0.0-standalone-baseline-anchored-eigate"
DEFAULT_SEEDS = (42, 123, 2024)
CACHE_SCHEMA_VERSION = "materon_elemnet_standalone_v14_cache_v1"
