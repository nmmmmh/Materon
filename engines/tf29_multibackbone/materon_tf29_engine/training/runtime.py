"""Runtime utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
import os,random
import numpy as np
import tensorflow as tf

def set_random_streams(seed): random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
def derived_seed(base,fold): return int(base)*1000+int(fold)
def configure_tensorflow_from_environment(require_one_gpu=True):
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("TF_NUM_INTRAOP_THREADS","2")))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get("TF_NUM_INTEROP_THREADS","1")))
    g=tf.config.list_physical_devices("GPU")
    if require_one_gpu and len(g)!=1: raise RuntimeError("Expected exactly one visible GPU, got %s"%g)
    for x in g:
        try: tf.config.experimental.set_memory_growth(x,True)
        except RuntimeError: pass
    return {"tensorflow":tf.__version__,"visible_gpus":[x.name for x in g],"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"intra_threads":os.environ.get("TF_NUM_INTRAOP_THREADS"),"inter_threads":os.environ.get("TF_NUM_INTEROP_THREADS")}
