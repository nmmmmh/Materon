"""TensorFlow runtime and seeding helpers shared by all training scripts."""
from __future__ import annotations

import os
import random


def configure_tensorflow(tf, intra: int = 2, inter: int = 2, tf32: bool = True) -> dict:
    """Bounded CPU thread pools and GPU memory growth; call before building models.

    ``tf32=False`` disables TensorFloat-32, as in the C2DB study.
    """
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra))
        tf.config.threading.set_inter_op_parallelism_threads(int(inter))
    except RuntimeError:
        pass
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if not tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    return {"visible_gpus": [str(x) for x in gpus], "intra": int(intra), "inter": int(inter)}


def set_seed(seed: int) -> None:
    """Python, NumPy and TensorFlow seeds (MEGNet and fixed-split runs)."""
    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
