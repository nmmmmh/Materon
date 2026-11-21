from __future__ import absolute_import, division, print_function

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, tf_module: Any = None) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    if tf_module is not None:
        tf_module.random.set_seed(int(seed))
