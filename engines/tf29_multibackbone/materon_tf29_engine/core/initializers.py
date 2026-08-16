# -*- coding: utf-8 -*-
"""Deterministic initializers that do not consume TensorFlow RNG state."""
from __future__ import annotations
from typing import Any
import numpy as np
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="MateronTF29")
class HarmonicInitializer(tf.keras.initializers.Initializer):
    def __init__(self, scale: float = 0.05, phase: float = 0.0):
        self.scale = float(scale); self.phase = float(phase)
    def __call__(self, shape, dtype=None, **kwargs):
        del kwargs
        size = int(np.prod(shape)); idx = np.arange(1, size+1, dtype=np.float32)
        arr = np.sin(idx * 0.731 + self.phase) + 0.5*np.cos(idx * 0.313 + 1.7*self.phase)
        arr = arr / max(float(np.sqrt(np.mean(arr*arr))), 1e-8) * self.scale
        return tf.convert_to_tensor(arr.reshape(tuple(shape)), dtype=dtype or tf.float32)
    def get_config(self): return {"scale": self.scale, "phase": self.phase}
