"""Constraints utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="MateronTF29")
class NonNegative(tf.keras.constraints.Constraint):
    def __call__(self, w): return tf.maximum(w, tf.cast(0.0, w.dtype))
    def get_config(self): return {}
