"""Bounded environment gate of the C2DB work-function study, Eq. (1) / Eq. (S2).

    g_ij = 1 + rho * tanh(h_j(x_i, c_i)),   rho = 0.5,   x~_ij = x_ij * g_ij

``h`` is an 8-unit swish layer followed by a zero-initialized 20-unit head, so
an untrained gate is the identity. Every factor lies in [1 - rho, 1 + rho].
A path mask m in [0, 1] scales the correction, x + m * x * (g - 1); m = 0
bypasses the gate exactly and m = 1 is the native model (Eq. S4).
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Materon")
class EnvironmentGate(tf.keras.layers.Layer):
    def __init__(self, feature_dim=20, context_dim=37, hidden_dim=8, rho=0.5, **kwargs):
        super().__init__(**kwargs)
        dims = (feature_dim, context_dim, hidden_dim)
        if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in dims):
            raise ValueError("feature_dim, context_dim, and hidden_dim must be positive integers")
        if not 0.0 < rho < 1.0:
            raise ValueError("rho must be between 0 and 1")
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.rho = float(rho)
        self.hidden = tf.keras.layers.Dense(hidden_dim, activation="swish", name="hidden")
        self.head = tf.keras.layers.Dense(
            feature_dim,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="head",
        )

    def call(self, inputs, return_readouts=False):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 3:
            raise ValueError("inputs must be [x, context, mask]")
        x, context, mask = inputs
        tensors = (x, context, mask)
        names = ("x", "context", "mask")
        widths = (self.feature_dim, self.context_dim, self.feature_dim)
        for value, name, width in zip(tensors, names, widths):
            if not isinstance(value, tf.RaggedTensor) or value.ragged_rank != 1:
                raise ValueError("%s must be a ragged-rank-1 tensor" % name)
            if value.shape.rank != 3:
                raise ValueError("%s must have rank 3" % name)
            if value.shape[-1] is not None and value.shape[-1] != width:
                raise ValueError("%s must have final dimension %d" % (name, width))
            if not value.dtype.is_floating:
                raise TypeError("%s must have a floating-point dtype" % name)

        if x.dtype != context.dtype or x.dtype != mask.dtype:
            raise TypeError("x, context, and mask must have the same dtype")

        checks = [
            tf.debugging.assert_equal(x.row_splits, context.row_splits,
                                      message="x and context row_splits differ"),
            tf.debugging.assert_equal(x.row_splits, mask.row_splits,
                                      message="x and mask row_splits differ"),
            tf.debugging.assert_equal(tf.shape(x.flat_values)[-1], self.feature_dim),
            tf.debugging.assert_equal(tf.shape(context.flat_values)[-1], self.context_dim),
            tf.debugging.assert_equal(tf.shape(mask.flat_values)[-1], self.feature_dim),
            tf.debugging.assert_greater_equal(mask.flat_values, tf.cast(0, mask.dtype),
                                              message="mask values must be in [0, 1]"),
            tf.debugging.assert_less_equal(mask.flat_values, tf.cast(1, mask.dtype),
                                           message="mask values must be in [0, 1]"),
        ]
        with tf.control_dependencies([check for check in checks if check is not None]):
            x_values = tf.debugging.assert_all_finite(x.flat_values, "x must be finite")
            context_values = tf.debugging.assert_all_finite(
                context.flat_values, "context must be finite")
            mask_values = tf.debugging.assert_all_finite(mask.flat_values, "mask must be finite")

        hidden = self.hidden(tf.concat([x_values, context_values], axis=-1))
        u = self.head(hidden)
        gate_values = 1.0 + tf.cast(self.rho, x_values.dtype) * tf.math.tanh(u)
        correction_values = mask_values * (x_values * gate_values - x_values)
        z_values = x_values + correction_values

        z = x.with_flat_values(z_values)
        if not return_readouts:
            return z
        return z, {
            "gate": x.with_flat_values(gate_values),
            "correction": x.with_flat_values(correction_values),
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "rho": self.rho,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="Materon")
class GlobalFeatureGate(tf.keras.layers.Layer):
    """Zero-initialized global feature scale with the same bounded gate range."""
    def __init__(self, feature_dim=20, rho=0.5, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if not 0.0 < rho < 1.0:
            raise ValueError("rho must be between 0 and 1")
        self.feature_dim = feature_dim
        self.rho = float(rho)

    def build(self, input_shape):
        self.b = self.add_weight(
            name="b", shape=(self.feature_dim,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("inputs must be [x, mask]")
        x, mask = inputs
        for value, name in ((x, "x"), (mask, "mask")):
            if not isinstance(value, tf.RaggedTensor) or value.ragged_rank != 1:
                raise ValueError("%s must be a ragged-rank-1 tensor" % name)
            if value.shape.rank != 3:
                raise ValueError("%s must have rank 3" % name)
            if value.shape[-1] is not None and value.shape[-1] != self.feature_dim:
                raise ValueError("%s must have final dimension %d" % (name, self.feature_dim))
            if not value.dtype.is_floating:
                raise TypeError("%s must have a floating-point dtype" % name)
        if x.dtype != mask.dtype:
            raise TypeError("x and mask must have the same dtype")

        checks = [
            tf.debugging.assert_equal(x.row_splits, mask.row_splits,
                                      message="x and mask row_splits differ"),
            tf.debugging.assert_greater_equal(mask.flat_values, tf.cast(0, mask.dtype),
                                              message="mask values must be in [0, 1]"),
            tf.debugging.assert_less_equal(mask.flat_values, tf.cast(1, mask.dtype),
                                           message="mask values must be in [0, 1]"),
        ]
        with tf.control_dependencies([check for check in checks if check is not None]):
            x_values = tf.debugging.assert_all_finite(x.flat_values, "x must be finite")
            mask_values = tf.debugging.assert_all_finite(mask.flat_values, "mask must be finite")

        scale = 1.0 + tf.cast(self.rho, x_values.dtype) * tf.math.tanh(self.b)
        result = x_values + mask_values * (x_values * scale - x_values)
        return x.with_flat_values(result)

    def get_config(self):
        config = super().get_config()
        config.update({"feature_dim": self.feature_dim, "rho": self.rho})
        return config


@tf.keras.utils.register_keras_serializable(package="Materon")
class ZeroEnvironment(tf.keras.layers.Layer):
    """Create a zero-valued ragged environment aligned to a physical node tensor."""
    def __init__(self, context_dim=37, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(context_dim, int) or isinstance(context_dim, bool) or context_dim <= 0:
            raise ValueError("context_dim must be a positive integer")
        self.context_dim = context_dim

    def call(self, inputs):
        if not isinstance(inputs, tf.RaggedTensor) or inputs.ragged_rank != 1:
            raise ValueError("inputs must be a ragged-rank-1 node tensor")
        flat_shape = tf.stack([tf.shape(inputs.flat_values)[0], self.context_dim])
        return inputs.with_flat_values(tf.zeros(flat_shape, dtype=inputs.dtype))

    def get_config(self):
        config = super().get_config()
        config.update({"context_dim": self.context_dim})
        return config
