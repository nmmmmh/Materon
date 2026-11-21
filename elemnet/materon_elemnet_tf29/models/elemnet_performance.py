from __future__ import absolute_import, division, print_function

from dataclasses import dataclass, field
import hashlib
from typing import Optional, Tuple

import tensorflow as tf

from .source_aligned_gate import GateConfig, SourceAlignedBiomimeticEIGate


@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_dim: int
    hidden_units: Tuple[int, ...] = (1024, 512, 256, 128)
    dropout: float = 0.10
    l2_regularization: float = 1.0e-5
    batch_norm: bool = True
    input_gate: Optional[GateConfig] = None
    post_gate: Optional[GateConfig] = None


class ElemNetPerformanceRegressor(tf.keras.Model):
    """Historical 1024-512-256-128 ElemNet with optional full input/post EIGates."""

    def __init__(self, config: ModelConfig, seed: int,
                 name="elemnet_performance_regressor", **kwargs):
        super(ElemNetPerformanceRegressor, self).__init__(name=name, **kwargs)
        self.model_config = config
        self.seed = int(seed)
        regularizer = (
            tf.keras.regularizers.l2(float(config.l2_regularization))
            if config.l2_regularization > 0 else None
        )

        # Preserve historical creation order: backbone layers are defined before gates.
        self.dense_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        for index, units in enumerate(config.hidden_units):
            initializer = (
                tf.keras.initializers.HeUniform(seed=self.seed + index * 101 + 1)
                if index == 0 else
                tf.keras.initializers.GlorotUniform(seed=self.seed + index * 101 + 1)
            )
            self.dense_layers.append(tf.keras.layers.Dense(
                int(units), activation="relu", kernel_initializer=initializer,
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="dense_%d" % index
            ))
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(
                    momentum=0.99, epsilon=1.0e-3, name="bn_%d" % index
                ) if config.batch_norm else None
            )
            self.dropout_layers.append(tf.keras.layers.Dropout(
                float(config.dropout), seed=self.seed + index * 101 + 2,
                name="dropout_%d" % index
            ))

        self.input_gate = (
            SourceAlignedBiomimeticEIGate(
                feature_dim=int(config.input_dim), config=config.input_gate,
                seed=self.seed + 10000, name="input_eigate"
            ) if config.input_gate is not None else None
        )
        self.post_gate = (
            SourceAlignedBiomimeticEIGate(
                feature_dim=int(config.hidden_units[0]), config=config.post_gate,
                seed=self.seed + 20000, name="post_eigate"
            ) if config.post_gate is not None else None
        )
        self.output_layer = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 99991),
            bias_initializer="zeros", kernel_regularizer=regularizer,
            name="output"
        )

    def call(self, inputs, training=None, return_analysis=False):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        analysis = {}
        if self.input_gate is not None:
            if return_analysis:
                x, values = self.input_gate(x, training=training, return_analysis=True)
                analysis["input_gate"] = values
            else:
                x = self.input_gate(x, training=training, return_analysis=False)

        for index, dense in enumerate(self.dense_layers):
            x = dense(x)
            if self.bn_layers[index] is not None:
                x = self.bn_layers[index](x, training=training)
            if index == 0 and self.post_gate is not None:
                if return_analysis:
                    x, values = self.post_gate(x, training=training, return_analysis=True)
                    analysis["post_gate"] = values
                else:
                    x = self.post_gate(x, training=training, return_analysis=False)
            # Historical model applies dropout only from the second hidden layer onward.
            if index > 0:
                x = self.dropout_layers[index](x, training=training)

        output = tf.squeeze(self.output_layer(x), axis=-1)
        if return_analysis:
            return output, analysis
        return output

    @staticmethod
    def _sha256(weights):
        digest = hashlib.sha256()
        for weight in weights:
            value = weight.numpy()
            digest.update(str(tuple(int(v) for v in value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def backbone_weights(self):
        output = []
        for dense, bn in zip(self.dense_layers, self.bn_layers):
            output.extend(dense.weights)
            if bn is not None:
                output.extend(bn.weights)
        output.extend(self.output_layer.weights)
        return output

    def gate_weights(self):
        output = []
        if self.input_gate is not None:
            output.extend(self.input_gate.weights)
        if self.post_gate is not None:
            output.extend(self.post_gate.weights)
        return output

    def backbone_sha256(self):
        return self._sha256(self.backbone_weights())

    def gate_sha256(self):
        return self._sha256(self.gate_weights())

    def gate_parameter_count(self):
        return int(sum(tf.keras.backend.count_params(v) for v in self.gate_weights()))


def build_model(config: ModelConfig, seed: int):
    tf.keras.backend.clear_session()
    try:
        tf.keras.utils.set_random_seed(int(seed))
    except AttributeError:
        tf.random.set_seed(int(seed))
    model = ElemNetPerformanceRegressor(config=config, seed=int(seed))
    model(tf.zeros((2, int(config.input_dim)), dtype=tf.float32), training=False)
    return model
