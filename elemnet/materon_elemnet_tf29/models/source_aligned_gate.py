from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf


class ScalarClip(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, value):
        return tf.clip_by_value(value, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


@dataclass(frozen=True)
class GateConfig:
    reduced_dim: int = 32
    comb_dropout: float = 0.10
    use_shared_expert: bool = True
    low_rank_gate: bool = True
    gate_low_rank_dim: int = 8
    use_dimension_gating: bool = True
    dimension_gating_type: str = "soft"
    dimension_sparsity_weight: float = 0.01
    dimension_parameterization: str = "historical_sigmoid"
    dimension_center_scale: float = 0.50
    dimension_neutral_weight: float = 0.0
    synergy_init: float = 0.10
    synergy_min: float = 0.01
    synergy_max: float = 0.50
    l2_regularization: float = 1.0e-5
    competition_sign: bool = True
    # Optional safety controls. Zero/False preserve the base gate path exactly.
    modulation_trust_ratio: float = 0.0
    use_competition_confidence: bool = False
    competition_confidence_floor: float = 0.50
    competition_confidence_power: float = 1.0


class SourceAlignedBiomimeticEIGate(tf.keras.layers.Layer):
    """Source-aligned EIGate with optional direction-preserving safety.

    The layer retains attention, dimension gating and E/I competition.
    It adds two post-gate controls that introduce no extra trainable parameters:
      1. competition-margin confidence attenuation;
      2. per-sample relative-L2 trust region on the complete gate-induced delta.

    The complete delta includes both historical dimension scaling and the
    promotion/inhibition expert residual. A ratio of zero preserves the exact
    historical path.
    """

    def __init__(self, feature_dim: int, config: Optional[GateConfig] = None,
                 seed: int = 0, name: str = "source_aligned_eigate", **kwargs):
        super(SourceAlignedBiomimeticEIGate, self).__init__(name=name, **kwargs)
        self.feature_dim = int(feature_dim)
        self.config_obj = config if config is not None else GateConfig()
        self.seed = int(seed)
        regularizer = (
            tf.keras.regularizers.l2(float(self.config_obj.l2_regularization))
            if self.config_obj.l2_regularization > 0 else None
        )

        if self.config_obj.low_rank_gate:
            self.gate_down = tf.keras.layers.Dense(
                int(self.config_obj.gate_low_rank_dim), activation=tf.nn.swish,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 11),
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="gate_lr_mlp_0"
            )
            self.gate_out = tf.keras.layers.Dense(
                2, activation=None,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 12),
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="gate_lr_mlp_1"
            )
        else:
            self.gate_down = None
            self.gate_out = tf.keras.layers.Dense(
                2, activation=None,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 13),
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="gate_dense"
            )

        if self.config_obj.use_shared_expert:
            self.shared_down = tf.keras.layers.Dense(
                int(self.config_obj.reduced_dim), activation=tf.nn.swish,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 21),
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="shared_expert_0"
            )
            self.shared_out = tf.keras.layers.Dense(
                self.feature_dim * 2, activation=None,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 22),
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="shared_expert_1"
            )
            self.promotion_down = None
            self.promotion_out = None
            self.inhibition_down = None
            self.inhibition_out = None
        else:
            self.shared_down = None
            self.shared_out = None
            self.promotion_down = tf.keras.layers.Dense(
                int(self.config_obj.reduced_dim), activation=tf.nn.swish,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 23),
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="promotion_mlp_0"
            )
            self.promotion_out = tf.keras.layers.Dense(
                self.feature_dim, activation=None,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 24),
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="promotion_mlp_1"
            )
            self.inhibition_down = tf.keras.layers.Dense(
                int(self.config_obj.reduced_dim), activation=tf.nn.swish,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 25),
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="inhibition_mlp_0"
            )
            self.inhibition_out = tf.keras.layers.Dense(
                self.feature_dim, activation=None,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 26),
                bias_initializer="zeros", kernel_regularizer=regularizer,
                name="inhibition_mlp_1"
            )

        self.comb_dropout = tf.keras.layers.Dropout(
            float(self.config_obj.comb_dropout), seed=self.seed + 31,
            name="comb_dropout"
        )

    def build(self, input_shape):
        self.attn_weights = self.add_weight(
            name="attn_weights", shape=(self.feature_dim, 1),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 1),
            trainable=True, dtype=tf.float32
        )
        if self.config_obj.use_dimension_gating:
            gate_type = str(self.config_obj.dimension_gating_type)
            parameterization = str(self.config_obj.dimension_parameterization)
            if gate_type == "soft" and parameterization == "identity_centered":
                initializer = "zeros"
                regularizer = None
            elif gate_type == "soft":
                initializer = "ones"
                regularizer = None
            elif gate_type == "sparse":
                initializer = tf.keras.initializers.Constant(0.5)
                regularizer = tf.keras.regularizers.l1(
                    float(self.config_obj.dimension_sparsity_weight)
                )
            else:
                initializer = tf.keras.initializers.Constant(0.5)
                regularizer = None
            self.dim_gate_weights = self.add_weight(
                name="dim_gate_weights", shape=(self.feature_dim,),
                initializer=initializer, regularizer=regularizer,
                trainable=True, dtype=tf.float32
            )
        else:
            self.dim_gate_weights = None
        self.synergy_scale = self.add_weight(
            name="synergy_scale", shape=(),
            initializer=tf.keras.initializers.Constant(float(self.config_obj.synergy_init)),
            trainable=True,
            constraint=ScalarClip(
                self.config_obj.synergy_min, self.config_obj.synergy_max
            )
        )
        super(SourceAlignedBiomimeticEIGate, self).build(input_shape)

    @property
    def synergy(self):
        return self.synergy_scale

    def _dimension_values(self, training):
        if not self.config_obj.use_dimension_gating:
            return tf.ones((self.feature_dim,), dtype=tf.float32)
        gate_type = str(self.config_obj.dimension_gating_type)
        parameterization = str(self.config_obj.dimension_parameterization)
        if gate_type == "soft" and parameterization == "identity_centered":
            scale = float(self.config_obj.dimension_center_scale)
            return 1.0 + scale * tf.math.tanh(self.dim_gate_weights)
        if gate_type in ("soft", "sparse"):
            return tf.math.sigmoid(self.dim_gate_weights)
        if gate_type == "hard":
            logits = tf.stack([self.dim_gate_weights, -self.dim_gate_weights], axis=-1)
            if training:
                uniform = tf.random.uniform(
                    tf.shape(logits), minval=1.0e-6, maxval=1.0 - 1.0e-6,
                    seed=self.seed + 41
                )
                gumbel = -tf.math.log(-tf.math.log(uniform))
                return tf.nn.softmax(logits + gumbel, axis=-1)[:, 0]
            return tf.one_hot(tf.argmax(logits, axis=-1), depth=2)[:, 0]
        raise ValueError("Unknown dimension_gating_type=%r" % gate_type)

    def _dimension_gate(self, x, training):
        values = tf.cast(self._dimension_values(training), x.dtype)
        neutral_weight = float(self.config_obj.dimension_neutral_weight)
        if self.config_obj.use_dimension_gating and neutral_weight > 0:
            self.add_loss(neutral_weight * tf.reduce_mean(tf.square(values - 1.0)))
        return x * values, values

    def _confidence(self, gate_weights):
        if not self.config_obj.use_competition_confidence:
            return tf.ones_like(gate_weights[:, 0:1])
        margin = tf.abs(gate_weights[:, 0:1] - gate_weights[:, 1:2])
        power = max(float(self.config_obj.competition_confidence_power), 1.0e-6)
        floor = min(max(float(self.config_obj.competition_confidence_floor), 0.0), 1.0)
        return floor + (1.0 - floor) * tf.pow(margin, power)

    def _trust_region(self, base, modulation):
        ratio = float(self.config_obj.modulation_trust_ratio)
        if ratio <= 0:
            return modulation, tf.ones_like(modulation[:, 0:1])
        eps = tf.cast(1.0e-7, modulation.dtype)
        base_norm = tf.norm(base, ord="euclidean", axis=-1, keepdims=True)
        modulation_norm = tf.norm(modulation, ord="euclidean", axis=-1, keepdims=True)
        cap = tf.cast(ratio, modulation.dtype) * tf.maximum(base_norm, eps)
        scale = tf.minimum(tf.ones_like(cap), cap / tf.maximum(modulation_norm, eps))
        return modulation * scale, scale

    def call(self, inputs, training=None, return_analysis=False):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        base, dimension_gates = self._dimension_gate(x, bool(training))
        attention_scores = tf.nn.sigmoid(tf.matmul(base, self.attn_weights))
        attended = base * attention_scores

        if self.config_obj.low_rank_gate:
            gate_logits = self.gate_out(self.gate_down(attended))
        else:
            gate_logits = self.gate_out(attended)
        gate_weights = tf.nn.softmax(gate_logits, axis=-1)

        if self.config_obj.use_shared_expert:
            shared = self.shared_out(self.shared_down(attended))
            promotion, inhibition = tf.split(shared, 2, axis=-1)
        else:
            promotion = self.promotion_out(self.promotion_down(attended))
            inhibition = self.inhibition_out(self.inhibition_down(attended))

        if self.config_obj.competition_sign:
            combined = (
                gate_weights[:, 0:1] * promotion -
                gate_weights[:, 1:2] * inhibition
            )
        else:
            combined = (
                gate_weights[:, 0:1] * promotion +
                gate_weights[:, 1:2] * inhibition
            )
        confidence = self._confidence(gate_weights)
        raw_modulation = self.synergy * combined
        dropped_modulation = self.comb_dropout(raw_modulation, training=training)

        # Safety is applied to the complete gate-induced change relative to the
        # untouched input, not only to the expert residual. This is essential
        # because historical dimension gating also changes the main path.
        historical_output = base + dropped_modulation
        raw_total_delta = historical_output - x
        confidence_delta = raw_total_delta * confidence
        safe_total_delta, trust_scale = self._trust_region(x, confidence_delta)
        output = x + safe_total_delta

        if not return_analysis:
            return output
        impact = promotion - inhibition
        importance = tf.abs(impact)
        importance_sum = tf.reduce_sum(importance, axis=-1, keepdims=True)
        normalized_importance = importance / tf.maximum(importance_sum, 1.0e-8)
        analysis = {
            "attention_scores": attention_scores,
            "dimension_gates": dimension_gates,
            "gate_weights": gate_weights,
            "promotion": promotion,
            "inhibition": inhibition,
            "combined": combined,
            "competition_confidence": confidence,
            "raw_modulation": raw_modulation,
            "historical_output": historical_output,
            "raw_total_delta": raw_total_delta,
            "confidence_delta": confidence_delta,
            "safe_total_delta": safe_total_delta,
            "modulation": safe_total_delta,
            "trust_scale": trust_scale,
            "synergy": self.synergy,
            "normalized_importance": normalized_importance,
        }
        return output, analysis

    def get_config(self):
        base = super(SourceAlignedBiomimeticEIGate, self).get_config()
        base.update({
            "feature_dim": self.feature_dim,
            "config": self.config_obj.__dict__,
            "seed": self.seed,
        })
        return base
