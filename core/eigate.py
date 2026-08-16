"""Canonical excitation--inhibition gate used by Materon graph backbones.

This module is the single source of truth for the graph E/I computation.
Historical CGCNN and MEGNet class names are implemented as thin subclasses in
``materon.compat``; workflow-local ``gates.py`` files only re-export those
classes and contain no gate mathematics.

The validated forward sequence is:

    physical features -> dimension gate -> node confidence
    -> promotion/inhibition experts -> E/I competition
    -> bounded synergy -> residual physical modulation.

Two compatibility profiles preserve old checkpoint layouts:
``cgcnn_legacy`` keeps the historical modulation-mask/context layout, while
``megnet_interpretable`` keeps feature/pathway/node-mask intervention state.
With all intervention masks equal to one, both profiles execute the same core
E/I equations for the corresponding frozen configuration.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="MateronCore")
class ScalarClip(tf.keras.constraints.Constraint):
    """Clip a scalar trainable variable to a closed interval."""

    def __init__(self, minimum: float, maximum: float):
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def __call__(self, value):
        return tf.clip_by_value(value, self.minimum, self.maximum)

    def get_config(self):
        return {"minimum": self.minimum, "maximum": self.maximum}


# Register the canonical constraint under historical serialized names as well.
tf.keras.utils.get_custom_objects()["MateronPhonons>ScalarClip"] = ScalarClip
tf.keras.utils.get_custom_objects()["MateronMEGNet>ScalarClip"] = ScalarClip


@tf.keras.utils.register_keras_serializable(package="MateronCore")
class EIGate(tf.keras.layers.Layer):
    """Dimension-resolved biomimetic excitation/inhibition gate.

    Parameters are intentionally compatible with the established graph-gate
    configurations. ``compatibility_profile`` is an implementation detail used
    by historical wrappers and should normally be left as ``"canonical"``.
    """

    _VALID_PROFILES = {"canonical", "cgcnn_legacy", "megnet_interpretable"}

    def __init__(
        self,
        feature_dim: int = 20,
        reduced_dim: int = 8,
        comb_dropout: float = 0.01,
        use_shared_expert: bool = True,
        low_rank_gate: bool = False,
        gate_low_rank_dim: int = 16,
        use_dimension_gating: bool = True,
        dimension_gating_type: str = "soft",
        dimension_sparsity_weight: float = 0.0,
        use_gate_context_proj: bool = True,
        gate_ctx_proj_units: int = 8,
        use_coord_proj: bool = False,
        use_qeff_in_gate: bool = False,
        use_emag_in_gate: bool = False,
        synergy_init: float = 0.1,
        synergy_min: float = 0.005,
        synergy_max: float = 0.5,
        l2_reg: float = 0.0,
        gate_seed: Optional[int] = 42,
        compatibility_profile: str = "canonical",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        profile = str(compatibility_profile)
        if profile not in self._VALID_PROFILES:
            raise ValueError(f"Unknown compatibility_profile={profile!r}")

        self.feature_dim = int(feature_dim)
        self.reduced_dim = int(reduced_dim)
        self.comb_dropout_rate = float(comb_dropout)
        self.use_shared_expert = bool(use_shared_expert)
        self.low_rank_gate = bool(low_rank_gate)
        self.gate_low_rank_dim = int(gate_low_rank_dim)
        self.use_dimension_gating = bool(use_dimension_gating)
        self.dimension_gating_type = str(dimension_gating_type)
        self.dimension_sparsity_weight = float(dimension_sparsity_weight)
        self.use_gate_context_proj = bool(use_gate_context_proj)
        self.gate_ctx_proj_units = int(gate_ctx_proj_units)
        self.use_coord_proj = bool(use_coord_proj)
        self.use_qeff_in_gate = bool(use_qeff_in_gate)
        self.use_emag_in_gate = bool(use_emag_in_gate)
        self.synergy_init = float(synergy_init)
        self.synergy_min = float(synergy_min)
        self.synergy_max = float(synergy_max)
        self.l2_reg = float(l2_reg)
        self.gate_seed = None if gate_seed is None else int(gate_seed)
        self.compatibility_profile = profile

        self._has_pathway_controls = profile in {"canonical", "megnet_interpretable"}
        self._has_context_layers = profile in {"canonical", "cgcnn_legacy"}

        regularizer = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None

        self.attn = tf.keras.layers.Dense(
            1, activation="sigmoid", use_bias=False, name="node_confidence"
        )
        if self.low_rank_gate:
            self.gate_down = tf.keras.layers.Dense(
                self.gate_low_rank_dim,
                activation="swish",
                kernel_regularizer=regularizer,
                name="gate_down",
            )
        self.gate_out = tf.keras.layers.Dense(
            2, activation=None, kernel_regularizer=regularizer, name="gate_out"
        )

        if self.use_shared_expert:
            self.shared_hidden = tf.keras.layers.Dense(
                self.reduced_dim,
                activation="swish",
                kernel_regularizer=regularizer,
                name="shared_hidden",
            )
            self.shared_out = tf.keras.layers.Dense(
                self.feature_dim * 2,
                activation=None,
                kernel_regularizer=regularizer,
                name="shared_out",
            )
        else:
            self.promo_hidden = tf.keras.layers.Dense(
                self.reduced_dim,
                activation="swish",
                kernel_regularizer=regularizer,
                name="promo_hidden",
            )
            self.promo_out = tf.keras.layers.Dense(
                self.feature_dim,
                activation=None,
                kernel_regularizer=regularizer,
                name="promo_out",
            )
            self.inhib_hidden = tf.keras.layers.Dense(
                self.reduced_dim,
                activation="swish",
                kernel_regularizer=regularizer,
                name="inhib_hidden",
            )
            self.inhib_out = tf.keras.layers.Dense(
                self.feature_dim,
                activation=None,
                kernel_regularizer=regularizer,
                name="inhib_out",
            )

        # Context projection belongs to the historical CGCNN gate layout. The
        # MEGNet compatibility profile deliberately does not create these
        # layers, preserving its established weight array count and ordering.
        if self._has_context_layers:
            if self.use_gate_context_proj:
                self.context_hidden = tf.keras.layers.Dense(
                    self.gate_ctx_proj_units,
                    activation="swish",
                    kernel_regularizer=regularizer,
                    name="context_hidden",
                )
                self.context_out = tf.keras.layers.Dense(
                    2, activation=None, kernel_regularizer=regularizer,
                    name="context_out",
                )
                self.context_direct = None
            else:
                self.context_hidden = None
                self.context_out = None
                self.context_direct = tf.keras.layers.Dense(
                    2, activation=None, kernel_regularizer=regularizer,
                    name="context_direct",
                )
            # The original CGCNN layout creates coord_proj even when its flag is
            # false. Keeping that detail preserves checkpoint weight ordering.
            self.coord_proj = tf.keras.layers.Dense(
                1, activation="relu", kernel_regularizer=regularizer,
                name="coord_proj",
            )
        else:
            self.context_hidden = None
            self.context_out = None
            self.context_direct = None
            self.coord_proj = None

        self.dropout = tf.keras.layers.Dropout(
            self.comb_dropout_rate, seed=self.gate_seed
        )

    @staticmethod
    def _flat(value):
        return value.flat_values if isinstance(value, tf.RaggedTensor) else value

    @staticmethod
    def _restore(reference, flat):
        return reference.with_flat_values(flat) if isinstance(reference, tf.RaggedTensor) else flat

    def build(self, input_shape):
        if self.use_dimension_gating:
            init = 1.0 if self.dimension_gating_type == "soft" else 0.5
            regularizer = (
                tf.keras.regularizers.l1(self.dimension_sparsity_weight)
                if self.dimension_gating_type == "sparse"
                else None
            )
            self.dimension_logits = self.add_weight(
                name="dimension_logits",
                shape=(self.feature_dim,),
                initializer=tf.keras.initializers.Constant(init),
                regularizer=regularizer,
                trainable=True,
            )

        self.synergy = self.add_weight(
            name="synergy",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.synergy_init),
            constraint=ScalarClip(self.synergy_min, self.synergy_max),
            trainable=True,
        )

        if self.compatibility_profile == "cgcnn_legacy":
            self.modulation_mask = self.add_weight(
                name="modulation_mask",
                shape=(self.feature_dim,),
                initializer="ones",
                trainable=False,
            )
        else:
            self.feature_intervention_mask = self.add_weight(
                name="feature_intervention_mask",
                shape=(self.feature_dim,),
                initializer="ones",
                trainable=False,
            )

        if self._has_pathway_controls:
            self.promotion_path_mask = self.add_weight(
                name="promotion_path_mask", shape=(), initializer="ones", trainable=False
            )
            self.inhibition_path_mask = self.add_weight(
                name="inhibition_path_mask", shape=(), initializer="ones", trainable=False
            )
            self.dimension_path_mask = self.add_weight(
                name="dimension_path_mask", shape=(), initializer="ones", trainable=False
            )
            self.synergy_path_mask = self.add_weight(
                name="synergy_path_mask", shape=(), initializer="ones", trainable=False
            )
        super().build(input_shape)

    def _feature_mask(self):
        if self.compatibility_profile == "cgcnn_legacy":
            return self.modulation_mask
        return self.feature_intervention_mask

    def set_feature_mask(self, mask):
        """Set an inference intervention mask over physical dimensions."""
        array = np.asarray(mask, dtype=np.float32).reshape(-1)
        if array.size != self.feature_dim:
            raise ValueError("feature mask length mismatch")
        self._feature_mask().assign(array)

    def reset_feature_mask(self):
        self._feature_mask().assign(tf.ones_like(self._feature_mask()))

    # Historical CGCNN API aliases.
    def set_modulation_mask(self, values):
        self.set_feature_mask(values)

    def reset_modulation_mask(self):
        self.reset_feature_mask()

    def set_pathway_state(
        self, promotion=1.0, inhibition=1.0, dimension=1.0, synergy=1.0
    ):
        if not self._has_pathway_controls:
            requested = (float(promotion), float(inhibition), float(dimension), float(synergy))
            if requested != (1.0, 1.0, 1.0, 1.0):
                raise RuntimeError(
                    "Pathway interventions are not part of the cgcnn_legacy checkpoint layout"
                )
            return
        self.promotion_path_mask.assign(float(promotion))
        self.inhibition_path_mask.assign(float(inhibition))
        self.dimension_path_mask.assign(float(dimension))
        self.synergy_path_mask.assign(float(synergy))

    def reset_pathway_state(self):
        self.set_pathway_state()

    def _context_logits(
        self,
        gate_context,
        coordination,
        q_eff,
        electrostatic_magnitude,
        training,
    ):
        if not self._has_context_layers:
            return None
        parts = []
        if gate_context is not None:
            parts.append(tf.cast(self._flat(gate_context), tf.float32))
        if self.use_coord_proj:
            if coordination is None:
                raise RuntimeError("use_coord_proj=True but coordination context was not supplied")
            parts.append(
                self.coord_proj(
                    tf.cast(self._flat(coordination), tf.float32), training=training
                )
            )
        if self.use_qeff_in_gate:
            if q_eff is None:
                raise RuntimeError("use_qeff_in_gate=True but q_eff context was not supplied")
            parts.append(tf.cast(self._flat(q_eff), tf.float32))
        if self.use_emag_in_gate:
            if electrostatic_magnitude is None:
                raise RuntimeError(
                    "use_emag_in_gate=True but electrostatic context was not supplied"
                )
            parts.append(tf.cast(self._flat(electrostatic_magnitude), tf.float32))
        if not parts:
            return None
        context = tf.concat(parts, axis=-1) if len(parts) > 1 else parts[0]
        if self.use_gate_context_proj:
            return self.context_out(
                self.context_hidden(context, training=training), training=training
            )
        return self.context_direct(context, training=training)

    def _dimension_gate(self, x, training):
        if not self.use_dimension_gating:
            dim = tf.ones((self.feature_dim,), tf.float32)
        elif self.dimension_gating_type in {"soft", "sparse"}:
            dim = tf.nn.sigmoid(self.dimension_logits)
        elif self.dimension_gating_type == "hard":
            logits = tf.stack([self.dimension_logits, -self.dimension_logits], axis=-1)
            if training:
                uniform = tf.random.uniform(
                    tf.shape(logits), minval=1.0e-6, maxval=1.0 - 1.0e-6
                )
                gumbel = -tf.math.log(-tf.math.log(uniform))
                dim = tf.nn.softmax(logits + gumbel, axis=-1)[:, 0]
            else:
                dim = tf.one_hot(
                    tf.argmax(logits, axis=-1), depth=2, dtype=tf.float32
                )[:, 0]
        else:
            raise ValueError(
                f"Unknown dimension_gating_type={self.dimension_gating_type!r}"
            )

        if self._has_pathway_controls:
            dim_eff = 1.0 + self.dimension_path_mask * (dim - 1.0)
        else:
            dim_eff = dim
        return x * dim_eff[None, :], dim, dim_eff

    def _split_inputs(self, inputs):
        if (
            self.compatibility_profile in {"canonical", "megnet_interpretable"}
            and isinstance(inputs, (list, tuple))
            and len(inputs) == 2
        ):
            return inputs[0], inputs[1]
        return inputs, None

    def call(
        self,
        inputs,
        training=False,
        return_analysis=False,
        gate_context=None,
        coordination=None,
        q_eff=None,
        electrostatic_magnitude=None,
    ):
        raw, node_mask = self._split_inputs(inputs)
        x = tf.cast(self._flat(raw), tf.float32)
        if node_mask is None:
            nm = tf.ones((tf.shape(x)[0], 1), tf.float32)
        else:
            nm = tf.cast(self._flat(node_mask), tf.float32)
            if nm.shape.rank == 1:
                nm = nm[:, None]

        x_dim, dim, dim_eff = self._dimension_gate(x, training)
        attention = self.attn(x_dim, training=training)
        x_att = x_dim * attention

        if self.low_rank_gate:
            gate_logits = self.gate_out(
                self.gate_down(x_att, training=training), training=training
            )
        else:
            gate_logits = self.gate_out(x_att, training=training)

        if self.use_shared_expert:
            both = self.shared_out(
                self.shared_hidden(x_att, training=training), training=training
            )
            promotion, inhibition = tf.split(both, 2, axis=-1)
        else:
            promotion = self.promo_out(
                self.promo_hidden(x_att, training=training), training=training
            )
            inhibition = self.inhib_out(
                self.inhib_hidden(x_att, training=training), training=training
            )

        context_logits = self._context_logits(
            gate_context, coordination, q_eff, electrostatic_magnitude, training
        )
        if context_logits is not None:
            gate_logits = gate_logits + context_logits

        weights = tf.nn.softmax(gate_logits, axis=-1)
        if self._has_pathway_controls:
            raw_delta = (
                self.promotion_path_mask * weights[:, 0:1] * promotion
                - self.inhibition_path_mask * weights[:, 1:2] * inhibition
            )
            applied = self.synergy_path_mask * self.synergy * raw_delta
        else:
            raw_delta = weights[:, 0:1] * promotion - weights[:, 1:2] * inhibition
            applied = self.synergy * raw_delta

        applied = self.dropout(applied, training=training)
        unmasked = x_dim + applied
        total_delta = (unmasked - x) * self._feature_mask()[None, :] * nm
        out = x + total_delta
        output = self._restore(raw, out)

        if not return_analysis:
            return output

        abs_delta = tf.abs(total_delta)
        relative = abs_delta / (
            tf.reduce_sum(abs_delta, axis=-1, keepdims=True) + 1.0e-8
        )
        pathway_state = (
            tf.stack(
                [
                    self.promotion_path_mask,
                    self.inhibition_path_mask,
                    self.dimension_path_mask,
                    self.synergy_path_mask,
                ]
            )
            if self._has_pathway_controls
            else tf.ones((4,), tf.float32)
        )
        analysis = {
            "input_features": self._restore(raw, x),
            "dimension_gates": dim,
            "effective_dimension_gates": dim_eff,
            "dimension_delta": self._restore(raw, x_dim - x),
            "attention_scores": self._restore(raw, attention),
            "promotion_features": self._restore(raw, promotion),
            "inhibition_features": self._restore(raw, inhibition),
            "gate_weights": self._restore(raw, weights),
            "raw_ei_delta": self._restore(raw, raw_delta),
            "applied_ei_delta": self._restore(raw, applied),
            "unmasked_total_delta": self._restore(raw, unmasked - x),
            "actual_total_delta": self._restore(raw, total_delta),
            "relative_actual_modulation": self._restore(raw, relative),
            "output_features": output,
            "synergy": tf.identity(self.synergy),
            "feature_intervention_mask": tf.identity(self._feature_mask()),
            "intervention_mask": tf.identity(self._feature_mask()),
            "pathway_state": pathway_state,
            "context_requested": {
                "coordination": self.use_coord_proj,
                "q_eff": self.use_qeff_in_gate,
                "electrostatic_magnitude": self.use_emag_in_gate,
                "generic": gate_context is not None,
            },
            "coordination_used": bool(self.use_coord_proj and coordination is not None),
            "q_eff_used": bool(self.use_qeff_in_gate and q_eff is not None),
            "emag_used": bool(
                self.use_emag_in_gate and electrostatic_magnitude is not None
            ),
            "generic_context_used": bool(gate_context is not None),
        }
        return output, analysis

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "feature_dim": self.feature_dim,
                "reduced_dim": self.reduced_dim,
                "comb_dropout": self.comb_dropout_rate,
                "use_shared_expert": self.use_shared_expert,
                "low_rank_gate": self.low_rank_gate,
                "gate_low_rank_dim": self.gate_low_rank_dim,
                "use_dimension_gating": self.use_dimension_gating,
                "dimension_gating_type": self.dimension_gating_type,
                "dimension_sparsity_weight": self.dimension_sparsity_weight,
                "use_gate_context_proj": self.use_gate_context_proj,
                "gate_ctx_proj_units": self.gate_ctx_proj_units,
                "use_coord_proj": self.use_coord_proj,
                "use_qeff_in_gate": self.use_qeff_in_gate,
                "use_emag_in_gate": self.use_emag_in_gate,
                "synergy_init": self.synergy_init,
                "synergy_min": self.synergy_min,
                "synergy_max": self.synergy_max,
                "l2_reg": self.l2_reg,
                "gate_seed": self.gate_seed,
                "compatibility_profile": self.compatibility_profile,
            }
        )
        return cfg
