"""Excitation/inhibition (E/I) input gates for the Matbench and fixed-split benchmarks.

All gates act on the named physical node attributes before message passing and
return the modulated attributes; with ``return_analysis=True`` they also return
their internal readouts.

``BiomimeticEIGate`` is the Materon E/I gate of Eq. (2) / Eq. (S1):

    x_d = x * sigmoid(dimension_logits)
    a   = sigmoid(W_a x_d)                       (node confidence)
    v+, v- = shared expert(a * x_d)
    w+, w- = softmax(gate_out(a * x_d) [+ context logits])
    x~  = x_d + s * (w+ v+ - w- v-),  s clipped to [synergy_min, synergy_max]

Two variable layouts are kept so that the trained checkpoints of both
backbones load unchanged:

* ``pathway_controls=False`` (CGCNN): one non-trainable feature mask
  (``modulation_mask``) and optional named context inputs
  (coordination, q_eff, electrostatic magnitude) for the context ablations.
* ``pathway_controls=True`` (MEGNet): the layer is called on ``[x, node_mask]``
  and carries a feature mask plus four scalar pathway masks (promotion,
  inhibition, dimension gate, synergy) used for pathway shutdown.

With every mask equal to one both layouts compute the same function.

The remaining gates (simple scaling, legacy sigmoid scaling, refined E/I) are
candidates of the CGCNN phonons configuration search and are kept so that the
inner-pool selection can be reproduced.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Materon")
class ScalarClip(tf.keras.constraints.Constraint):
    def __init__(self, minimum, maximum):
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def __call__(self, value):
        return tf.clip_by_value(value, self.minimum, self.maximum)

    def get_config(self):
        return {"minimum": self.minimum, "maximum": self.maximum}


class _RaggedGateBase(tf.keras.layers.Layer):
    """Shared ragged handling and the per-attribute intervention mask."""

    mask_variable_name = "modulation_mask"

    @staticmethod
    def _flat(value):
        return value.flat_values if isinstance(value, tf.RaggedTensor) else value

    @staticmethod
    def _restore(reference, flat):
        return reference.with_flat_values(flat) if isinstance(reference, tf.RaggedTensor) else flat

    def _feature_mask(self):
        return getattr(self, self.mask_variable_name)

    def set_feature_mask(self, values):
        """Scale the gate correction of each attribute (1 = native, 0 = disabled)."""
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.shape[0] != self.feature_dim:
            raise ValueError("feature mask has wrong length")
        self._feature_mask().assign(array)

    def reset_feature_mask(self):
        mask = self._feature_mask()
        mask.assign(tf.ones_like(mask))


@tf.keras.utils.register_keras_serializable(package="Materon")
class SimpleFeatureGate(_RaggedGateBase):
    """Identity-initialized direct feature scaling (search candidate)."""

    def __init__(self, feature_dim, init_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = int(feature_dim)
        self.init_value = float(init_value)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="feature_scale", shape=(self.feature_dim,),
            initializer=tf.keras.initializers.Constant(self.init_value), trainable=True)
        self.modulation_mask = self.add_weight(
            name="modulation_mask", shape=(self.feature_dim,), initializer="ones", trainable=False)
        super().build(input_shape)

    def call(self, inputs, training=False, return_analysis=False, **kwargs):
        del training, kwargs
        x = tf.cast(self._flat(inputs), tf.float32)
        scale = tf.cast(self.scale, tf.float32)
        raw_out = x * scale
        out = x + (raw_out - x) * self.modulation_mask[None, :]
        output = self._restore(inputs, out)
        if not return_analysis:
            return output
        analysis = {
            "input_features": self._restore(inputs, x),
            "feature_scale": scale,
            "intervention_mask": tf.identity(self.modulation_mask),
            "unmasked_delta": self._restore(inputs, raw_out - x),
            "actual_delta": self._restore(inputs, out - x),
            "actual_total_delta": self._restore(inputs, out - x),
            "output_features": output,
        }
        return output, analysis

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"feature_dim": self.feature_dim, "init_value": self.init_value})
        return cfg


@tf.keras.utils.register_keras_serializable(package="Materon")
class LegacySigmoidSimpleGate(_RaggedGateBase):
    """Legacy scaling x * sigmoid(w); init 1.0 gives sigmoid(1) = 0.731 (search candidate)."""

    def __init__(self, feature_dim, init_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = int(feature_dim)
        self.init_value = float(init_value)

    def build(self, input_shape):
        self.gate_weights = self.add_weight(
            name="gate_weights", shape=(self.feature_dim,),
            initializer=tf.keras.initializers.Constant(self.init_value), trainable=True)
        self.modulation_mask = self.add_weight(
            name="modulation_mask", shape=(self.feature_dim,), initializer="ones", trainable=False)
        super().build(input_shape)

    def call(self, inputs, training=False, return_analysis=False, **kwargs):
        del training, kwargs
        x = tf.cast(self._flat(inputs), tf.float32)
        scale = tf.nn.sigmoid(self.gate_weights)
        raw = x * scale
        out = x + (raw - x) * self.modulation_mask[None, :]
        output = self._restore(inputs, out)
        if not return_analysis:
            return output
        analysis = {
            "input_features": self._restore(inputs, x),
            "feature_scale": scale,
            "unmasked_delta": self._restore(inputs, raw - x),
            "actual_delta": self._restore(inputs, out - x),
            "actual_total_delta": self._restore(inputs, out - x),
            "output_features": output,
            "legacy_sigmoid_semantics": True,
        }
        return output, analysis

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"feature_dim": self.feature_dim, "init_value": self.init_value})
        return cfg


@tf.keras.utils.register_keras_serializable(package="Materon")
class BiomimeticEIGate(_RaggedGateBase):
    """Materon E/I gate (see module docstring)."""

    def __init__(
        self,
        feature_dim,
        reduced_dim=8,
        comb_dropout=0.01,
        use_shared_expert=True,
        low_rank_gate=True,
        gate_low_rank_dim=8,
        use_dimension_gating=True,
        dimension_gating_type="soft",
        dimension_sparsity_weight=0.0,
        use_gate_context_proj=True,
        gate_ctx_proj_units=8,
        use_coord_proj=False,
        use_qeff_in_gate=False,
        use_emag_in_gate=False,
        synergy_init=0.1,
        synergy_min=0.005,
        synergy_max=0.5,
        l2_reg=0.0,
        gate_seed=None,
        pathway_controls=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.pathway_controls = bool(pathway_controls)
        self.mask_variable_name = "feature_intervention_mask" if self.pathway_controls else "modulation_mask"
        if self.pathway_controls and (self.use_coord_proj or self.use_qeff_in_gate or self.use_emag_in_gate):
            raise ValueError("named gate contexts are only available without pathway controls")
        if self.pathway_controls and self.dimension_gating_type == "hard":
            raise ValueError("hard dimension gating is only available without pathway controls")
        reg = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None

        # Sublayers are created in the original order; unused ones hold no weights.
        self.attn = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=False, name="node_confidence")
        if self.low_rank_gate:
            self.gate_down = tf.keras.layers.Dense(
                self.gate_low_rank_dim, activation="swish", kernel_regularizer=reg, name="gate_down")
        self.gate_out = tf.keras.layers.Dense(2, activation=None, kernel_regularizer=reg, name="gate_out")
        if self.use_shared_expert:
            self.shared_hidden = tf.keras.layers.Dense(
                self.reduced_dim, activation="swish", kernel_regularizer=reg, name="shared_hidden")
            self.shared_out = tf.keras.layers.Dense(
                self.feature_dim * 2, activation=None, kernel_regularizer=reg, name="shared_out")
        else:
            self.promo_hidden = tf.keras.layers.Dense(
                self.reduced_dim, activation="swish", kernel_regularizer=reg, name="promo_hidden")
            self.promo_out = tf.keras.layers.Dense(
                self.feature_dim, activation=None, kernel_regularizer=reg, name="promo_out")
            self.inhib_hidden = tf.keras.layers.Dense(
                self.reduced_dim, activation="swish", kernel_regularizer=reg, name="inhib_hidden")
            self.inhib_out = tf.keras.layers.Dense(
                self.feature_dim, activation=None, kernel_regularizer=reg, name="inhib_out")
        if not self.pathway_controls:
            if self.use_gate_context_proj:
                self.context_hidden = tf.keras.layers.Dense(
                    self.gate_ctx_proj_units, activation="swish", kernel_regularizer=reg, name="context_hidden")
                self.context_out = tf.keras.layers.Dense(
                    2, activation=None, kernel_regularizer=reg, name="context_out")
                self.context_direct = None
            else:
                self.context_hidden = None
                self.context_out = None
                self.context_direct = tf.keras.layers.Dense(
                    2, activation=None, kernel_regularizer=reg, name="context_direct")
            self.coord_proj = tf.keras.layers.Dense(1, activation="relu", kernel_regularizer=reg, name="coord_proj")
        self.dropout = tf.keras.layers.Dropout(self.comb_dropout_rate, seed=self.gate_seed)

    def build(self, input_shape):
        if self.use_dimension_gating:
            # CGCNN layout: 1.0 for soft gating, 0.5 otherwise; MEGNet layout: always 1.0.
            init = 1.0 if (self.pathway_controls or self.dimension_gating_type == "soft") else 0.5
            reg = (tf.keras.regularizers.l1(self.dimension_sparsity_weight)
                   if self.dimension_gating_type == "sparse" else None)
            self.dimension_logits = self.add_weight(
                name="dimension_logits", shape=(self.feature_dim,),
                initializer=tf.keras.initializers.Constant(init), regularizer=reg, trainable=True)
        self.synergy = self.add_weight(
            name="synergy", shape=(), initializer=tf.keras.initializers.Constant(self.synergy_init),
            constraint=ScalarClip(self.synergy_min, self.synergy_max), trainable=True)
        if self.pathway_controls:
            self.feature_intervention_mask = self.add_weight(
                name="feature_intervention_mask", shape=(self.feature_dim,), initializer="ones", trainable=False)
            self.promotion_path_mask = self.add_weight(
                name="promotion_path_mask", shape=(), initializer="ones", trainable=False)
            self.inhibition_path_mask = self.add_weight(
                name="inhibition_path_mask", shape=(), initializer="ones", trainable=False)
            self.dimension_path_mask = self.add_weight(
                name="dimension_path_mask", shape=(), initializer="ones", trainable=False)
            self.synergy_path_mask = self.add_weight(
                name="synergy_path_mask", shape=(), initializer="ones", trainable=False)
        else:
            self.modulation_mask = self.add_weight(
                name="modulation_mask", shape=(self.feature_dim,), initializer="ones", trainable=False)
        super().build(input_shape)

    # Pathway shutdown (MEGNet layout only).
    def set_pathway_state(self, promotion=1.0, inhibition=1.0, dimension=1.0, synergy=1.0):
        if not self.pathway_controls:
            raise RuntimeError("pathway masks exist only with pathway_controls=True")
        self.promotion_path_mask.assign(float(promotion))
        self.inhibition_path_mask.assign(float(inhibition))
        self.dimension_path_mask.assign(float(dimension))
        self.synergy_path_mask.assign(float(synergy))

    def reset_pathway_state(self):
        self.set_pathway_state()

    def _context_logits(self, gate_context, coordination, q_eff, electrostatic_magnitude, training):
        parts = []
        if gate_context is not None:
            parts.append(tf.cast(self._flat(gate_context), tf.float32))
        if self.use_coord_proj:
            if coordination is None:
                raise RuntimeError("use_coord_proj=True but coordination context was not supplied")
            parts.append(self.coord_proj(tf.cast(self._flat(coordination), tf.float32), training=training))
        if self.use_qeff_in_gate:
            if q_eff is None:
                raise RuntimeError("use_qeff_in_gate=True but q_eff context was not supplied")
            parts.append(tf.cast(self._flat(q_eff), tf.float32))
        if self.use_emag_in_gate:
            if electrostatic_magnitude is None:
                raise RuntimeError("use_emag_in_gate=True but electrostatic context was not supplied")
            parts.append(tf.cast(self._flat(electrostatic_magnitude), tf.float32))
        if not parts:
            return None
        context = tf.concat(parts, axis=-1) if len(parts) > 1 else parts[0]
        if self.use_gate_context_proj:
            return self.context_out(self.context_hidden(context, training=training), training=training)
        return self.context_direct(context, training=training)

    def _dimension_gate(self, x, training):
        if not self.use_dimension_gating:
            dim = tf.ones((self.feature_dim,), tf.float32)
            return dim, dim, x
        if self.dimension_gating_type in {"soft", "sparse"}:
            dim = tf.nn.sigmoid(self.dimension_logits)
        elif self.dimension_gating_type == "hard":
            logits = tf.stack([self.dimension_logits, -self.dimension_logits], axis=-1)
            if training:
                uniform = tf.random.uniform(tf.shape(logits), minval=1.0e-6, maxval=1.0 - 1.0e-6)
                gumbel = -tf.math.log(-tf.math.log(uniform))
                dim = tf.nn.softmax(logits + gumbel, axis=-1)[:, 0]
            else:
                dim = tf.one_hot(tf.argmax(logits, axis=-1), depth=2, dtype=tf.float32)[:, 0]
        else:
            raise ValueError("Unknown dimension_gating_type: %s" % self.dimension_gating_type)
        if self.pathway_controls:
            dim_eff = 1.0 + self.dimension_path_mask * (dim - 1.0)
            return dim, dim_eff, x * dim_eff[None, :]
        return dim, dim, x * dim

    def call(self, inputs, training=False, return_analysis=False,
             gate_context=None, coordination=None, q_eff=None, electrostatic_magnitude=None):
        if self.pathway_controls:
            raw, node_mask = inputs
            node_scale = tf.cast(self._flat(node_mask), tf.float32)
        else:
            raw, node_scale = inputs, None
        x = tf.cast(self._flat(raw), tf.float32)
        dim, dim_eff, x_dim = self._dimension_gate(x, training)

        attention = self.attn(x_dim, training=training)
        x_att = x_dim * attention
        if self.low_rank_gate:
            gate_logits = self.gate_out(self.gate_down(x_att, training=training), training=training)
        else:
            gate_logits = self.gate_out(x_att, training=training)
        if self.use_shared_expert:
            both = self.shared_out(self.shared_hidden(x_att, training=training), training=training)
            promotion, inhibition = tf.split(both, 2, axis=-1)
        else:
            promotion = self.promo_out(self.promo_hidden(x_att, training=training), training=training)
            inhibition = self.inhib_out(self.inhib_hidden(x_att, training=training), training=training)
        if not self.pathway_controls:
            context_logits = self._context_logits(gate_context, coordination, q_eff,
                                                  electrostatic_magnitude, training)
            if context_logits is not None:
                gate_logits = gate_logits + context_logits
        weights = tf.nn.softmax(gate_logits, axis=-1)

        if self.pathway_controls:
            raw_delta = (self.promotion_path_mask * weights[:, 0:1] * promotion
                         - self.inhibition_path_mask * weights[:, 1:2] * inhibition)
            applied = self.synergy_path_mask * self.synergy * raw_delta
        else:
            raw_delta = weights[:, 0:1] * promotion - weights[:, 1:2] * inhibition
            applied = self.synergy * raw_delta
        applied = self.dropout(applied, training=training)
        unmasked = x_dim + applied
        if self.pathway_controls:
            total_delta = (unmasked - x) * self.feature_intervention_mask[None, :] * node_scale
            out = x + total_delta
        else:
            out = x + (unmasked - x) * self.modulation_mask[None, :]
            total_delta = out - x
        output = self._restore(raw, out)
        if not return_analysis:
            return output

        abs_delta = tf.abs(total_delta)
        relative = abs_delta / (tf.reduce_sum(abs_delta, axis=-1, keepdims=True) + 1.0e-8)
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
        }
        if self.pathway_controls:
            analysis["pathway_state"] = tf.stack([
                self.promotion_path_mask, self.inhibition_path_mask,
                self.dimension_path_mask, self.synergy_path_mask])
        else:
            analysis.update({
                "coordination_used": bool(self.use_coord_proj and coordination is not None),
                "q_eff_used": bool(self.use_qeff_in_gate and q_eff is not None),
                "emag_used": bool(self.use_emag_in_gate and electrostatic_magnitude is not None),
            })
        return output, analysis

    def get_config(self):
        cfg = super().get_config()
        cfg.update({k: getattr(self, k) for k in [
            "feature_dim", "reduced_dim", "use_shared_expert", "low_rank_gate",
            "gate_low_rank_dim", "use_dimension_gating", "dimension_gating_type",
            "dimension_sparsity_weight", "use_gate_context_proj", "gate_ctx_proj_units",
            "use_coord_proj", "use_qeff_in_gate", "use_emag_in_gate", "synergy_init",
            "synergy_min", "synergy_max", "l2_reg", "gate_seed", "pathway_controls"]})
        cfg["comb_dropout"] = self.comb_dropout_rate
        return cfg


@tf.keras.utils.register_keras_serializable(package="Materon")
class RefinedBiomimeticEIGate(_RaggedGateBase):
    """Identity-capable feature-wise E/I variant (search candidate)."""

    def __init__(self, feature_dim, rank=8, alpha_init=0.0, alpha_max=0.2,
                 use_shared_expert=True, low_rank_gate=True, use_dimension_gating=True,
                 context_dim=0, dropout=0.0, l2_reg=0.0, gate_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.alpha_init = float(alpha_init)
        self.alpha_max = float(alpha_max)
        self.use_shared_expert = bool(use_shared_expert)
        self.low_rank_gate = bool(low_rank_gate)
        self.use_dimension_gating = bool(use_dimension_gating)
        self.context_dim = int(context_dim)
        self.dropout_rate = float(dropout)
        self.l2_reg = float(l2_reg)
        self.gate_seed = None if gate_seed is None else int(gate_seed)
        reg = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        self.down = tf.keras.layers.Dense(self.rank, activation="swish", kernel_regularizer=reg, name="down")
        if self.use_shared_expert:
            self.expert = tf.keras.layers.Dense(
                self.feature_dim * 2, activation=None, kernel_regularizer=reg, name="shared_expert")
        else:
            self.promotion = tf.keras.layers.Dense(
                self.feature_dim, activation=None, kernel_regularizer=reg, name="promotion")
            self.inhibition = tf.keras.layers.Dense(
                self.feature_dim, activation=None, kernel_regularizer=reg, name="inhibition")
        self.competition = tf.keras.layers.Dense(
            self.feature_dim * 2, activation=None, kernel_regularizer=reg, name="feature_competition")
        self.context_projection = (tf.keras.layers.Dense(
            self.feature_dim * 2, activation=None, kernel_regularizer=reg, name="context_projection")
            if self.context_dim > 0 else None)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, seed=self.gate_seed)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=(), initializer=tf.keras.initializers.Constant(self.alpha_init),
            constraint=ScalarClip(0.0, self.alpha_max), trainable=True)
        if self.use_dimension_gating:
            self.dimension_logits = self.add_weight(
                name="dimension_logits", shape=(self.feature_dim,), initializer="zeros", trainable=True)
        self.modulation_mask = self.add_weight(
            name="modulation_mask", shape=(self.feature_dim,), initializer="ones", trainable=False)
        super().build(input_shape)

    def call(self, inputs, training=False, return_analysis=False, gate_context=None, **kwargs):
        del kwargs
        x = tf.cast(self._flat(inputs), tf.float32)
        hidden = self.down(x, training=training)
        if self.use_shared_expert:
            promotion, inhibition = tf.split(self.expert(hidden, training=training), 2, axis=-1)
        else:
            promotion = self.promotion(hidden, training=training)
            inhibition = self.inhibition(hidden, training=training)
        logits = self.competition(hidden if self.low_rank_gate else x, training=training)
        if self.context_dim > 0:
            if gate_context is None:
                raise RuntimeError("Refined gate requires gate_context but none was supplied")
            logits = logits + self.context_projection(
                tf.cast(self._flat(gate_context), tf.float32), training=training)
        weights = tf.nn.softmax(tf.reshape(logits, (-1, self.feature_dim, 2)), axis=-1)
        raw_delta = weights[:, :, 0] * promotion - weights[:, :, 1] * inhibition
        if self.use_dimension_gating:
            dimension_gate = tf.nn.sigmoid(self.dimension_logits)
            raw_delta = raw_delta * dimension_gate[None, :]
        else:
            dimension_gate = tf.ones((self.feature_dim,), tf.float32)
        applied = self.dropout(self.alpha * raw_delta, training=training)
        unmasked_out = x + applied
        out = x + (unmasked_out - x) * self.modulation_mask[None, :]
        output = self._restore(inputs, out)
        if not return_analysis:
            return output
        abs_delta = tf.abs(out - x)
        relative = abs_delta / (tf.reduce_sum(abs_delta, axis=-1, keepdims=True) + 1.0e-8)
        analysis = {
            "input_features": self._restore(inputs, x),
            "dimension_gates": dimension_gate,
            "promotion_features": self._restore(inputs, promotion),
            "inhibition_features": self._restore(inputs, inhibition),
            "feature_gate_weights": self._restore(inputs, weights),
            "raw_ei_delta": self._restore(inputs, raw_delta),
            "applied_ei_delta": self._restore(inputs, applied),
            "unmasked_total_delta": self._restore(inputs, unmasked_out - x),
            "intervention_mask": tf.identity(self.modulation_mask),
            "actual_total_delta": self._restore(inputs, out - x),
            "relative_actual_modulation": self._restore(inputs, relative),
            "output_features": output,
            "alpha": tf.identity(self.alpha),
        }
        return output, analysis

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "feature_dim": self.feature_dim, "rank": self.rank, "alpha_init": self.alpha_init,
            "alpha_max": self.alpha_max, "use_shared_expert": self.use_shared_expert,
            "low_rank_gate": self.low_rank_gate, "use_dimension_gating": self.use_dimension_gating,
            "context_dim": self.context_dim, "dropout": self.dropout_rate, "l2_reg": self.l2_reg,
            "gate_seed": self.gate_seed})
        return cfg


def make_gate(config, feature_dim, seed, name="input_gate", pathway_controls=False):
    """Build an input gate from a configuration dict with a ``type`` key."""
    if not config:
        return None
    cfg = dict(config)
    kind = cfg.pop("type")
    cfg["feature_dim"] = int(feature_dim)
    if kind == "simple":
        return SimpleFeatureGate(name=name, **cfg)
    if kind == "legacy_simple":
        return LegacySigmoidSimpleGate(name=name, **cfg)
    cfg["gate_seed"] = int(seed)
    if kind in {"legacy", "legacy_biomimetic_ei"}:
        return BiomimeticEIGate(name=name, pathway_controls=pathway_controls, **cfg)
    if kind == "refined":
        return RefinedBiomimeticEIGate(name=name, **cfg)
    raise ValueError("Unknown gate type: %s" % kind)
