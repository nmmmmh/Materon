"""CGCNN models: conventional (kgcnn), physical-attribute (no gate) and Materon (input gate).

Physics and Materon models share ``LegacyCGCNNCore``; the gate only replaces
the node attributes before the input projection. ``build_paired_model`` copies
the core weights of a freshly built no-gate model into the Materon model, so
both start from identical message-passing weights.
"""
from __future__ import annotations

import hashlib

import numpy as np
import tensorflow as tf
from kgcnn.layers.conv.cgcnn_conv import CGCNNLayer
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.literature.CGCNN import make_crystal_model

from materon.gates.ei import make_gate


class LegacyCGCNNCore(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=64, num_conv=3, head_dim=64, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_conv = int(num_conv)
        self.input_proj = (tf.keras.layers.Dense(self.hidden_dim, activation="linear", name="input_proj")
                           if self.input_dim != self.hidden_dim
                           else tf.keras.layers.Lambda(lambda x: x, name="input_identity"))
        self.convs = [CGCNNLayer(units=self.hidden_dim, activation_s="softplus", activation_out="softplus",
                                 batch_normalization=True, name="cgcnn_conv_%d" % i) for i in range(self.num_conv)]
        self.dropout = tf.keras.layers.Dropout(float(dropout), name="backbone_dropout")
        self.pool = PoolingNodes(pooling_method="mean", name="mean_pool")
        self.head_hidden = tf.keras.layers.Dense(int(head_dim), activation="softplus", name="head_hidden")
        self.head_output = tf.keras.layers.Dense(1, activation="linear", use_bias=False, name="head_output")

    def convolve(self, node_features, edge_features, edge_indices, training=False):
        h = self.input_proj(node_features, training=training)
        for layer in self.convs:
            h = layer([h, edge_features, edge_indices], training=training)
        return self.dropout(h, training=training)

    def readout(self, h, training=False):
        return self.head_output(self.head_hidden(self.pool(h), training=training), training=training)

    def call(self, inputs, training=False):
        node, edge_idx, edge_feat = inputs
        return self.readout(self.convolve(node, edge_feat, edge_idx, training), training)


class MateronCGCNN(tf.keras.Model):
    """Inputs: [node, edge_indices, edge_features, coordination, q_eff, electrostatic_magnitude]."""

    def __init__(self, feature_dim=20, input_gate=None, context_mode="none", seed=0, profile=None, **kwargs):
        super().__init__(**kwargs)
        p = profile or {}
        self.feature_dim = int(feature_dim)
        self.context_mode = str(context_mode)
        hidden = int(p.get("hidden_dim", 64))
        self.input_gate = make_gate(input_gate, feature_dim, seed)
        self.core = LegacyCGCNNCore(feature_dim, hidden_dim=hidden, num_conv=p.get("depth", 3),
                                    head_dim=p.get("head_dim", hidden), dropout=p.get("dropout", 0.0),
                                    name="cgcnn_core")

    def _kwargs(self, coord, qeff, emag):
        return {"none": {}, "coord": {"coordination": coord}, "qeff": {"q_eff": qeff},
                "emag": {"electrostatic_magnitude": emag},
                "all": {"coordination": coord, "q_eff": qeff, "electrostatic_magnitude": emag}}[self.context_mode]

    def call(self, inputs, training=False, return_analysis=False):
        node, edge_idx, edge_feat, coord, qeff, emag = inputs
        kwargs = self._kwargs(coord, qeff, emag)
        analysis = {"raw_physical_features": node, "context_mode": self.context_mode}
        if return_analysis:
            h, item = self.input_gate(node, training=training, return_analysis=True, **kwargs)
            analysis["input_gate"] = item
        else:
            h = self.input_gate(node, training=training, **kwargs)
        out = self.core.readout(self.core.convolve(h, edge_feat, edge_idx, training), training)
        return (out, analysis) if return_analysis else out


class PhysicsCGCNN(tf.keras.Model):
    """No-gate model on physical attributes; optional extra MLP for the parameter-matched control."""

    def __init__(self, feature_dim=20, profile=None, extra_capacity_units=0, **kwargs):
        super().__init__(**kwargs)
        p = profile or {}
        self.feature_dim = int(feature_dim)
        self.core = LegacyCGCNNCore(feature_dim, hidden_dim=p.get("hidden_dim", 64), num_conv=p.get("depth", 3),
                                    head_dim=p.get("head_dim", 64), dropout=p.get("dropout", 0.0),
                                    name="cgcnn_core")
        u = int(extra_capacity_units)
        self.extra_capacity_units = u
        self.capacity_mlp = None if u <= 0 else tf.keras.Sequential(
            [tf.keras.layers.Dense(u, activation="swish"),
             tf.keras.layers.Dense(int(p.get("hidden_dim", 64)), activation="linear")],
            name="parameter_match_capacity")

    def call(self, inputs, training=False, return_analysis=False):
        node, edge_idx, edge_feat = inputs
        h = self.core.convolve(node, edge_feat, edge_idx, training)
        if self.capacity_mlp is not None:
            h = h + self.capacity_mlp(h, training=training)
        out = self.core.readout(h, training)
        if return_analysis:
            return out, {"raw_physical_features": node, "extra_capacity_units": self.extra_capacity_units}
        return out


def build_standard_cgcnn(profile, name="cgcnn_baseline"):
    hidden = int(profile["hidden_dim"])
    depth = int(profile["depth"])
    head = int(profile.get("head_dim", hidden))
    return make_crystal_model(
        name=name,
        inputs=[{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "node_frac_coordinates", "dtype": "float32", "ragged": True},
                {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32", "ragged": False},
                {"shape": (None, 3), "name": "range_image", "dtype": "float32", "ragged": True}],
        input_embedding={"node": {"input_dim": 95, "output_dim": hidden}}, representation="unit",
        expand_distance=True, make_distances=True,
        gauss_args={"bins": 40, "distance": 8.0, "offset": 0.0, "sigma": 0.4}, depth=depth, verbose=0,
        conv_layer_args={"units": hidden, "activation_s": "softplus", "activation_out": "softplus",
                         "batch_normalization": True},
        node_pooling_args={"pooling_method": "mean"}, output_embedding="graph",
        output_mlp={"use_bias": [True, False], "units": [head, 1], "activation": ["softplus", "linear"]})


def feature_dim(cfg):
    return 20 if cfg.get("feature_mode") in {"full20", "shuffled_full20"} else 12


def build_model(cfg, profile, seed, extra_capacity_units=0):
    kind = cfg["model_type"]
    if kind == "cgcnn":
        return build_standard_cgcnn(profile, name=cfg["id"])
    if kind == "physics":
        return PhysicsCGCNN(feature_dim(cfg), profile, extra_capacity_units=extra_capacity_units, name=cfg["id"])
    if kind == "materon":
        return MateronCGCNN(feature_dim(cfg), cfg.get("input_gate"), cfg.get("context_mode", "none"), seed,
                            profile, name=cfg["id"])
    raise ValueError(kind)


def common_layers(model):
    core = getattr(model, "core", None)
    if core is None:
        return []
    return [core.input_proj] + list(core.convs) + [core.head_hidden, core.head_output]


def copy_common_weights(reference, target):
    records = []
    for a, b in zip(common_layers(reference), common_layers(target)):
        if len(a.weights) != len(b.weights):
            raise RuntimeError("Common layer weight-count mismatch")
        b.set_weights(a.get_weights())
        records.append((a.name, b.name, len(a.weights)))
    return records


def common_fingerprint(model):
    h = hashlib.sha256()
    n = 0
    for layer in common_layers(model):
        for w in layer.weights:
            h.update(np.asarray(w.numpy()).tobytes())
            n += 1
    return {"sha256": h.hexdigest(), "weight_tensors": n}


def build_paired_model(cfg, profile, sample, seed, extra_capacity_units=0):
    """Build a model; Materon models receive the core weights of a matching no-gate model."""
    if cfg["model_type"] != "materon":
        model = build_model(cfg, profile, seed, extra_capacity_units)
        _ = model(sample, training=False)
        return model, None, common_fingerprint(model) if hasattr(model, "core") else None
    ref_cfg = {"id": "paired_physics_reference", "model_type": "physics",
               "feature_mode": cfg.get("feature_mode", "full20"), "context_mode": "none", "input_gate": None}
    ref = build_model(ref_cfg, profile, seed)
    _ = ref(sample[:3], training=False)
    model = build_model(cfg, profile, seed)
    _ = model(sample, training=False)
    records = copy_common_weights(ref, model)
    if common_fingerprint(ref) != common_fingerprint(model):
        raise RuntimeError("Paired initialization mismatch")
    return model, records, common_fingerprint(model)


def estimate_parameter_match_units(base_params, target_params, hidden_dim):
    delta = max(0, int(target_params) - int(base_params))
    denom = 2 * int(hidden_dim) + 1
    return max(1, int(round(max(0, delta - int(hidden_dim)) / float(denom)))) if delta > 0 else 0
