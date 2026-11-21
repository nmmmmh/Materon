"""kgcnn 2.1 MEGNet models with the fixed Matbench hyper-parameters.

* megnet_baseline: atomic-number embedding (official kgcnn model).
* materon_nogate_megnet / materon_pca_megnet: full20 / grouped-PCA node attributes.
* materon_explain_megnet: full20 attributes -> E/I gate (S1-L0-D1) -> MEGNet.
  ``build_model`` returns (train_model, analysis_model, ANALYSIS_NAMES); the
  analysis model exposes the gate readouts listed in ANALYSIS_NAMES.
"""
from __future__ import annotations

import copy

import tensorflow as tf
from kgcnn.literature.Megnet import make_crystal_model

from materon.gates.ei import BiomimeticEIGate

ANALYSIS_NAMES = ["prediction", "attention_scores", "dimension_gates", "dimension_delta", "promotion_features",
                  "inhibition_features", "gate_weights", "raw_ei_delta", "applied_ei_delta", "actual_total_delta",
                  "relative_actual_modulation", "output_features", "synergy", "pathway_state"]


def official_inputs(node_shape, node_name="node_number"):
    return [
        {"shape": node_shape, "name": node_name, "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
        {"shape": [1], "name": "charge", "dtype": "float32", "ragged": False},
        {"shape": (None, 3), "name": "range_image", "dtype": "int64", "ragged": True},
        {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32", "ragged": False},
    ]


def megnet_config(node_shape, node_name, name):
    return {
        "name": name, "inputs": official_inputs(node_shape, node_name),
        "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}, "graph": {"input_dim": 100, "output_dim": 64}},
        "make_distance": True, "expand_distance": True,
        "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4},
        "meg_block_args": {"node_embed": [64, 32, 32], "edge_embed": [64, 32, 32], "env_embed": [64, 32, 32],
                           "activation": "kgcnn>softplus2"},
        "set2set_args": {"channels": 16, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
        "node_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
        "edge_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
        "state_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
        "nblocks": 3, "has_ff": True, "dropout": None, "use_set2set": True, "verbose": 10,
        "output_embedding": "graph",
        "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1],
                       "activation": ["kgcnn>softplus2", "kgcnn>softplus2", "linear"]},
    }


def find_physical_megnet_backbone(model):
    """The nested MEGNet model of materon_explain_megnet (matched by structure, not by name)."""
    try:
        return model.get_layer("MEGNet_physical_backbone")
    except Exception:
        pass
    candidates = []
    for layer in getattr(model, "layers", []):
        if not isinstance(layer, tf.keras.Model):
            continue
        inputs = list(getattr(layer, "inputs", []) or [])
        if len(inputs) != 6:
            continue
        first_name = str(getattr(inputs[0], "name", ""))
        try:
            last_dim = int(inputs[0].shape[-1]) if inputs[0].shape[-1] is not None else None
        except Exception:
            last_dim = None
        if "node_features" in first_name or last_dim == 20:
            candidates.append(layer)
    if len(candidates) != 1:
        raise RuntimeError(f"Unable to uniquely locate physical MEGNet backbone; candidates={len(candidates)}")
    return candidates[0]


def build_model(model_id, protocol, seed=42):
    if model_id == "megnet_baseline":
        return make_crystal_model(**megnet_config((None,), "node_number", "MEGNet_baseline")), None, None
    if model_id == "materon_nogate_megnet":
        return make_crystal_model(**megnet_config((None, 20), "node_full20", "Materon_NoGate_MEGNet")), None, None
    if model_id == "materon_pca_megnet":
        return make_crystal_model(**megnet_config((None, 12), "node_grouped_pca", "Materon_PCA_MEGNet")), None, None
    if model_id != "materon_explain_megnet":
        raise KeyError(model_id)
    phys = tf.keras.layers.Input(shape=(None, 20), dtype="float32", ragged=True, name="node_full20")
    coords = tf.keras.layers.Input(shape=(None, 3), dtype="float32", ragged=True, name="node_coordinates")
    edges = tf.keras.layers.Input(shape=(None, 2), dtype="int64", ragged=True, name="range_indices")
    charge = tf.keras.layers.Input(shape=(1,), dtype="float32", name="charge")
    image = tf.keras.layers.Input(shape=(None, 3), dtype="int64", ragged=True, name="range_image")
    lattice = tf.keras.layers.Input(shape=(3, 3), dtype="float32", name="graph_lattice")
    node_mask = tf.keras.layers.Input(shape=(None, 1), dtype="float32", ragged=True, name="node_gate_mask")
    gate_cfg = copy.deepcopy(protocol["fixed_gate"])
    gate_cfg.pop("type", None)
    gate_cfg["gate_seed"] = int(seed)
    gate = BiomimeticEIGate(name="biomimetic_ei_gate", pathway_controls=True, **gate_cfg)
    gated, analysis = gate([phys, node_mask], return_analysis=True)
    backbone = make_crystal_model(**megnet_config((None, 20), "node_features", "MEGNet_physical_backbone"))
    prediction = backbone([gated, coords, edges, charge, image, lattice])
    inputs = [phys, coords, edges, charge, image, lattice, node_mask]
    train_model = tf.keras.Model(inputs=inputs, outputs=prediction, name="Materon_Explain_MEGNet")
    outputs = [prediction] + [analysis[name] for name in ANALYSIS_NAMES[1:]]
    analysis_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Materon_Explain_MEGNet_Analysis")
    return train_model, analysis_model, ANALYSIS_NAMES
