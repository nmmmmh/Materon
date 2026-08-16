# -*- coding: utf-8 -*-
"""Semantic backbone-weight pairing for official backbone and Materon models.

The target model wraps the official node embedding inside
``PriorAwareNodeEmbedding``.  Consequently, the flattened order of
``model.weights`` is not guaranteed to be the same as in the official model.
CGCNN contains many repeated convolution and BatchNorm tensors with identical
shapes, so global positional copying can silently assign weights to the wrong
same-shaped layer.

This module pairs direct functional-model layers in topological order.  At the
single Materon wrapper position, the source official embedding is paired with
``wrapper.base_layer``.  All remaining official layers must match by class and
by their local weight list.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from .embedding_wrapper import PriorAwareNodeEmbedding
from materon_tf29_engine.core.composition_ei import CompositionSimplexEIGate

ADAPTER_TYPES = (PriorAwareNodeEmbedding, CompositionSimplexEIGate)


def find_adapter_layers(model):
    matches = []
    for layer in model.submodules:
        if isinstance(layer, ADAPTER_TYPES) and layer not in matches:
            matches.append(layer)
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one Materon adapter, found %d" % len(matches)
        )
    return matches


def find_adapter_layer(model):
    return find_adapter_layers(model)[0]


def _ref(variable):
    try:
        return variable.ref()
    except Exception:
        return id(variable)


def _class_key(layer):
    cls = layer.__class__
    return "%s.%s" % (cls.__module__, cls.__name__)


def _effective_target_layer(layer):
    if isinstance(layer, PriorAwareNodeEmbedding):
        return layer.base_layer
    return layer


def _direct_layer_pairs(source_model, target_model):
    """Pair baseline backbone layers with Materon backbone layers.

    Two supported integration geometries exist:

    Graph backbones
        ``PriorAwareNodeEmbedding`` replaces the official embedding at the
        same direct graph location.  Direct layer counts remain equal and the
        wrapper's ``base_layer`` is the effective backbone layer.

    ElemNet
        ``CompositionSimplexEIGate`` is inserted as one additional direct layer
        between the composition input and the first Dense layer.  The adapter
        is removed from the target topology before pairing; all remaining
        Input/Dense layers must then match the baseline in order and class.
    """
    source_layers = list(source_model.layers)
    target_layers_all = list(target_model.layers)

    direct_node_wrappers = [
        layer for layer in target_layers_all
        if isinstance(layer, PriorAwareNodeEmbedding)
    ]
    direct_composition_adapters = [
        layer for layer in target_layers_all
        if isinstance(layer, CompositionSimplexEIGate)
    ]
    direct_adapter_count = (
        len(direct_node_wrappers) + len(direct_composition_adapters)
    )
    if direct_adapter_count != 1:
        raise RuntimeError(
            "Expected exactly one direct Materon adapter: "
            "node_wrappers=%d composition_adapters=%d"
            % (
                len(direct_node_wrappers),
                len(direct_composition_adapters),
            )
        )

    integration_mode = (
        "node_wrapper"
        if direct_node_wrappers
        else "composition_insert"
    )

    if integration_mode == "node_wrapper":
        target_layers = target_layers_all
        if len(source_layers) != len(target_layers):
            raise RuntimeError(
                "Graph direct layer-count mismatch: baseline=%d materon=%d"
                % (len(source_layers), len(target_layers))
            )
    else:
        if len(target_layers_all) != len(source_layers) + 1:
            raise RuntimeError(
                "ElemNet direct layer-count mismatch: baseline=%d "
                "materon=%d; expected exactly one inserted adapter"
                % (len(source_layers), len(target_layers_all))
            )
        target_layers = [
            layer for layer in target_layers_all
            if not isinstance(layer, CompositionSimplexEIGate)
        ]
        if len(target_layers) != len(source_layers):
            raise RuntimeError(
                "ElemNet filtered layer-count mismatch: baseline=%d "
                "materon_backbone=%d"
                % (len(source_layers), len(target_layers))
            )

    pairs = []
    for index, (source_layer, target_outer_layer) in enumerate(
        zip(source_layers, target_layers)
    ):
        target_layer = _effective_target_layer(target_outer_layer)

        source_key = _class_key(source_layer)
        target_key = _class_key(target_layer)
        if source_key != target_key:
            raise RuntimeError(
                "Backbone layer-class mismatch at paired index %d "
                "(mode=%s): %s (%s) vs %s (%s)"
                % (
                    index,
                    integration_mode,
                    source_layer.name,
                    source_key,
                    target_outer_layer.name,
                    target_key,
                )
            )

        source_weights = list(source_layer.weights)
        target_weights = list(target_layer.weights)
        if len(source_weights) != len(target_weights):
            raise RuntimeError(
                "Local weight-count mismatch at paired layer %d: "
                "%s=%d vs %s=%d"
                % (
                    index,
                    source_layer.name,
                    len(source_weights),
                    target_outer_layer.name,
                    len(target_weights),
                )
            )

        for local_index, (source_weight, target_weight) in enumerate(
            zip(source_weights, target_weights)
        ):
            if tuple(source_weight.shape) != tuple(target_weight.shape):
                raise RuntimeError(
                    "Local weight-shape mismatch at paired layer %d "
                    "weight %d: %s %s vs %s %s"
                    % (
                        index,
                        local_index,
                        source_weight.name,
                        tuple(source_weight.shape),
                        target_weight.name,
                        tuple(target_weight.shape),
                    )
                )

        pairs.append(
            {
                "topology_index": index,
                "integration_mode": integration_mode,
                "source_outer": source_layer,
                "target_outer": target_outer_layer,
                "target_effective": target_layer,
                "source_weights": source_weights,
                "target_weights": target_weights,
            }
        )

    return pairs

def _validate_weight_coverage(source_model, target_model, pairs):
    source_expected = {_ref(v) for v in source_model.weights}
    source_paired = {
        _ref(v)
        for pair in pairs
        for v in pair["source_weights"]
    }
    if source_expected != source_paired:
        raise RuntimeError(
            "Source backbone weight coverage failed: expected=%d paired=%d "
            "missing=%d extra=%d"
            % (
                len(source_expected),
                len(source_paired),
                len(source_expected - source_paired),
                len(source_paired - source_expected),
            )
        )

    adapter = find_adapter_layer(target_model)
    adapter_only = {_ref(v) for v in adapter.adapter_only_weights}
    target_expected = {
        _ref(v) for v in target_model.weights if _ref(v) not in adapter_only
    }
    target_paired = {
        _ref(v)
        for pair in pairs
        for v in pair["target_weights"]
    }
    if target_expected != target_paired:
        raise RuntimeError(
            "Target backbone weight coverage failed: expected=%d paired=%d "
            "missing=%d extra=%d"
            % (
                len(target_expected),
                len(target_paired),
                len(target_expected - target_paired),
                len(target_paired - target_expected),
            )
        )


def target_backbone_weights(model):
    """Return target backbone weights without relying on their order."""
    adapter = find_adapter_layer(model)
    excluded = {_ref(v) for v in adapter.adapter_only_weights}
    return [v for v in model.weights if _ref(v) not in excluded]


def copy_backbone_weights(source_model, target_model):
    """Copy official backbone weights by semantic layer location."""
    pairs = _direct_layer_pairs(source_model, target_model)
    _validate_weight_coverage(source_model, target_model, pairs)

    copied = 0
    for pair in pairs:
        for source_weight, target_weight in zip(
            pair["source_weights"], pair["target_weights"]
        ):
            target_weight.assign(source_weight)
            copied += 1
    return copied


def assert_backbone_parity(source_model, target_model, atol=0.0):
    """Verify semantically corresponding backbone tensors are equal."""
    try:
        pairs = _direct_layer_pairs(source_model, target_model)
        _validate_weight_coverage(source_model, target_model, pairs)
    except Exception as exc:
        return False, {"reason": "pairing", "detail": repr(exc)}

    worst = 0.0
    worst_record = None
    tensor_count = 0
    layer_records = []

    for pair in pairs:
        local_worst = 0.0
        for local_index, (source_weight, target_weight) in enumerate(
            zip(pair["source_weights"], pair["target_weights"])
        ):
            difference = float(
                np.max(np.abs(source_weight.numpy() - target_weight.numpy()))
            )
            tensor_count += 1
            local_worst = max(local_worst, difference)
            if difference >= worst:
                worst = difference
                worst_record = {
                    "topology_index": pair["topology_index"],
                    "local_weight_index": local_index,
                    "source_layer": pair["source_outer"].name,
                    "target_layer": pair["target_outer"].name,
                    "source_weight": source_weight.name,
                    "target_weight": target_weight.name,
                    "difference": difference,
                }

        layer_records.append(
            {
                "topology_index": pair["topology_index"],
                "source_layer": pair["source_outer"].name,
                "target_layer": pair["target_outer"].name,
                "effective_target_class": _class_key(
                    pair["target_effective"]
                ),
                "weight_tensors": len(pair["source_weights"]),
                "max_abs_difference": local_worst,
            }
        )

    return worst <= atol, {
        "max_abs_difference": worst,
        "weight_tensors": tensor_count,
        "direct_layers": len(pairs),
        "worst_record": worst_record,
        "layer_records": layer_records,
    }


def _same_architecture_layer_pairs(source_model, target_model):
    """Pair two independently built models with the same full architecture.

    Unlike baseline-to-Materon pairing, no embedding unwrapping is performed:
    both models are expected to contain the same adapter architecture.
    """
    source_layers = list(source_model.layers)
    target_layers = list(target_model.layers)
    if len(source_layers) != len(target_layers):
        raise RuntimeError(
            "Full-model direct layer-count mismatch: source=%d target=%d"
            % (len(source_layers), len(target_layers))
        )

    pairs = []
    for index, (source_layer, target_layer) in enumerate(
        zip(source_layers, target_layers)
    ):
        source_key = _class_key(source_layer)
        target_key = _class_key(target_layer)
        if source_key != target_key:
            raise RuntimeError(
                "Full-model layer-class mismatch at topology index %d: "
                "%s (%s) vs %s (%s)"
                % (
                    index,
                    source_layer.name,
                    source_key,
                    target_layer.name,
                    target_key,
                )
            )

        source_weights = list(source_layer.weights)
        target_weights = list(target_layer.weights)
        if len(source_weights) != len(target_weights):
            raise RuntimeError(
                "Full-model local weight-count mismatch at layer %d: "
                "%s=%d vs %s=%d"
                % (
                    index,
                    source_layer.name,
                    len(source_weights),
                    target_layer.name,
                    len(target_weights),
                )
            )

        for local_index, (source_weight, target_weight) in enumerate(
            zip(source_weights, target_weights)
        ):
            if tuple(source_weight.shape) != tuple(target_weight.shape):
                raise RuntimeError(
                    "Full-model weight-shape mismatch at layer %d weight %d: "
                    "%s %s vs %s %s"
                    % (
                        index,
                        local_index,
                        source_weight.name,
                        tuple(source_weight.shape),
                        target_weight.name,
                        tuple(target_weight.shape),
                    )
                )

        pairs.append(
            {
                "topology_index": index,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "source_weights": source_weights,
                "target_weights": target_weights,
            }
        )

    source_expected = {_ref(v) for v in source_model.weights}
    source_paired = {
        _ref(v)
        for pair in pairs
        for v in pair["source_weights"]
    }
    target_expected = {_ref(v) for v in target_model.weights}
    target_paired = {
        _ref(v)
        for pair in pairs
        for v in pair["target_weights"]
    }
    if source_expected != source_paired:
        raise RuntimeError(
            "Full-model source coverage failed: expected=%d paired=%d"
            % (len(source_expected), len(source_paired))
        )
    if target_expected != target_paired:
        raise RuntimeError(
            "Full-model target coverage failed: expected=%d paired=%d"
            % (len(target_expected), len(target_paired))
        )
    return pairs


def assert_full_model_weight_parity(source_model, target_model, atol=0.0):
    """Compare every weight of two independently built identical models."""
    try:
        pairs = _same_architecture_layer_pairs(source_model, target_model)
    except Exception as exc:
        return False, {"reason": "pairing", "detail": repr(exc)}

    worst = 0.0
    worst_record = None
    tensor_count = 0
    layer_records = []

    for pair in pairs:
        layer_worst = 0.0
        for local_index, (source_weight, target_weight) in enumerate(
            zip(pair["source_weights"], pair["target_weights"])
        ):
            difference = float(
                np.max(
                    np.abs(
                        source_weight.numpy() - target_weight.numpy()
                    )
                )
            )
            tensor_count += 1
            layer_worst = max(layer_worst, difference)
            if worst_record is None or difference >= worst:
                worst = difference
                worst_record = {
                    "topology_index": pair["topology_index"],
                    "local_weight_index": local_index,
                    "source_layer": pair["source_layer"].name,
                    "target_layer": pair["target_layer"].name,
                    "source_weight": source_weight.name,
                    "target_weight": target_weight.name,
                    "difference": difference,
                }

        layer_records.append(
            {
                "topology_index": pair["topology_index"],
                "source_layer": pair["source_layer"].name,
                "target_layer": pair["target_layer"].name,
                "weight_tensors": len(pair["source_weights"]),
                "max_abs_difference": layer_worst,
            }
        )

    return worst <= atol, {
        "max_abs_difference": worst,
        "weight_tensors": tensor_count,
        "direct_layers": len(pairs),
        "worst_record": worst_record,
        "layer_records": layer_records,
    }


def snapshot_weights(model):
    """Return an ordered immutable snapshot for same-object reload tests."""
    return [np.array(variable.numpy(), copy=True) for variable in model.weights]


def assert_snapshot_restored(model, snapshot, atol=0.0):
    """Check that reloading restored every weight of the same model object."""
    current = list(model.weights)
    if len(current) != len(snapshot):
        return False, {
            "reason": "count",
            "current": len(current),
            "snapshot": len(snapshot),
        }

    worst = 0.0
    worst_record = None
    for index, (variable, expected) in enumerate(zip(current, snapshot)):
        if tuple(variable.shape) != tuple(expected.shape):
            return False, {
                "reason": "shape",
                "index": index,
                "weight": variable.name,
                "current": tuple(variable.shape),
                "snapshot": tuple(expected.shape),
            }
        difference = float(
            np.max(np.abs(variable.numpy() - expected))
        )
        if worst_record is None or difference >= worst:
            worst = difference
            worst_record = {
                "index": index,
                "weight": variable.name,
                "difference": difference,
            }

    return worst <= atol, {
        "max_abs_difference": worst,
        "weight_tensors": len(current),
        "worst_record": worst_record,
    }


def perturb_first_floating_weight(model, amount=0.125):
    """Deterministically perturb one trainable floating tensor before reload."""
    for variable in model.trainable_weights:
        if getattr(variable.dtype, "is_floating", False):
            original = variable.numpy()
            perturbation = np.zeros_like(original)
            perturbation.flat[0] = np.asarray(amount, dtype=original.dtype)
            variable.assign(original + perturbation)
            return {
                "weight": variable.name,
                "amount": float(amount),
            }
    raise RuntimeError("No floating trainable weight found to perturb")
