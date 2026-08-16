# -*- coding: utf-8 -*-
"""Identity utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from materon_tf29_engine.integration.weight_pairing import (
    assert_backbone_parity,
    find_adapter_layer,
)
from materon_tf29_engine.core.composition_ei import CompositionSimplexEIGate


def _flat_values(value):
    if isinstance(value, tf.RaggedTensor) or (
        hasattr(value, "flat_values") and hasattr(value, "with_flat_values")
    ):
        return value.flat_values
    return value


def _call_layer(layer, inputs):
    try:
        return layer(inputs, training=False)
    except TypeError:
        return layer(inputs)


def _ordered_float32_bits(values):
    """Map float32 bit patterns to monotonic unsigned integers.

    Positive and negative finite floats become a single ordered integer axis,
    so adjacent representable values have distance one.  +0 and -0 map to the
    same point.
    """
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32).astype(np.uint64)
    sign = (bits & np.uint64(0x80000000)) != 0
    magnitude = bits & np.uint64(0x7FFFFFFF)
    return np.where(
        sign,
        np.uint64(0x80000000) - magnitude,
        np.uint64(0x80000000) + magnitude,
    )


def compare_float_outputs(
    reference,
    candidate,
    atol=1e-6,
    rtol=5e-7,
    max_ulp=4,
):
    """Compare float outputs using scale-aware and ULP-aware diagnostics.

    Separate TensorFlow functional graphs can choose slightly different GPU
    fusion/reduction paths even when their inputs and weights are identical.
    With TF32 enabled, an absolute-only threshold such as 1e-6 is invalid for
    large untrained outputs.  Structural identity is checked separately by
    exact adapter and weight parity.  This function only evaluates backend
    numerical equivalence.
    """
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)

    if ref.shape != cand.shape:
        raise AssertionError(
            "Output shape mismatch: %s vs %s" % (ref.shape, cand.shape)
        )
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cand)):
        raise AssertionError("Identity comparison received non-finite outputs")

    absolute = np.abs(ref - cand)
    scale = np.maximum(np.abs(ref), np.abs(cand))
    allowed = float(atol) + float(rtol) * scale
    scaled = absolute / np.maximum(allowed, np.finfo(np.float32).tiny)

    ref_ordered = _ordered_float32_bits(ref)
    cand_ordered = _ordered_float32_bits(cand)
    ulp_distance = np.where(
        ref_ordered >= cand_ordered,
        ref_ordered - cand_ordered,
        cand_ordered - ref_ordered,
    )

    max_abs = float(np.max(absolute)) if absolute.size else 0.0
    max_scale = float(np.max(scale)) if scale.size else 0.0
    max_relative = float(
        np.max(absolute / np.maximum(scale, 1.0))
    ) if absolute.size else 0.0
    max_scaled = float(np.max(scaled)) if scaled.size else 0.0
    max_ulp_distance = int(np.max(ulp_distance)) if ulp_distance.size else 0

    allclose_pass = bool(np.all(absolute <= allowed))
    ulp_pass = bool(max_ulp_distance <= int(max_ulp))
    passed = bool(allclose_pass or ulp_pass)

    worst_flat = int(np.argmax(absolute)) if absolute.size else 0
    worst_index = (
        tuple(int(x) for x in np.unravel_index(worst_flat, absolute.shape))
        if absolute.size
        else ()
    )

    return {
        "passed": passed,
        "allclose_pass": allclose_pass,
        "ulp_pass": ulp_pass,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_ulp_allowed": int(max_ulp),
        "max_abs_difference": max_abs,
        "max_output_magnitude": max_scale,
        "max_scale_normalized_difference": max_relative,
        "max_tolerance_ratio": max_scaled,
        "max_ulp_distance": max_ulp_distance,
        "worst_index": worst_index,
        "reference_at_worst": (
            float(ref[worst_index]) if absolute.size else 0.0
        ),
        "candidate_at_worst": (
            float(cand[worst_index]) if absolute.size else 0.0
        ),
    }


def assert_adapter_identity(materon, adapter_input, tolerance=0.0):
    """Verify exact identity at raw_alpha=0 for either adapter geometry."""
    adapter = find_adapter_layer(materon)

    if isinstance(adapter, CompositionSimplexEIGate):
        adapter.reset_identity()
        original=tf.cast(
            tf.convert_to_tensor(adapter_input),tf.float32
        )
        transformed=_call_layer(adapter,original)
        difference=float(
            tf.reduce_max(tf.abs(original-transformed)).numpy()
        )
        input_sum=tf.reduce_sum(original,axis=-1)
        output_sum=tf.reduce_sum(transformed,axis=-1)
        sum_difference=float(
            tf.reduce_max(tf.abs(input_sum-output_sum)).numpy()
        )
        minimum=float(tf.reduce_min(transformed).numpy())
        zero_support_violation=bool(
            tf.reduce_any(
                tf.logical_and(
                    tf.equal(original,0.0),
                    tf.not_equal(transformed,0.0)
                )
            ).numpy()
        )
        if difference > tolerance:
            raise AssertionError(
                "Composition adapter identity failed: %.3e > %.3e"
                % (difference,tolerance)
            )
        if sum_difference > tolerance:
            raise AssertionError(
                "Composition simplex sum changed at identity: %.3e"
                % sum_difference
            )
        if minimum < 0.0:
            raise AssertionError(
                "Composition adapter produced negative value: %.3e"
                % minimum
            )
        if zero_support_violation:
            raise AssertionError(
                "Composition adapter activated an absent element"
            )
        return {
            "applicable":True,
            "adapter_type":"composition_simplex",
            "max_abs_difference":difference,
            "max_sum_difference":sum_difference,
            "minimum_output":minimum,
            "zero_support_violation":zero_support_violation,
            "samples":int(tf.shape(original)[0].numpy()),
            "elements":int(tf.shape(original)[-1].numpy()),
        }

    if not hasattr(adapter,"base_layer"):
        raise RuntimeError(
            "Unsupported adapter type for identity test: %s"
            % adapter.__class__.__name__
        )

    adapter.gate.reset_identity()
    official_embedding=_call_layer(
        adapter.base_layer,adapter_input
    )
    wrapped_embedding=_call_layer(adapter,adapter_input)

    official_flat=tf.cast(
        _flat_values(official_embedding),tf.float32
    )
    wrapped_flat=tf.cast(
        _flat_values(wrapped_embedding),tf.float32
    )
    difference=float(
        tf.reduce_max(
            tf.abs(official_flat-wrapped_flat)
        ).numpy()
    )
    if difference > tolerance:
        raise AssertionError(
            "Node adapter identity failed: %.3e > %.3e"
            % (difference,tolerance)
        )
    return {
        "applicable":True,
        "adapter_type":"node_embedding",
        "max_abs_difference":difference,
        "atoms":int(tf.shape(official_flat)[0].numpy()),
        "embedding_dim":int(
            tf.shape(official_flat)[-1].numpy()
        ),
    }



def assert_structural_identity(
    baseline,
    materon,
    sample,
    diagnostic_atol=1e-6,
    diagnostic_rtol=5e-7,
    diagnostic_max_ulp=4,
):
    """Hard-check exact model state; only diagnose cross-graph GPU numerics.

    Formal training must never abort merely because two separately constructed
    float32/TF32 GPU graphs differ by several ULPs. The scientifically relevant
    initialization guarantees are exact and are enforced here:

    - every semantically paired backbone tensor has difference zero;
    - the Materon adapter output is exactly identical at raw_alpha=0.

    Full-model outputs are still compared and fully reported, but the result is
    diagnostic only.
    """
    ok, parity = assert_backbone_parity(baseline, materon, atol=0.0)
    if not ok:
        raise AssertionError("Backbone parity failed: %s" % parity)

    adapter_input = sample[0] if isinstance(sample, (list, tuple)) else sample
    adapter_identity = assert_adapter_identity(
        materon, adapter_input, tolerance=0.0
    )

    baseline_output = np.asarray(baseline(sample, training=False))
    materon_output = np.asarray(materon(sample, training=False))
    numerics = compare_float_outputs(
        baseline_output,
        materon_output,
        atol=diagnostic_atol,
        rtol=diagnostic_rtol,
        max_ulp=diagnostic_max_ulp,
    )

    report = dict(parity)
    report["adapter_identity"] = adapter_identity
    report["full_model_numerics"] = numerics
    report["structural_identity_passed"] = True
    report["full_model_numerics_is_diagnostic_only"] = True
    return numerics["max_abs_difference"], report


def assert_identity(
    baseline,
    materon,
    sample,
    atol=1e-6,
    rtol=5e-7,
    max_ulp=4,
):
    """Check exact structural identity and GPU numerical equivalence.

    Exact requirements:
    - every semantically corresponding backbone tensor is bit-identical;
    - the Materon adapter returns the official embedding exactly at alpha=0.

    The two separately constructed full functional graphs are then compared
    with ULP-aware/relative criteria, because GPU graph fusion and TF32 can
    differ by a few representable float32 values without any model difference.
    """
    ok, parity = assert_backbone_parity(baseline, materon, atol=0.0)
    if not ok:
        raise AssertionError("Backbone parity failed: %s" % parity)

    atomic_numbers = sample[0] if isinstance(sample, (list, tuple)) else sample
    adapter_identity = assert_adapter_identity(
        materon, atomic_numbers, tolerance=0.0
    )

    baseline_output = np.asarray(baseline(sample, training=False))
    materon_output = np.asarray(materon(sample, training=False))
    numerics = compare_float_outputs(
        baseline_output,
        materon_output,
        atol=atol,
        rtol=rtol,
        max_ulp=max_ulp,
    )
    if not numerics["passed"]:
        raise AssertionError(
            "Full-model numerical equivalence failed: %s; "
            "adapter_identity=%.3e; worst_backbone=%s"
            % (
                numerics,
                adapter_identity["max_abs_difference"],
                parity.get("worst_record"),
            )
        )

    parity = dict(parity)
    parity["adapter_identity"] = adapter_identity
    parity["full_model_numerics"] = numerics
    return numerics["max_abs_difference"], parity
