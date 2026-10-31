"""Per-attribute summary of the CGCNN input-gate readouts on a test sample."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def flat(value):
    return value.flat_values if isinstance(value, tf.RaggedTensor) else value


def mean_abs(value):
    return np.mean(np.abs(np.asarray(flat(value).numpy(), dtype=np.float32)), axis=0).tolist()


def _feature_count(gate):
    for key in ["actual_total_delta", "output_features", "raw_ei_delta"]:
        if key in gate:
            return int(flat(gate[key]).shape[-1])
    return 0


def export_analysis_summary(model, sample, feature_names, output_path):
    """Write mean absolute gate readouts (e.g. actual_total_delta) per attribute to JSON."""
    try:
        prediction, analysis = model(sample, training=False, return_analysis=True)
    except TypeError:
        return None
    payload = {"prediction_shape": list(prediction.shape), "input_feature_names": list(feature_names),
               "context_mode": analysis.get("context_mode", "none")}
    raw = analysis.get("raw_physical_features")
    if raw is not None:
        payload["raw_physical_mean_abs"] = mean_abs(raw)
    gate = analysis.get("input_gate")
    if gate:
        count = _feature_count(gate)
        names = list(feature_names) if len(feature_names) == count else [f"feature_{i:03d}" for i in range(count)]
        item = {"feature_names": names}
        for key in ["actual_total_delta", "unmasked_total_delta", "applied_ei_delta", "raw_ei_delta",
                    "promotion_features", "inhibition_features", "relative_actual_modulation", "attention_scores",
                    "gate_weights", "feature_gate_weights", "dimension_delta"]:
            if key in gate:
                item[key + "_mean_abs"] = mean_abs(gate[key])
        for key in ["synergy", "alpha"]:
            if key in gate:
                item[key] = float(tf.keras.backend.get_value(gate[key]))
        for key in ["coordination_used", "q_eff_used", "emag_used", "legacy_sigmoid_semantics"]:
            if key in gate:
                item[key] = bool(gate[key])
        payload["input_gate"] = item
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
