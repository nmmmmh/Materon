from __future__ import absolute_import, division, print_function

from dataclasses import asdict

from .elemnet_performance import ModelConfig
from .source_aligned_gate import GateConfig


def gate_to_dict(value):
    return None if value is None else asdict(value)


def gate_from_dict(value):
    return None if value is None else GateConfig(**value)


def model_config_to_dict(value):
    return {
        "name": str(value.name),
        "input_dim": int(value.input_dim),
        "hidden_units": [int(v) for v in value.hidden_units],
        "dropout": float(value.dropout),
        "l2_regularization": float(value.l2_regularization),
        "batch_norm": bool(value.batch_norm),
        "input_gate": gate_to_dict(value.input_gate),
        "post_gate": gate_to_dict(value.post_gate),
    }


def model_config_from_dict(value):
    return ModelConfig(
        name=str(value["name"]),
        input_dim=int(value["input_dim"]),
        hidden_units=tuple(int(v) for v in value.get("hidden_units", [1024, 512, 256, 128])),
        dropout=float(value.get("dropout", 0.10)),
        l2_regularization=float(value.get("l2_regularization", 1.0e-5)),
        batch_norm=bool(value.get("batch_norm", True)),
        input_gate=gate_from_dict(value.get("input_gate")),
        post_gate=gate_from_dict(value.get("post_gate")),
    )
