from __future__ import absolute_import, division, print_function

from dataclasses import replace

from .elemnet_performance import ModelConfig


def clone_config(base, name, input_gate=None, post_gate=None):
    return ModelConfig(
        name=str(name), input_dim=int(base.input_dim),
        hidden_units=tuple(int(v) for v in base.hidden_units),
        dropout=float(base.dropout), l2_regularization=float(base.l2_regularization),
        batch_norm=bool(base.batch_norm),
        input_gate=base.input_gate if input_gate is None else input_gate,
        post_gate=base.post_gate if post_gate is None else post_gate,
    )


def plain_additive_control(base):
    """Exact-parameter control removing only signed E/I subtraction."""
    def convert(gate):
        return None if gate is None else replace(gate, competition_sign=False)
    name = "%s__plain_additive_control" % base.name
    return clone_config(base, name, convert(base.input_gate), convert(base.post_gate))
