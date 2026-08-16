"""Frozen named configurations for the canonical Materon EIGate."""
from __future__ import annotations

EIGATE_S1_L0_D1 = {
    "feature_dim": 20,
    "reduced_dim": 8,
    "comb_dropout": 0.01,
    "use_shared_expert": True,
    "low_rank_gate": False,
    "gate_low_rank_dim": 16,
    "use_dimension_gating": True,
    "dimension_gating_type": "soft",
    "dimension_sparsity_weight": 0.0,
    "synergy_init": 0.1,
    "synergy_min": 0.005,
    "synergy_max": 0.5,
    "l2_reg": 0.0,
    "gate_seed": 42,
}


def make_eigate_config(**overrides):
    """Return a copy of the frozen S1-L0-D1 configuration with explicit overrides."""
    config = dict(EIGATE_S1_L0_D1)
    config.update(overrides)
    return config
