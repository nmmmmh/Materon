from __future__ import absolute_import, division, print_function

from typing import Dict, Optional, Sequence, Tuple

from .elemnet_performance import ModelConfig
from .source_aligned_gate import GateConfig


def _hidden(values: Optional[Sequence[int]]) -> Tuple[int, ...]:
    return tuple(int(v) for v in (values or (1024, 512, 256, 128)))


def _gate(reduced_dim=32, dropout=0.10, shared=True, low_rank=False,
          rank=8, dimension=True, dimension_type="soft", synergy=0.10,
          trust_ratio=0.0):
    return GateConfig(
        reduced_dim=int(reduced_dim),
        comb_dropout=float(dropout),
        use_shared_expert=bool(shared),
        low_rank_gate=bool(low_rank),
        gate_low_rank_dim=int(rank),
        use_dimension_gating=bool(dimension),
        dimension_gating_type=str(dimension_type),
        synergy_init=float(synergy),
        dimension_parameterization="historical_sigmoid",
        dimension_neutral_weight=0.0,
        modulation_trust_ratio=float(trust_ratio),
        use_competition_confidence=False,
        competition_confidence_floor=0.50,
        competition_confidence_power=1.0,
        l2_regularization=1.0e-5,
    )


def _common(input_dim, hidden_units=None):
    return dict(
        input_dim=int(input_dim),
        hidden_units=_hidden(hidden_units),
        dropout=0.10,
        l2_regularization=1.0e-5,
        batch_norm=True,
    )


def baseline_registry(input_dim: int, hidden_units=None) -> Dict[str, ModelConfig]:
    return {
        "H0_exact_baseline": ModelConfig(
            name="H0_exact_baseline", **_common(input_dim, hidden_units)
        )
    }


def v14_candidate_registry(input_dim: int, hidden_units=None) -> Dict[str, ModelConfig]:
    """Small evidence-based pool for baseline-anchored blending.

    V14 deliberately does not perform a broad hyperparameter sweep. Each
    candidate represents a distinct behavior already worth testing:
    compact historical input+post, strong regularized input+post, low-dropout
    input+post, trust-region input+post, and a post-free input-only fallback.
    """
    common = _common(input_dim, hidden_units)
    values = {}

    def add(name, input_gate, post_gate):
        values[name] = ModelConfig(
            name=name, input_gate=input_gate, post_gate=post_gate, **common
        )

    add("V14_G0_H5_R32_D0p10", _gate(32, 0.10), _gate(32, 0.10))
    add("V14_G1_I64P64_D0p10", _gate(64, 0.10), _gate(64, 0.10))
    add("V14_G2_I64P64_PD0p025", _gate(64, 0.0), _gate(64, 0.025))
    add(
        "V14_G3_I64P64_PD0p025_TR0p10",
        _gate(64, 0.0),
        _gate(64, 0.025, trust_ratio=0.10),
    )
    add(
        "V14_G4_INPUT_ONLY_FULL_R32_D0p10",
        _gate(32, 0.10, low_rank=True),
        None,
    )
    return values


def get_registry(stage: str, input_dim: int, hidden_units=None) -> Dict[str, ModelConfig]:
    if stage == "baseline":
        return baseline_registry(input_dim, hidden_units)
    if stage in ("v14_candidates", "gates"):
        return v14_candidate_registry(input_dim, hidden_units)
    if stage == "all":
        values = baseline_registry(input_dim, hidden_units)
        values.update(v14_candidate_registry(input_dim, hidden_units))
        return values
    raise KeyError("Unknown stage=%r" % stage)
