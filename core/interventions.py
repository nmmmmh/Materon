"""Safe temporary intervention helpers for EIGate explainability experiments."""
from __future__ import annotations
from contextlib import contextmanager


@contextmanager
def temporary_intervention(gate, feature_mask=None, pathway_state=None):
    """Apply inference-only EIGate interventions and always restore the default state."""
    try:
        if feature_mask is not None:
            gate.set_feature_mask(feature_mask)
        if pathway_state is not None:
            gate.set_pathway_state(**dict(pathway_state))
        yield gate
    finally:
        gate.reset_feature_mask()
        try:
            gate.reset_pathway_state()
        except RuntimeError:
            pass
