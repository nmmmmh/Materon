"""Contract and regression checks for test elemnet semantic pairing static in the TF2.9 multi-backbone engine."""

from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_elemnet_inserted_adapter_is_supported():
    text=(ROOT/"materon"/"integration"/"weight_pairing.py").read_text(
        encoding="utf-8"
    )
    assert "composition_insert" in text
    assert "len(target_layers_all) != len(source_layers) + 1" in text
    assert "CompositionSimplexEIGate" in text
    assert "target_layers = [" in text


def test_composition_identity_correction_is_present():
    text=(ROOT/"materon"/"core"/"composition_ei.py").read_text(
        encoding="utf-8"
    )
    assert "base_normalized=c/denominator" in text
    assert "y=c+(transformed-base_normalized)" in text
    assert "simplex_input_sum" in text
    assert "simplex_output_sum" in text


def test_composition_identity_is_hard_checked():
    text=(ROOT/"materon"/"training"/"identity.py").read_text(
        encoding="utf-8"
    )
    assert "CompositionSimplexEIGate" in text
    assert "Composition adapter identity failed" in text
    assert "zero_support_violation" in text
