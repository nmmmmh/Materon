"""Contract and regression checks for test semantic weight pairing static in the TF2.9 multi-backbone engine."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_global_positional_copy_removed():
    text = (
        ROOT
        / "materon"
        / "integration"
        / "weight_pairing.py"
    ).read_text(encoding="utf-8")
    assert "zip(src,dst)" not in text.replace(" ", "")
    assert "_direct_layer_pairs" in text
    assert "target_effective" in text
    assert "source_outer" in text


def test_identity_checks_adapter_separately():
    text = (
        ROOT
        / "materon"
        / "training"
        / "identity.py"
    ).read_text(encoding="utf-8")
    assert "assert_adapter_identity" in text
    assert "adapter.base_layer" in text
    assert "baseline(sample, training=False)" in text
    assert "materon(sample, training=False)" in text
