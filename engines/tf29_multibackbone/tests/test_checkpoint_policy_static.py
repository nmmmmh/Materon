"""Contract and regression checks for test checkpoint policy static in the TF2.9 multi-backbone engine."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_uses_exact_weight_parity():
    text = (ROOT / "scripts" / "acceptance_test.py").read_text(
        encoding="utf-8"
    )
    assert "assert_snapshot_restored" in text
    assert "assert_full_model_weight_parity" in text
    assert "perturb_first_floating_weight" in text
    assert 'atol=0.0' in text
    assert "checkpoint_cross_graph_output_diagnostic" in text


def test_production_identity_api_is_current():
    text = (ROOT / "scripts" / "run_seed.py").read_text(
        encoding="utf-8"
    )
    assert "--identity-atol" in text
    assert "--identity-rtol" in text
    assert "--identity-max-ulp" in text
    assert "tolerance=args.identity_tolerance" not in text
    assert "identity_report" in text


def test_smoke_identity_api_is_current():
    text = (ROOT / "scripts" / "smoke_test_identity.py").read_text(
        encoding="utf-8"
    )
    assert "--max-ulp" in text
    assert "max_ulp=int(a.max_ulp)" in text
