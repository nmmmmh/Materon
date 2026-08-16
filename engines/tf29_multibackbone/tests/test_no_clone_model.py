"""Contract and regression checks for test no clone model in the TF2.9 multi-backbone engine."""

from pathlib import Path

def test_no_clone_model():
    root=Path(__file__).resolve().parents[1]/"materon"
    text="\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py"))
    assert "clone_model(" not in text
    assert "model_surgery" not in text
