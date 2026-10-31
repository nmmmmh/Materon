"""Protocol of the fixed-split 2D study (configs/fixed_split/protocol.json)."""
from __future__ import annotations

import json
from pathlib import Path

from materon.attributes import config_dir


def protocol_path() -> Path:
    return config_dir() / "fixed_split" / "protocol.json"


def load_protocol(path=None) -> dict:
    data = json.loads(Path(path or protocol_path()).read_text(encoding="utf-8"))
    split = data["split"]
    total = sum(float(split[k]) for k in ("train_fraction", "validation_fraction", "test_fraction"))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to one, got {total}")
    return data
