"""Fixed Matbench MEGNet protocol (configs/megnet/matbench_protocol.json)."""
from __future__ import annotations

import json

from materon.attributes import config_dir


def load_protocol() -> dict:
    return json.loads((config_dir() / "megnet" / "matbench_protocol.json").read_text(encoding="utf-8"))
