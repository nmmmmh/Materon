from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent))
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(handle.name, str(path))
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def read_json(path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_file_hashes(root, exclude_manifest: bool = True) -> Dict[str, str]:
    root = Path(root)
    output = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if exclude_manifest and relative == "manifest.json":
            continue
        output[relative] = sha256_file(path)
    return output
