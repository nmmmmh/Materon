#!/usr/bin/env python3
"""Fixed composition-group-aware 70/15/15 splits of HOIP and JDFT2D (seed 42).

JDFT2D structures are read from the MEGNet Matbench cache (scripts/megnet/build_caches.py).
Writes <output-root>/splits/<dataset>_split_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hoip-path", required=True, help="CoRE 2D-HOIP band-gap JSON.GZ (499 structures)")
    p.add_argument("--megnet-cache-root", required=True)
    p.add_argument("--output-root", required=True)
    a = p.parse_args()

    from materon.fixed_split.config import load_protocol
    from materon.fixed_split.datasets import bundle_manifest, load_from_megnet_cache, load_hoip
    from materon.fixed_split.splits import make_group_aware_split, write_split

    protocol = load_protocol()
    out = Path(a.output_root)
    hoip_cfg = protocol["datasets"]["hoip_bandgap"]
    jd_cfg = protocol["datasets"]["jdft2d_exfoliation"]
    bundles = {"hoip_bandgap": load_hoip(a.hoip_path, int(hoip_cfg["expected_n"])),
               "jdft2d_exfoliation": load_from_megnet_cache(a.megnet_cache_root, jd_cfg["reuse_megnet_task"],
                                                            "jdft2d_exfoliation", jd_cfg["unit"])}
    for name, bundle in bundles.items():
        frame = bundle_manifest(bundle)
        if len(frame) != int(protocol["datasets"][name]["expected_n"]):
            raise RuntimeError(f"Dataset size mismatch for {name}")
        split = make_group_aware_split(frame, protocol["split"])
        split.pop("content_sha256", None)
        split.update({"protocol_id": protocol["protocol_id"], "dataset": name, "unit": bundle.unit,
                      "source_sha256": bundle.source_sha256})
        # Split identity used by the training and interpretation records.
        split["content_sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        write_split(out / "splits" / f"{name}_split_manifest.json", split)
        print(f"[SPLIT] {name} {split['counts']} groups={frame['group'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
