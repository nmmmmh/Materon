#!/usr/bin/env bash
# Materon workflow: TF2.9 multi-backbone.
# Purpose: preflight all
set -euo pipefail
GPU="${1:-0}"
python scripts/check_environment.py
python scripts/inspect_kgcnn_contract.py
CUDA_VISIBLE_DEVICES="$GPU" python scripts/acceptance_test.py --backbones megnet,cgcnn,elemnet
echo "MATERON PREFLIGHT: PASS"
