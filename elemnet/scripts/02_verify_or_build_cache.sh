#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$PROJECT/workspace"
CACHE="$WORK/cache/matbench_steels_v14"
ACCEPTANCE="$WORK/acceptance/TARGET_RUNTIME_ACCEPTANCE.json"
PROTOCOL="$WORK/results/protocol/SELECTED_PROTOCOL.json"
INNER_BASELINE="$WORK/results/inner_baseline"
TUNING="$WORK/results/blend_tuning"
FORMAL="$WORK/results/formal"
SUMMARY="$WORK/results/formal_summary"
INITIAL="$WORK/initial_weights/backbone"
SELECTED_MODELS="$WORK/selected_models"
ARTIFACTS="$PROJECT/artifacts"
mkdir -p "$WORK" "$ARTIFACTS"
if [[ "${CONDA_DEFAULT_ENV:-}" != "kgcnn_old" ]]; then
  source /usr/local/anaconda3/etc/profile.d/conda.sh
  conda activate kgcnn_old
fi
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
if [[ -f "$CACHE/manifest.json" ]]; then
  python - "$CACHE" <<'PYLOCAL'
import sys
from materon_elemnet_tf29.data.cache import CachedDataset
cache = CachedDataset.load(sys.argv[1], verify_hashes=True)
assert cache.manifest.get("formal_official_folds") is True, cache.manifest
assert cache.feature_view("legacy_engineered").shape == (312, 258), cache.feature_view("legacy_engineered").shape
print("[LOCAL CACHE REUSE PASS]", cache.root)
PYLOCAL
else
  python -m materon_elemnet_tf29.data.cache_builder \
    --source matbench \
    --cache-dir "$CACHE" \
    --include-engineered \
    --max-atomic-number 118 \
    --seeds 42 123 2024 \
    --val-fraction 0.15
fi
