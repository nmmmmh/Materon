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
STAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$ARTIFACTS/Materon_ElemNet_V14_slim_results_${STAMP}.tar.gz"
MODELS="$ARTIFACTS/Materon_ElemNet_V14_selected_models_${STAMP}.tar.gz"
tar -C "$PROJECT" \
  --exclude='*.weights.h5' --exclude='*.joblib' --exclude='*_resident_worker_state/*' \
  -czf "$RESULTS" workspace/results workspace/acceptance
tar -C "$PROJECT" -czf "$MODELS" workspace/selected_models
sha256sum "$RESULTS" "$MODELS" | tee "$RESULTS.sha256"
tar -tzf "$RESULTS" >/dev/null
tar -tzf "$MODELS" >/dev/null
echo '[ARCHIVE PASS]'
ls -lh "$RESULTS" "$MODELS" "$RESULTS.sha256"
