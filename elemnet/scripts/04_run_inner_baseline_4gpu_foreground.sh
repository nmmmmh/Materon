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
python -m materon_elemnet_tf29.training.launch_resident \
  --mode config_screen --cache-dir "$CACHE" --outdir "$INNER_BASELINE" \
  --initial-weights-dir "$INITIAL" --acceptance-report "$ACCEPTANCE" \
  --protocol-json "$PROTOCOL" --stage baseline --configs H0_exact_baseline \
  --screen-seeds 42 123 2024 --folds 0 1 2 3 4 \
  --gpu-ids 0,1,2,3 --verbose 0
