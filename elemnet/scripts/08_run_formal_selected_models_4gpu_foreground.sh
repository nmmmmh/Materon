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
PLAN="$TUNING/SELECTED_BASELINE_ANCHORED_MODEL.json"
GATE="$FORMAL/BASELINE_INTEGRITY.json"
python - "$PLAN" "$GATE" <<'PYLOCAL'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
g = json.load(open(sys.argv[2], encoding="utf-8"))
assert p.get("all_pass"), p
assert g.get("all_pass"), g
assert len(p["save_model_configs"]) == 1, p
print("[SELECTED GATE]", p["primary_gate_config"])
print("[BLEND LAMBDA ]", p["blend_lambda"])
print("[CONTROL      ]", p["mechanism_control_config"])
PYLOCAL
python -m materon_elemnet_tf29.training.launch_resident \
  --mode plan_formal --cache-dir "$CACHE" --outdir "$FORMAL" \
  --initial-weights-dir "$INITIAL" --acceptance-report "$ACCEPTANCE" \
  --protocol-json "$PROTOCOL" --plan-json "$PLAN" \
  --selected-models-dir "$SELECTED_MODELS" \
  --folds 0 1 2 3 4 --seeds 42 123 2024 \
  --gpu-ids 0,1,2,3 --verbose 0
