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
REPORT="$SUMMARY/FORMAL_SUMMARY.json"
python - "$PLAN" "$REPORT" "$SELECTED_MODELS" <<'PYLOCAL'
import json, shutil, sys
from pathlib import Path
plan = json.load(open(sys.argv[1], encoding="utf-8"))
report = json.load(open(sys.argv[2], encoding="utf-8"))
root = Path(sys.argv[3])
baseline = plan["baseline_config"]
gate = plan["primary_gate_config"]
keep = {baseline}
if report.get("deployment_gate_pass", False) and float(plan["blend_lambda"]) > 0:
    keep.add(gate)
root.mkdir(parents=True, exist_ok=True)
for path in list(root.iterdir()):
    if path.is_dir() and path.name not in keep:
        shutil.rmtree(path)
found = {p.name for p in root.iterdir() if p.is_dir()}
if found != keep:
    raise SystemExit("MODEL RETENTION MISMATCH: found=%r keep=%r" % (sorted(found), sorted(keep)))
weights = list(root.rglob("*.weights.h5"))
expected = 15 * len(keep)
if len(weights) != expected:
    raise SystemExit("EXPECTED %d selected weight files, found %d" % (expected, len(weights)))
print("[MODEL RETENTION PASS] configs=%s weights=%d" % (sorted(keep), len(weights)))
PYLOCAL
if find "$WORK/results" -type f \( -name '*.weights.h5' -o -name '*.joblib' \) | grep -q .; then
  echo 'Unexpected disposable model/preprocessor files under workspace/results/' >&2
  find "$WORK/results" -type f \( -name '*.weights.h5' -o -name '*.joblib' \)
  exit 2
fi
