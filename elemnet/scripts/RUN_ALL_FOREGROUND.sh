#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
bash scripts/00_environment_preflight.sh
bash scripts/01_target_runtime_acceptance.sh
bash scripts/02_verify_or_build_cache.sh
bash scripts/03_prepare_v14_protocol.sh
bash scripts/04_run_inner_baseline_4gpu_foreground.sh
bash scripts/05_run_blend_tuning_4gpu_foreground.sh
bash scripts/06_run_formal_baseline_4gpu_foreground.sh
bash scripts/07_check_formal_baseline_integrity.sh
bash scripts/08_run_formal_selected_models_4gpu_foreground.sh
bash scripts/09_summarize_formal_results.sh
bash scripts/10_prune_and_verify_saved_models.sh
bash scripts/11_pack_slim_results_and_selected_models.sh
