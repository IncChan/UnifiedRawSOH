#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This entry point intentionally runs only E2-FULL-D w/o cycle auxiliary.
export DIAGNOSTICS="e2_full_d_no_cycle_aux"
export SEEDS="${SEEDS:-42 52 62}"
export SUMMARY_SEEDS="${SUMMARY_SEEDS:-42 52 62}"
export GPU_IDS="${GPU_IDS:-1}"
export MAX_PARALLEL="${MAX_PARALLEL:-3}"

bash "${SCRIPT_DIR}/run_e2_diagnostics.sh"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
  cd "${PROJECT_ROOT}"
  PYTHON_BIN="$("${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh")"
  "${PYTHON_BIN}" -m UnifiedRawSOH.evaluation.paper_v1.compare_diagnostics \
    --baseline_root UnifiedRawSOH/outputs/Paper-v1/v1_diagnostics/e2_full_d \
    --ablation_root UnifiedRawSOH/outputs/Paper-v1/v1_diagnostics/e2_full_d_no_cycle_aux \
    --output_root UnifiedRawSOH/outputs/Paper-v1/v1_diagnostics/e2_full_d_no_cycle_aux
fi
