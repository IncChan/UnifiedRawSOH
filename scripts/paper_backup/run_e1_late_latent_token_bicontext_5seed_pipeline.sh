#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}"
STAGE="${1:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/Paper-Backup/E1-Late-LatentToken-BiContext-5Seed}"
RAW_DUAL_ROOT="${RAW_DUAL_ROOT:-${REPO_ROOT}/outputs/Paper-Backup/E1-Final-Interaction-5Seed}"
MEAN_BICONTEXT_ROOT="${MEAN_BICONTEXT_ROOT:-${REPO_ROOT}/outputs/Paper-Backup/E1-BiContext-5Seed}"
SEED_SPEC="${SEEDS:-42 52 62 72 82}"
RUN_TAG="${RUN_TIME:-e1_late_latent_token_bicontext_5seed_$(date +%Y%m%dT%H%M%S)}"

validate_all() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_late_latent_token_bicontext_model.py"
  OUTPUT_ROOT="${OUTPUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    DRY_RUN=1 CHECK_DATA=1 SEEDS="${SEED_SPEC}" \
    bash "${SCRIPT_DIR}/run_e1_late_latent_token_bicontext_5seed.sh"
}

train_all() {
  OUTPUT_ROOT="${OUTPUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    SEEDS="${SEED_SPEC}" RUN_TIME="${RUN_TAG}" \
    bash "${SCRIPT_DIR}/run_e1_late_latent_token_bicontext_5seed.sh"
}

summarize_all() {
  local summary_root="${OUTPUT_ROOT}/summaries"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --experiment e1_late_latent_token_bicontext_5seed \
    --seeds "${SEED_SPEC}" --root "${OUTPUT_ROOT}" --output-dir "${summary_root}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/render_late_latent_token_bicontext_tables.py" \
    --new-summary-dir "${summary_root}" \
    --raw-dual-summary-dir "${RAW_DUAL_ROOT}/summaries" \
    --mean-bicontext-summary-dir "${MEAN_BICONTEXT_ROOT}/summaries"
}

diagnose_all() {
  local summary_root="${OUTPUT_ROOT}/summaries"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_latent_token_bicontext_diagnostics.py" \
    --root "${OUTPUT_ROOT}" --output-dir "${summary_root}" \
    --seeds "${SEED_SPEC}" --device "${DEVICE_OVERRIDE:-cuda:0}"
}

case "${STAGE}" in
  validate) validate_all ;;
  train) validate_all; train_all ;;
  summary) summarize_all ;;
  diagnostics) diagnose_all ;;
  all) validate_all; train_all; summarize_all ;;
  *) echo "Usage: $0 {validate|train|summary|diagnostics|all}" >&2; exit 2 ;;
esac
