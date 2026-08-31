#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}"
STAGE="${1:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/Paper-Backup/E2-Final-256Budget}"
SEED_SPEC="${SEEDS:-42 52 62 72 82 92 102 112 122 123}"
RUN_TAG="${RUN_TIME:-e2_final_256budget}"

preprocess_data() {
  PYTHON_BIN="${PYTHON_BIN}" PAPER_BACKUP_WORKERS="${PAPER_BACKUP_WORKERS:-8}" \
    bash "${REPO_ROOT}/preprocess/paper_backup/run_preprocess_e2_final_256budget.sh" preprocess
}

validate_all() {
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${REPO_ROOT}/preprocess/paper_backup/run_preprocess_e2_final_256budget.sh" validate
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_e2_final_parameter_match.py"
  OUTPUT_ROOT="${OUTPUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    DRY_RUN=1 CHECK_DATA=1 SEEDS="${SEED_SPEC}" \
    bash "${SCRIPT_DIR}/run_e2_final_256budget.sh"
}

train_all() {
  OUTPUT_ROOT="${OUTPUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    SEEDS="${SEED_SPEC}" RUN_TIME="${RUN_TAG}" \
    bash "${SCRIPT_DIR}/run_e2_final_256budget.sh"
}

summarize_all() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --experiment e2_final_256budget \
    --seeds "${SEED_SPEC}" \
    --root "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/summaries"
}

case "${STAGE}" in
  preprocess) preprocess_data ;;
  validate) validate_all ;;
  train) validate_all; train_all ;;
  summary) summarize_all ;;
  all) preprocess_data; validate_all; train_all; summarize_all ;;
  *)
    echo "Usage: $0 {preprocess|validate|train|summary|all}" >&2
    exit 2
    ;;
esac
