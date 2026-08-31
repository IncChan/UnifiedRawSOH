#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}"
STAGE="${1:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/Paper-Backup/CRateV2-128x128}"

preprocess_data() {
  PYTHON_BIN="${PYTHON_BIN}" bash "${REPO_ROOT}/preprocess/paper_backup/run_preprocess_128x128.sh" terminal
}

validate_data() {
  PYTHON_BIN="${PYTHON_BIN}" bash "${REPO_ROOT}/preprocess/paper_backup/run_preprocess_128x128.sh" validate
}

train_models() {
  OUTPUT_ROOT="${OUTPUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/run_e1_crate_128x128.sh"
}

summarize_results() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --experiment e1_crate_128x128 \
    --seeds "${SEEDS:-42 52 62}" \
    --root "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/summaries"
}

case "${STAGE}" in
  preprocess)
    preprocess_data
    validate_data
    ;;
  validate)
    validate_data
    ;;
  train)
    validate_data
    train_models
    ;;
  summary)
    summarize_results
    ;;
  all)
    preprocess_data
    validate_data
    train_models
    summarize_results
    ;;
  *)
    echo "Usage: $0 {preprocess|validate|train|summary|all}" >&2
    exit 2
    ;;
esac
