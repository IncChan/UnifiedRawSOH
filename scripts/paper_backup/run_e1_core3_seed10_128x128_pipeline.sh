#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}"
STAGE="${1:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/Paper-Backup/CRateV2-128x128}"
ALL_SEEDS="${SEEDS:-42 52 62 72 82 92 102 112 122 123}"
ADDITIONAL_SEQUENCE_SEEDS="${ADDITIONAL_SEQUENCE_SEEDS:-72 82 92 102 112 122 123}"
RUN_TAG="${RUN_TIME:-e1_core3_seed10_128x128}"

validate_data() {
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${REPO_ROOT}/preprocess/paper_backup/run_preprocess_128x128.sh" validate
}

train_missing_runs() {
  # The feature MLP has no runs in the original 128x128 suite, so it needs all
  # ten seeds. Ours and Smaller Transformer already have 42/52/62 and only
  # need the seven additional seeds.
  OUTPUT_ROOT="${OUTPUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    MODELS="hi_mlp" SEEDS="${ALL_SEEDS}" RUN_TIME="${RUN_TAG}_hi_mlp" \
    bash "${SCRIPT_DIR}/run_e1_core3_128x128.sh"

  OUTPUT_ROOT="${OUTPUT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    MODELS="smaller_transformer ours_pointbridge" \
    SEEDS="${ADDITIONAL_SEQUENCE_SEEDS}" \
    RUN_TIME="${RUN_TAG}_sequence_new_seeds" \
    bash "${SCRIPT_DIR}/run_e1_core3_128x128.sh"
}

summarize_results() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
    --experiment e1_core3_128x128 \
    --seeds "${ALL_SEEDS}" \
    --root "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/summaries"
}

case "${STAGE}" in
  validate)
    validate_data
    ;;
  train)
    validate_data
    train_missing_runs
    ;;
  summary)
    summarize_results
    ;;
  all)
    validate_data
    train_missing_runs
    summarize_results
    ;;
  *)
    echo "Usage: $0 {validate|train|summary|all}" >&2
    exit 2
    ;;
esac
