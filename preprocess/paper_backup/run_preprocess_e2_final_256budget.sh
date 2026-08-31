#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CALLER_PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -f "${REPO_ROOT}/preprocess/paths.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/preprocess/paths.env"
fi
PYTHON_BIN="${CALLER_PYTHON_BIN:-${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}}"
OUTPUT_ROOT="${PAPER_BACKUP_E2_PREPROCESSED_ROOT:-${REPO_ROOT}/datasets/PaperBackup_preprocessed_v2_e2_final}"
MODE="${1:-preprocess}"

case "${MODE}" in
  preprocess)
    : "${XJTU_SOURCE_ROOT:?Configure XJTU_SOURCE_ROOT in preprocess/paths.env}"
    : "${SMARTHEALTH_SOURCE_ROOT:?Configure SMARTHEALTH_SOURCE_ROOT in preprocess/paths.env}"
    args=(
      --domains xjtu smarthealth_lishen40 smarthealth_catl280
      --include-full xjtu smarthealth_lishen40 smarthealth_catl280
      --output-root "${OUTPUT_ROOT}"
      --schema-version 2
      --cc-len 128
      --cv-len 128
      --full-joint-len 256
      --workers "${PAPER_BACKUP_WORKERS:-8}"
      --xjtu-full-source-root "${XJTU_SOURCE_ROOT}"
      --smarthealth-full-source-root "${SMARTHEALTH_SOURCE_ROOT}"
    )
    [[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_preprocessed.py" "${args[@]}"
    ;;
  validate)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_preprocessed.py" \
      --root "${OUTPUT_ROOT}" \
      --domains xjtu smarthealth_lishen40 smarthealth_catl280
    ;;
  *)
    echo "Usage: $0 {preprocess|validate}" >&2
    exit 2
    ;;
esac
