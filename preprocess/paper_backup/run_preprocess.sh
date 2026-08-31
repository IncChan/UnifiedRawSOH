#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT

CALLER_PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -f "${REPO_ROOT}/preprocess/paths.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/preprocess/paths.env"
fi

PYTHON_BIN="${CALLER_PYTHON_BIN:-${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}}"
MODE="${1:-all}"
SCHEMA_VERSION="${PAPER_BACKUP_SCHEMA_VERSION:-1}"
if [[ "${SCHEMA_VERSION}" == "2" ]]; then
  DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/datasets/PaperBackup_preprocessed_v2"
else
  DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/datasets/PaperBackup_preprocessed"
fi
OUTPUT_ROOT="${PAPER_BACKUP_PREPROCESSED_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
CC_LEN="${CC_LEN:-128}"
CV_LEN="${CV_LEN:-256}"
OVERWRITE="${OVERWRITE:-0}"
MAX_RECORDS="${MAX_RECORDS:-}"
WORKERS="${PAPER_BACKUP_WORKERS:-${PREPROCESS_WORKERS:-1}}"

common=(--output-root "${OUTPUT_ROOT}" --schema-version "${SCHEMA_VERSION}" --cc-len "${CC_LEN}" --cv-len "${CV_LEN}" --workers "${WORKERS}")
[[ "${OVERWRITE}" == "1" ]] && common+=(--overwrite)
[[ -n "${MAX_RECORDS}" ]] && common+=(--max-records "${MAX_RECORDS}")

case "${MODE}" in
  terminal)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_preprocessed.py" --domains all "${common[@]}"
    ;;
  full)
    : "${XJTU_SOURCE_ROOT:?Set XJTU_SOURCE_ROOT in preprocess/paths.env or the environment}"
    : "${SMARTHEALTH_SOURCE_ROOT:?Set SMARTHEALTH_SOURCE_ROOT in preprocess/paths.env or the environment}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_preprocessed.py" \
      --domains xjtu smarthealth_lishen40 smarthealth_catl280 \
      --include-full xjtu smarthealth_lishen40 smarthealth_catl280 \
      --xjtu-full-source-root "${XJTU_SOURCE_ROOT}" \
      --smarthealth-full-source-root "${SMARTHEALTH_SOURCE_ROOT}" \
      "${common[@]}"
    ;;
  all)
    : "${XJTU_SOURCE_ROOT:?Set XJTU_SOURCE_ROOT in preprocess/paths.env or the environment}"
    : "${SMARTHEALTH_SOURCE_ROOT:?Set SMARTHEALTH_SOURCE_ROOT in preprocess/paths.env or the environment}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_preprocessed.py" \
      --domains all \
      --include-full xjtu smarthealth_lishen40 smarthealth_catl280 \
      --xjtu-full-source-root "${XJTU_SOURCE_ROOT}" \
      --smarthealth-full-source-root "${SMARTHEALTH_SOURCE_ROOT}" \
      "${common[@]}"
    ;;
  validate)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_preprocessed.py" --root "${OUTPUT_ROOT}"
    ;;
  *)
    echo "Usage: $0 {terminal|full|all|validate}" >&2
    exit 2
    ;;
esac
