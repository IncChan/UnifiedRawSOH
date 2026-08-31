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
OUTPUT_ROOT="${PAPER_BACKUP_E2_PREPROCESSED_ROOT:-${REPO_ROOT}/datasets/PaperBackup_preprocessed_v2_e2_5domain_256}"
MIT_FULL_ROOT="${MIT_FULL_CSV_ROOT:-${REPO_ROOT}/datasets/MIT_full_cccv_physical124}"
MODE="${1:-preprocess}"

export_mit_full() {
  : "${MIT_SOURCE_ROOT:?Configure MIT_SOURCE_ROOT in preprocess/paths.env}"
  if [[ "${OVERWRITE_MIT_FULL:-0}" != "1" && -d "${MIT_FULL_ROOT}" ]]; then
    mapfile -t existing_mit_full < <(find "${MIT_FULL_ROOT}" -maxdepth 1 -type f -name 'MIT_*_full_cccv.csv' | sort)
    if (( ${#existing_mit_full[@]} == 124 )); then
      echo "[MIT FULL] reusing complete 124-cell export: ${MIT_FULL_ROOT}"
      return
    fi
    if (( ${#existing_mit_full[@]} > 0 )); then
      echo "Incomplete MIT FULL export (${#existing_mit_full[@]}/124): ${MIT_FULL_ROOT}" >&2
      echo "Set OVERWRITE_MIT_FULL=1 to rebuild it." >&2
      exit 2
    fi
  fi
  args=(--input-root "${MIT_SOURCE_ROOT}" --output-root "${MIT_FULL_ROOT}")
  [[ "${OVERWRITE_MIT_FULL:-0}" == "1" ]] && args+=(--overwrite)
  "${PYTHON_BIN}" "${SCRIPT_DIR}/export_mit_full_cccv.py" "${args[@]}"
}

build_all() {
  : "${XJTU_SOURCE_ROOT:?Configure XJTU_SOURCE_ROOT in preprocess/paths.env}"
  : "${SMARTHEALTH_SOURCE_ROOT:?Configure SMARTHEALTH_SOURCE_ROOT in preprocess/paths.env}"
  [[ -d "${MIT_FULL_ROOT}" ]] || {
    echo "MIT FULL export is missing: ${MIT_FULL_ROOT}. Run '$0 export-mit' first." >&2
    exit 2
  }
  args=(
    --domains xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280
    --include-full xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280
    --output-root "${OUTPUT_ROOT}"
    --schema-version 2 --cc-len 128 --cv-len 128 --full-joint-len 256
    --workers "${PAPER_BACKUP_WORKERS:-8}"
    --xjtu-full-source-root "${XJTU_SOURCE_ROOT}"
    --mit-full-source-root "${MIT_FULL_ROOT}"
    --smarthealth-full-source-root "${SMARTHEALTH_SOURCE_ROOT}"
  )
  [[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_preprocessed.py" "${args[@]}"
}

validate_all() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_preprocessed.py" \
    --root "${OUTPUT_ROOT}" \
    --domains xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280
}

case "${MODE}" in
  export-mit) export_mit_full ;;
  preprocess) build_all ;;
  all) export_mit_full; build_all; validate_all ;;
  validate) validate_all ;;
  *) echo "Usage: $0 {export-mit|preprocess|validate|all}" >&2; exit 2 ;;
esac
