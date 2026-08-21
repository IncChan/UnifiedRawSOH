#!/usr/bin/env bash
# Unified source-to-product launcher.  It never reads Git-tracked datasets;
# all source and output locations come from preprocess/paths.env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
CONFIG_FILE="${PREPROCESS_CONFIG:-${SCRIPT_DIR}/paths.env}"

usage() {
  cat <<'EOF'
Usage:
  bash preprocess/run_preprocess.sh [--config PATH] <domain> <stage> [--workers N] [extractor options]

Domains:
  xjtu
  mit
  smarthealth_lishen40 | smarthealth_catl280 | smarthealth_eve280
  smarthealth                       # run all three SmartHealth families sequentially
  smarthealth_validate

Stages:
  xjtu: raw | features | all
  mit: all                           # canonical physical124 raw + features are one pipeline
  SmartHealth family/all: raw | features | all
  smarthealth_validate: validate

Examples:
  bash preprocess/run_preprocess.sh xjtu all --workers 4
  bash preprocess/run_preprocess.sh mit all --workers 4
  bash preprocess/run_preprocess.sh smarthealth_lishen40 all --workers 8 --overwrite
  bash preprocess/run_preprocess.sh smarthealth all --workers 8 --overwrite
  bash preprocess/run_preprocess.sh smarthealth_validate validate

Pass --overwrite only when deliberately replacing an existing domain product.
EOF
}

if [[ "${1:-}" == "--config" ]]; then
  [[ $# -ge 2 ]] || { usage >&2; exit 2; }
  CONFIG_FILE="$2"
  shift 2
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  printf 'Missing configuration: %s\nCopy preprocess/paths.env.example to preprocess/paths.env first.\n' "${CONFIG_FILE}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${PREPROCESS_WORKERS:-1}"
DOMAIN="$1"
STAGE="$2"
shift 2
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      [[ $# -ge 2 ]] || { printf '%s\n' '--workers requires an integer.' >&2; exit 2; }
      WORKERS="$2"
      shift 2
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done
case "${WORKERS}" in
  ''|*[!0-9]*) printf 'workers must be a positive integer, got %q\n' "${WORKERS}" >&2; exit 2 ;;
esac
if (( WORKERS < 1 )); then
  printf 'workers must be at least 1.\n' >&2
  exit 2
fi

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf 'Required configuration variable is empty: %s\n' "${name}" >&2
    exit 2
  fi
}

require_directory() {
  local name="$1"
  require_value "${name}"
  if [[ ! -d "${!name}" ]]; then
    printf 'Configured source directory does not exist: %s=%s\n' "${name}" "${!name}" >&2
    exit 2
  fi
}

run_xjtu() {
  require_directory XJTU_SOURCE_ROOT
  require_value XJTU_RAW_OUTPUT
  require_value XJTU_FEATURE_OUTPUT
  case "${STAGE}" in
    raw)
      "${PYTHON_BIN}" "${SCRIPT_DIR}/extract_xjtu_cycle_raw_with_temperature.py" \
        --input-root "${XJTU_SOURCE_ROOT}" --output-dir "${XJTU_RAW_OUTPUT}" \
        --report-csv "${XJTU_RAW_OUTPUT}/extraction_report.csv" --workers "${WORKERS}" \
        "${FORWARD_ARGS[@]}"
      ;;
    features)
      "${PYTHON_BIN}" "${SCRIPT_DIR}/extract_xjtu_features.py" \
        --input-root "${XJTU_SOURCE_ROOT}" --output-dir "${XJTU_FEATURE_OUTPUT}" \
        --report-csv "${XJTU_FEATURE_OUTPUT}/extraction_report.csv" --workers "${WORKERS}" \
        "${FORWARD_ARGS[@]}"
      ;;
    all)
      STAGE=raw run_xjtu
      STAGE=features run_xjtu
      ;;
    *) usage >&2; exit 2 ;;
  esac
}

run_mit() {
  require_directory MIT_SOURCE_ROOT
  require_value MIT_RAW_OUTPUT
  require_value MIT_FEATURE_OUTPUT
  require_value MIT_SPLIT_OUTPUT
  if [[ "${STAGE}" != "all" ]]; then
    printf 'MIT physical124 exports raw and features together; use: mit all\n' >&2
    exit 2
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/extract_mit_physical_with_temperature.py" \
    --input-root "${MIT_SOURCE_ROOT}" --raw-output-dir "${MIT_RAW_OUTPUT}" \
    --feature-output-dir "${MIT_FEATURE_OUTPUT}" --split-json "${MIT_SPLIT_OUTPUT}" \
    --cohort paper124 --workers "${WORKERS}" "${FORWARD_ARGS[@]}"
}

smarthealth_raw_script() {
  case "$1" in
    smarthealth_lishen40) printf '%s\n' process_smarthealth_lishen40_raw.py ;;
    smarthealth_catl280) printf '%s\n' process_smarthealth_catl280_raw.py ;;
    smarthealth_eve280) printf '%s\n' process_smarthealth_eve280_raw.py ;;
    *) return 1 ;;
  esac
}

smarthealth_feature_script() {
  case "$1" in
    smarthealth_lishen40) printf '%s\n' extract_smarthealth_lishen40_features.py ;;
    smarthealth_catl280) printf '%s\n' extract_smarthealth_catl280_features.py ;;
    smarthealth_eve280) printf '%s\n' extract_smarthealth_eve280_features.py ;;
    *) return 1 ;;
  esac
}

run_smarthealth_family() {
  local raw_script feature_script
  require_directory SMARTHEALTH_SOURCE_ROOT
  require_value SMARTHEALTH_RAW_OUTPUT
  require_value SMARTHEALTH_FEATURE_OUTPUT
  require_value SMARTHEALTH_SPLITS_OUTPUT
  raw_script="$(smarthealth_raw_script "${DOMAIN}")"
  feature_script="$(smarthealth_feature_script "${DOMAIN}")"
  case "${STAGE}" in
    raw)
      "${PYTHON_BIN}" "${SCRIPT_DIR}/${raw_script}" \
        --input-root "${SMARTHEALTH_SOURCE_ROOT}" --raw-output-root "${SMARTHEALTH_RAW_OUTPUT}" \
        --splits-output-root "${SMARTHEALTH_SPLITS_OUTPUT}" --workers "${WORKERS}" \
        "${FORWARD_ARGS[@]}"
      ;;
    features)
      "${PYTHON_BIN}" "${SCRIPT_DIR}/${feature_script}" \
        --raw-input-root "${SMARTHEALTH_RAW_OUTPUT}" \
        --feature-output-root "${SMARTHEALTH_FEATURE_OUTPUT}" "${FORWARD_ARGS[@]}"
      ;;
    all)
      STAGE=raw run_smarthealth_family
      STAGE=features run_smarthealth_family
      ;;
    *) usage >&2; exit 2 ;;
  esac
}

case "${DOMAIN}" in
  xjtu) run_xjtu ;;
  mit) run_mit ;;
  smarthealth_lishen40|smarthealth_catl280|smarthealth_eve280) run_smarthealth_family ;;
  smarthealth)
    for family in smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280; do
      DOMAIN="${family}" run_smarthealth_family
    done
    ;;
  smarthealth_validate)
    if [[ "${STAGE}" != "validate" ]]; then
      printf 'Use: smarthealth_validate validate\n' >&2
      exit 2
    fi
    require_value SMARTHEALTH_RAW_OUTPUT
    require_value SMARTHEALTH_FEATURE_OUTPUT
    require_value SMARTHEALTH_SPLITS_OUTPUT
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_smarthealth_canonical_v2.py" \
      --raw-root "${SMARTHEALTH_RAW_OUTPUT}" --feature-root "${SMARTHEALTH_FEATURE_OUTPUT}" \
      --split-root "${SMARTHEALTH_SPLITS_OUTPUT}" "${FORWARD_ARGS[@]}"
    ;;
  *) usage >&2; exit 2 ;;
esac
