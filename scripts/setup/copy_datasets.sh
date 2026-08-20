#!/usr/bin/env bash
set -euo pipefail

# Provision local data without assuming that an adjacent historical repository
# exists. This script never downloads, versions, or redistributes a dataset.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
XJTU_SOURCE="${XJTU_SOURCE:-}"
XJTU_FEATURE_SOURCE="${XJTU_FEATURE_SOURCE:-}"
MIT_RAW_SOURCE="${MIT_RAW_SOURCE:-}"
MIT_FEATURE_SOURCE="${MIT_FEATURE_SOURCE:-}"
XJTU_TARGET="${REPO_ROOT}/datasets/XJTU_raw"
XJTU_FEATURE_TARGET="${REPO_ROOT}/datasets/XJTU_features"
MIT_RAW_TARGET="${REPO_ROOT}/datasets/MIT_raw"
MIT_FEATURE_TARGET="${REPO_ROOT}/datasets/MIT_features"

has_csv() {
  local directory="$1"
  [[ -d "${directory}" ]] && find "${directory}" -type f -name '*.csv' -print -quit | grep -q .
}

copy_if_missing() {
  local label="$1"
  local source="$2"
  local target="$3"
  if has_csv "${target}"; then
    echo "${label} already available locally: ${target}"
    return 0
  fi
  [[ -n "${source}" ]] || {
    echo "${label} is absent. Set the corresponding *_SOURCE environment variable to a local authorized copy." >&2
    return 1
  }
  [[ -d "${source}" ]] || { echo "Missing ${label} source: ${source}" >&2; return 1; }
  mkdir -p "${target}"
  cp -a "${source}/." "${target}/"
  has_csv "${target}" || { echo "${label} copy produced no CSV files: ${target}" >&2; return 1; }
  echo "Provisioned ${label}: ${target}"
}

copy_if_missing "XJTU raw" "${XJTU_SOURCE}" "${XJTU_TARGET}"
copy_if_missing "XJTU features" "${XJTU_FEATURE_SOURCE}" "${XJTU_FEATURE_TARGET}"

if [[ -n "${MIT_RAW_SOURCE}" || -n "${MIT_FEATURE_SOURCE}" ]]; then
  [[ -n "${MIT_RAW_SOURCE}" && -n "${MIT_FEATURE_SOURCE}" ]] || {
    echo "Set both MIT_RAW_SOURCE and MIT_FEATURE_SOURCE, or neither." >&2
    exit 1
  }
fi
copy_if_missing "MIT phase-aware raw" "${MIT_RAW_SOURCE}" "${MIT_RAW_TARGET}"
copy_if_missing "MIT Only-F features" "${MIT_FEATURE_SOURCE}" "${MIT_FEATURE_TARGET}"

SMARTHEALTH_RAW="${REPO_ROOT}/datasets/SmartHealth_raw"
SMARTHEALTH_FEATURES="${REPO_ROOT}/datasets/SmartHealth_features"
has_csv "${SMARTHEALTH_RAW}" || {
  echo "SmartHealth canonical raw product is absent: ${SMARTHEALTH_RAW}" >&2
  exit 1
}
has_csv "${SMARTHEALTH_FEATURES}" || {
  echo "SmartHealth canonical feature product is absent: ${SMARTHEALTH_FEATURES}" >&2
  exit 1
}
echo "SmartHealth canonical RAW/features already available locally."
