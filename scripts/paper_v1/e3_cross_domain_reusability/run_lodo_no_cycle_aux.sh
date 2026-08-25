#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LODO_CONFIG_ROOT="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e3_cross_domain_reusability/leave_one_domain_out"

# Select one fold through LEFT_OUT_DOMAIN or the first positional argument.
# Use "all" to execute all five folds sequentially.
LEFT_OUT_DOMAIN="${LEFT_OUT_DOMAIN:-${1:-}}"
if [[ -z "${LEFT_OUT_DOMAIN}" ]]; then
  echo "Usage: LEFT_OUT_DOMAIN=<xjtu|mit|smarthealth_lishen40|smarthealth_catl280|smarthealth_eve280|all> bash $0" >&2
  exit 2
fi

case "${LEFT_OUT_DOMAIN}" in
  lishen40) LEFT_OUT_DOMAIN="smarthealth_lishen40" ;;
  catl280) LEFT_OUT_DOMAIN="smarthealth_catl280" ;;
  eve280) LEFT_OUT_DOMAIN="smarthealth_eve280" ;;
esac

case "${LEFT_OUT_DOMAIN}" in
  xjtu|mit|smarthealth_lishen40|smarthealth_catl280|smarthealth_eve280)
    FOLDS=("${LEFT_OUT_DOMAIN}")
    ;;
  all)
    FOLDS=(
      xjtu
      mit
      smarthealth_lishen40
      smarthealth_catl280
      smarthealth_eve280
    )
    ;;
  *)
    echo "Unknown LEFT_OUT_DOMAIN: ${LEFT_OUT_DOMAIN}" >&2
    exit 2
    ;;
esac

SEEDS="${SEEDS:-42 52 62}"
GPU_IDS="${GPU_IDS:-7}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
fi
PYTHON_BIN="$("${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh")"
export SEEDS GPU_IDS MAX_PARALLEL PYTHON_BIN
export TRAIN_MODULE="UnifiedRawSOH.main"
export REQUIRE_OFFICIAL_MAMBA="${REQUIRE_OFFICIAL_MAMBA:-1}"

if [[ "${DRY_RUN:-0}" != "1" && "${SKIP_DATA_CHECK:-0}" != "1" ]]; then
  SOURCE_CONFIGS=(
    "${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json"
    "${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_mit.json"
    "${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_smarthealth_lishen40.json"
    "${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_smarthealth_catl280.json"
    "${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_smarthealth_eve280.json"
  )
  for source_config in "${SOURCE_CONFIGS[@]}"; do
    "${PYTHON_BIN}" "${PROJECT_ROOT}/UnifiedRawSOH/scripts/setup/check_e1_dataset_ready.py" --config "${source_config}" --mode raw
  done
fi

for fold in "${FOLDS[@]}"; do
  export CONFIG_SOURCE="${LODO_CONFIG_ROOT}/lodo_no_cycle_aux_${fold}.json"
  echo "[LODO fold] left_out=${fold}; config=${CONFIG_SOURCE}"
  bash "${PROJECT_ROOT}/UnifiedRawSOH/scripts/run_seed_batch.sh"
done
