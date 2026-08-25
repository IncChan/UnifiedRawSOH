#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LODO_CONFIG_ROOT="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e3_cross_domain_reusability/leave_one_domain_out"

# Select one fold through LEFT_OUT_DOMAIN or the first positional argument.
# Use "all" to assign the five folds across the configured GPU pool.
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
SEED_SPEC="${SEEDS//,/ }"
read -r -a SEED_LIST <<< "${SEED_SPEC}"
GPU_SPEC="${GPU_IDS:-7}"
GPU_SPEC="${GPU_SPEC//,/ }"
read -r -a GPU_LIST <<< "${GPU_SPEC}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

if (( ${#SEED_LIST[@]} == 0 )); then
  echo "SEEDS must contain at least one integer seed." >&2
  exit 2
fi
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU ID." >&2
  exit 2
fi
if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer per GPU." >&2
  exit 2
fi
for gpu_index in "${!GPU_LIST[@]}"; do
  if [[ -z "${GPU_LIST[${gpu_index}]}" ]]; then
    echo "GPU_IDS contains an empty GPU ID." >&2
    exit 2
  fi
  for previous_index in "${!GPU_LIST[@]}"; do
    if (( previous_index >= gpu_index )); then
      break
    fi
    if [[ "${GPU_LIST[${gpu_index}]}" == "${GPU_LIST[${previous_index}]}" ]]; then
      echo "GPU_IDS contains duplicate GPU ID: ${GPU_LIST[${gpu_index}]}" >&2
      exit 2
    fi
  done
done

if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
fi
PYTHON_BIN="$("${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh")"
export SEEDS PYTHON_BIN
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

GPU_COUNT=${#GPU_LIST[@]}
FOLD_COUNT=${#FOLDS[@]}
WORKER_COUNT=${GPU_COUNT}
if (( WORKER_COUNT > FOLD_COUNT )); then
  WORKER_COUNT=${FOLD_COUNT}
fi

ACTIVE_LIMIT_PER_GPU=${MAX_PARALLEL}
if (( ACTIVE_LIMIT_PER_GPU > ${#SEED_LIST[@]} )); then
  ACTIVE_LIMIT_PER_GPU=${#SEED_LIST[@]}
fi

echo "LODO folds: ${FOLDS[*]}"
echo "GPU pool: ${GPU_LIST[*]}"
echo "seeds per fold: ${SEEDS}"
echo "max parallel processes per GPU: ${MAX_PARALLEL}"
echo "maximum aggregate processes: $((WORKER_COUNT * ACTIVE_LIMIT_PER_GPU))"
for fold_index in "${!FOLDS[@]}"; do
  assigned_gpu="${GPU_LIST[$((fold_index % GPU_COUNT))]}"
  echo "[schedule] gpu=${assigned_gpu}; fold=${FOLDS[${fold_index}]}; seeds=${SEEDS}; max_parallel=${MAX_PARALLEL}"
done

run_fold() {
  local fold="$1"
  local gpu="$2"
  local config_source
  config_source="${LODO_CONFIG_ROOT}/lodo_no_cycle_aux_${fold}.json"
  echo "[LODO fold] left_out=${fold}; gpu=${gpu}; config=${config_source}"
  CONFIG_SOURCE="${config_source}" GPU_IDS="${gpu}" MAX_PARALLEL="${MAX_PARALLEL}" \
    bash "${PROJECT_ROOT}/UnifiedRawSOH/scripts/run_seed_batch.sh"
}

run_gpu_worker() {
  local gpu="$1"
  local worker_index="$2"
  local worker_status=0
  local fold_index
  local fold
  for ((fold_index = worker_index; fold_index < FOLD_COUNT; fold_index += GPU_COUNT)); do
    fold="${FOLDS[${fold_index}]}"
    if ! run_fold "${fold}" "${gpu}"; then
      echo "[LODO failure] left_out=${fold}; gpu=${gpu}" >&2
      worker_status=1
    fi
  done
  return "${worker_status}"
}

worker_pids=()
for ((worker_index = 0; worker_index < WORKER_COUNT; worker_index++)); do
  run_gpu_worker "${GPU_LIST[${worker_index}]}" "${worker_index}" &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "One or more LODO folds failed." >&2
  exit "${status}"
fi

echo "All selected LODO folds completed."
