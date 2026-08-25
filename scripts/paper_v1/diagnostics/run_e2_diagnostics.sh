#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${PROJECT_ROOT}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
fi
PYTHON_BIN="$(${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh)"
GPU_SPEC="${GPU_IDS:-${GPU_ID:-3,6,7}}"
GPU_SPEC="${GPU_SPEC//,/ }"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-cuda}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"
SEED_TEXT="${SEEDS:-42 52 62}"
SEED_TEXT="${SEED_TEXT//,/ }"
SUMMARY_SEED_TEXT="${SUMMARY_SEEDS:-${SEED_TEXT}}"
SUMMARY_SEED_TEXT="${SUMMARY_SEED_TEXT//,/ }"
DIAGNOSTIC_TEXT="${DIAGNOSTICS:-e2_full_b e2_full_d}"
DIAGNOSTIC_TEXT="${DIAGNOSTIC_TEXT//,/ }"
read -r -a SEED_LIST <<< "${SEED_TEXT}"
read -r -a SUMMARY_SEED_LIST <<< "${SUMMARY_SEED_TEXT}"
read -r -a GPU_LIST <<< "${GPU_SPEC}"
read -r -a DIAGNOSTIC_LIST <<< "${DIAGNOSTIC_TEXT}"

if (( ${#SEED_LIST[@]} == 0 )); then
  echo "SEEDS must contain at least one integer seed." >&2
  exit 2
fi
if (( ${#SUMMARY_SEED_LIST[@]} == 0 )); then
  echo "SUMMARY_SEEDS must contain at least one integer seed." >&2
  exit 2
fi
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPU_IDS or GPU_ID must contain at least one GPU ID." >&2
  exit 2
fi
if (( ${#DIAGNOSTIC_LIST[@]} == 0 )); then
  echo "DIAGNOSTICS must contain at least one diagnostic ID." >&2
  exit 2
fi
for diagnostic_id in "${DIAGNOSTIC_LIST[@]}"; do
  case "${diagnostic_id}" in
    e2_full_b|e2_full_d) ;;
    *)
      echo "Unknown diagnostic ID: ${diagnostic_id}; expected e2_full_b or e2_full_d." >&2
      exit 2
      ;;
  esac
done

MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_LIST[@]}}"
if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer." >&2
  exit 2
fi
if [[ -n "${CPU_THREADS_PER_JOB:-}" ]] && ! [[ "${CPU_THREADS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS_PER_JOB must be a positive integer." >&2
  exit 2
fi

GPU_CSV="${GPU_LIST[*]}"
GPU_CSV="${GPU_CSV// /,}"

if [[ "${DRY_RUN:-0}" != "1" && "${DEVICE_OVERRIDE}" == cuda* && "${SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_CSV}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable for formal V1 diagnostics.")
PY
fi

if [[ "${DRY_RUN:-0}" != "1" && "${DEVICE_OVERRIDE}" == cuda* && -z "${BACKEND_OVERRIDE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_CSV}" "${PYTHON_BIN}" - <<'PY'
from UnifiedRawSOH.models.c5b_model import require_official_mamba

print(f"official Mamba backend: {require_official_mamba()}")
PY
fi

echo "diagnostics: ${DIAGNOSTIC_LIST[*]}"
echo "worker seeds: ${SEED_LIST[*]}"
echo "summary seeds: ${SUMMARY_SEED_LIST[*]}"
echo "GPU IDs: ${GPU_LIST[*]}"
echo "max parallel seeds: ${MAX_PARALLEL}"

run_diagnostic() {
  local diagnostic_id="$1"
  local seed="$2"
  local gpu_id="$3"
  local config
  config="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/diagnostics/${diagnostic_id}.json"
  local -a args=(
    --config "${config}"
    --device_override "${DEVICE_OVERRIDE}"
    --seed "${seed}"
    --worker_mode
  )
  if [[ -n "${BACKEND_OVERRIDE}" ]]; then
    args+=(--backend_override "${BACKEND_OVERRIDE}")
  fi
  if [[ -n "${MAX_SAMPLES_PER_DOMAIN:-}" ]]; then
    args+=(--max_samples_per_domain "${MAX_SAMPLES_PER_DOMAIN}")
  fi
  if [[ "${SKIP_GRADIENTS:-0}" == "1" ]]; then
    args+=(--skip_gradients)
  fi
  echo "[worker] ${diagnostic_id}; seed=${seed}; gpu=${gpu_id}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  local -a command=("${PYTHON_BIN}" -m UnifiedRawSOH.evaluation.paper_v1.domain_diagnostics "${args[@]}")
  if [[ "${DEVICE_OVERRIDE}" == cpu* && -n "${CPU_THREADS_PER_JOB:-}" ]]; then
    OMP_NUM_THREADS="${CPU_THREADS_PER_JOB}" MKL_NUM_THREADS="${CPU_THREADS_PER_JOB}" "${command[@]}"
  else
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}"
  fi
}

run_seed_worker() {
  local seed="$1"
  local gpu_id="$2"
  echo "[seed worker] seed=${seed}; gpu=${gpu_id}; diagnostics=${DIAGNOSTIC_LIST[*]}"
  for diagnostic_id in "${DIAGNOSTIC_LIST[@]}"; do
    run_diagnostic "${diagnostic_id}" "${seed}" "${gpu_id}" || return $?
  done
}

aggregate_diagnostic() {
  local diagnostic_id="$1"
  local config
  config="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/diagnostics/${diagnostic_id}.json"
  local -a args=(--config "${config}" --aggregate_only)
  for seed in "${SUMMARY_SEED_LIST[@]}"; do
    args+=(--seed "${seed}")
  done
  if [[ "${SKIP_GRADIENTS:-0}" == "1" ]]; then
    args+=(--skip_gradients)
  fi
  echo "[aggregate] ${diagnostic_id}; seeds=${SUMMARY_SEED_LIST[*]}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" -m UnifiedRawSOH.evaluation.paper_v1.domain_diagnostics "${args[@]}"
}

active_pids=()
status=0
for seed_index in "${!SEED_LIST[@]}"; do
  while (( ${#active_pids[@]} >= MAX_PARALLEL )); do
    first_pid="${active_pids[0]}"
    wait "${first_pid}" || status=1
    active_pids=("${active_pids[@]:1}")
  done
  seed="${SEED_LIST[${seed_index}]}"
  gpu_id="${GPU_LIST[$((seed_index % ${#GPU_LIST[@]}))]}"
  run_seed_worker "${seed}" "${gpu_id}" &
  active_pids+=("$!")
done

for pid in "${active_pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo "At least one diagnostic worker failed; shared summaries were not rewritten." >&2
  exit "${status}"
fi
for diagnostic_id in "${DIAGNOSTIC_LIST[@]}"; do
  aggregate_diagnostic "${diagnostic_id}"
done
