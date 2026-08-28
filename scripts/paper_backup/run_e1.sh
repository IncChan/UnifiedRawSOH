#!/usr/bin/env bash
set -euo pipefail

# E1 is intentionally limited to HI-MLP, Transformer and Ours.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}"
DRY_RUN="${DRY_RUN:-1}"
SEEDS="${SEEDS:-42}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/Paper-Backup}"
RUN_TIME="${RUN_TIME:-e1_main_estimation}"
CHECK_DATA="${CHECK_DATA:-1}"

read -r -a SEED_LIST <<< "${SEEDS//,/ }"
mapfile -t CONFIGS < <(find "${REPO_ROOT}/configs/paper_backup/e1_main_estimation" -type f -name '*.json' | sort)
for config in "${CONFIGS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    args=(--config "${config}" --seed "${seed}" --output_root "${OUTPUT_ROOT}" --run_time "${RUN_TIME}")
    [[ -n "${DEVICE_OVERRIDE}" ]] && args+=(--device_override "${DEVICE_OVERRIDE}")
    [[ -n "${BACKEND_OVERRIDE}" ]] && args+=(--backend_override "${BACKEND_OVERRIDE}")
    [[ "${CHECK_DATA}" == "1" ]] && args+=(--check_data)
    if [[ "${DRY_RUN}" == "1" ]]; then
      args+=(--validate_only)
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_experiment.py" "${args[@]}"
  done
done
