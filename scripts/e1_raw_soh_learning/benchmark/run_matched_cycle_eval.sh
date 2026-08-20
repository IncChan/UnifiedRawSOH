#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# Direct settings.  Update these explicit E1 runtime directories when a new
# formal E1 run should become the matched-cycle reference.  This script only
# loads checkpoints and evaluates; it never starts training.
PYTHON_BIN="/home/chenyanxi/.conda/envs/pinn/bin/python"
GPU_ID="0"
SEEDS="42 52 62"

XJTU_RAW_RUN_DIR="${PROJECT_ROOT}/UnifiedRawSOH/outputs/Paper-v1/e1_single_domain/RawMamba/XJTU_raw/runtime_260819-132218"
XJTU_ONLYF_RUN_DIR="${PROJECT_ROOT}/UnifiedRawSOH/outputs/Paper-v1/e1_single_domain/PINN4SOH-noLeak-OnlyF/XJTU_features/runtime_260819-142332"
# Canonical 124-cell MIT E1 runs.  Keep these explicit so a later formal run
# can be selected by changing only the two runtime paths below.  Never point
# them at the deleted, incorrect MIT v1 data runs.
MIT_RAW_RUN_DIR="${PROJECT_ROOT}/UnifiedRawSOH/outputs/Paper-v1/e1_single_domain/RawMamba/MIT_raw/runtime_260819-195617"
MIT_ONLYF_RUN_DIR="${PROJECT_ROOT}/UnifiedRawSOH/outputs/Paper-v1/e1_single_domain/PINN4SOH-noLeak-OnlyF/MIT_features/runtime_260819-193655"
RESULT_PATH="${PROJECT_ROOT}/UnifiedRawSOH/outputs/e1_matched_cycle/result.json"

for run_dir in "${XJTU_RAW_RUN_DIR}" "${XJTU_ONLYF_RUN_DIR}" "${MIT_RAW_RUN_DIR}" "${MIT_ONLYF_RUN_DIR}"; do
  if [[ ! -d "${run_dir}" ]]; then
    echo "Configured E1 runtime directory does not exist: ${run_dir}" >&2
    exit 2
  fi
done

echo "Matched-cycle E1 checkpoint evaluation (no training)"
echo "GPU: ${GPU_ID}; seeds: ${SEEDS}"
echo "XJTU RawMamba: ${XJTU_RAW_RUN_DIR}"
echo "XJTU Only-F: ${XJTU_ONLYF_RUN_DIR}"
echo "MIT RawMamba: ${MIT_RAW_RUN_DIR}"
echo "MIT Only-F: ${MIT_ONLYF_RUN_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m UnifiedRawSOH.evaluation.matched_cycle \
  --xjtu-raw-run-dir "${XJTU_RAW_RUN_DIR}" \
  --xjtu-onlyf-run-dir "${XJTU_ONLYF_RUN_DIR}" \
  --mit-raw-run-dir "${MIT_RAW_RUN_DIR}" \
  --mit-onlyf-run-dir "${MIT_ONLYF_RUN_DIR}" \
  --seeds ${SEEDS} \
  --device cuda \
  --output "${RESULT_PATH}"
