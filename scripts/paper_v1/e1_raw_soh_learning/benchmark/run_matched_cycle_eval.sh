#!/usr/bin/env bash
set -euo pipefail
# Keep launcher logs safe under non-UTF-8 scheduler locales.
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

# Direct settings.  Update these explicit E1 runtime directories when a new
# formal E1 run should become the matched-cycle reference.  This script only
# loads checkpoints and evaluates; it never starts training.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
fi
PYTHON_BIN="$(${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh)"
GPU_ID="2"
SEEDS="42 52 62"

XJTU_RAW_RUN_DIR="${PROJECT_ROOT}/UnifiedRawSOH/outputs/Paper-v1/e1_raw_soh_learning/RawMamba/xjtu/runtime_260819-145050"
XJTU_ONLYF_RUN_DIR="${PROJECT_ROOT}/UnifiedRawSOH/outputs/Paper-v1/e1_raw_soh_learning/PINN4SOH-noLeak-OnlyF/xjtu/runtime_260819-142332"
# Canonical 124-cell MIT E1 runs.  Keep these explicit so a later formal run
# can be selected by changing only the two runtime paths below.  Never point
# them at the deleted, incorrect MIT v1 data runs.
MIT_RAW_RUN_DIR="${PROJECT_ROOT}/UnifiedRawSOH/outputs/Paper-v1/e1_raw_soh_learning/RawMamba/mit/runtime_260821-120322"
MIT_ONLYF_RUN_DIR="${PROJECT_ROOT}/UnifiedRawSOH/outputs/Paper-v1/e1_raw_soh_learning/PINN4SOH-noLeak-OnlyF/mit/runtime_260821-215915"
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

PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m UnifiedRawSOH.evaluation.matched_cycle \
  --xjtu-raw-run-dir "${XJTU_RAW_RUN_DIR}" \
  --xjtu-onlyf-run-dir "${XJTU_ONLYF_RUN_DIR}" \
  --mit-raw-run-dir "${MIT_RAW_RUN_DIR}" \
  --mit-onlyf-run-dir "${MIT_ONLYF_RUN_DIR}" \
  --seeds ${SEEDS} \
  --device cuda \
  --output "${RESULT_PATH}"
