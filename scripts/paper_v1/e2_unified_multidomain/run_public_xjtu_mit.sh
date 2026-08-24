#!/usr/bin/env bash
set -euo pipefail
# Keep launcher logs safe under non-UTF-8 scheduler locales.
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# E2-Pilot Domain-Balanced: XJTU + MIT. One worker on one GPU is the safe default. To run the
# three seeds concurrently, set distinct GPU IDs and increase MAX_PARALLEL,
# for example: GPU_IDS="0 1 2" MAX_PARALLEL=3 bash "$0".
SEEDS="${SEEDS:-42 52 62}"
GPU_IDS="${GPU_IDS:-7}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
fi
PYTHON_BIN="$(${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh)"
export SEEDS GPU_IDS MAX_PARALLEL PYTHON_BIN

export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e2_unified_multidomain/unified/public_xjtu_mit_domain_balanced.json"
export TRAIN_MODULE="UnifiedRawSOH.main"
export REQUIRE_OFFICIAL_MAMBA="${REQUIRE_OFFICIAL_MAMBA:-1}"

# Fail before any seed is launched when a canonical raw export is incomplete.
for source_config in \
  "${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json" \
  "${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_mit.json"; do
  "${PYTHON_BIN}" "${PROJECT_ROOT}/UnifiedRawSOH/scripts/setup/check_e1_dataset_ready.py" \
    --config "${source_config}" --mode raw
done

exec bash "${PROJECT_ROOT}/UnifiedRawSOH/scripts/run_seed_batch.sh"
