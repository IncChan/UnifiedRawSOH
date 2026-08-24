#!/usr/bin/env bash
set -euo pipefail
# Keep launcher logs safe under non-UTF-8 scheduler locales.
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# E2-Full Domain-Balanced: XJTU + MIT + LISHEN40 + CATL280 + EVE280. One worker on one GPU
# is the safe default. Use distinct GPU IDs to run seeds concurrently, for
# example: GPU_IDS="0 1 2" MAX_PARALLEL=3 bash "$0".
SEEDS="${SEEDS:-42 52 62}"
GPU_IDS="${GPU_IDS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
fi
PYTHON_BIN="$(${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh)"
export SEEDS GPU_IDS MAX_PARALLEL PYTHON_BIN

export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/e2_unified_multidomain/unified/public_all_domains_domain_balanced.json"
export TRAIN_MODULE="UnifiedRawSOH.main"
export REQUIRE_OFFICIAL_MAMBA="${REQUIRE_OFFICIAL_MAMBA:-1}"

# Every domain keeps its E1 data contract. Reuse the read-only E1 guard for
# each contract before creating the shared-model training jobs.
for source_config in \
  "${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json" \
  "${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/raw_mamba_mit.json" \
  "${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/raw_mamba_smarthealth_lishen40.json" \
  "${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/raw_mamba_smarthealth_catl280.json" \
  "${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/raw_mamba_smarthealth_eve280.json"; do
  "${PYTHON_BIN}" "${PROJECT_ROOT}/UnifiedRawSOH/scripts/setup/check_e1_dataset_ready.py" \
    --config "${source_config}" --mode raw
done

exec bash "${PROJECT_ROOT}/UnifiedRawSOH/scripts/run_seed_batch.sh"
