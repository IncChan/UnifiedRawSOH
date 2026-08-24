#!/usr/bin/env bash
set -euo pipefail
# Keep launcher logs safe under non-UTF-8 scheduler locales.
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

# Direct-launch settings. Change this exact line, then run this script without
# additional command-line arguments. It intentionally does not inherit a
# terminal DOMAIN variable, so the selected config and output directory agree.
# Change only this line for the E1 battery domain to train.
DOMAIN="smarthealth_eve280"  # choices: xjtu, mit, smarthealth_lishen40, smarthealth_catl280, smarthealth_eve280
SEEDS="42 52 62"
GPU_IDS="2"
MAX_PARALLEL="3"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
fi
PYTHON_BIN="$(${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh)"
export SEEDS GPU_IDS MAX_PARALLEL PYTHON_BIN

case "${DOMAIN}" in
  xjtu)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_xjtu.json"
    ;;
  mit)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_mit.json"
    ;;
  smarthealth_lishen40)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_smarthealth_lishen40.json"
    ;;
  smarthealth_catl280)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_smarthealth_catl280.json"
    ;;
  smarthealth_eve280)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/benchmark/raw_mamba_smarthealth_eve280.json"
    ;;
  *)
    echo "Unsupported E1 raw DOMAIN=${DOMAIN}." >&2
    exit 2
    ;;
esac

# Stop before spawning one process per seed if a canonical product is absent,
# header-only, or copied incompletely. This check never trains or rewrites data.
"${PYTHON_BIN}" "${PROJECT_ROOT}/UnifiedRawSOH/scripts/setup/check_e1_dataset_ready.py" \
  --config "${CONFIG_SOURCE}" --mode raw

export TRAIN_MODULE="UnifiedRawSOH.main"
export REQUIRE_OFFICIAL_MAMBA="${REQUIRE_OFFICIAL_MAMBA:-1}"
exec bash "${PROJECT_ROOT}/UnifiedRawSOH/scripts/run_seed_batch.sh"
