#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# Direct-launch settings. Change this exact line, then run this script without
# additional command-line arguments. It intentionally does not inherit a
# terminal DOMAIN variable, so the selected config and output directory agree.
# Change only this line for the E1 battery domain to train.
DOMAIN="mit"  # choices: xjtu, mit, smarthealth_lishen40, smarthealth_catl280, smarthealth_eve280
SEEDS="42 52 62"
GPU_IDS="2"
MAX_PARALLEL="3"
PYTHON_BIN="${PYTHON_BIN:-/home/chenyanxi/.conda/envs/pinn/bin/python}"
export SEEDS GPU_IDS MAX_PARALLEL PYTHON_BIN

case "${DOMAIN}" in
  xjtu)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/pinn4soh_onlyf_xjtu.json"
    ;;
  mit)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/pinn4soh_onlyf_mit.json"
    ;;
  smarthealth_lishen40)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/pinn4soh_onlyf_smarthealth_lishen40.json"
    ;;
  smarthealth_catl280)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/pinn4soh_onlyf_smarthealth_catl280.json"
    ;;
  smarthealth_eve280)
    export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/e1_raw_soh_learning/benchmark/pinn4soh_onlyf_smarthealth_eve280.json"
    ;;
  *)
    echo "Unsupported E1 Only-F DOMAIN=${DOMAIN}." >&2
    exit 2
    ;;
esac

# This does not train or change data. It verifies that the selected canonical
# feature product is complete before the multi-seed launcher is allowed to fork.
"${PYTHON_BIN}" "${PROJECT_ROOT}/UnifiedRawSOH/scripts/setup/check_e1_dataset_ready.py" \
  --config "${CONFIG_SOURCE}" --mode onlyf

export TRAIN_MODULE="UnifiedRawSOH.main_baseline"
export REQUIRE_OFFICIAL_MAMBA="0"
exec bash "${PROJECT_ROOT}/UnifiedRawSOH/scripts/run_seed_batch.sh"
