#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

# Change only this name to run one implemented E1 control.  The main proposed
# model remains under benchmark/run_raw_mamba_benchmark.sh.
ABLATION="no_degradation_aux"  # vi_only, delta_t, t0, independent_cc_cv, no_degradation_aux
SEEDS="42 52 62"
GPU_IDS="0"
MAX_PARALLEL="3"
export SEEDS GPU_IDS MAX_PARALLEL

case "${ABLATION}" in
  vi_only) CONFIG_NAME="temperature_vi_only.json" ;;
  delta_t) CONFIG_NAME="temperature_delta_t.json" ;;
  t0) CONFIG_NAME="temperature_t0.json" ;;
  independent_cc_cv) CONFIG_NAME="independent_cc_cv.json" ;;
  no_degradation_aux) CONFIG_NAME="no_degradation_aux.json" ;;
  *)
    echo "Unsupported E1 ablation ${ABLATION}" >&2
    exit 2
    ;;
esac

export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/ablation/${CONFIG_NAME}"
export TRAIN_MODULE="UnifiedRawSOH.main"
export REQUIRE_OFFICIAL_MAMBA="${REQUIRE_OFFICIAL_MAMBA:-1}"
exec bash "${PROJECT_ROOT}/UnifiedRawSOH/scripts/run_seed_batch.sh"
