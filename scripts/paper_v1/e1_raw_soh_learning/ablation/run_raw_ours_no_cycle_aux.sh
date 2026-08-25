#!/usr/bin/env bash
set -euo pipefail
# Keep launcher logs safe under non-UTF-8 scheduler locales.
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

# Direct-launch settings. Edit these four lines for the desired formal run.
DOMAIN="xjtu"  # xjtu, mit, smarthealth_lishen40, smarthealth_catl280, smarthealth_eve280
SEEDS="42 52 62"
GPU_IDS="0"
MAX_PARALLEL="1"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
fi
PYTHON_BIN="$("${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh")"
export SEEDS GPU_IDS MAX_PARALLEL PYTHON_BIN

case "${DOMAIN}" in
  xjtu)
    CONFIG_NAME="raw_ours_no_cycle_aux_xjtu.json"
    ;;
  mit)
    CONFIG_NAME="raw_ours_no_cycle_aux_mit.json"
    ;;
  smarthealth_lishen40)
    CONFIG_NAME="raw_ours_no_cycle_aux_smarthealth_lishen40.json"
    ;;
  smarthealth_catl280)
    CONFIG_NAME="raw_ours_no_cycle_aux_smarthealth_catl280.json"
    ;;
  smarthealth_eve280)
    CONFIG_NAME="raw_ours_no_cycle_aux_smarthealth_eve280.json"
    ;;
  *)
    echo "Unsupported E1 RawOurs no-cycle-aux DOMAIN=${DOMAIN}." >&2
    exit 2
    ;;
esac

export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/e1_raw_soh_learning/ablation/${CONFIG_NAME}"

# Keep the same canonical raw-data readiness guard as the E1 RawMamba
# benchmark before spawning the configured seed processes.
"${PYTHON_BIN}" "${PROJECT_ROOT}/UnifiedRawSOH/scripts/setup/check_e1_dataset_ready.py" \
  --config "${CONFIG_SOURCE}" --mode raw

export TRAIN_MODULE="UnifiedRawSOH.main"
export REQUIRE_OFFICIAL_MAMBA="${REQUIRE_OFFICIAL_MAMBA:-1}"
exec bash "${PROJECT_ROOT}/UnifiedRawSOH/scripts/run_seed_batch.sh"
