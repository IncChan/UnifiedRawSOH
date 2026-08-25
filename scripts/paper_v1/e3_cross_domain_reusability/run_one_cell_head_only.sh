#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8:backslashreplace"

# ======================== USER CONFIGURATION ========================
# Edit values here, then run this file with:
#   bash scripts/paper_v1/e3_cross_domain_reusability/run_one_cell_head_only.sh
TARGET_DOMAINS="all"
GPU_IDS="0 1 2 3 4"
JOBS_PER_GPU=3
SUPPORT_SEEDS="42 52 62"

CHECKPOINT_XJTU="/path/to/lodo_xjtu/seed_x/best.pt"
CHECKPOINT_MIT="/path/to/lodo_mit/seed_x/best.pt"
CHECKPOINT_LISHEN40="/path/to/lodo_lishen40/seed_x/best.pt"
CHECKPOINT_CATL280="/path/to/lodo_catl280/seed_x/best.pt"
CHECKPOINT_EVE280="/path/to/lodo_eve280/seed_x/best.pt"

ONE_CELL_OUTPUT_ROOT="UnifiedRawSOH/outputs"
RUN_TIME=""
DRY_RUN=0
RESUME=0

DEVICE_OVERRIDE="cuda:0"
BACKEND_OVERRIDE=""
PYTHON_BIN=""
# ==================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$("${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh")"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Configured PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

export TARGET_DOMAINS GPU_IDS JOBS_PER_GPU SUPPORT_SEEDS
export CHECKPOINT_XJTU CHECKPOINT_MIT CHECKPOINT_LISHEN40
export CHECKPOINT_CATL280 CHECKPOINT_EVE280
export ONE_CELL_OUTPUT_ROOT RUN_TIME DRY_RUN RESUME
export DEVICE_OVERRIDE BACKEND_OVERRIDE PYTHON_BIN

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m UnifiedRawSOH.trainers.one_cell_launcher
