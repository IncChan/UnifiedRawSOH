#!/usr/bin/env bash
# Plot one canonical capacity-versus-cycle mosaic for all XJTU conditions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${XJTU_RAW_ROOT:-${SCRIPT_DIR}/../datasets/XJTU_raw}"
SPLIT_FILE="${XJTU_CAPACITY_SPLIT_FILE:-${SCRIPT_DIR}/../splits/xjtu/paper_v1_mixed_split.json}"
OUTPUT_DIR="${XJTU_CAPACITY_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/xjtu_capacity_trajectories}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/unifiedrawsoh_mpl}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_xjtu_capacity_trajectories.py" \
  --raw-root "${RAW_ROOT}" \
  --split-file "${SPLIT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
