#!/usr/bin/env bash
# Plot one canonical capacity-versus-global-cycle mosaic for MIT physical cells.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${MIT_RAW_ROOT:-${SCRIPT_DIR}/../datasets/MIT_raw}"
SPLIT_FILE="${MIT_CAPACITY_SPLIT_FILE:-${SCRIPT_DIR}/../splits/mit/mit_paper_physical124_v2_split.json}"
OUTPUT_DIR="${MIT_CAPACITY_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/mit_capacity_trajectories}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/unifiedrawsoh_mpl}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_mit_capacity_trajectories.py" \
  --raw-root "${RAW_ROOT}" \
  --split-file "${SPLIT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
