#!/usr/bin/env bash
# Plot one canonical capacity-versus-cycle mosaic for each SmartHealth domain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${SMARTHEALTH_RAW_ROOT:-${SCRIPT_DIR}/../datasets/SmartHealth_raw}"
OUTPUT_DIR="${SMARTHEALTH_CAPACITY_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/smarthealth_capacity_trajectories}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/unifiedrawsoh_mpl}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_smarthealth_capacity_trajectories.py" \
  --raw-root "${RAW_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
