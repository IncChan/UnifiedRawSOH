#!/usr/bin/env bash
# Generate raw CC/CV diagnostic figures from all canonical SmartHealth cycles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SMARTHEALTH_SOURCE_ROOT:-/data1/chenyanxi/lb_project/datasets/SmartHealth}"
OUTPUT_DIR="${SMARTHEALTH_PROFILE_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/smarthealth_source_charge_profiles}"
# Some shared compute environments make ~/.config read-only.  Keep Matplotlib's
# local font/config cache disposable and avoid a noisy warning on every run.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/unifiedrawsoh_mpl}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_smarthealth_charge_profiles.py" \
  --source-root "${SOURCE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
