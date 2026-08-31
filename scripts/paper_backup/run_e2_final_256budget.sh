#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_SUITE="e2_final_256budget"
exec bash "${SCRIPT_DIR}/run_e2.sh" "$@"
