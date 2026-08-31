#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_SUITE="e1_shared_crate_fullvi"
exec bash "${SCRIPT_DIR}/run_e1.sh" "$@"
