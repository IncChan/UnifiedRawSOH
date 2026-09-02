#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_SUITE=e1_bicontext_cycle_mtl_5seed
exec bash "${SCRIPT_DIR}/run_e1.sh" "$@"
