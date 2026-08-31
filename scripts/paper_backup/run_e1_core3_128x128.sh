#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_SUITE="e1_core3_128x128"
exec bash "${SCRIPT_DIR}/run_e1.sh" "$@"
