#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_SUITE=e1_late_latent_token_bicontext_5seed
export SEEDS="${SEEDS:-42 52 62 72 82}"
export EPOCHS="${EPOCHS:-600}"
export PATIENCE="${PATIENCE:-30}"
exec bash "${SCRIPT_DIR}/run_e1.sh" "$@"
