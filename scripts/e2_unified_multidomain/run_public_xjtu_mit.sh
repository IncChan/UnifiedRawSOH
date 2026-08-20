#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# This is the currently available two-domain E2 composition.  It is not
# launched automatically; edit these fixed settings before an intentional run.
SEEDS="42 52 62"
GPU_IDS="0"
MAX_PARALLEL="3"
export SEEDS GPU_IDS MAX_PARALLEL
export CONFIG_SOURCE="${PROJECT_ROOT}/UnifiedRawSOH/configs/e2_unified_multidomain/unified/public_xjtu_mit.json"
export TRAIN_MODULE="UnifiedRawSOH.main"
export REQUIRE_OFFICIAL_MAMBA="${REQUIRE_OFFICIAL_MAMBA:-1}"
exec bash "${PROJECT_ROOT}/UnifiedRawSOH/scripts/run_seed_batch.sh"
