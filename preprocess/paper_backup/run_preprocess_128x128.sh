#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# This wrapper deliberately owns a different, fixed root from the 128x256
# C-rate v2 product.  Keeping the path fixed also prevents preprocessing and
# training configs from silently pointing at different products.
export PAPER_BACKUP_SCHEMA_VERSION=2
export PAPER_BACKUP_PREPROCESSED_ROOT="${REPO_ROOT}/datasets/PaperBackup_preprocessed_v2_128x128"
export CC_LEN=128
export CV_LEN=128

exec bash "${SCRIPT_DIR}/run_preprocess.sh" "${1:-terminal}"
