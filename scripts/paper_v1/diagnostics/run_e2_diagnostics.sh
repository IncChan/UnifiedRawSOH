#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${PROJECT_ROOT}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN
fi
PYTHON_BIN="$(${PROJECT_ROOT}/UnifiedRawSOH/scripts/resolve_python_bin.sh)"
GPU_ID="${GPU_ID:-0}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-cuda}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"
SEED_TEXT="${SEEDS:-42 52 62}"
SEED_TEXT="${SEED_TEXT//,/ }"
read -r -a SEED_LIST <<< "${SEED_TEXT}"

if (( ${#SEED_LIST[@]} == 0 )); then
  echo "SEEDS must contain at least one integer seed." >&2
  exit 2
fi

if [[ "${DEVICE_OVERRIDE}" == cuda* && "${SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable for formal V1 diagnostics.")
PY
fi

if [[ "${DEVICE_OVERRIDE}" == cuda* && -z "${BACKEND_OVERRIDE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
from UnifiedRawSOH.models.c5b_model import require_official_mamba

print(f"official Mamba backend: {require_official_mamba()}")
PY
fi

for diagnostic_id in e2_full_b e2_full_d; do
  config="${PROJECT_ROOT}/UnifiedRawSOH/configs/paper_v1/diagnostics/${diagnostic_id}.json"
  args=(
    --config "${config}"
    --device_override "${DEVICE_OVERRIDE}"
  )
  for seed in "${SEED_LIST[@]}"; do
    args+=(--seed "${seed}")
  done
  if [[ -n "${BACKEND_OVERRIDE}" ]]; then
    args+=(--backend_override "${BACKEND_OVERRIDE}")
  fi
  if [[ -n "${MAX_SAMPLES_PER_DOMAIN:-}" ]]; then
    args+=(--max_samples_per_domain "${MAX_SAMPLES_PER_DOMAIN}")
  fi
  if [[ "${SKIP_GRADIENTS:-0}" == "1" ]]; then
    args+=(--skip_gradients)
  fi
  echo "[diagnose] ${diagnostic_id}; seeds=${SEED_LIST[*]}; gpu=${GPU_ID}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_ID}"     "${PYTHON_BIN}" -m UnifiedRawSOH.evaluation.paper_v1.domain_diagnostics "${args[@]}"
done
