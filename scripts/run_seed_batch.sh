#!/usr/bin/env bash
set -euo pipefail

# Reusable Paper-v1 multi-seed launcher.
# Required environment: CONFIG_SOURCE and TRAIN_MODULE.
# Optional environment: PYTHON_BIN, OUTPUT_ROOT, SEEDS, GPU_IDS,
# MAX_PARALLEL, RUN_TIME, DEVICE_OVERRIDE, BACKEND_OVERRIDE, EPOCHS,
# PATIENCE, DEBUG_NUM_SAMPLES, REQUIRE_OFFICIAL_MAMBA.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

: "${CONFIG_SOURCE:?CONFIG_SOURCE must point to a Paper-v1 JSON config}"
: "${TRAIN_MODULE:?TRAIN_MODULE must be a Python module with a main entry point}"

PYTHON_BIN="${PYTHON_BIN:-/home/chenyanxi/.conda/envs/pinn/bin/python}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-cuda}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"
SEED_TEXT="${SEEDS:-42 52 62}"
SEED_TEXT="${SEED_TEXT//,/ }"
GPU_SPEC="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_SPEC="${GPU_SPEC//,/ }"
read -r -a SEED_LIST <<< "${SEED_TEXT}"
read -r -a GPU_LIST <<< "${GPU_SPEC}"
GPU_CSV="${GPU_LIST[*]}"
GPU_CSV="${GPU_CSV// /,}"

if (( ${#SEED_LIST[@]} == 0 )); then
  echo "SEEDS must contain at least one integer seed." >&2
  exit 2
fi
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPU_IDS or CUDA_VISIBLE_DEVICES must contain at least one GPU ID." >&2
  exit 2
fi

MAX_PARALLEL="${MAX_PARALLEL:-3}"
if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer." >&2
  exit 2
fi

if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$(${PYTHON_BIN} - "${CONFIG_SOURCE}" <<'PY'
import sys
from UnifiedRawSOH.utils.config import load_config

config = load_config(sys.argv[1])
print(config.get("experiment", {}).get("output_root", "UnifiedRawSOH/outputs"))
PY
  )"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi

read -r PAPER_VERSION EXPERIMENT_ID MODEL_ID DATA_ID EXPERIMENT_NAME RESOLVED_DOMAIN_ID RESOLVED_DATA_ROOT <<< "$(${PYTHON_BIN} - "${CONFIG_SOURCE}" <<'PY'
import sys
from UnifiedRawSOH.utils.config import load_config
from UnifiedRawSOH.utils.output_layout import output_identity

config = load_config(sys.argv[1])
experiment = config.get("experiment", {})
data = config.get("data", {})
output = output_identity(config)
domain_id = experiment.get("domain_id")
if domain_id is None:
    domain_ids = experiment.get("domain_ids", ())
    domain_id = "+".join(str(value) for value in domain_ids) if domain_ids else experiment.get("dataset_id", "unknown")
data_root = data.get("data_root")
if data_root is None:
    roots = data.get("data_roots", {})
    data_root = "+".join(f"{key}:{value}" for key, value in sorted(roots.items())) or "unknown"
print(
    output["paper_version"],
    output["experiment_id"],
    output["model_id"],
    output["data_id"],
    experiment.get("name", "paper_v1_experiment"),
    domain_id,
    data_root,
)
PY
)"

BASE_RUN_TIME="${RUN_TIME:-$(date +%y%m%d-%H%M%S)}"
if [[ "${BASE_RUN_TIME}" != runtime_* ]]; then
  BASE_RUN_TIME="runtime_${BASE_RUN_TIME}"
fi
RUN_TIME="${BASE_RUN_TIME}"
BATCH_ROOT="${OUTPUT_ROOT}/${PAPER_VERSION}/${EXPERIMENT_ID}/${MODEL_ID}/${DATA_ID}/${RUN_TIME}"
run_namespace_exists() {
  [[ -e "${BATCH_ROOT}" ]] && return 0
  return 1
}
if run_namespace_exists; then
  suffix=1
  while :; do
    RUN_TIME="${BASE_RUN_TIME}-${suffix}"
    BATCH_ROOT="${OUTPUT_ROOT}/${PAPER_VERSION}/${EXPERIMENT_ID}/${MODEL_ID}/${DATA_ID}/${RUN_TIME}"
    if ! run_namespace_exists; then
      break
    fi
    suffix=$((suffix + 1))
  done
fi
mkdir -p "${BATCH_ROOT}"
cp "${CONFIG_SOURCE}" "${BATCH_ROOT}/source_config.json"
"${PYTHON_BIN}" - "${CONFIG_SOURCE}" "${BATCH_ROOT}/resolved_launcher_config.json" <<'PY'
import sys
from pathlib import Path

from UnifiedRawSOH.utils.config import load_config, save_json

save_json(Path(sys.argv[2]), load_config(sys.argv[1]))
PY
"${PYTHON_BIN}" - "${CONFIG_SOURCE}" "${OUTPUT_ROOT}" "${RUN_TIME}" "${BATCH_ROOT}/run_manifest.json" <<'PY'
import sys
from pathlib import Path

from UnifiedRawSOH.utils.config import load_config, save_json
from UnifiedRawSOH.utils.output_layout import build_run_manifest

config = load_config(sys.argv[1])
save_json(Path(sys.argv[4]), build_run_manifest(config, sys.argv[2], sys.argv[3]))
PY

echo "paper version: ${PAPER_VERSION}"
echo "experiment: ${EXPERIMENT_ID} (${EXPERIMENT_NAME})"
echo "model: ${MODEL_ID}"
echo "domain: ${DATA_ID} (configured domain: ${RESOLVED_DOMAIN_ID})"
echo "data root: ${RESOLVED_DATA_ROOT}"
echo "config: ${CONFIG_SOURCE}"
echo "output root: ${OUTPUT_ROOT}"
echo "batch run time: ${RUN_TIME}"
echo "output directory: ${BATCH_ROOT}"
echo "seeds: ${SEED_LIST[*]}"
echo "GPU IDs: ${GPU_LIST[*]}"
echo "max parallel: ${MAX_PARALLEL}"

if [[ "${DRY_RUN:-0}" != "1" && "${DEVICE_OVERRIDE}" == cuda* && "${SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_CSV}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Set up the CUDA environment before formal Paper-v1 training, "
        "or use DEVICE_OVERRIDE=cpu only for a structural smoke test."
    )
print(f"visible CUDA devices: {torch.cuda.device_count()}")
PY
fi

if [[ "${DRY_RUN:-0}" != "1" && "${REQUIRE_OFFICIAL_MAMBA:-0}" == "1" && -z "${BACKEND_OVERRIDE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_CSV}" "${PYTHON_BIN}" - <<'PY'
from UnifiedRawSOH.models.c5b_model import require_official_mamba

print(f"official Mamba backend: {require_official_mamba()}")
PY
fi

base_args=(
  --config "${CONFIG_SOURCE}"
  --output_root "${OUTPUT_ROOT}"
  --run_time "${RUN_TIME}"
  --device_override "${DEVICE_OVERRIDE}"
)
if [[ -n "${BACKEND_OVERRIDE}" ]]; then
  base_args+=(--backend_override "${BACKEND_OVERRIDE}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  base_args+=(--epochs "${EPOCHS}")
fi
if [[ -n "${PATIENCE:-}" ]]; then
  base_args+=(--patience "${PATIENCE}")
fi
if [[ -n "${DEBUG_NUM_SAMPLES:-}" ]]; then
  base_args+=(--debug_num_samples "${DEBUG_NUM_SAMPLES}")
fi

run_job() {
  local seed="$1"
  local gpu="$2"
  echo "[launch] module=${TRAIN_MODULE} seed=${seed} gpu=${gpu} run_time=${RUN_TIME}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -m "${TRAIN_MODULE}" \
    "${base_args[@]}" --seed "${seed}"
}

pids=()
status=0
job_index=0

wait_for_slot() {
  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
    wait -n || status=1
  done
}

for seed in "${SEED_LIST[@]}"; do
  wait_for_slot
  gpu="${GPU_LIST[$((job_index % ${#GPU_LIST[@]}))]}"
  run_job "${seed}" "${gpu}" &
  pids+=("$!")
  job_index=$((job_index + 1))
done

for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

if [[ "${status}" -ne 0 ]]; then
  echo "one or more Paper-v1 training jobs failed; no aggregate summary was generated." >&2
  exit "${status}"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry run complete; no training process was started."
  exit 0
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_batch_runs.py" \
  --batch_root "${BATCH_ROOT}" \
  --expected_seeds "${SEED_LIST[@]}"

echo "batch summary: ${BATCH_ROOT}/summary_mean_std.json"
exit 0
