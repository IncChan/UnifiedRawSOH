#!/usr/bin/env bash
set -euo pipefail

# One-seed, multi-GPU curated-SMVIC comparison: six domains x three models.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}"
CONFIG_ROOT="${REPO_ROOT}/configs/paper_backup/e4_industrial_external/smvic"
STAGE="${1:-all}"

SEED="${SEED:-42}"
GPU_SPEC="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_SPEC="${GPU_SPEC//,/ }"
read -r -a GPU_LIST <<< "${GPU_SPEC}"
# MAX_PARALLEL is intentionally per physical GPU.  Aggregate concurrency is
# len(GPU_LIST) * MAX_PARALLEL.
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MODELS_SPEC="${MODELS:-all}"
DOMAINS_SPEC="${DOMAINS:-all}"
MODELS_SPEC="${MODELS_SPEC//,/ }"
DOMAINS_SPEC="${DOMAINS_SPEC//,/ }"
read -r -a REQUESTED_MODELS <<< "${MODELS_SPEC}"
read -r -a REQUESTED_DOMAINS <<< "${DOMAINS_SPEC}"

EPOCHS="${EPOCHS:-600}"
PATIENCE="${PATIENCE:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECK_DATA="${CHECK_DATA:-1}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-cuda:0}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/Paper-Backup/E4-SMVIC-Curated-OneSeed}"
RUN_TIME="${RUN_TIME:-smvic_curated_one_seed_${SEED}_$(date +%Y%m%dT%H%M%S)}"
RUN_TIME="${RUN_TIME//\//_}"

AVAILABLE_MODELS=(hi_mlp raw_vanilla bicontext)
AVAILABLE_DOMAINS=(
  smvic_e72_69ah
  smvic_s5e891_51ah
  smvic_type1_18ah
  smvic_type2_150ah_t40
  smvic_type3_108ah
  smvic_type4_11ah
)

select_values() {
  local kind="$1" requested_name="$2" available_name="$3" output_name="$4"
  local -n requested_ref="${requested_name}"
  local -n available_ref="${available_name}"
  local -n output_ref="${output_name}"
  output_ref=()
  if (( ${#requested_ref[@]} == 1 )) && [[ "${requested_ref[0],,}" == "all" ]]; then
    output_ref=("${available_ref[@]}")
    return
  fi
  local requested available found duplicate selected
  for requested in "${requested_ref[@]}"; do
    requested="${requested,,}"
    found=0
    for available in "${available_ref[@]}"; do
      [[ "${requested}" == "${available}" ]] && found=1
    done
    (( found == 1 )) || {
      echo "Unknown ${kind} '${requested}'. Allowed: all ${available_ref[*]}" >&2
      exit 2
    }
    duplicate=0
    for selected in "${output_ref[@]:-}"; do
      [[ "${requested}" == "${selected}" ]] && duplicate=1
    done
    (( duplicate == 1 )) || output_ref+=("${requested}")
  done
}

SELECTED_MODELS=()
SELECTED_DOMAINS=()
select_values model REQUESTED_MODELS AVAILABLE_MODELS SELECTED_MODELS
select_values domain REQUESTED_DOMAINS AVAILABLE_DOMAINS SELECTED_DOMAINS

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "SEED must be one non-negative integer" >&2; exit 2; }
[[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_PARALLEL must be a positive per-GPU process limit" >&2; exit 2; }
[[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || { echo "NUM_WORKERS must be non-negative" >&2; exit 2; }
[[ "${CHECK_DATA}" =~ ^[01]$ ]] || { echo "CHECK_DATA must be 0 or 1" >&2; exit 2; }
[[ "${DRY_RUN}" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
(( ${#GPU_LIST[@]} > 0 )) || { echo "GPU_IDS must not be empty" >&2; exit 2; }
for variable in EPOCHS PATIENCE BATCH_SIZE; do
  value="${!variable}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "${variable} must be positive" >&2; exit 2; }
done
for gpu in "${GPU_LIST[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: ${gpu}" >&2; exit 2; }
done

CONFIGS=()
TASK_KEYS=()
TASK_SPLITS=()
TASK_DATA_IDS=()
for model in "${SELECTED_MODELS[@]}"; do
  for domain in "${SELECTED_DOMAINS[@]}"; do
    config="${CONFIG_ROOT}/${model}/${domain}.json"
    [[ -f "${config}" ]] || { echo "Missing config: ${config}" >&2; exit 2; }
    if [[ "${domain}" == "smvic_type3_108ah" ]]; then
      variants=(test_seed420 test_seed421)
    else
      variants=(test_cell01 test_cell02)
    fi
    for variant in "${variants[@]}"; do
      split="splits/smvic/${domain}__${variant}.json"
      [[ -f "${REPO_ROOT}/${split}" ]] || { echo "Missing split: ${split}" >&2; exit 2; }
      CONFIGS+=("${config}")
      TASK_KEYS+=("${model}__${domain}__${variant}")
      TASK_SPLITS+=("${split}")
      TASK_DATA_IDS+=("${domain}__${variant}")
    done
  done
done

validate_all() {
  local index=0 config
  for config in "${CONFIGS[@]}"; do
    index=$((index + 1))
    echo "[preflight ${index}/${#CONFIGS[@]}] ${TASK_KEYS[$((index - 1))]}"
    args=(
      --config "${config}"
      --split_file_override "${TASK_SPLITS[$((index - 1))]}"
      --data_id_override "${TASK_DATA_IDS[$((index - 1))]}"
      --seed "${SEED}"
      --output_root "${OUTPUT_ROOT}"
      --run_time "${RUN_TIME}"
      --epochs "${EPOCHS}"
      --patience "${PATIENCE}"
      --batch_size "${BATCH_SIZE}"
      --num_workers "${NUM_WORKERS}"
      --validate_only
    )
    [[ "${CHECK_DATA}" == 1 ]] && args+=(--check_data)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_experiment.py" "${args[@]}" >/dev/null
  done
  echo "[preflight] all ${#CONFIGS[@]} configs passed"
}

summarize_all() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_smvic_one_seed.py" \
    --root "${OUTPUT_ROOT}" --seed "${SEED}"
}

train_all() {
  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "[dry-run] ${#CONFIGS[@]} tasks validated; no training started"
    return
  fi
  wait_help="$(help wait 2>/dev/null || true)"
  [[ "${wait_help}" == *"-p"* ]] || {
    echo "Bash 5.1+ with wait -n -p is required; current=${BASH_VERSION}" >&2
    exit 2
  }

  # Construct exactly MAX_PARALLEL lanes for every physical GPU. Iterating
  # GPUs inside each slot keeps launch order balanced across devices.
  LANE_GPU=()
  for ((slot=0; slot<MAX_PARALLEL; slot++)); do
    for gpu in "${GPU_LIST[@]}"; do
      LANE_GPU+=("${gpu}")
    done
  done
  lane_count=${#LANE_GPU[@]}
  LAUNCHER_ROOT="${OUTPUT_ROOT}/_launcher_logs/smvic_one_seed/runtime_${RUN_TIME}"
  mkdir -p "${LAUNCHER_ROOT}"

  run_one() {
    local task_index="$1" gpu="$2"
    local config="${CONFIGS[$task_index]}" key="${TASK_KEYS[$task_index]}"
    local task_dir="${LAUNCHER_ROOT}/${key}"
    local log_file="${task_dir}/seed_${SEED}.log"
    mkdir -p "${task_dir}"
    args=(
      --config "${config}"
      --split_file_override "${TASK_SPLITS[$task_index]}"
      --data_id_override "${TASK_DATA_IDS[$task_index]}"
      --seed "${SEED}"
      --output_root "${OUTPUT_ROOT}"
      --run_time "${RUN_TIME}"
      --device_override "${DEVICE_OVERRIDE}"
      --epochs "${EPOCHS}"
      --patience "${PATIENCE}"
      --batch_size "${BATCH_SIZE}"
      --num_workers "${NUM_WORKERS}"
    )
    [[ "${CHECK_DATA}" == 1 ]] && args+=(--check_data)
    [[ -n "${BACKEND_OVERRIDE}" ]] && args+=(--backend_override "${BACKEND_OVERRIDE}")
    echo "[launch] ${key}; seed=${SEED}; gpu=${gpu}; log=${log_file}"
    local code=0
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/run_experiment.py" "${args[@]}" \
      >"${log_file}" 2>&1 || code=$?
    if (( code == 0 )); then
      echo "[complete] ${key}; gpu=${gpu}"
      return 0
    fi
    echo "[failed] ${key}; gpu=${gpu}; exit=${code}; log=${log_file}" >&2
    return "${code}"
  }

  LANE_PIDS=()
  for ((lane=0; lane<lane_count; lane++)); do LANE_PIDS+=(""); done
  declare -A PID_LANE=()
  next_task=0
  active=0
  overall=0
  while (( next_task < ${#CONFIGS[@]} || active > 0 )); do
    while (( next_task < ${#CONFIGS[@]} && active < lane_count )); do
      free=-1
      for ((candidate=0; candidate<lane_count; candidate++)); do
        [[ -z "${LANE_PIDS[$candidate]}" ]] && { free=${candidate}; break; }
      done
      (( free >= 0 )) || break
      run_one "${next_task}" "${LANE_GPU[$free]}" &
      pid=$!
      LANE_PIDS[$free]="${pid}"
      PID_LANE[$pid]="${free}"
      next_task=$((next_task + 1))
      active=$((active + 1))
    done
    (( active > 0 )) || break
    finished=""
    wait -n -p finished || overall=1
    lane="${PID_LANE[$finished]:-}"
    [[ -n "${lane}" ]] || { echo "Unknown completed PID: ${finished}" >&2; exit 1; }
    unset "PID_LANE[${finished}]"
    LANE_PIDS[$lane]=""
    active=$((active - 1))
  done
  (( overall == 0 )) || { echo "One or more SMVIC tasks failed; inspect ${LAUNCHER_ROOT}" >&2; exit 1; }
  echo "[training] all ${#CONFIGS[@]} tasks completed"
}

echo "Curated SMVIC one-seed comparison"
echo "seed: ${SEED}"
echo "models: ${SELECTED_MODELS[*]}"
echo "domains: ${SELECTED_DOMAINS[*]}"
echo "GPUs: ${GPU_LIST[*]}"
echo "max processes/GPU: ${MAX_PARALLEL}"
echo "maximum aggregate processes: $(( ${#GPU_LIST[@]} * MAX_PARALLEL ))"
echo "epochs/patience: ${EPOCHS}/${PATIENCE}"
echo "output: ${OUTPUT_ROOT}"

case "${STAGE}" in
  validate) validate_all ;;
  train) validate_all; train_all ;;
  summary) summarize_all ;;
  all) validate_all; train_all; [[ "${DRY_RUN}" == 1 ]] || summarize_all ;;
  *) echo "Usage: $0 {validate|train|summary|all}" >&2; exit 2 ;;
esac
