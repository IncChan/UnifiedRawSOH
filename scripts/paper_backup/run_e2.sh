#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

# Dynamic multi-GPU launcher for the Paper-Backup E2 matrix. All five E2
# views use the offline FULL-matched physical-cycle cohort. Every config/seed
# pair is one schedulable job; each GPU owns JOBS_PER_GPU lanes.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}"

DRY_RUN="${DRY_RUN:-0}"
CHECK_DATA="${CHECK_DATA:-0}"
EXPERIMENT_SUITE="${EXPERIMENT_SUITE:-e2_charging_information}"
if [[ "${EXPERIMENT_SUITE}" == "e2_final_interaction_5seed" ]]; then
  AVAILABLE_MODELS=(full_vanilla raw_dual_vanilla ours_interaction)
  DEFAULT_SEEDS="42 52 62 72 82"
  DEFAULT_EPOCHS=600
  DEFAULT_PATIENCE=30
  DEFAULT_NUM_WORKERS=4
  DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/Paper-Backup/E2-Final-Interaction-5Seed"
elif [[ "${EXPERIMENT_SUITE}" == "e2_final_256budget" ]]; then
  AVAILABLE_MODELS=(full_vanilla_256 terminal_vanilla_sep_128x128 ours_cc_only_128 ours_cv_only_128 ours_pointbridge_128x128)
  DEFAULT_SEEDS="42 52 62 72 82 92 102 112 122 123"
  DEFAULT_PATIENCE=30
  DEFAULT_NUM_WORKERS=4
  DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/Paper-Backup/E2-Final-256Budget"
  DEFAULT_EPOCHS=400
else
  AVAILABLE_MODELS=(full_vanilla terminal_cc_only terminal_cv_only terminal_ours terminal_vanilla)
  DEFAULT_SEEDS="42 52 62"
  DEFAULT_PATIENCE=20
  DEFAULT_NUM_WORKERS=1
  DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs/Paper-Backup"
  DEFAULT_EPOCHS=400
fi
SEED_SPEC="${SEEDS:-${DEFAULT_SEEDS}}"
GPU_SPEC="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-4 5 6 7}}"
MODEL_SPEC="${MODELS:-all}"
JOBS_PER_GPU="${JOBS_PER_GPU:-${MAX_PARALLEL:-3}}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-cuda:0}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
PATIENCE="${PATIENCE:-${DEFAULT_PATIENCE}}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
RUN_TIME="${RUN_TIME:-${EXPERIMENT_SUITE}_$(date +%Y%m%dT%H%M%S)}"

SEED_SPEC="${SEED_SPEC//,/ }"
GPU_SPEC="${GPU_SPEC//,/ }"
MODEL_SPEC="${MODEL_SPEC//,/ }"
read -r -a SEED_LIST <<< "${SEED_SPEC}"
read -r -a GPU_LIST <<< "${GPU_SPEC}"
read -r -a REQUESTED_MODELS <<< "${MODEL_SPEC}"
SELECTED_MODELS=()
if (( ${#REQUESTED_MODELS[@]} == 1 )) && [[ "${REQUESTED_MODELS[0],,}" == "all" ]]; then
  SELECTED_MODELS=("${AVAILABLE_MODELS[@]}")
else
  for requested in "${REQUESTED_MODELS[@]}"; do
    requested="${requested,,}"
    [[ "${requested}" != "all" ]] || {
      echo "MODELS=all cannot be combined with explicit E2 models." >&2
      exit 2
    }
    found=0
    for available in "${AVAILABLE_MODELS[@]}"; do
      if [[ "${requested}" == "${available}" ]]; then
        found=1
        break
      fi
    done
    (( found == 1 )) || {
      echo "Unknown E2 model '${requested}'. Allowed: all ${AVAILABLE_MODELS[*]}" >&2
      exit 2
    }
    duplicate=0
    for selected in "${SELECTED_MODELS[@]}"; do
      [[ "${requested}" == "${selected}" ]] && duplicate=1
    done
    (( duplicate == 1 )) || SELECTED_MODELS+=("${requested}")
  done
fi
(( ${#SELECTED_MODELS[@]} > 0 )) || {
  echo "MODELS must select at least one E2 model." >&2
  exit 2
}
mapfile -t CONFIGS < <(
  for model in "${SELECTED_MODELS[@]}"; do
    find "${REPO_ROOT}/configs/paper_backup/${EXPERIMENT_SUITE}/${model}" \
      -maxdepth 1 -type f -name '*.json'
  done | sort
)

[[ "${DRY_RUN}" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1." >&2; exit 2; }
[[ "${CHECK_DATA}" =~ ^[01]$ ]] || { echo "CHECK_DATA must be 0 or 1." >&2; exit 2; }
[[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS_PER_GPU must be a positive integer." >&2; exit 2; }
(( ${#SEED_LIST[@]} > 0 )) || { echo "SEEDS must not be empty." >&2; exit 2; }
(( ${#GPU_LIST[@]} > 0 )) || { echo "GPU_IDS must not be empty." >&2; exit 2; }
(( ${#CONFIGS[@]} > 0 )) || { echo "No Paper-Backup E2 configs found." >&2; exit 2; }
for seed in "${SEED_LIST[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "Invalid seed: ${seed}" >&2; exit 2; }
done
for variable in EPOCHS PATIENCE BATCH_SIZE; do
  value="${!variable:-}"
  [[ -z "${value}" || "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${variable} must be a positive integer when set." >&2
    exit 2
  }
done
[[ -z "${DEBUG_NUM_SAMPLES:-}" || "${DEBUG_NUM_SAMPLES}" =~ ^[0-9]+$ ]] || {
  echo "DEBUG_NUM_SAMPLES must be a non-negative integer when set." >&2
  exit 2
}
[[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || {
  echo "NUM_WORKERS must be a non-negative integer." >&2
  exit 2
}
for gpu_index in "${!GPU_LIST[@]}"; do
  [[ -n "${GPU_LIST[$gpu_index]}" ]] || { echo "GPU_IDS contains an empty value." >&2; exit 2; }
  for previous_index in "${!GPU_LIST[@]}"; do
    (( previous_index < gpu_index )) || break
    if [[ "${GPU_LIST[$gpu_index]}" == "${GPU_LIST[$previous_index]}" ]]; then
      echo "GPU_IDS contains duplicate GPU ID: ${GPU_LIST[$gpu_index]}" >&2
      exit 2
    fi
  done
done

[[ "${OUTPUT_ROOT}" == /* ]] || OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
RUN_TIME="${RUN_TIME//\//_}"
RUN_TIME="${RUN_TIME#runtime_}"
TOTAL_LANES=$(( ${#GPU_LIST[@]} * JOBS_PER_GPU ))
LANE_GPU=()
for gpu in "${GPU_LIST[@]}"; do
  for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
    LANE_GPU+=("${gpu}")
  done
done

echo "Paper-Backup E2 dynamic launcher"
echo "models: ${SELECTED_MODELS[*]}"
echo "configs: ${#CONFIGS[@]}"
echo "seeds: ${SEED_LIST[*]}"
echo "GPU IDs: ${GPU_LIST[*]}"
echo "jobs per GPU: ${JOBS_PER_GPU}"
echo "maximum aggregate processes: ${TOTAL_LANES}"
echo "device inside each child: ${DEVICE_OVERRIDE}"
echo "epochs: ${EPOCHS}"
echo "early-stop patience: ${PATIENCE}"
echo "batch size: ${BATCH_SIZE}"
echo "DataLoader workers per training: ${NUM_WORKERS}"
echo "run time: ${RUN_TIME}"
echo "output root: ${OUTPUT_ROOT}"
echo "dry run: ${DRY_RUN}"

# Validate each unique config once before starting any training process. This
# prevents one missing FULL/terminal product or split from causing three
# identical seed failures after the GPU queue has already started.
config_index=0
for config in "${CONFIGS[@]}"; do
  config_index=$((config_index + 1))
  relative="${config#${REPO_ROOT}/configs/paper_backup/${EXPERIMENT_SUITE}/}"
  echo "[preflight ${config_index}/${#CONFIGS[@]}] ${relative}"
  validate_args=(
    --config "${config}"
    --output_root "${OUTPUT_ROOT}"
    --run_time "${RUN_TIME}"
    --validate_only
  )
  [[ "${CHECK_DATA}" == 1 ]] && validate_args+=(--check_data)
  validate_args+=(--epochs "${EPOCHS}")
  validate_args+=(--patience "${PATIENCE}")
  validate_args+=(--batch_size "${BATCH_SIZE}")
  validate_args+=(--num_workers "${NUM_WORKERS}")
  [[ -n "${DEBUG_NUM_SAMPLES:-}" ]] && validate_args+=(--debug_num_samples "${DEBUG_NUM_SAMPLES}")
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_experiment.py" "${validate_args[@]}" >/dev/null
done
echo "[preflight] all ${#CONFIGS[@]} unique configs passed"

TASK_CONFIG=()
TASK_SEED=()
TASK_KEY=()
for config in "${CONFIGS[@]}"; do
  relative="${config#${REPO_ROOT}/configs/paper_backup/${EXPERIMENT_SUITE}/}"
  key="${relative%.json}"
  key="${key//\//__}"
  for seed in "${SEED_LIST[@]}"; do
    TASK_CONFIG+=("${config}")
    TASK_SEED+=("${seed}")
    TASK_KEY+=("${key}")
    echo "[job] key=${key}; seed=${seed}; GPU=dynamic; config=${config}"
  done
done

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "dry run complete; ${#TASK_CONFIG[@]} jobs validated and no training process was started."
  exit 0
fi

wait_help="$(help wait 2>/dev/null || true)"
if [[ "${wait_help}" != *"-p"* ]]; then
  echo "Dynamic scheduling requires Bash 5.1+ wait -n -p; current Bash: ${BASH_VERSION}" >&2
  exit 2
fi

LAUNCHER_ROOT="${OUTPUT_ROOT}/_launcher_logs/${EXPERIMENT_SUITE}/runtime_${RUN_TIME}"
mkdir -p "${LAUNCHER_ROOT}"

run_one() {
  local task_index="$1" gpu="$2"
  local config="${TASK_CONFIG[$task_index]}"
  local seed="${TASK_SEED[$task_index]}"
  local key="${TASK_KEY[$task_index]}"
  local task_dir="${LAUNCHER_ROOT}/${key}"
  local log_file="${task_dir}/seed_${seed}.log"
  local success_file="${task_dir}/seed_${seed}.completed"
  local failure_file="${task_dir}/seed_${seed}.failed"
  mkdir -p "${task_dir}"
  rm -f "${success_file}" "${failure_file}"

  local -a args=(
    --config "${config}"
    --seed "${seed}"
    --output_root "${OUTPUT_ROOT}"
    --run_time "${RUN_TIME}"
    --device_override "${DEVICE_OVERRIDE}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
  )
  [[ -n "${BACKEND_OVERRIDE}" ]] && args+=(--backend_override "${BACKEND_OVERRIDE}")
  [[ "${CHECK_DATA}" == 1 ]] && args+=(--check_data)
  [[ -n "${DEBUG_NUM_SAMPLES:-}" ]] && args+=(--debug_num_samples "${DEBUG_NUM_SAMPLES}")

  echo "[launch] key=${key}; seed=${seed}; gpu=${gpu}; log=${log_file}"
  local code=0
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_experiment.py" "${args[@]}" \
    >"${log_file}" 2>&1 || code=$?
  if (( code == 0 )); then
    printf 'completed\n' > "${success_file}"
    echo "[complete] key=${key}; seed=${seed}; gpu=${gpu}"
    return 0
  fi
  printf 'failed:%s\n' "${code}" > "${failure_file}"
  echo "[failed] key=${key}; seed=${seed}; gpu=${gpu}; exit=${code}; log=${log_file}" >&2
  return "${code}"
}

declare -a LANE_PIDS=()
declare -A PID_LANE=()
declare -A PID_TASK=()
for ((lane=0; lane<TOTAL_LANES; lane++)); do LANE_PIDS+=(""); done

terminate_children() {
  trap - INT TERM
  echo "[scheduler] interruption received; terminating active training jobs" >&2
  local pid
  for pid in "${!PID_LANE[@]}"; do kill "${pid}" 2>/dev/null || true; done
  wait || true
  exit 130
}
trap terminate_children INT TERM

task_count=${#TASK_CONFIG[@]}
next_task=0
active_jobs=0
overall_status=0

while (( next_task < task_count || active_jobs > 0 )); do
  # Fill every free lane before waiting. A lane is permanently associated
  # with one physical GPU, so per-GPU concurrency cannot exceed the limit.
  while (( next_task < task_count && active_jobs < TOTAL_LANES )); do
    free_lane=-1
    for ((candidate=0; candidate<TOTAL_LANES; candidate++)); do
      if [[ -z "${LANE_PIDS[$candidate]:-}" ]]; then
        free_lane=${candidate}
        break
      fi
    done
    if (( free_lane < 0 )); then
      echo "[scheduler-error] no free lane with active_jobs=${active_jobs}" >&2
      overall_status=1
      break
    fi

    gpu="${LANE_GPU[$free_lane]}"
    run_one "${next_task}" "${gpu}" &
    pid=$!
    LANE_PIDS[$free_lane]="${pid}"
    PID_LANE[$pid]="${free_lane}"
    PID_TASK[$pid]="${next_task}"
    next_task=$((next_task + 1))
    active_jobs=$((active_jobs + 1))
  done

  (( active_jobs > 0 )) || break
  finished_pid=""
  if wait -n -p finished_pid; then
    :
  else
    overall_status=1
  fi
  finished_lane=""
  if [[ -n "${finished_pid}" ]]; then
    finished_lane="${PID_LANE[$finished_pid]:-}"
  fi
  if [[ -z "${finished_lane}" ]]; then
    echo "[scheduler-error] completed PID ${finished_pid:-unknown} has no GPU lane" >&2
    overall_status=1
    for pid in "${!PID_LANE[@]}"; do
      if ! wait "${pid}"; then overall_status=1; fi
    done
    break
  fi
  unset "PID_LANE[${finished_pid}]" "PID_TASK[${finished_pid}]"
  LANE_PIDS[$finished_lane]=""
  active_jobs=$((active_jobs - 1))
done

trap - INT TERM
if (( overall_status != 0 )); then
  echo "one or more Paper-Backup E2 jobs failed; inspect ${LAUNCHER_ROOT}" >&2
  exit 1
fi

echo "Paper-Backup E2 matrix completed successfully."
echo "launcher logs: ${LAUNCHER_ROOT}"
SUMMARY_SELECTOR="e2"
[[ "${EXPERIMENT_SUITE}" == "e2_final_256budget" ]] && SUMMARY_SELECTOR="e2_final_256budget"
echo "summary: ${PYTHON_BIN} ${SCRIPT_DIR}/summarize_results.py --experiment ${SUMMARY_SELECTOR} --seeds '${SEED_LIST[*]}'"
