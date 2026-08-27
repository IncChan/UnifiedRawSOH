#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}"

# ======================== 可调参数说明 ========================
# 运行模式与安全开关：
# DRY_RUN=1 只校验 fold/config 并打印任务，不训练、不创建 Paper-v2 输出；
#           确认 fold、模型和 seed 后再设为 0。
# RESUME=1 跳过已经有 completed.status、best.pt、test_metrics.json 和
#          split_info.json 的完整 seed；改配置后设为 0 才会强制重跑。
# CHECK_DATA_READINESS=1 在 DRY_RUN=0 时检查所有声明的数据目录和 split 文件。
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
CHECK_DATA_READINESS="${CHECK_DATA_READINESS:-1}"

# 随机种子：每个 seed 是一套独立训练/episode 随机流。调试时可设 SEEDS=42；
# 正式重复实验建议保留默认的 42/52/62，支持空格或逗号分隔。
SEED_SPEC="${SEEDS:-42 52 62}"

# GPU/并发：GPU_IDS 是物理 GPU 编号，可写 GPU_IDS="0 1"；JOBS_PER_GPU
# 控制每张卡同时跑几个 fold。E3 raw sequence、MoE 和 DG 较占显存，默认值
# 偏保守；显存足够时可提高，显存不足时设为 1。进程内 DEVICE_OVERRIDE
# 通常保持 cuda:0，因为 CUDA_VISIBLE_DEVICES 会重映射物理 GPU。
GPU_SPEC="${GPU_IDS:-6 7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"

# E3 模型/训练器消融：
# base_erm=Base-ERM；dense_adapter_erm=Dense Adapter-ERM；
# moe_erm=Residual MoE-ERM；moe_dg=Residual MoE + first-order MLDG。
# 可写 MODEL_VARIANTS=moe_dg 只跑 DG，或列出多个 variant 做对照。
MODEL_SPEC="${MODEL_VARIANTS:-base_erm dense_adapter_erm moe_erm moe_dg}"

# TARGET_DOMAINS 表示“留出的真实 target fold”，不是额外训练数据。
# all 会生成五折：每折使用其余四个 domain 做 source train/val；调试某一折
# 可写 TARGET_DOMAINS=xjtu，多个 fold 可写 TARGET_DOMAINS="xjtu mit"。
TARGET_SPEC="${TARGET_DOMAINS:-all}"

# 输出/backend：每个 fold、model、seed 都有独立目录；RUN_TIME 改名可避免
# 新实验复用旧目录。BACKEND_OVERRIDE 留空使用正式 mamba_ssm.Mamba；仅 CPU
# bounded smoke 使用 torch_reference。正式训练不要静默切换到 reference backend。
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
RUN_TIME_BASE="${RUN_TIME:-paper_v2_e3_zero_cell}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-cuda:0}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"

# 可选的 child-process 覆盖：EPOCHS/PATIENCE 调整训练轮数和早停；
# DEBUG_NUM_SAMPLES 仅用于极小规模 bounded smoke/debug，正式运行请不要设置。

# 允许 SEEDS/GPU_IDS/MODEL_VARIANTS/TARGET_DOMAINS 使用逗号或空格分隔。

SEED_SPEC="${SEED_SPEC//,/ }"
GPU_SPEC="${GPU_SPEC//,/ }"
MODEL_SPEC="${MODEL_SPEC//,/ }"
TARGET_SPEC="${TARGET_SPEC//,/ }"
read -r -a SEED_LIST <<< "${SEED_SPEC}"
read -r -a GPU_LIST <<< "${GPU_SPEC}"
read -r -a MODEL_LIST <<< "${MODEL_SPEC}"
read -r -a TARGET_LIST <<< "${TARGET_SPEC}"
[[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS_PER_GPU must be positive." >&2; exit 2; }
(( ${#SEED_LIST[@]} > 0 )) || { echo "SEEDS must not be empty." >&2; exit 2; }
(( ${#GPU_LIST[@]} > 0 )) || { echo "GPU_IDS must not be empty." >&2; exit 2; }

ALL_DOMAINS=(xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280)
canonical_domain() {
  case "$1" in
    xjtu|mit|smarthealth_lishen40|smarthealth_catl280|smarthealth_eve280) echo "$1" ;;
    lishen40|smarthealth_lishen) echo smarthealth_lishen40 ;;
    catl280|smarthealth_catl) echo smarthealth_catl280 ;;
    eve280|smarthealth_eve) echo smarthealth_eve280 ;;
    *) echo "" ;;
  esac
}
canonical_model() {
  case "$1" in
    base|base_erm) echo base_erm ;;
    dense|dense_adapter|dense_adapter_erm) echo dense_adapter_erm ;;
    moe|moe_erm|residual_moe) echo moe_erm ;;
    dg|moe_dg|residual_moe_dg) echo moe_dg ;;
    *) echo "" ;;
  esac
}
model_id_for() {
  case "$1" in
    base_erm) echo RawMambaV2-Base-ERM ;;
    dense_adapter_erm) echo RawMambaV2-DenseAdapter-ERM ;;
    moe_erm) echo RawMambaV2-ResidualMoE-ERM ;;
    moe_dg) echo RawMambaV2-ResidualMoE-DG ;;
  esac
}

SELECTED_DOMAINS=()
if [[ "${TARGET_LIST[0]:-all}" == all ]]; then
  SELECTED_DOMAINS=("${ALL_DOMAINS[@]}")
else
  for value in "${TARGET_LIST[@]}"; do
    domain="$(canonical_domain "${value}")"
    [[ -n "${domain}" ]] || { echo "Unsupported TARGET_DOMAINS entry: ${value}" >&2; exit 2; }
    SELECTED_DOMAINS+=("${domain}")
  done
fi

declare -a JOB_MODEL=() JOB_DOMAIN=() JOB_CONFIG=() JOB_RUNTIME=() JOB_MODEL_ID=()
for requested_model in "${MODEL_LIST[@]}"; do
  model="$(canonical_model "${requested_model}")"
  [[ -n "${model}" ]] || { echo "Unsupported MODEL_VARIANTS entry: ${requested_model}" >&2; exit 2; }
  for domain in "${SELECTED_DOMAINS[@]}"; do
    config="${REPO_ROOT}/configs/paper_v2/e3_lodo_zero_cell/${model}/lodo_${domain}.json"
    runtime="${RUN_TIME_BASE}_${model}_to_${domain}"
    runtime="${runtime//\//_}"
    runtime="${runtime#runtime_}"
    JOB_MODEL+=("${model}")
    JOB_DOMAIN+=("${domain}")
    JOB_CONFIG+=("${config}")
    JOB_RUNTIME+=("${runtime}")
    JOB_MODEL_ID+=("$(model_id_for "${model}")")
  done
done

TOTAL_LANES=$(( ${#GPU_LIST[@]} * JOBS_PER_GPU ))
# 每个 lane 固定绑定一张物理 GPU；总并发上限为 GPU 数 × JOBS_PER_GPU。
LANE_GPU=()
for gpu in "${GPU_LIST[@]}"; do
  for ((slot=0; slot<JOBS_PER_GPU; slot++)); do LANE_GPU+=("${gpu}"); done
done
[[ "${OUTPUT_ROOT}" == /* ]] || OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
echo "Paper-v2 E3 zero-cell launcher"
echo "models: ${MODEL_LIST[*]}"
echo "target folds: ${SELECTED_DOMAINS[*]}"
echo "seeds: ${SEED_LIST[*]}"
echo "GPU IDs: ${GPU_LIST[*]}"
echo "jobs per GPU: ${JOBS_PER_GPU}"
echo "maximum aggregate processes: ${TOTAL_LANES}"
echo "dry run: ${DRY_RUN}"
echo "resume: ${RESUME}"
echo "data readiness check for formal jobs: ${CHECK_DATA_READINESS}"
echo "output root: ${OUTPUT_ROOT}"

is_complete() {
  local run_dir="$1"
  [[ -s "${run_dir}/completed.status" ]] && grep -q '^completed$' "${run_dir}/completed.status" \
    && [[ -s "${run_dir}/best.pt" ]] && [[ -s "${run_dir}/test_metrics.json" ]] \
    && [[ -s "${run_dir}/split_info.json" ]]
}

for job_index in "${!JOB_CONFIG[@]}"; do
  validate_args=(--config "${JOB_CONFIG[$job_index]}" --output_root "${OUTPUT_ROOT}" --validate_only)
  if [[ "${DRY_RUN}" != 1 && "${CHECK_DATA_READINESS}" == 1 ]]; then
    validate_args+=(--validate_data_readiness)
  fi
  "${PYTHON_BIN}" -m UnifiedRawSOH.scripts.paper_v2.train "${validate_args[@]}" >/dev/null
  for seed in "${SEED_LIST[@]}"; do
    run_dir="${OUTPUT_ROOT}/Paper-v2/e3_lodo_zero_cell/${JOB_MODEL_ID[$job_index]}/lodo_zero_cell_to_${JOB_DOMAIN[$job_index]}/runtime_${JOB_RUNTIME[$job_index]}/seed_${seed}"
    if [[ "${RESUME}" == 1 ]] && is_complete "${run_dir}"; then
      echo "[resume] model=${JOB_MODEL[$job_index]}; fold=${JOB_DOMAIN[$job_index]}; seed=${seed}; GPU=dynamic; output=${run_dir}"
    else
      echo "[job] model=${JOB_MODEL[$job_index]}; fold=${JOB_DOMAIN[$job_index]}; seed=${seed}; GPU=dynamic; config=${JOB_CONFIG[$job_index]}; output=${run_dir}"
    fi
  done
done

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "dry run complete; no training process was started and no Paper-v2 output was written."
  exit 0
fi

run_one() {
  local job_index="$1" seed="$2" gpu="$3"
  local config="${JOB_CONFIG[$job_index]}"
  local runtime="${JOB_RUNTIME[$job_index]}"
  local model_id="${JOB_MODEL_ID[$job_index]}"
  local domain="${JOB_DOMAIN[$job_index]}"
  local model="${JOB_MODEL[$job_index]}"
  local batch_root="${OUTPUT_ROOT}/Paper-v2/e3_lodo_zero_cell/${model_id}/lodo_zero_cell_to_${domain}/runtime_${runtime}"
  local run_dir="${batch_root}/seed_${seed}"
  local log_dir="${batch_root}/logs"
  mkdir -p "${log_dir}" "${run_dir}"
  local -a args=(
    -m UnifiedRawSOH.scripts.paper_v2.train
    --config "${config}"
    --output_root "${OUTPUT_ROOT}"
    --run_time "${runtime}"
    --seed "${seed}"
    --device_override "${DEVICE_OVERRIDE}"
  )
  [[ -n "${BACKEND_OVERRIDE}" ]] && args+=(--backend_override "${BACKEND_OVERRIDE}")
  [[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")
  [[ -n "${PATIENCE:-}" ]] && args+=(--patience "${PATIENCE}")
  [[ -n "${DEBUG_NUM_SAMPLES:-}" ]] && args+=(--debug_num_samples "${DEBUG_NUM_SAMPLES}")
  echo "[launch] model=${model}; fold=${domain}; seed=${seed}; gpu=${gpu}; output=${run_dir}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${args[@]}" >"${log_dir}/seed_${seed}.log" 2>&1
}

declare -a PIDS=() PENDING_JOB=() PENDING_SEED=()
for ((lane=0; lane<TOTAL_LANES; lane++)); do PIDS+=(""); done
for job_index in "${!JOB_CONFIG[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    run_dir="${OUTPUT_ROOT}/Paper-v2/e3_lodo_zero_cell/${JOB_MODEL_ID[$job_index]}/lodo_zero_cell_to_${JOB_DOMAIN[$job_index]}/runtime_${JOB_RUNTIME[$job_index]}/seed_${seed}"
    if [[ "${RESUME}" == 1 ]] && is_complete "${run_dir}"; then continue; fi
    PENDING_JOB+=("${job_index}")
    PENDING_SEED+=("${seed}")
  done
done

pending_count=${#PENDING_JOB[@]}
next_pending=0
active=0
overall_status=0
while (( next_pending < pending_count || active > 0 )); do
  while (( next_pending < pending_count && active < TOTAL_LANES )); do
    free_lane=-1
    for ((candidate=0; candidate<TOTAL_LANES; candidate++)); do
      if [[ -z "${PIDS[$candidate]}" ]]; then free_lane=$candidate; break; fi
    done
    job_index="${PENDING_JOB[$next_pending]}"
    seed="${PENDING_SEED[$next_pending]}"
    run_one "${job_index}" "${seed}" "${LANE_GPU[$free_lane]}" &
    PIDS[$free_lane]=$!
    next_pending=$((next_pending + 1))
    active=$((active + 1))
  done
  for ((lane=0; lane<TOTAL_LANES; lane++)); do
    if [[ -n "${PIDS[$lane]}" ]]; then
      if ! wait "${PIDS[$lane]}"; then overall_status=1; fi
      PIDS[$lane]=""
      active=$((active - 1))
      break
    fi
  done
done

if (( overall_status != 0 )); then
  echo "one or more E3 Paper-v2 jobs failed; no aggregate summary was generated." >&2
  exit 1
fi
echo "E3 Paper-v2 jobs completed."
