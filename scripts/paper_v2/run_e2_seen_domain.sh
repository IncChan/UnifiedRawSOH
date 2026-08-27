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
# DRY_RUN=1 只解析配置、打印 job matrix，不启动训练，也不创建 Paper-v2 输出；
#           第一次调整参数时建议保持 1。确认任务矩阵后再设为 0。
# RESUME=1 发现某个 seed 已同时存在 completed.status、best.pt、
#          test_metrics.json 和 metrics_by_domain.csv 时跳过它；改配置或想
#          重跑时设为 0。所有 seed 完成后仍会自动补写批次级平均结果。
# CHECK_DATA_READINESS=1 在真正训练前检查配置声明的数据目录和 split 文件；
#                      仅对 DRY_RUN=0 生效，适合防止批量任务启动后才发现路径错误。
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
CHECK_DATA_READINESS="${CHECK_DATA_READINESS:-1}"

# 随机性：三个 seed 用于独立重复实验。调试可写 SEEDS=42，正式矩阵可写
# SEEDS="42 52 62" 或 SEEDS=42,52,62；每个 seed 会产生独立 checkpoint/output。
SEED_SPEC="${SEEDS:-42 52 62}"

# GPU/并发：GPU_IDS 使用物理 GPU 编号，例如 GPU_IDS="0 1" 或 GPU_IDS=0,1。
# JOBS_PER_GPU 是每张卡允许同时运行的 child process 数，实际总并发为
# GPU 数 × JOBS_PER_GPU。显存不足时优先改成 1；提高它只会提高吞吐，
# 不会改变单个任务的 batch size。DEVICE_OVERRIDE 仍应使用本地 cuda:0，
# 因为下面的 CUDA_VISIBLE_DEVICES 会把物理 GPU 映射成进程内的 cuda:0。
GPU_SPEC="${GPU_IDS:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"

# E2 模型消融：base=Base-ERM，dense_adapter=参数匹配 Dense-Adapter-ERM，
# moe_erm=Residual-MoE-ERM。可用 MODEL_VARIANTS=dense_adapter 只跑一个对照，
# 或 MODEL_VARIANTS="base moe_erm" 跑指定组合。TARGET_DOMAINS 必须是 all，
# 因为 E2 的协议是五个 domain 联合训练，而不是单个 held-out fold。
MODEL_SPEC="${MODEL_VARIANTS:-moe_erm}" # base dense_adapter moe_erm
TARGET_SPEC="${TARGET_DOMAINS:-all}"

# 输出与 backend：OUTPUT_ROOT 最好为本次实验专用目录，避免不同矩阵互相覆盖；
# RUN_TIME 用于区分同一 model/data/seed 的不同运行批次，改超参数后建议改名。
# BACKEND_OVERRIDE 留空时使用 config 中的正式 mamba_ssm.Mamba；只有 CPU bounded
# smoke 才设置 BACKEND_OVERRIDE=torch_reference，不要用它替代正式 backend。
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
RUN_TIME_BASE="${RUN_TIME:-paper_v2_e2_seen_domain}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-cuda:0}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"

# 以下三个变量默认不覆盖 config：
# EPOCHS/PATIENCE 可用于 pilot 调整训练长度和早停；DEBUG_NUM_SAMPLES 只应在
# bounded smoke/debug 时设置，例如 DEBUG_NUM_SAMPLES=2，正式实验请保持为空。

# 以下逗号替换允许 shell 参数同时支持空格分隔和逗号分隔；一般不需要修改。

SEED_SPEC="${SEED_SPEC//,/ }"
GPU_SPEC="${GPU_SPEC//,/ }"
MODEL_SPEC="${MODEL_SPEC//,/ }"
read -r -a SEED_LIST <<< "${SEED_SPEC}"
read -r -a GPU_LIST <<< "${GPU_SPEC}"
read -r -a MODEL_LIST <<< "${MODEL_SPEC}"
[[ "${TARGET_SPEC}" == "all" ]] || {
  echo "E2 seen-domain requires TARGET_DOMAINS=all; got ${TARGET_SPEC}" >&2
  exit 2
}
[[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS_PER_GPU must be positive." >&2; exit 2; }
(( ${#SEED_LIST[@]} > 0 )) || { echo "SEEDS must not be empty." >&2; exit 2; }
(( ${#GPU_LIST[@]} > 0 )) || { echo "GPU_IDS must not be empty." >&2; exit 2; }

ALL_DOMAINS=(xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280)

canonical_model() {
  case "$1" in
    base|base_erm) echo base ;;
    dense|dense_adapter|dense_adapter_erm) echo dense_adapter ;;
    moe|moe_erm|residual_moe) echo moe_erm ;;
    *) echo "" ;;
  esac
}
model_id_for() {
  case "$1" in
    base) echo RawMambaV2-Base-ERM ;;
    dense_adapter) echo RawMambaV2-DenseAdapter-ERM ;;
    moe_erm) echo RawMambaV2-ResidualMoE-ERM ;;
  esac
}
data_id_for() {
  case "$1" in
    base) echo full_domain_base_erm ;;
    dense_adapter) echo full_domain_dense_adapter_erm ;;
    moe_erm) echo full_domain_residual_moe_erm ;;
  esac
}

declare -a JOB_MODEL=() JOB_MODEL_ID=() JOB_DATA_ID=() JOB_CONFIG=() JOB_RUNTIME=()
for requested_model in "${MODEL_LIST[@]}"; do
  model="$(canonical_model "${requested_model}")"
  [[ -n "${model}" ]] || { echo "Unsupported MODEL_VARIANTS entry: ${requested_model}" >&2; exit 2; }
  case "${model}" in
    base) config="${REPO_ROOT}/configs/paper_v2/e2_full_domain/base/config.json" ;;
    dense_adapter) config="${REPO_ROOT}/configs/paper_v2/e2_full_domain/dense_adapter/config.json" ;;
    moe_erm) config="${REPO_ROOT}/configs/paper_v2/e2_full_domain/moe_erm/config.json" ;;
  esac
  runtime="${RUN_TIME_BASE}_${model}"
  runtime="${runtime//\//_}"
  runtime="${runtime#runtime_}"
  JOB_MODEL+=("${model}")
  JOB_MODEL_ID+=("$(model_id_for "${model}")")
  JOB_DATA_ID+=("$(data_id_for "${model}")")
  JOB_CONFIG+=("${config}")
  JOB_RUNTIME+=("${runtime}")
done

TOTAL_LANES=$(( ${#GPU_LIST[@]} * JOBS_PER_GPU ))
# 每个 lane 绑定一个 GPU 槽位；TOTAL_LANES 只限制并发，不会改变实验数量。
LANE_GPU=()
for gpu in "${GPU_LIST[@]}"; do
  for ((slot=0; slot<JOBS_PER_GPU; slot++)); do LANE_GPU+=("${gpu}"); done
done
[[ "${OUTPUT_ROOT}" == /* ]] || OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
echo "Paper-v2 E2 seen-domain launcher"
echo "models: ${JOB_MODEL[*]}"
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
    && [[ -s "${run_dir}/metrics_by_domain.csv" ]]
}

for job_index in "${!JOB_CONFIG[@]}"; do
  validate_args=(--config "${JOB_CONFIG[$job_index]}" --output_root "${OUTPUT_ROOT}" --validate_only)
  if [[ "${DRY_RUN}" != 1 && "${CHECK_DATA_READINESS}" == 1 ]]; then
    validate_args+=(--validate_data_readiness)
  fi
  "${PYTHON_BIN}" -m UnifiedRawSOH.scripts.paper_v2.train "${validate_args[@]}" >/dev/null
  for seed in "${SEED_LIST[@]}"; do
    run_dir="${OUTPUT_ROOT}/Paper-v2/e2_full_domain/${JOB_MODEL_ID[$job_index]}/${JOB_DATA_ID[$job_index]}/runtime_${JOB_RUNTIME[$job_index]}/seed_${seed}"
    if [[ "${RESUME}" == 1 ]] && is_complete "${run_dir}"; then
      echo "[resume] model=${JOB_MODEL[$job_index]}; seed=${seed}; GPU=dynamic; output=${run_dir}"
    else
      echo "[job] model=${JOB_MODEL[$job_index]}; seed=${seed}; GPU=dynamic; config=${JOB_CONFIG[$job_index]}; output=${run_dir}"
    fi
  done
done

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "dry run complete; no training process was started and no Paper-v2 output was written."
  exit 0
fi

run_one() {
  local job_index="$1" seed="$2" gpu="$3"
  local model="${JOB_MODEL[$job_index]}"
  local config="${JOB_CONFIG[$job_index]}"
  local runtime="${JOB_RUNTIME[$job_index]}"
  local model_id="${JOB_MODEL_ID[$job_index]}"
  local data_id="${JOB_DATA_ID[$job_index]}"
  local batch_root="${OUTPUT_ROOT}/Paper-v2/e2_full_domain/${model_id}/${data_id}/runtime_${runtime}"
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
  echo "[launch] model=${model}; seed=${seed}; gpu=${gpu}; config=${config}; output=${run_dir}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${args[@]}" >"${log_dir}/seed_${seed}.log" 2>&1
}

declare -a PIDS=() PENDING_JOB=() PENDING_SEED=()
for ((lane=0; lane<TOTAL_LANES; lane++)); do PIDS+=(""); done
for job_index in "${!JOB_CONFIG[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    run_dir="${OUTPUT_ROOT}/Paper-v2/e2_full_domain/${JOB_MODEL_ID[$job_index]}/${JOB_DATA_ID[$job_index]}/runtime_${JOB_RUNTIME[$job_index]}/seed_${seed}"
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
  echo "one or more E2 Paper-v2 jobs failed; no aggregate summary was generated." >&2
  exit 1
fi

AGGREGATOR="${SCRIPT_DIR}/aggregate_metrics_by_domain.py"
for job_index in "${!JOB_CONFIG[@]}"; do
  batch_root="${OUTPUT_ROOT}/Paper-v2/e2_full_domain/${JOB_MODEL_ID[$job_index]}/${JOB_DATA_ID[$job_index]}/runtime_${JOB_RUNTIME[$job_index]}"
  echo "[aggregate] model=${JOB_MODEL[$job_index]}; seeds=${SEED_LIST[*]}; output=${batch_root}/metrics_by_domain.csv"
  "${PYTHON_BIN}" "${AGGREGATOR}" \
    --batch_root "${batch_root}" \
    --expected_seeds "${SEED_LIST[@]}" \
    --expected_domains "${ALL_DOMAINS[@]}"
done

echo "E2 Paper-v2 jobs completed; batch-level metrics_by_domain.csv generated."
