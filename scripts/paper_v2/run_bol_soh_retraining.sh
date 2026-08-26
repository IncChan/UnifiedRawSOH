#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8:backslashreplace}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-$(${REPO_ROOT}/scripts/resolve_python_bin.sh)}"

# ======================== 部分实验控制 ========================
# 所有选项都可以在命令行前临时覆盖，例如：
#
#   STAGE=e1_feature TARGET_DOMAINS=smarthealth_lishen40 SEEDS=42 \
#     DRY_RUN=0 bash scripts/paper_v2/run_bol_soh_retraining.sh
#
# STAGE：选择实验阶段。
#   e1_feature  - FeatureMLP-BOL 单域实验
#   e1_raw      - RawMamba 单域实验
#   e2_full     - 全域 RawMamba 实验；该阶段必须使用 TARGET_DOMAINS=all
#   e3_lodo     - Leave-One-Domain-Out 实验
#   all         - 按 e1_feature -> e1_raw -> e2_full -> e3_lodo 全部运行
#   如果只选择部分域，请同时指定单个阶段；STAGE=all 会包含要求全域的 e2_full。
#
# TARGET_DOMAINS：选择数据集，可写 all，或用逗号/空格分隔多个域。
#   可用域：xjtu、mit、smarthealth_lishen40、smarthealth_catl280、
#          smarthealth_eve280；也接受 lishen40/catl280/eve280 等别名。
#   例：TARGET_DOMAINS=smarthealth_lishen40
#       TARGET_DOMAINS=smarthealth_catl280,smarthealth_eve280
#
# SEEDS：选择随机种子，默认 42 52 62；只跑一个 seed 时写 SEEDS=42。
# GPU_IDS：提供物理 GPU 编号，逗号或空格分隔；例如 GPU_IDS=3,7。
# JOBS_PER_GPU：每张 GPU 同时运行的任务数。调试或显存紧张时建议设为 1。
#
# DRY_RUN=1：只打印将要运行的任务，不启动训练；确认筛选范围后改为 0。
# RESUME=1：跳过已经完整成功的 seed；失败任务仍会重跑。RESUME=0 强制重跑全部。
# OUTPUT_ROOT：输出根目录；不要让两个同时运行的 launcher 使用同一批输出目录。
# 如果 SmartHealth 预处理正在使用 --overwrite 写入数据，请等待它完成并通过
# smarthealth_validate 后再启动训练，避免训练读取到不完整的 RAW/FEATURE 产品。
#
# 可直接复制的常用例子：
#   # 只跑 Lishen40 的 FeatureMLP，seed 42
#   STAGE=e1_feature TARGET_DOMAINS=smarthealth_lishen40 SEEDS=42 \
#     GPU_IDS=7 JOBS_PER_GPU=1 DRY_RUN=0 \
#     bash scripts/paper_v2/run_bol_soh_retraining.sh
#
#   # 跑 CATL/EVE 的 FeatureMLP，三个 seed
#   STAGE=e1_feature TARGET_DOMAINS=smarthealth_catl280,smarthealth_eve280 \
#     SEEDS="42 52 62" GPU_IDS="3 7" JOBS_PER_GPU=1 DRY_RUN=0 \
#     bash scripts/paper_v2/run_bol_soh_retraining.sh
#
#   # 只跑 XJTU 和 MIT 的 RawMamba
#   STAGE=e1_raw TARGET_DOMAINS="xjtu mit" SEEDS="42 52 62" \
#     bash scripts/paper_v2/run_bol_soh_retraining.sh
#
#   # 只预览某一部分任务，不产生训练输出
#   STAGE=e1_feature TARGET_DOMAINS=smarthealth_lishen40 SEEDS=42 \
#     DRY_RUN=1 bash scripts/paper_v2/run_bol_soh_retraining.sh
#
# 注意：脚本设置 CUDA_VISIBLE_DEVICES=<GPU_IDS 中的物理编号> 后，训练进程
# 只看到本地 cuda:0，因此 DEVICE_OVERRIDE 默认应保持 cuda:0，不要改成物理编号。

STAGE="${STAGE:-all}"
TARGET_SPEC="${TARGET_DOMAINS:-all}"
SEED_SPEC="${SEEDS:-42 52 62}"
GPU_SPEC="${GPU_IDS:-7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-cuda:0}"
BACKEND_OVERRIDE="${BACKEND_OVERRIDE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"

SEED_SPEC="${SEED_SPEC//,/ }"
GPU_SPEC="${GPU_SPEC//,/ }"
TARGET_SPEC="${TARGET_SPEC//,/ }"
read -r -a SEED_LIST <<< "${SEED_SPEC}"
read -r -a GPU_LIST <<< "${GPU_SPEC}"
read -r -a TARGET_LIST <<< "${TARGET_SPEC}"
[[ "${OUTPUT_ROOT}" == /* ]] || OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"

case "${STAGE}" in
  e1_feature|e1_raw|e2_full|e3_lodo|all) ;;
  *) echo "STAGE must be e1_feature|e1_raw|e2_full|e3_lodo|all." >&2; exit 2 ;;
esac
[[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS_PER_GPU must be positive." >&2; exit 2; }
(( ${#SEED_LIST[@]} > 0 )) || { echo "SEEDS must not be empty." >&2; exit 2; }
(( ${#GPU_LIST[@]} > 0 )) || { echo "GPU_IDS must not be empty." >&2; exit 2; }
for seed in "${SEED_LIST[@]}"; do [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "Invalid seed: ${seed}" >&2; exit 2; }; done

ALL_DOMAINS=(xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280)
canonical_domain() {
  case "$1" in
    lishen40|smarthealth_lishen) echo smarthealth_lishen40 ;;
    catl280|smarthealth_catl) echo smarthealth_catl280 ;;
    eve280|smarthealth_eve) echo smarthealth_eve280 ;;
    xjtu|mit|smarthealth_lishen40|smarthealth_catl280|smarthealth_eve280) echo "$1" ;;
    *) echo "" ;;
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

RUN_TIME_BASE="${RUN_TIME:-paper_v2_retraining}"
RUN_TIME_BASE="${RUN_TIME_BASE//\//_}"
RUN_TIME_BASE="${RUN_TIME_BASE#runtime_}"

JOB_STAGE=(); JOB_DOMAIN=(); JOB_CONFIG=(); JOB_MODULE=(); JOB_RUNTIME=(); JOB_BATCH_ROOT=()
BATCH_ROOTS=()
declare -A BATCH_ROOT_SEEN=()
add_job() {
  local stage="$1" domain="$2" config="$3" module="$4" runtime="$5" experiment="$6" model="$7" data="$8"
  local batch_root="${OUTPUT_ROOT}/Paper-v2/${experiment}/${model}/${data}/runtime_${runtime}"
  JOB_STAGE+=("${stage}"); JOB_DOMAIN+=("${domain}"); JOB_CONFIG+=("${config}"); JOB_MODULE+=("${module}"); JOB_RUNTIME+=("${runtime}"); JOB_BATCH_ROOT+=("${batch_root}")
  if [[ -z "${BATCH_ROOT_SEEN[${batch_root}]+x}" ]]; then
    BATCH_ROOT_SEEN["${batch_root}"]=1
    BATCH_ROOTS+=("${batch_root}")
  fi
}

for requested_stage in e1_feature e1_raw e2_full e3_lodo; do
  [[ "${STAGE}" == all || "${STAGE}" == "${requested_stage}" ]] || continue
  case "${requested_stage}" in
    e1_feature)
      for domain in "${SELECTED_DOMAINS[@]}"; do
        add_job e1_feature "${domain}" "${REPO_ROOT}/configs/paper_v2/e1_single_domain/feature_mlp/${domain}.json" UnifiedRawSOH.main_baseline "${RUN_TIME_BASE}_e1_feature_${domain}" e1_single_domain FeatureMLP-BOL "${domain}"
      done ;;
    e1_raw)
      for domain in "${SELECTED_DOMAINS[@]}"; do
        add_job e1_raw "${domain}" "${REPO_ROOT}/configs/paper_v2/e1_single_domain/raw_mamba/${domain}.json" UnifiedRawSOH.main "${RUN_TIME_BASE}_e1_raw_${domain}" e1_single_domain RawMamba-noCycleAux "${domain}"
      done ;;
    e2_full)
      [[ "${TARGET_LIST[0]:-all}" == all ]] || { echo "e2_full requires TARGET_DOMAINS=all." >&2; exit 2; }
      add_job e2_full all "${REPO_ROOT}/configs/paper_v2/e2_full_domain/raw_mamba_domain_balanced.json" UnifiedRawSOH.main "${RUN_TIME_BASE}_e2_full" e2_full_domain RawMamba-noCycleAux full_domain ;;
    e3_lodo)
      for domain in "${SELECTED_DOMAINS[@]}"; do
        add_job e3_lodo "${domain}" "${REPO_ROOT}/configs/paper_v2/e3_lodo_zero_cell/lodo_${domain}.json" UnifiedRawSOH.main "${RUN_TIME_BASE}_e3_lodo_${domain}" e3_lodo_zero_cell RawMamba-noCycleAux "lodo_zero_cell_to_${domain}"
      done ;;
  esac
done
(( ${#JOB_CONFIG[@]} > 0 )) || { echo "No jobs selected." >&2; exit 2; }

TOTAL_LANES=$(( ${#GPU_LIST[@]} * JOBS_PER_GPU ))
LANE_GPU=()
for gpu in "${GPU_LIST[@]}"; do for ((slot=0; slot<JOBS_PER_GPU; slot++)); do LANE_GPU+=("${gpu}"); done; done

echo "Paper-v2 launcher"
echo "stage: ${STAGE}"
echo "target domains: ${SELECTED_DOMAINS[*]}"
echo "seeds: ${SEED_LIST[*]}"
echo "GPU IDs: ${GPU_LIST[*]}"
echo "jobs per GPU: ${JOBS_PER_GPU}"
echo "maximum aggregate processes: ${TOTAL_LANES}"
echo "output root: ${OUTPUT_ROOT}"
echo "dry run: ${DRY_RUN}"
echo "resume: ${RESUME}"

is_complete() {
  local run_dir="$1"
  [[ -f "${run_dir}/completed.status" ]] && grep -q '^completed$' "${run_dir}/completed.status" && [[ -s "${run_dir}/test_metrics.json" ]] && [[ -s "${run_dir}/metrics_by_cell.csv" ]] && [[ -s "${run_dir}/metrics_by_group.csv" ]] && [[ -s "${run_dir}/metrics_by_domain.csv" ]]
}

preview_index=0
for job_index in "${!JOB_CONFIG[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    lane=$((preview_index % TOTAL_LANES)); gpu="${LANE_GPU[${lane}]}"; run_dir="${JOB_BATCH_ROOT[${job_index}]}/seed_${seed}"
    if [[ "${RESUME}" == 1 ]] && is_complete "${run_dir}"; then
      echo "[resume] stage=${JOB_STAGE[${job_index}]} domain=${JOB_DOMAIN[${job_index}]}; seed=${seed}; GPU=${gpu}; output=${run_dir}"
    else
      echo "[job] stage=${JOB_STAGE[${job_index}]}; domain/fold=${JOB_DOMAIN[${job_index}]}; seed=${seed}; GPU=${gpu}; config=${JOB_CONFIG[${job_index}]}; output=${run_dir}"
    fi
    preview_index=$((preview_index + 1))
  done
done

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "dry run complete; no training process was started and no Paper-v2 output was written."
  exit 0
fi

run_one() {
  local job_index="$1" seed="$2" gpu="$3"
  local config="${JOB_CONFIG[$job_index]}"
  local module="${JOB_MODULE[$job_index]}"
  local runtime="${JOB_RUNTIME[$job_index]}"
  local batch_root="${JOB_BATCH_ROOT[$job_index]}"
  local run_dir="${batch_root}/seed_${seed}"
  local log_dir="${batch_root}/logs"
  local log_file="${log_dir}/seed_${seed}.log"
  mkdir -p "${log_dir}" "${run_dir}"
  local -a args=(--config "${config}" --output_root "${OUTPUT_ROOT}" --run_time "${runtime}" --device_override "${DEVICE_OVERRIDE}" --seed "${seed}")
  if [[ "${module}" == UnifiedRawSOH.main && -n "${BACKEND_OVERRIDE}" ]]; then args+=(--backend_override "${BACKEND_OVERRIDE}"); fi
  [[ -n "${EPOCHS:-}" ]] && args+=(--epochs "${EPOCHS}")
  [[ -n "${PATIENCE:-}" ]] && args+=(--patience "${PATIENCE}")
  [[ -n "${DEBUG_NUM_SAMPLES:-}" ]] && args+=(--debug_num_samples "${DEBUG_NUM_SAMPLES}")
  echo "[launch] stage=${JOB_STAGE[${job_index}]} domain=${JOB_DOMAIN[${job_index}]} seed=${seed} gpu=${gpu} config=${config} output=${run_dir} log=${log_file}"
  local code=0
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m "${module}" "${args[@]}" >"${log_file}" 2>&1 || code=$?
  if (( code == 0 )); then
    printf 'completed\n' > "${run_dir}/completed.status.tmp"
    mv -f "${run_dir}/completed.status.tmp" "${run_dir}/completed.status"
    rm -f "${run_dir}/failed.status"
    return 0
  fi
  printf 'failed:%s\n' "${code}" > "${run_dir}/failed.status"
  rm -f "${run_dir}/completed.status"
  echo "[failed] stage=${JOB_STAGE[${job_index}]} domain=${JOB_DOMAIN[${job_index}]} seed=${seed}; see ${log_file}" >&2
  return "${code}"
}

declare -a LANE_PIDS=()
for ((lane=0; lane<TOTAL_LANES; lane++)); do LANE_PIDS+=(""); done
overall_status=0
launch_index=0
for job_index in "${!JOB_CONFIG[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    lane=$((launch_index % TOTAL_LANES)); gpu="${LANE_GPU[${lane}]}"; run_dir="${JOB_BATCH_ROOT[${job_index}]}/seed_${seed}"
    if [[ "${RESUME}" == 1 ]] && is_complete "${run_dir}"; then launch_index=$((launch_index + 1)); continue; fi
    if [[ -n "${LANE_PIDS[$lane]:-}" ]]; then
      if ! wait "${LANE_PIDS[$lane]}"; then overall_status=1; fi
      LANE_PIDS[$lane]=
    fi
    run_one "${job_index}" "${seed}" "${gpu}" &
    LANE_PIDS[$lane]=$!
    launch_index=$((launch_index + 1))
  done
done
for ((lane=0; lane<TOTAL_LANES; lane++)); do
  if [[ -n "${LANE_PIDS[$lane]:-}" ]]; then if ! wait "${LANE_PIDS[$lane]}"; then overall_status=1; fi; fi
done

if (( overall_status != 0 )); then
  echo "one or more Paper-v2 jobs failed; no aggregate summary was generated." >&2
  exit 1
fi

SUMMARIZER="${SCRIPT_DIR}/summarize_bol_soh.py"
for batch_root in "${BATCH_ROOTS[@]}"; do
  expected_domains=""
  for job_index in "${!JOB_BATCH_ROOT[@]}"; do
    if [[ "${JOB_BATCH_ROOT[${job_index}]}" == "${batch_root}" ]]; then
      case "${JOB_STAGE[${job_index}]}" in
        e1_feature|e1_raw|e3_lodo) expected_domains="${JOB_DOMAIN[${job_index}]}" ;;
        e2_full) expected_domains="${ALL_DOMAINS[*]}" ;;
      esac
      break
    fi
  done
  read -r -a expected_domain_list <<< "${expected_domains}"
  "${PYTHON_BIN}" "${SUMMARIZER}" --batch_root "${batch_root}" --expected_seeds "${SEED_LIST[@]}" --expected_domains "${expected_domain_list[@]}"
done

if [[ "${STAGE}" == all && "${TARGET_LIST[0]:-all}" == all ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_main_table.py" --output_root "${OUTPUT_ROOT}" --expected_seeds "${SEED_LIST[@]}"
fi

echo "Paper-v2 retraining launcher completed successfully."
