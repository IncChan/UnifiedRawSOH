#!/usr/bin/env bash
# Portable source-to-summary launcher for the final five-seed E1/E2 suites.
#
# This script intentionally uses only the currently activated Conda Python.
# Vendor source paths are supplied by flags or environment variables, so no
# machine-specific preprocess/paths.env is required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STAGE="${STAGE:-all}"
XJTU_SOURCE_ROOT="${XJTU_SOURCE_ROOT:-}"
MIT_SOURCE_ROOT="${MIT_SOURCE_ROOT:-}"
SMARTHEALTH_SOURCE_ROOT="${SMARTHEALTH_SOURCE_ROOT:-}"
GPU_IDS="${GPU_IDS:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-4}"
CANONICAL_MODE="${CANONICAL_MODE:-auto}"
PAPER_MODE="${PAPER_MODE:-auto}"
E1_OUTPUT_ROOT="${E1_OUTPUT_ROOT:-${REPO_ROOT}/outputs/Paper-Backup/E1-Final-Interaction-5Seed}"
E2_OUTPUT_ROOT="${E2_OUTPUT_ROOT:-${REPO_ROOT}/outputs/Paper-Backup/E2-Final-Interaction-5Seed}"
E1_DATA_ROOT="${REPO_ROOT}/datasets/PaperBackup_preprocessed_v2_128x128"
E2_DATA_ROOT="${REPO_ROOT}/datasets/PaperBackup_preprocessed_v2_e2_5domain_256"
MIT_FULL_ROOT="${MIT_FULL_CSV_ROOT:-${REPO_ROOT}/datasets/MIT_full_cccv_physical124}"
RUNTIME_CONFIG_FILE=""

cleanup() {
  if [[ -n "${RUNTIME_CONFIG_FILE}" && -f "${RUNTIME_CONFIG_FILE}" ]]; then
    rm -f "${RUNTIME_CONFIG_FILE}"
  fi
}
trap cleanup EXIT

usage() {
  cat <<'EOF'
Usage:
  conda activate <environment>
  bash scripts/paper_backup/run_final_interaction_5seed_portable.sh [options]

Required for stages that include preprocessing:
  --xjtu-source PATH
  --mit-source PATH
  --smarthealth-source PATH

Options:
  --stage all|preprocess|train|e1|e2|summary   default: all
  --gpus "0 1 2 3"                            default: 0
  --jobs-per-gpu N                            default: 1
  --workers N                                 default: 4
  --canonical-mode auto|rebuild|skip          default: auto
  --paper-mode auto|rebuild|skip              default: auto
  -h, --help

Equivalent environment variables are also accepted: XJTU_SOURCE_ROOT,
MIT_SOURCE_ROOT, SMARTHEALTH_SOURCE_ROOT, GPU_IDS, JOBS_PER_GPU,
PREPROCESS_WORKERS, CANONICAL_MODE and PAPER_MODE.

Modes:
  auto     reuse a complete product, build a missing product, reject partial data
  rebuild  deliberately overwrite the corresponding generated product
  skip     trust an existing product and proceed to validation/training
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --xjtu-source) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; XJTU_SOURCE_ROOT="$2"; shift 2 ;;
    --mit-source) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; MIT_SOURCE_ROOT="$2"; shift 2 ;;
    --smarthealth-source) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; SMARTHEALTH_SOURCE_ROOT="$2"; shift 2 ;;
    --gpus) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; GPU_IDS="$2"; shift 2 ;;
    --jobs-per-gpu) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; JOBS_PER_GPU="$2"; shift 2 ;;
    --workers) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; PREPROCESS_WORKERS="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; STAGE="$2"; shift 2 ;;
    --canonical-mode) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; CANONICAL_MODE="$2"; shift 2 ;;
    --paper-mode) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; PAPER_MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${STAGE}" in all|preprocess|train|e1|e2|summary) ;; *) echo "Invalid --stage: ${STAGE}" >&2; exit 2 ;; esac
case "${CANONICAL_MODE}" in auto|rebuild|skip) ;; *) echo "Invalid --canonical-mode: ${CANONICAL_MODE}" >&2; exit 2 ;; esac
case "${PAPER_MODE}" in auto|rebuild|skip) ;; *) echo "Invalid --paper-mode: ${PAPER_MODE}" >&2; exit 2 ;; esac
[[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs-per-gpu must be positive" >&2; exit 2; }
[[ "${PREPROCESS_WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo "--workers must be positive" >&2; exit 2; }

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  echo "No active Conda Python detected. Run 'conda activate <environment>' before this script." >&2
  exit 2
fi
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
export PYTHON_BIN

"${PYTHON_BIN}" - <<'PY'
import sys
import torch
print(f"[environment] python={sys.executable}")
print(f"[environment] torch={torch.__version__}; cuda_available={torch.cuda.is_available()}")
PY

need_sources() {
  local name
  for name in XJTU_SOURCE_ROOT MIT_SOURCE_ROOT SMARTHEALTH_SOURCE_ROOT; do
    [[ -n "${!name}" ]] || { echo "Missing required source path: ${name}" >&2; exit 2; }
    [[ -d "${!name}" ]] || { echo "Source directory does not exist: ${name}=${!name}" >&2; exit 2; }
  done
}

has_csv() {
  [[ -d "$1" ]] && find "$1" -type f -name '*.csv' -print -quit | grep -q .
}

domain_manifests() {
  local root="$1" domain
  for domain in xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280; do
    [[ -f "${root}/${domain}/manifest.json" ]] || return 1
  done
}

canonical_state() {
  local complete=0 populated=0 path
  local paths=(
    "${REPO_ROOT}/datasets/XJTU_raw"
    "${REPO_ROOT}/datasets/XJTU_features"
    "${REPO_ROOT}/datasets/MIT_raw"
    "${REPO_ROOT}/datasets/MIT_features"
    "${REPO_ROOT}/datasets/SmartHealth_raw"
    "${REPO_ROOT}/datasets/SmartHealth_features"
  )
  for path in "${paths[@]}"; do
    if has_csv "${path}"; then populated=$((populated + 1)); fi
  done
  (( populated == ${#paths[@]} )) && complete=1
  if (( complete == 1 )); then printf '%s\n' complete
  elif (( populated == 0 )); then printf '%s\n' missing
  else printf '%s\n' partial
  fi
}

write_runtime_config() {
  local target="$1"
  mkdir -p "$(dirname "${target}")"
  {
    printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
    printf 'XJTU_SOURCE_ROOT=%q\n' "${XJTU_SOURCE_ROOT}"
    printf 'MIT_SOURCE_ROOT=%q\n' "${MIT_SOURCE_ROOT}"
    printf 'SMARTHEALTH_SOURCE_ROOT=%q\n' "${SMARTHEALTH_SOURCE_ROOT}"
    printf 'XJTU_RAW_OUTPUT=%q\n' "${REPO_ROOT}/datasets/XJTU_raw"
    printf 'XJTU_FEATURE_OUTPUT=%q\n' "${REPO_ROOT}/datasets/XJTU_features"
    printf 'MIT_RAW_OUTPUT=%q\n' "${REPO_ROOT}/datasets/MIT_raw"
    printf 'MIT_FEATURE_OUTPUT=%q\n' "${REPO_ROOT}/datasets/MIT_features"
    printf 'SMARTHEALTH_RAW_OUTPUT=%q\n' "${REPO_ROOT}/datasets/SmartHealth_raw"
    printf 'SMARTHEALTH_FEATURE_OUTPUT=%q\n' "${REPO_ROOT}/datasets/SmartHealth_features"
    printf 'MIT_SPLIT_OUTPUT=%q\n' "${REPO_ROOT}/splits/mit/mit_paper_physical124_v2_split.json"
    printf 'SMARTHEALTH_SPLITS_OUTPUT=%q\n' "${REPO_ROOT}/splits/smarthealth"
    printf 'PREPROCESS_WORKERS=%q\n' "${PREPROCESS_WORKERS}"
  } >"${target}"
}

preprocess_canonical() {
  local state overwrite_args=()
  state="$(canonical_state)"
  case "${CANONICAL_MODE}" in
    skip) echo "[canonical] skipped by request"; return ;;
    auto)
      if [[ "${state}" == complete ]]; then echo "[canonical] reusing existing products"; return; fi
      if [[ "${state}" == partial ]]; then
        echo "Canonical products are partial. Use --canonical-mode rebuild or repair them." >&2
        exit 2
      fi
      ;;
    rebuild) overwrite_args=(--overwrite) ;;
  esac
  RUNTIME_CONFIG_FILE="${REPO_ROOT}/outputs/Paper-Backup/_portable_runtime/paths.$$.env"
  write_runtime_config "${RUNTIME_CONFIG_FILE}"
  echo "[canonical] extracting XJTU, MIT physical124 and all SmartHealth families"
  bash "${REPO_ROOT}/preprocess/run_preprocess.sh" --config "${RUNTIME_CONFIG_FILE}" xjtu all --workers "${PREPROCESS_WORKERS}" "${overwrite_args[@]}"
  bash "${REPO_ROOT}/preprocess/run_preprocess.sh" --config "${RUNTIME_CONFIG_FILE}" mit all --workers "${PREPROCESS_WORKERS}" "${overwrite_args[@]}"
  bash "${REPO_ROOT}/preprocess/run_preprocess.sh" --config "${RUNTIME_CONFIG_FILE}" smarthealth all --workers "${PREPROCESS_WORKERS}" "${overwrite_args[@]}"
  cleanup
  RUNTIME_CONFIG_FILE=""
}

build_e1_data() {
  if [[ "${PAPER_MODE}" == skip ]]; then echo "[E1 data] skipped by request"; return; fi
  if domain_manifests "${E1_DATA_ROOT}" && [[ "${PAPER_MODE}" == auto ]]; then
    echo "[E1 data] reusing ${E1_DATA_ROOT}"
    "${PYTHON_BIN}" "${REPO_ROOT}/preprocess/paper_backup/validate_preprocessed.py" --root "${E1_DATA_ROOT}"
    return
  fi
  local args=(
    --domains all --output-root "${E1_DATA_ROOT}" --schema-version 2
    --cc-len 128 --cv-len 128 --workers "${PREPROCESS_WORKERS}"
  )
  [[ "${PAPER_MODE}" == rebuild ]] && args+=(--overwrite)
  "${PYTHON_BIN}" "${REPO_ROOT}/preprocess/paper_backup/build_preprocessed.py" "${args[@]}"
  "${PYTHON_BIN}" "${REPO_ROOT}/preprocess/paper_backup/validate_preprocessed.py" --root "${E1_DATA_ROOT}"
}

build_mit_full() {
  local count=0 args
  if [[ -d "${MIT_FULL_ROOT}" ]]; then
    count="$(find "${MIT_FULL_ROOT}" -maxdepth 1 -type f -name 'MIT_*_full_cccv.csv' | wc -l)"
  fi
  if [[ "${PAPER_MODE}" == auto && "${count}" == 124 ]]; then
    echo "[MIT FULL] reusing ${MIT_FULL_ROOT}"
    return
  fi
  if [[ "${PAPER_MODE}" == auto && "${count}" != 0 ]]; then
    echo "MIT FULL export is partial (${count}/124). Use --paper-mode rebuild." >&2
    exit 2
  fi
  args=(--input-root "${MIT_SOURCE_ROOT}" --output-root "${MIT_FULL_ROOT}")
  [[ "${PAPER_MODE}" == rebuild ]] && args+=(--overwrite)
  "${PYTHON_BIN}" "${REPO_ROOT}/preprocess/paper_backup/export_mit_full_cccv.py" "${args[@]}"
}

build_e2_data() {
  if [[ "${PAPER_MODE}" == skip ]]; then echo "[E2 data] skipped by request"; return; fi
  if domain_manifests "${E2_DATA_ROOT}" && [[ "${PAPER_MODE}" == auto ]]; then
    echo "[E2 data] reusing ${E2_DATA_ROOT}"
    "${PYTHON_BIN}" "${REPO_ROOT}/preprocess/paper_backup/validate_preprocessed.py" \
      --root "${E2_DATA_ROOT}" \
      --domains xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280
    return
  fi
  build_mit_full
  local args=(
    --domains xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280
    --include-full xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280
    --output-root "${E2_DATA_ROOT}" --schema-version 2
    --cc-len 128 --cv-len 128 --full-joint-len 256
    --workers "${PREPROCESS_WORKERS}"
    --xjtu-full-source-root "${XJTU_SOURCE_ROOT}"
    --mit-full-source-root "${MIT_FULL_ROOT}"
    --smarthealth-full-source-root "${SMARTHEALTH_SOURCE_ROOT}"
  )
  [[ "${PAPER_MODE}" == rebuild ]] && args+=(--overwrite)
  "${PYTHON_BIN}" "${REPO_ROOT}/preprocess/paper_backup/build_preprocessed.py" "${args[@]}"
  "${PYTHON_BIN}" "${REPO_ROOT}/preprocess/paper_backup/validate_preprocessed.py" \
    --root "${E2_DATA_ROOT}" \
    --domains xjtu mit smarthealth_lishen40 smarthealth_catl280 smarthealth_eve280
}

preprocess_all() {
  need_sources
  preprocess_canonical
  build_e1_data
  build_e2_data
}

train_e1() {
  GPU_IDS="${GPU_IDS}" JOBS_PER_GPU="${JOBS_PER_GPU}" MODELS=all \
    PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${E1_OUTPUT_ROOT}" \
    bash "${SCRIPT_DIR}/run_e1_final_interaction_5seed_pipeline.sh" train
  PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${E1_OUTPUT_ROOT}" \
    bash "${SCRIPT_DIR}/run_e1_final_interaction_5seed_pipeline.sh" summary
}

train_e2() {
  GPU_IDS="${GPU_IDS}" JOBS_PER_GPU="${JOBS_PER_GPU}" MODELS=all \
    PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${E2_OUTPUT_ROOT}" \
    bash "${SCRIPT_DIR}/run_e2_final_interaction_5seed_pipeline.sh" train
  PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${E2_OUTPUT_ROOT}" \
    bash "${SCRIPT_DIR}/run_e2_final_interaction_5seed_pipeline.sh" summary
}

summarize_all() {
  PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${E1_OUTPUT_ROOT}" \
    bash "${SCRIPT_DIR}/run_e1_final_interaction_5seed_pipeline.sh" summary
  PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${E2_OUTPUT_ROOT}" \
    bash "${SCRIPT_DIR}/run_e2_final_interaction_5seed_pipeline.sh" summary
}

mkdir -p "${E1_OUTPUT_ROOT}" "${E2_OUTPUT_ROOT}"
echo "[portable] stage=${STAGE}; GPUs=${GPU_IDS}; jobs/GPU=${JOBS_PER_GPU}; workers=${PREPROCESS_WORKERS}"
case "${STAGE}" in
  preprocess) preprocess_all ;;
  train) train_e1; train_e2 ;;
  e1) train_e1 ;;
  e2) train_e2 ;;
  summary) summarize_all ;;
  all) preprocess_all; train_e1; train_e2 ;;
esac
echo "[portable] completed stage=${STAGE}"
