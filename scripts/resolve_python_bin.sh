#!/usr/bin/env bash
# Resolve the Python interpreter used by repository launchers.
#
# Priority:
#   1. explicit PYTHON_BIN;
#   2. the currently active Conda environment;
#   3. conda run for CONDA_ENV_NAME (defaults to the project pinn env).
#
# The resolver intentionally does not silently fall back to /usr/bin/python:
# that can make a training job fail later with a misleading missing-package
# or missing-CUDA error.
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "PYTHON_BIN is not an executable file: ${PYTHON_BIN}" >&2
    exit 2
  fi
  printf "%s\n" "${PYTHON_BIN}"
  exit 0
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  printf "%s\n" "${CONDA_PREFIX}/bin/python"
  exit 0
fi

CONDA_COMMAND="${CONDA_EXE:-}"
if [[ -z "${CONDA_COMMAND}" ]]; then
  CONDA_COMMAND="$(command -v conda 2>/dev/null || true)"
fi

if [[ -n "${CONDA_COMMAND}" && -x "${CONDA_COMMAND}" ]]; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-pinn}}"
  RESOLVED_PYTHON="$("${CONDA_COMMAND}" run --no-capture-output -n "${CONDA_ENV_NAME}" \
    python -c "import sys; print(sys.executable)" 2>/dev/null | tail -n 1)"
  if [[ -n "${RESOLVED_PYTHON}" && -x "${RESOLVED_PYTHON}" ]]; then
    printf "%s\n" "${RESOLVED_PYTHON}"
    exit 0
  fi
fi

cat >&2 <<EOF
Could not resolve a Conda Python interpreter.
Activate the intended environment first, for example:
  conda activate pinn
or set an explicit interpreter:
  PYTHON_BIN=/path/to/conda/env/bin/python
You may also set CONDA_EXE and CONDA_ENV_NAME for non-interactive launches.
EOF
exit 2
