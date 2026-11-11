#!/usr/bin/env bash
# Run a quantization script inside the NGC vllm 26.09 container using the quant venv. Run ON the Spark.
#   quant/run_quant.sh <module> [args...]      e.g.  quant/run_quant.sh quant.list_linear_modules <release-dir>
# Mounts: repo /bench (ro), models /models (rw), results /results, HF cache /hf (ro), venv /venv-quant.
# Do not launch for the 27B model unless you are the orchestrator (GPU rule in PLAN.md).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../serve/docker_env.sh"
NAME="clefbench-quant-$$"
docker run --rm --name "$NAME" $(clef_docker_flags) -v "$CLEF_ROOT/venv-quant:/venv-quant:ro" "$CLEF_IMAGE" \
  /venv-quant/bin/python -m "$@" 2>&1 | tr '\r' '\n' | awk '!/Loading weights/ && !/^$/'
