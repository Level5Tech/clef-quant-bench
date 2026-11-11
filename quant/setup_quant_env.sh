#!/usr/bin/env bash
# One-time: build the quantization venv on the Spark (run ON the Spark).
# A venv with --system-site-packages inside the NGC vllm 26.09 container, so torch 2.14 / transformers 5.16.1 /
# CUDA come from the image and only llm-compressor 0.14.0 + its pinned compressed-tensors 0.19.0 (+ datasets,
# auto-round, ...) are installed into ~/clef-quant/venv-quant. The serving environment (stock image,
# compressed-tensors 0.17.0) is untouched.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../serve/docker_env.sh"
mkdir -p "$CLEF_ROOT/venv-quant" "$CLEF_ROOT/cache"
docker run --rm --name clefbench-quant-setup --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$CLEF_ROOT/venv-quant:/venv-quant" -v "$CLEF_ROOT/cache:/cache" -e PIP_CACHE_DIR=/cache/pip \
  "$CLEF_IMAGE" bash -c '
    set -e
    python -m venv --system-site-packages /venv-quant
    /venv-quant/bin/pip install "llmcompressor==0.14.0"
    /venv-quant/bin/pip list 2>/dev/null | grep -i -E "^(llmcompressor|compressed-tensors|transformers|torch|datasets|auto-round|accelerate|numpy) "
  '
