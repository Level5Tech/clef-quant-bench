# Sourced by the clefbench scripts. Common docker flags for NGC vLLM 26.09 containers on the Spark.
# Layout on the Spark: ~/clef-quant/{bench (this repo, rsynced), models, results, pylibs, pylibs-fla}
CLEF_IMAGE="${CLEF_IMAGE:-nvcr.io/nvidia/vllm:26.09-py3}"
CLEF_ROOT="${CLEF_ROOT:-$HOME/clef-quant}"
# --gpus all, HF cache read-only, bench read-only, models rw (quant output), results rw.
# HF_HUB_OFFLINE=1: weights come from the mounted cache only.
# CLEF_PYTHONPATH_EXTRA (optional, caller's environment): appended to the container's PYTHONPATH. The sweep sets it to
# the identity-mounted flash-linear-attention install for fla-kernel native workers; nothing else sets it.
clef_docker_flags() {
  echo --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$HOME/.cache/huggingface:/hf:ro" \
    -v "$CLEF_ROOT/bench:/bench:ro" \
    -v "$CLEF_ROOT/models:/models" \
    -v "$CLEF_ROOT/results:/results" \
    -v "$CLEF_ROOT/pylibs:/pylibs:ro" \
    -v "$CLEF_ROOT/cache:/cache" \
    -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e XDG_CACHE_HOME=/cache/xdg \
    -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/bench:/pylibs${CLEF_PYTHONPATH_EXTRA:+:$CLEF_PYTHONPATH_EXTRA} -e PYTHONDONTWRITEBYTECODE=1 \
    ${CLEF_MPS_DIR:+-v "$CLEF_MPS_DIR:/mps" -e CUDA_MPS_PIPE_DIRECTORY=/mps} \
    -w /bench
}
# Snapshot dir (inside the container) for a Cloudflare release: clef-flash | clef
clef_snapshot() {
  echo "/hf/hub/models--Cloudflare--$1/snapshots/$(ls "$HOME/.cache/huggingface/hub/models--Cloudflare--$1/snapshots" | head -1)"
}
