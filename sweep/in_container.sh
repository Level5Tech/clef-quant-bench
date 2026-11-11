#!/usr/bin/env bash
# Run a python module in the NGC vLLM container with extra identity bind-mounts (host path == container path).
#   sweep/in_container.sh <container-name> <mounts> <module> [args...]
# <mounts> is a comma-separated list of host paths; a ":ro" suffix mounts read-only. Used for native_ref and the
# native latency job so they can read the private records and write the private results (mode 700 dirs) directly.
# Run ON the Spark. Container names start with clefbench-sweep- so the driver can find and stop its own.
# CLEF_PYTHONPATH_EXTRA in the environment is appended to the container's PYTHONPATH (docker_env.sh); the driver sets
# it for fla-kernel workers together with an identity mount of the flash-linear-attention install.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../serve/docker_env.sh"
NAME="$1"; MOUNTS="$2"; shift 2
EXTRA=()
IFS=',' read -r -a MS <<< "$MOUNTS"
for m in "${MS[@]}"; do
  [ -z "$m" ] && continue
  if [[ "$m" == *:ro ]]; then p="${m%:ro}"; EXTRA+=(-v "$p:$p:ro"); else EXTRA+=(-v "$m:$m"); fi
done
docker run --rm --name "$NAME" $(clef_docker_flags) "${EXTRA[@]}" "$CLEF_IMAGE" python -m "$@" 2>&1 \
  | tr '\r' '\n' | awk '!/Loading weights/ && !/^$/ {print; fflush()}'
