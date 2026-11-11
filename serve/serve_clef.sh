#!/usr/bin/env bash
# Serve a Clef model through vLLM 0.29 (NGC 26.09 container) + clef_bench_vllm plugin. Run ON the Spark.
#   serve/serve_clef.sh <clef-flash|clef> <bf16|fp8|nvfp4> <port> [server args...]
# Container name: clefbench-serve-<model>-<fmt>. Stop it with: docker stop clefbench-serve-<model>-<fmt>
# Ports in use on the Spark (do not use): 22 53 631 5678 7583 8001 8002 8004 8080 8768 8770 8771 9111 9292 11434 18770 3000
#   bf16 serves the Cloudflare release snapshot directly (read-only HF cache); fp8/nvfp4 serve /models/<model>-<fmt>.
# GPU RULE: "clef" (27B) is launched only by the orchestrator, one job at a time.
# Runs in the foreground; use `-d`-style backgrounding by the caller (nohup ... &). Health: curl :<port>/health
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/docker_env.sh"
MODEL="$1"; FMT="$2"; PORT="$3"; shift 3
case "$FMT" in
  bf16) MDIR="$(clef_snapshot "$MODEL")";;
  fp8|nvfp4) MDIR="/models/${MODEL}-${FMT}";;
  *) echo "format must be bf16|fp8|nvfp4" >&2; exit 2;;
esac
NAME="${CLEF_CONTAINER_NAME:-clefbench-serve-${MODEL}-${FMT}}"   # override lets several servers of one config coexist
# The plugin is pip-installed into a throwaway dir at container start (entry point registers it in every vLLM process).
exec docker run --rm --name "$NAME" -p "${PORT}:${PORT}" $(clef_docker_flags) "$CLEF_IMAGE" bash -c "
  set -e
  rm -rf /tmp/plugin_src /tmp/plugin_site && cp -r /bench/serve/plugin /tmp/plugin_src
  pip install -q --no-deps --no-build-isolation --target /tmp/plugin_site /tmp/plugin_src 2>&1 | grep -v -i warning || true
  export PYTHONPATH=/tmp/plugin_site:\$PYTHONPATH
  exec python -m clef_bench_vllm.server --model '$MDIR' --port $PORT --served-model-name '$MODEL' $*
"
