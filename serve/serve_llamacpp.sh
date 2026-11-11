#!/usr/bin/env bash
# Serve a Clef GGUF with llama.cpp's llama-server (/v1/systemone). Run ON the Spark. Native process (not a container).
#   serve/serve_llamacpp.sh <clef-flash|clef> <BF16|Q8_0|Q4_K_M> <port> [extra llama-server args]
# Stop it with: kill $(cat ~/clef-quant/logs/llama-<model>-<quant>.pid)   (PID file written below)
# llama.cpp: ~/clef-quant/llama.cpp (master ba3c05f7, includes PR 29831). Avoid ports 8001 8002 8004 9292 11434 18770 8770 8771 3000.
# -b/-ub 16384 are REQUIRED: the whole prompt must fit one physical batch (default 512 gives HTTP 500 "input is too large").
# GPU RULE: the 27B GGUFs ("clef") are launched only by the orchestrator, one job at a time.
set -euo pipefail
MODEL="$1"; QUANT="$2"; PORT="$3"; shift 3
BIN="$HOME/clef-quant/llama.cpp/build/bin"
GGUF="$HOME/clef-quant/models/gguf/${MODEL}-${QUANT}.gguf"
PIDFILE="${CLEF_PIDFILE:-$HOME/clef-quant/logs/llama-${MODEL}-${QUANT}.pid}"   # override: one pidfile per concurrent server
echo $$ > "$PIDFILE"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
exec "$BIN/llama-server" -m "$GGUF" --port "$PORT" --host 127.0.0.1 -ngl 99 -c 16384 -b 16384 -ub 16384 "$@"
