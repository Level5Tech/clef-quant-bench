#!/usr/bin/env bash
# Run the BF16 reference runner inside the NGC vLLM container. Run ON the Spark (ssh spark).
#   runners/run_native_ref.sh <clef-flash|clef> [native_ref args...]
# Paths inside the container: repo = /bench (ro), outputs = /results (= ~/clef-quant/results), models = /models.
# Example (Clef-flash, smoke):
#   runners/run_native_ref.sh clef-flash --records /bench/data/smoke/smoke.jsonl --out /results/clef-flash/ref/smoke.jsonl
# Full sweep, one model load:
#   runners/run_native_ref.sh clef-flash --records /bench/data/records/*.jsonl --out-dir /results/clef-flash/ref
# Batched throughput mode: add --batch-size 8 (and --config ref-b8 --out-dir ... so it does not overwrite bs=1).
# GPU RULE: Clef 27B ("clef") is started only by the orchestrator, one job at a time.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../serve/docker_env.sh"
MODEL="$1"; shift
docker run --rm --name "clefbench-native-${MODEL}-$$" $(clef_docker_flags) "$CLEF_IMAGE" \
  python -m runners.native_ref --model "$(clef_snapshot "$MODEL")" --model-name "$MODEL" "$@" 2>&1 \
  | tr '\r' '\n' | awk '!/Loading weights/ && !/^$/'
