#!/usr/bin/env bash
# Build one quantized checkpoint. Run ON the Spark.
#   quant/build_quants.sh <clef-flash|clef> <fp8|nvfp4> [extra quantize args]
# Output: ~/clef-quant/models/<model>-<fmt>/
#
# CLEF-FLASH (9B) commands, verified in this repo's build:
#   quant/build_quants.sh clef-flash fp8
#   quant/build_quants.sh clef-flash nvfp4 --calibration /bench/data/calibration.jsonl --num-samples 512
# CLEF 27B commands (WRITTEN, NOT RUN: GPU rule, the orchestrator runs these one at a time):
#   quant/build_quants.sh clef fp8
#   quant/build_quants.sh clef nvfp4 --calibration /bench/data/calibration.jsonl --num-samples 512
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../serve/docker_env.sh"
MODEL="$1"; FMT="$2"; shift 2
case "$FMT" in fp8) SCHEME=FP8_DYNAMIC;; nvfp4) SCHEME=NVFP4;; *) echo "format must be fp8|nvfp4" >&2; exit 2;; esac
REL="$(clef_snapshot "$MODEL")"
"$HERE/run_quant.sh" quant.quantize --release "$REL" --scheme "$SCHEME" --out "/models/${MODEL}-${FMT}" "$@"
"$HERE/run_quant.sh" quant.verify_checkpoint "/models/${MODEL}-${FMT}" --release "$REL"
