#!/usr/bin/env bash
# Run on the Mac: push ONLY the GPU-side code (runners/, quant/, serve/, third_party/, data/smoke/, PLAN.md)
# to spark:~/clef-quant/bench/. Never touches datasets/, bench/, data/records, private*/ (other engineer's areas).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ssh spark 'mkdir -p ~/clef-quant/bench'
rsync -a --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.venv' \
  PLAN.md runners quant serve third_party spark:~/clef-quant/bench/
ssh spark 'mkdir -p ~/clef-quant/bench/data/smoke'
rsync -a data/smoke/ spark:~/clef-quant/bench/data/smoke/
echo "synced"
