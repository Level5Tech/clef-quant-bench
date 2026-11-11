#!/usr/bin/env bash
# Prove the NGC container imports flash-linear-attention the way a fla-kernel native worker gets it: identity mount of
# ~/clef-quant/pylibs-fla plus CLEF_PYTHONPATH_EXTRA through docker_env.sh (not an explicit -e PYTHONPATH override).
# Run ON the Spark. Exit 0 only if the import succeeds and reports the expected version. Touches no GPU memory.
#   sweep/check_fla_container.sh [expected-version, default 0.5.2]
# 2026-10-18: the first version ran docker inside a nested `bash -c`, which cannot see clef_docker_flags or CLEF_IMAGE
# from this shell; it failed silently and swap_driver.sh correctly refused to relaunch. Everything runs in this shell now.
set -euo pipefail
WANT="${1:-0.5.2}"
cd ~/clef-quant/bench && source serve/docker_env.sh
FLA="$HOME/clef-quant/pylibs-fla"
[ -d "$FLA/fla" ] || { echo "missing $FLA/fla"; exit 1; }
export CLEF_PYTHONPATH_EXTRA="$FLA"   # read by clef_docker_flags below, exactly as the driver sets it for fla workers
set +e
out="$(timeout 600 docker run --rm --name clefbench-flacheck $(clef_docker_flags) -v "$FLA:$FLA:ro" "$CLEF_IMAGE" \
  python -c 'import os, fla, fla.ops; print("PYTHONPATH", os.environ["PYTHONPATH"]); print("container imports fla", fla.__version__, fla.__file__)' 2>&1 | tail -2)"
st=$?
set -e
echo "$out"
[ "$st" -eq 0 ] || { echo "docker run exited $st"; exit 1; }
grep -q "container imports fla $WANT $FLA/" <<<"$out"
