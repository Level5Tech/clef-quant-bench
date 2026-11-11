"""Optional NVIDIA MPS (spatial GPU sharing between our worker processes).

Without MPS, kernels from different processes are time-sliced on the GPU and never overlap, so extra workers do not
add throughput (measured: 2 native workers = 0.90x of one). MPS lets kernels from several client processes run on
different SMs at once. Kernels and their math are unchanged; the determinism proofs are re-run under MPS.

Isolation: the daemon uses a PRIVATE pipe/log directory under ~/clef-quant/mps. Only processes that inherit
CUDA_MPS_PIPE_DIRECTORY from us (our containers via serve/docker_env.sh `CLEF_MPS_DIR`, our llama-server) connect to
it; Ollama, the PAI memory server and everything else keep using the GPU exactly as before. The default pipe directory
(/tmp/nvidia-mps) is never used. Not available while a latency run is measuring: the driver stops it first.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any


def env(root: str | Path) -> dict[str, str]:
    base = Path(root) / "mps"
    return {"CLEF_MPS_DIR": str(base / "pipe"), "CUDA_MPS_PIPE_DIRECTORY": str(base / "pipe"),
            "CUDA_MPS_LOG_DIRECTORY": str(base / "log")}


def _ctl(root: str | Path, stdin: str | None = None, args: list[str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["nvidia-cuda-mps-control", *(args or [])], input=stdin, capture_output=True, text=True,
                          env={**os.environ, **env(root)}, timeout=60)


def running(root: str | Path) -> bool:
    return (Path(env(root)["CUDA_MPS_PIPE_DIRECTORY"]) / "control").exists()


def start(root: str | Path, log: Any = None) -> None:
    e = env(root)
    for d in (e["CUDA_MPS_PIPE_DIRECTORY"], e["CUDA_MPS_LOG_DIRECTORY"]):
        Path(d).mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
    if not running(root):
        r = _ctl(root, args=["-d"])
        if r.returncode != 0:
            raise RuntimeError(f"nvidia-cuda-mps-control -d failed: {r.stderr.strip()[:200]}")
        for _ in range(20):
            if running(root):
                break
            time.sleep(0.5)
    if not running(root):
        raise RuntimeError("MPS control socket did not appear")
    os.environ.update(e)    # children (docker scripts, llama-server) inherit the private pipe dir
    if log:
        log("MPS_START", pipe=e["CUDA_MPS_PIPE_DIRECTORY"])


def stop(root: str | Path, log: Any = None) -> None:
    for k in env(root):
        os.environ.pop(k, None)
    if running(root):
        _ctl(root, stdin="quit\n")
        for _ in range(40):
            if not running(root):
                break
            time.sleep(0.5)
    if log:
        log("MPS_STOP", still_running=running(root))
