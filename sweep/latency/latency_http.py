"""Latency benchmark against a running /v1/systemone server (vLLM plugin or llama.cpp). Run ON the Spark host.

Fixed sample (stats.SAMPLE_PER_SET records per set, seeded), batch size 1, concurrency 1, 20 warm-up requests
(round-robin over all sets, not timed), then every sampled request timed as the HTTP round trip (client and server
on one host). Writes ONE json with p50/p95/p99, decisions/sec, input-token buckets and peak memory. Private sets
appear only as per-set aggregates (no ids, no text, no error strings).

Peak memory: vLLM = plugin-reported weights + peak activation (GET /v1/engine, read AFTER the run) and, separately,
nvidia-smi summed over the container's processes (includes the preallocated KV pool); llama.cpp = nvidia-smi of
the server pid (includes KV and compute buffers).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from runners.common import read_records, weights_size_gb
from runners.http_systemone import Client, engine_info, parse_nvidia_smi_apps, parse_response
from sweep.latency import stats


def smi_apps() -> dict[int, int]:
    try:
        text = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return parse_nvidia_smi_apps(text)


def container_pids(name: str) -> set[int]:
    try:
        out = subprocess.run(["docker", "top", name, "-eo", "pid"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {int(x) for x in out.split()[1:] if x.isdigit()}


class Sampler(threading.Thread):
    def __init__(self, pid: int | None, container: str | None) -> None:
        super().__init__(daemon=True)
        self.pid, self.container, self.peak, self._halt = pid, container, 0, threading.Event()

    def sample(self) -> None:
        pids = container_pids(self.container) if self.container else {self.pid}
        apps = smi_apps()
        self.peak = max(self.peak, sum(v for k, v in apps.items() if k in pids))

    def run(self) -> None:
        while not self._halt.is_set():
            self.sample()
            self._halt.wait(0.5)

    def finish(self) -> int:
        self._halt.set()
        self.sample()
        return self.peak


def load_sample(path: str, key: str) -> list[dict[str, Any]]:
    recs = list(read_records(path))
    return [recs[i] for i in stats.sample_indices(len(recs), stats.SAMPLE_PER_SET, key)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--private-records-dir", default=None)
    ap.add_argument("--public-sets", nargs="*", default=[])
    ap.add_argument("--private-sets", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--container", default=None, help="vLLM container name (sums GPU memory over its pids)")
    ap.add_argument("--server-pid", type=int, default=None, help="llama-server host pid")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--model-file", default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)

    samples: dict[str, list[dict[str, Any]]] = {}
    private_flag: dict[str, bool] = {}
    for s in a.public_sets:
        samples[s] = load_sample(f"{a.records_dir}/{s}.jsonl", s)
        private_flag[s] = False
    for s in a.private_sets:
        samples[s] = load_sample(f"{a.private_records_dir}/{s}.jsonl", s)
        private_flag[s] = True

    client = Client(a.url, a.timeout)
    info = engine_info(client, None, None)
    foreign = {k: v for k, v in smi_apps().items()
               if k not in (container_pids(a.container) if a.container else {a.server_pid})}

    warm = stats.warmup_order({s: v for s, v in samples.items()}, stats.WARMUP_REQUESTS)
    warm_errors = 0
    for rec in warm:
        try:
            client.request("POST", "/v1/systemone", rec["request"])
        except Exception:  # noqa: BLE001 - counted, not fatal
            warm_errors += 1

    sampler = Sampler(a.server_pid, a.container)
    sampler.start()
    obs: list[dict[str, Any]] = []
    wall0 = time.perf_counter()
    for name, recs in samples.items():
        for rec in recs:
            t0 = time.perf_counter()
            err, tokens, ms = None, None, None
            try:
                resp = client.request("POST", "/v1/systemone", rec["request"])
                ms = (time.perf_counter() - t0) * 1e3
                tokens = (resp.get("usage") or {}).get("input_tokens")
                parse_response(rec["request"], resp)
            except Exception as exc:  # noqa: BLE001 - kept only as a count in the report
                err = type(exc).__name__
                ms = None
            obs.append({"set": name, "private": private_flag[name], "tokens": tokens, "ms": ms, "error": err})
    wall = time.perf_counter() - wall0
    smi_peak = sampler.finish() / 1e9

    final = engine_info(client, None, None)  # re-read: the plugin's activation peak grows during the run
    details = final.get("details") or {}
    plugin_peak = details.get("peak_memory_gb") if isinstance(details, dict) else None
    report = stats.build_report(obs, wall)
    size = (weights_size_gb(a.model_dir) if a.model_dir else
            (Path(a.model_file).stat().st_size / 1e9 if a.model_file else None))
    out = {
        "model": a.model_name, "config": a.config, "engine": final["engine"], "engine_version": final["engine_version"],
        "batch_size": 1, "concurrency": 1,
        "timing": "HTTP round trip to localhost (includes server-side tokenisation and encoding)",
        "warmup_requests": len(warm), "warmup_errors": warm_errors,
        "sample_per_set": stats.SAMPLE_PER_SET, "sample_seed": stats.SAMPLE_SEED,
        "public_sets": a.public_sets, "private_sets": a.private_sets,
        **report,
        "peak_memory_gb": round(plugin_peak if plugin_peak is not None else smi_peak, 3),
        "peak_memory_source": "vllm plugin (weights + peak activation)" if plugin_peak is not None
        else "nvidia-smi compute-apps of the server process",
        "nvidia_smi_peak_gb": round(smi_peak, 3),
        "file_size_gb": round(size, 3) if size is not None else None,
        "gpu_memory_utilization": a.gpu_memory_utilization,
        "other_gpu_processes_at_start": {"count": len(foreign), "total_mib": round(sum(foreign.values()) / 2**20)},
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"overall": report["overall"], "peak_memory_gb": out["peak_memory_gb"]}))


if __name__ == "__main__":
    main()
