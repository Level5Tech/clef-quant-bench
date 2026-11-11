"""HTTP client runner for /v1/systemone endpoints (vLLM + clef_bench_vllm plugin, or llama.cpp `llama-server`).

    python -m runners.http_systemone --url http://127.0.0.1:18951 --records data/smoke/smoke.jsonl \
        --out-dir results/clef-flash/bf16-srv --model-name clef-flash --config bf16-srv \
        --model-dir /path/to/weights --engine-name vllm

Sends each record's `request` to POST <url>/v1/systemone one at a time (latency fidelity; --concurrency N for
throughput) and writes runner-output JSONL (<out-dir>/<records stem>.jsonl) plus <out-dir>/meta.json
({file_size_gb, peak_memory_gb, engine, engine_version, ...}) for the analysis.

Precision: if the response has a "raw" block (our vLLM plugin), logits are exact float32 and probs are a float64
softmax of them. Otherwise the per-option "probabilities" (choice/score) or "noul" (true) values are used as sent,
`logits` is null, and `probs_rounded` records whether every value is representable at 4 decimals (the signature of
Cloudflare's reference rounding). llama.cpp's server sends full-precision probabilities, so probs_rounded is false.

latency_ms = wall time of the HTTP round trip, including server-side tokenisation (client and server on the same
host). input_tokens comes from usage.input_tokens.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import sys
import threading
import time
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runners.common import (
    answers_from_logits,
    build_meta,
    make_row,
    plan_outputs,
    read_record_range,
    read_records,
    sniff_option_ids,
    softmax,
    validate_row,
    weights_size_gb,
    write_meta,
)

__all__ = ["question_option_ids", "parse_response", "build_meta", "weights_size_gb", "parse_nvidia_smi_apps"]


def question_option_ids(request: Mapping[str, Any]) -> dict[str, list[str]]:
    return {qid: sniff_option_ids(q) for qid, q in request["questions"].items()}


def _is_rounded(values: list[float]) -> bool:
    return all(round(v, 4) == v for v in values)


def parse_response(request: Mapping[str, Any], response: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Turn a /v1/systemone response into runner `answers`. Returns (answers, probs_rounded)."""
    ids = question_option_ids(request)
    raw = response.get("raw")
    if raw:
        missing = [q for q in ids if q not in raw]
        if missing:
            raise ValueError(f"raw block lacks questions {missing}")
        for q, expected in ids.items():
            if list(raw[q]["option_ids"]) != expected:
                raise ValueError(f"{q}: raw option ids {raw[q]['option_ids']} != expected {expected}")
        return answers_from_logits(ids, {q: raw[q]["logits"] for q in ids}), False

    answers_resp = response.get("answers") or {}
    answers: dict[str, Any] = {}
    all_values: list[float] = []
    for qid, option_ids in ids.items():
        item = answers_resp.get(qid)
        if item is None:
            raise ValueError(f"response lacks answer for question {qid!r}")
        if "probabilities" in item:
            probs = [float(item["probabilities"][o]) for o in option_ids]
        elif "noul" in item:
            p_true = float(item["noul"])
            probs = [p_true, 1.0 - p_true]
        else:
            raise ValueError(f"cannot read probabilities for {qid!r}: keys {sorted(item)}")
        all_values.extend(probs[:1] if "noul" in item and "probabilities" not in item else probs)
        answers[qid] = {"option_ids": list(option_ids), "logits": None, "probs": probs}
    return answers, _is_rounded(all_values)


def parse_nvidia_smi_apps(text: str) -> dict[int, int]:
    """Parse `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` into {pid: bytes}."""
    out: dict[int, int] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not parts[1].endswith("MiB"):
            continue
        try:
            out[int(parts[0])] = int(parts[1].split()[0]) * 2**20
        except ValueError:
            continue
    return out


class GpuPeakSampler(threading.Thread):
    """Samples one process's GPU memory (nvidia-smi compute-apps) every 0.5 s and keeps the maximum, in bytes."""

    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.pid, self.peak, self._halt = pid, 0, threading.Event()

    def sample(self) -> None:
        import subprocess

        try:
            text = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return
        self.peak = max(self.peak, parse_nvidia_smi_apps(text).get(self.pid, 0))

    def run(self) -> None:
        while not self._halt.is_set():
            self.sample()
            self._halt.wait(0.5)

    def finish(self) -> int:
        self._halt.set()
        self.sample()
        return self.peak


class Client:
    """One keep-alive HTTP connection per thread."""

    def __init__(self, url: str, timeout: float) -> None:
        parts = urllib.parse.urlparse(url)
        self.host, self.port, self.timeout = parts.hostname, parts.port or 80, timeout
        self._local = threading.local()

    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return conn

    def request(self, method: str, path: str, body: Any = None) -> Any:
        payload = None if body is None else json.dumps(body).encode()
        for attempt in (0, 1):
            conn = self._conn()
            try:
                conn.request(method, path, body=payload, headers={"content-type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {data[:300]!r}")
                return json.loads(data)
            except (http.client.HTTPException, ConnectionError, OSError):
                conn.close()
                self._local.conn = None
                if attempt:
                    raise
        raise AssertionError("unreachable")


def engine_info(client: Client, name: str | None, version: str | None) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for path in ("/v1/engine", "/props"):
        try:
            info = client.request("GET", path)
            break
        except Exception:  # noqa: BLE001 - endpoint absent on this server
            continue
    if "engine" in info:  # our vLLM plugin
        return {"engine": name or info["engine"], "engine_version": version or info.get("engine_version", "unknown"),
                "details": info}
    build = info.get("build_info") if isinstance(info, dict) else None
    return {"engine": name or "llama.cpp", "engine_version": version or (str(build) if build else "unknown"),
            "details": info or None}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:18951")
    ap.add_argument("--records", required=True, nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--engine-name", default=None, help="override the engine field (default: asked from the server)")
    ap.add_argument("--engine-version", default=None)
    ap.add_argument("--model-dir", default=None, help="weights dir (sums *.safetensors) for meta.file_size_gb")
    ap.add_argument("--model-file", default=None, help="a single weights file (e.g. a .gguf) for meta.file_size_gb")
    ap.add_argument("--peak-memory-gb", type=float, default=None, help="override; default: server-reported if any")
    ap.add_argument("--server-pid", type=int, default=None,
                    help="host PID of the server process; its peak GPU memory (nvidia-smi) becomes meta.peak_memory_gb "
                         "(includes cache/compute buffers the engine preallocates) unless the server reports its own")
    ap.add_argument("--request-model", default=None, help="value of request.model (default: record's own)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), default=None,
                    help="only records [START, END) of the (single) records file: a contiguous shard for concurrent workers")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args(argv)

    plan = plan_outputs(args.records, args.out, args.out_dir)
    client = Client(args.url, args.timeout)
    info = engine_info(client, args.engine_name, args.engine_version)
    engine, engine_version = info["engine"], info["engine_version"]

    def call(rec: dict[str, Any]) -> dict[str, Any]:
        request = dict(rec["request"])
        if args.request_model:
            request["model"] = args.request_model
        t0 = time.perf_counter()
        try:
            resp = client.request("POST", "/v1/systemone", request)
            latency_ms = (time.perf_counter() - t0) * 1e3
            answers, rounded = parse_response(request, resp)
            row = make_row(rec["id"], rec["task"], args.model_name, args.config, answers, latency_ms,
                           (resp.get("usage") or {}).get("input_tokens"), engine, engine_version,
                           probs_rounded=rounded, concurrency=args.concurrency)
        except Exception as exc:  # noqa: BLE001 - one bad record must not end the run
            row = make_row(rec["id"], rec["task"], args.model_name, args.config, {},
                           (time.perf_counter() - t0) * 1e3, None, engine, engine_version, error=repr(exc),
                           probs_rounded=None, concurrency=args.concurrency)
        validate_row(row)
        return row

    sampler = GpuPeakSampler(args.server_pid) if args.server_pid else None
    if sampler:
        sampler.start()
    summary: list[dict[str, Any]] = []
    first = True
    for records_path, out_path in plan:
        records = (list(read_record_range(records_path, *args.range)) if args.range
                   else list(read_records(records_path, args.limit)))
        if first:
            for rec in records[: args.warmup]:
                try:
                    client.request("POST", "/v1/systemone", rec["request"])
                except Exception as exc:  # noqa: BLE001
                    print(f"warm-up error on {rec['id']}: {exc!r}", file=sys.stderr)
            first = False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wall0 = time.perf_counter()
        if args.concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
                rows = list(pool.map(call, records))
        else:
            rows = [call(r) for r in records]
        wall_s = time.perf_counter() - wall0
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        errors = sum(1 for r in rows if r["error"])
        summary.append({"file": str(out_path), "records": len(rows), "errors": errors, "wall_seconds": round(wall_s, 3),
                        "records_per_second": round(len(rows) / wall_s, 3) if wall_s else None})
        print(json.dumps(summary[-1]), flush=True)

    # Re-read engine details now: the vLLM plugin's peak activation grows while requests run, so the value
    # fetched before the first request would under-report peak memory.
    info = engine_info(client, args.engine_name, args.engine_version)
    details = info.get("details") or {}
    smi_peak_gb = sampler.finish() / 1e9 if sampler else None
    peak = args.peak_memory_gb
    if peak is None and isinstance(details, dict):
        peak = details.get("peak_memory_gb")
    if peak is None and smi_peak_gb:
        peak = smi_peak_gb
    if args.model_dir:
        size = weights_size_gb(args.model_dir)
    elif args.model_file:
        size = Path(args.model_file).stat().st_size / 1e9
    else:
        size = None
    out_dir = plan[0][1].parent
    meta = build_meta(engine, engine_version, size, peak, extra={
        "model": args.model_name, "config": args.config, "url": args.url, "runs": summary,
        "memory_note": (details.get("memory_note") if isinstance(details, dict) and details.get("memory_note")
                        else ("peak = server process GPU memory via nvidia-smi compute-apps (includes KV cache and "
                              "compute buffers)" if smi_peak_gb else None)),
        "nvidia_smi_peak_gb": round(smi_peak_gb, 3) if smi_peak_gb else None,
        "server_details": details or None,
    })
    print("meta:", write_meta(out_dir, meta), flush=True)


if __name__ == "__main__":
    main()
