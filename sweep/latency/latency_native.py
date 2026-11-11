"""Latency benchmark for the native BF16 reference (Cloudflare's load_release_model), batch size 1. Runs in the container.

Same sampling, warm-up and report as latency_http.py. latency = tokenise/encode + collate + forward + cuda sync
(the same window runners/native_ref.py times; no HTTP). Peak memory = torch.cuda.max_memory_allocated after the
warm-up reset (weights + activations, no allocator cache), plus max_memory_reserved.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from runners.common import read_records, weights_size_gb
from runners.native_ref import import_release_module, resolve_model_dir
from sweep.latency import stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--private-records-dir", default=None)
    ap.add_argument("--public-sets", nargs="*", default=[])
    ap.add_argument("--private-sets", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=16384)
    a = ap.parse_args(argv)

    import torch
    import transformers

    samples: dict[str, list[dict[str, Any]]] = {}
    private_flag: dict[str, bool] = {}
    for names, root, priv in ((a.public_sets, a.records_dir, False), (a.private_sets, a.private_records_dir, True)):
        for s in names:
            recs = list(read_records(f"{root}/{s}.jsonl"))
            samples[s] = [recs[i] for i in stats.sample_indices(len(recs), stats.SAMPLE_PER_SET, s)]
            private_flag[s] = priv

    model_dir = resolve_model_dir(a.model)
    release = import_release_module(model_dir)
    model, processor = release.load_release_model(model_dir, device="cuda", dtype=torch.bfloat16)
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device

    @torch.inference_mode()
    def run(rec: dict[str, Any]) -> int:
        enc = release.encode_record(tokenizer, rec["request"], max_length=a.max_length, processor=processor)
        model(release.collate_records([enc], tokenizer.pad_token_id, device))
        torch.cuda.synchronize()
        return len(enc.input_ids)

    warm = stats.warmup_order(samples, stats.WARMUP_REQUESTS)
    warm_errors = 0
    for rec in warm:
        try:
            run(rec)
        except Exception:  # noqa: BLE001
            warm_errors += 1
    torch.cuda.reset_peak_memory_stats()

    obs: list[dict[str, Any]] = []
    wall0 = time.perf_counter()
    for name, recs in samples.items():
        for rec in recs:
            t0 = time.perf_counter()
            try:
                tokens = run(rec)
                obs.append({"set": name, "private": private_flag[name], "tokens": tokens,
                            "ms": (time.perf_counter() - t0) * 1e3, "error": None})
            except Exception as exc:  # noqa: BLE001 - count only
                obs.append({"set": name, "private": private_flag[name], "tokens": None, "ms": None,
                            "error": type(exc).__name__})
    wall = time.perf_counter() - wall0
    report = stats.build_report(obs, wall)
    out = {
        "model": a.model_name, "config": a.config, "engine": "transformers-native",
        "engine_version": f"transformers {transformers.__version__}", "batch_size": 1, "concurrency": 1,
        "timing": "encode + collate + forward + cuda sync in-process (no HTTP)",
        "warmup_requests": len(warm), "warmup_errors": warm_errors,
        "sample_per_set": stats.SAMPLE_PER_SET, "sample_seed": stats.SAMPLE_SEED,
        "public_sets": a.public_sets, "private_sets": a.private_sets, **report,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_memory_source": "torch.cuda.max_memory_allocated (weights + activations, no cache pool)",
        "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "file_size_gb": round(weights_size_gb(model_dir), 3),
        "gpu": torch.cuda.get_device_name(0),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"overall": report["overall"], "peak_memory_gb": out["peak_memory_gb"]}))


if __name__ == "__main__":
    main()
