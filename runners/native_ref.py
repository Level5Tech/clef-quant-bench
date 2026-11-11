"""Reference runner: Cloudflare's own BF16 `load_release_model`, raw logits out.

Calls encode_record + collate_records + model(batch) directly and does the softmax itself in
float64. It does NOT use `systemone()` (that rounds probabilities to 4 decimals).

    python -m runners.native_ref --model Cloudflare/clef-flash --records data/smoke/smoke.jsonl \
        --out results/clef-flash/ref/smoke.jsonl [--batch-size 1] [--warmup 2] [--limit N]
    python -m runners.native_ref --model ... --records a.jsonl b.jsonl --out-dir results/clef-flash/ref
        (several records files, one model load; writes <out-dir>/<stem>.jsonl for each)

Batch size 1 (default) is the latency-faithful mode. --batch-size N > 1 is the optional throughput
mode: consecutive records are right-padded and run together; each row's latency_ms is then the
wall time of its batch and the row carries batch_size.

latency_ms = tokenise/encode + collate + forward + cuda sync (excludes softmax and JSON).
A sidecar <out>.meta.json records versions, peak GPU memory, wall time and throughput.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from runners.common import (
    answers_from_logits,
    build_meta,
    iter_pairs,
    make_row,
    plan_outputs,
    read_records,
    weights_size_gb,
    write_meta,
    write_row,
)
from runners import gdn_kernels


def import_release_module(model_dir: Path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", model_dir / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def resolve_model_dir(model: str) -> Path:
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id (e.g. Cloudflare/clef-flash) or a local release dir")
    ap.add_argument("--records", required=True, nargs="+")
    ap.add_argument("--out", default=None, help="output file (single --records file)")
    ap.add_argument("--out-dir", default=None, help="output directory (any number of --records files)")
    ap.add_argument("--model-name", default=None, help="'model' field in the output (default: basename of --model)")
    ap.add_argument("--config", default="ref")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2, help="untimed warm-up passes over the first records")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gdn-kernel", choices=gdn_kernels.KINDS, default="torch",
                    help="Gated DeltaNet kernel: transformers' torch fallback or flash-linear-attention (fla)")
    args = ap.parse_args(argv)

    import torch
    import transformers

    gdn = gdn_kernels.install(args.gdn_kernel)
    model_dir = resolve_model_dir(args.model)
    release = import_release_module(model_dir)
    model_name = args.model_name or Path(args.model.rstrip("/")).name.lower()

    t_load = time.perf_counter()
    model, processor = release.load_release_model(model_dir, device=args.device, dtype=torch.bfloat16)
    load_s = time.perf_counter() - t_load
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    engine_version = f"transformers {transformers.__version__}; gdn={gdn}"

    plan = plan_outputs(args.records, args.out, args.out_dir)

    @torch.inference_mode()
    def run(batch: list[dict[str, Any]]):
        """Returns (encoded list, logits per record per question) for one batch of PLAN records."""
        encoded = [release.encode_record(tokenizer, r["request"], max_length=args.max_length, processor=processor)
                   for r in batch]
        logits = model(release.collate_records(encoded, tokenizer.pad_token_id, device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return encoded, logits

    warmup_done = False
    peaks: list[float] = []
    runs: list[dict[str, Any]] = []
    for records_path, out_path in plan:
        records = list(read_records(records_path, args.limit))
        if not warmup_done:  # kernel compile / allocator warm-up: not timed, not written
            for r in records[: args.warmup]:
                try:
                    run([r])
                except Exception as exc:  # noqa: BLE001 - failures resurface in the timed pass
                    print(f"warm-up error on {r['id']}: {exc!r}", file=sys.stderr)
            warmup_done = True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_err = 0
        wall0 = time.perf_counter()
        with open(out_path, "w", encoding="utf-8") as fh:
            for batch in iter_pairs(records, max(1, args.batch_size)):
                t0 = time.perf_counter()
                try:
                    encoded, logits = run(batch)
                    latency_ms = (time.perf_counter() - t0) * 1e3
                    for rec, enc, rec_logits in zip(batch, encoded, logits):
                        ids = {q.question_id: list(q.option_ids) for q in enc.questions}
                        lg = {q.question_id: l.float().cpu().tolist() for q, l in zip(enc.questions, rec_logits)}
                        write_row(fh, make_row(
                            rec["id"], rec["task"], model_name, args.config, answers_from_logits(ids, lg),
                            latency_ms, len(enc.input_ids), "transformers-native", engine_version,
                            batch_size=len(batch), probs_rounded=False))
                except Exception as exc:  # noqa: BLE001 - one bad record must not end the run
                    latency_ms = (time.perf_counter() - t0) * 1e3
                    for rec in batch:
                        n_err += 1
                        write_row(fh, make_row(rec["id"], rec["task"], model_name, args.config, {}, latency_ms, None,
                                               "transformers-native", engine_version, error=repr(exc),
                                               batch_size=len(batch), probs_rounded=False))
        wall_s = time.perf_counter() - wall0
        meta = {
            "engine": "transformers-native",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": "bfloat16",
            "model_dir": str(model_dir),
            "model_name": model_name,
            "config": args.config,
            "records_file": str(records_path),
            "batch_size": args.batch_size,
            "gdn_kernel": gdn,
            "records": len(records),
            "errors": n_err,
            "load_seconds": round(load_s, 2),
            "wall_seconds": round(wall_s, 3),
            "records_per_second": round(len(records) / wall_s, 3) if wall_s else None,
            "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else None,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        }
        Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta))
        if meta["peak_gpu_mem_gib"] is not None:
            peaks.append(meta["peak_gpu_mem_gib"])
        runs.append({k: meta[k] for k in ("records_file", "records", "errors", "wall_seconds", "records_per_second")})

    # meta.json beside the results (analysis input): weights on disk + peak GPU memory (torch allocator, GB = 1e9).
    peak_gb = max(peaks) * 2**30 / 1e9 if peaks else None
    write_meta(plan[0][1].parent, build_meta(
        "transformers-native", engine_version, weights_size_gb(model_dir), peak_gb,
        extra={"model": model_name, "config": args.config, "dtype": "bfloat16", "torch": torch.__version__,
               "batch_size": args.batch_size, "gdn_kernel": gdn, "runs": runs,
               "memory_note": "peak = torch.cuda.max_memory_allocated (weights + activations, no cache pool)"}))


if __name__ == "__main__":
    main()
