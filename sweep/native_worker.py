"""Concurrent-safe BF16 reference worker (Cloudflare's load_release_model), run inside the NGC container.

One process = one model copy. It walks a plan of units (contiguous record ranges), atomically claims the next
unclaimed one (mkdir in --claim-dir), runs it, and exits when nothing is left. Several containers can run the same
plan; they share the work. The forward pass is byte-for-byte the one in runners/native_ref.py
(encode_record -> collate_records -> model -> float32 logits -> float64 softmax); only the loop around it differs.

    python -m sweep.native_worker --model <release> --model-name clef-flash --config ref --batch-size 1 \
        --plan plan.json --claim-dir <dir> --owner w0 [--ready-file F] [--go-file F] [--stop-file F]

plan.json: [{"id": "...", "records": "<jsonl>", "start": 0, "end": 376, "out": "<unit>/<set>.jsonl"}, ...]
A unit's output appears under its final name only when complete (written to .part, then renamed), with a sidecar
<out>.meta.json (records, errors, wall_seconds, peak_gpu_mem_gib, started_at, ended_at, ...).
--ready-file is touched after load + warm-up; --go-file (benchmark barrier) delays the first unit until it exists.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from runners.common import answers_from_logits, iter_pairs, make_row, read_record_range, write_row
from runners import gdn_kernels
from runners.native_ref import import_release_module, resolve_model_dir
from sweep import units as U


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--claim-dir", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--go-file", default=None)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--gdn-kernel", choices=gdn_kernels.KINDS, default="torch",
                    help="Gated DeltaNet kernel: transformers' torch fallback or flash-linear-attention (fla)")
    a = ap.parse_args(argv)

    import torch
    import transformers

    plan: list[dict[str, Any]] = json.loads(Path(a.plan).read_text())
    claim_dir = Path(a.claim_dir)
    stop = Path(a.stop_file) if a.stop_file else None

    gdn = gdn_kernels.install(a.gdn_kernel)
    model_dir = resolve_model_dir(a.model)
    release = import_release_module(model_dir)
    t_load = time.perf_counter()
    model, processor = release.load_release_model(model_dir, device="cuda", dtype=torch.bfloat16)
    load_s = time.perf_counter() - t_load
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    engine_version = f"transformers {transformers.__version__}; gdn={gdn}"

    @torch.inference_mode()
    def run(batch: list[dict[str, Any]]):
        encoded = [release.encode_record(tokenizer, r["request"], max_length=a.max_length, processor=processor)
                   for r in batch]
        logits = model(release.collate_records(encoded, tokenizer.pad_token_id, device))
        torch.cuda.synchronize()
        return encoded, logits

    def remaining() -> list[dict[str, Any]]:
        return [u for u in plan if not U.unit_complete(Path(u["out"]), u["end"] - u["start"])]

    todo = remaining()
    if todo and a.warmup:  # kernel compile / allocator warm-up: untimed, not written
        first = todo[0]
        for r in itertools.islice(read_record_range(first["records"], first["start"], first["end"]), a.warmup):
            try:
                run([r])
            except Exception as exc:  # noqa: BLE001 - failures resurface in the timed pass
                print(f"warm-up error on {r['id']}: {exc!r}", file=sys.stderr, flush=True)
        gdn_kernels.assert_used(a.gdn_kernel)
    if a.ready_file:
        Path(a.ready_file).parent.mkdir(parents=True, exist_ok=True)
        Path(a.ready_file).write_text(str(time.time()))
    if a.go_file:
        deadline = time.time() + 3600
        while not Path(a.go_file).exists() and time.time() < deadline:
            time.sleep(0.2)

    done = 0
    for u in plan:
        if stop is not None and stop.exists():
            print("stop file present: exiting between units", flush=True)
            break
        out, n_expected = Path(u["out"]), u["end"] - u["start"]
        if U.unit_complete(out, n_expected) or not U.try_claim(claim_dir, u["id"], a.owner):
            continue
        records = list(read_record_range(u["records"], u["start"], u["end"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        torch.cuda.reset_peak_memory_stats()
        n_err = 0
        t_start = time.time()
        wall0 = time.perf_counter()
        with open(part, "w", encoding="utf-8") as fh:
            for batch in iter_pairs(records, max(1, a.batch_size)):
                t0 = time.perf_counter()
                try:
                    encoded, logits = run(batch)
                    latency_ms = (time.perf_counter() - t0) * 1e3
                    for rec, enc, rec_logits in zip(batch, encoded, logits):
                        ids = {q.question_id: list(q.option_ids) for q in enc.questions}
                        lg = {q.question_id: l.float().cpu().tolist() for q, l in zip(enc.questions, rec_logits)}
                        write_row(fh, make_row(
                            rec["id"], rec["task"], a.model_name, a.config, answers_from_logits(ids, lg),
                            latency_ms, len(enc.input_ids), "transformers-native", engine_version,
                            batch_size=len(batch), probs_rounded=False))
                except Exception as exc:  # noqa: BLE001 - one bad record must not end the run
                    latency_ms = (time.perf_counter() - t0) * 1e3
                    for rec in batch:
                        n_err += 1
                        write_row(fh, make_row(rec["id"], rec["task"], a.model_name, a.config, {}, latency_ms, None,
                                               "transformers-native", engine_version, error=repr(exc),
                                               batch_size=len(batch), probs_rounded=False))
        wall_s = time.perf_counter() - wall0
        os.replace(part, out)
        meta = {
            "engine": "transformers-native", "transformers": transformers.__version__, "torch": torch.__version__,
            "dtype": "bfloat16", "model_dir": str(model_dir), "model_name": a.model_name, "config": a.config,
            "records_file": u["records"], "start": u["start"], "end": u["end"], "batch_size": a.batch_size,
            "gdn_kernel": gdn,
            "records": len(records), "errors": n_err, "load_seconds": round(load_s, 2),
            "wall_seconds": round(wall_s, 3), "records_per_second": round(len(records) / wall_s, 3) if wall_s else None,
            "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "gpu": torch.cuda.get_device_name(0), "owner": a.owner, "started_at": t_start, "ended_at": time.time(),
        }
        Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({k: meta[k] for k in ("records_file", "start", "end", "records", "errors", "wall_seconds")}),
              flush=True)
        done += 1
    print(f"worker {a.owner} done: {done} units", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
