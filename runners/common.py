"""Shared record / runner-output helpers (PLAN.md 'Shared interface').

Pure Python (no torch) so it runs on the Mac and inside the containers.

Runner output row, one per record:
  {"id", "task", "model", "config",
   "answers": {qid: {"option_ids": [...], "logits": [...] | null, "probs": [...]}},
   "latency_ms", "input_tokens", "engine", "engine_version", "error": null | str, ...extras}
Probabilities are full precision (float64 softmax of the raw logits). When an engine exposes only
rounded probabilities, `logits` is null and the runner says so in `engine` and `probs_rounded`.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax in float64 (Python floats)."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = math.fsum(exps)
    return [e / total for e in exps]


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_records(path: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield PLAN.md records ({id, task, request, gold})."""
    for index, rec in enumerate(read_jsonl(path)):
        if limit is not None and index >= limit:
            return
        yield rec


def read_record_range(path: str | Path, start: int, end: int) -> Iterator[dict[str, Any]]:
    """Yield records [start, end) of a records file (a contiguous shard)."""
    for index, rec in enumerate(read_jsonl(path)):
        if index >= end:
            return
        if index >= start:
            yield rec


def answers_from_logits(
    option_ids: Mapping[str, Sequence[str]], logits: Mapping[str, Sequence[float]]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for qid, ids in option_ids.items():
        lg = [float(x) for x in logits[qid]]
        out[qid] = {"option_ids": list(ids), "logits": lg, "probs": softmax(lg)}
    return out


def answers_from_probs(
    option_ids: Mapping[str, Sequence[str]], probs: Mapping[str, Sequence[float]]
) -> dict[str, dict[str, Any]]:
    """For engines that expose only probabilities: logits stay null."""
    return {
        qid: {"option_ids": list(ids), "logits": None, "probs": [float(p) for p in probs[qid]]}
        for qid, ids in option_ids.items()
    }


def make_row(
    record_id: str,
    task: str,
    model: str,
    config: str,
    answers: Mapping[str, Any],
    latency_ms: float,
    input_tokens: int | None,
    engine: str,
    engine_version: str,
    error: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": record_id,
        "task": task,
        "model": model,
        "config": config,
        "answers": dict(answers),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "engine": engine,
        "engine_version": engine_version,
        "error": error,
    }
    row.update(extra)
    return row


def validate_row(row: Mapping[str, Any]) -> None:
    """Raise ValueError if the row does not follow the shared interface."""
    for key in ("id", "task", "model", "config", "answers", "latency_ms", "input_tokens",
                "engine", "engine_version", "error"):
        if key not in row:
            raise ValueError(f"row {row.get('id')!r} missing key {key!r}")
    for qid, ans in row["answers"].items():
        n = len(ans["option_ids"])
        if len(ans["probs"]) != n:
            raise ValueError(f"{row['id']}/{qid}: {len(ans['probs'])} probs for {n} options")
        if ans["logits"] is not None and len(ans["logits"]) != n:
            raise ValueError(f"{row['id']}/{qid}: {len(ans['logits'])} logits for {n} options")


def write_row(fh, row: Mapping[str, Any]) -> None:
    validate_row(row)
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def sniff_option_ids(question: Mapping[str, Any]) -> list[str]:
    """Option ids in the order Cloudflare's encoder scores them (mirrors joint_schema_model.question_options)."""
    qtype = str(question["type"])
    if qtype == "noul":
        return ["true", "false"]
    if qtype == "choice":
        return sorted(str(k) for k in question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def iter_pairs(items: Iterable[Any], n: int) -> Iterator[list[Any]]:
    """Yield consecutive chunks of size n."""
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def plan_outputs(records: Sequence[str], out: str | None, out_dir: str | None) -> list[tuple[str, Path]]:
    """Map records files to output files: one records file + --out, or any number + --out-dir (<stem>.jsonl)."""
    if (out is None) == (out_dir is None):
        raise ValueError("give exactly one of --out (single records file) or --out-dir")
    if out is not None:
        if len(records) != 1:
            raise ValueError("--out needs exactly one records file; use --out-dir for several")
        return [(records[0], Path(out))]
    return [(r, Path(out_dir) / f"{Path(r).stem}.jsonl") for r in records]


def weights_size_gb(model_dir: str | Path) -> float:
    """On-disk size (GB = 1e9 bytes) of all *.safetensors under model_dir (symlinks followed), head files included."""
    total = 0
    for p in Path(model_dir).rglob("*.safetensors"):
        total += p.stat().st_size
    return total / 1e9


def build_meta(engine: str, engine_version: str, file_size_gb: float | None, peak_memory_gb: float | None,
               extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The meta.json the analysis expects beside results/<model>/<config>/ (plus any extra keys)."""
    meta: dict[str, Any] = {
        "file_size_gb": None if file_size_gb is None else round(file_size_gb, 3),
        "peak_memory_gb": None if peak_memory_gb is None else round(peak_memory_gb, 3),
        "engine": engine,
        "engine_version": engine_version,
    }
    meta.update(extra or {})
    return meta


def write_meta(out_dir: str | Path, meta: Mapping[str, Any]) -> Path:
    path = Path(out_dir) / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))
    return path
