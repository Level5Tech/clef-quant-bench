"""Pure helpers for the concurrent sweep: shard units, atomic claims, ordered merge, memory model, launch rule.

A "unit" is a contiguous record range of one set. Every quality job runs at batch size 1 (ref-r2 at 8 with unit
starts that are multiples of 8, so its batches are identical to the unsharded run); concurrency only changes how
many units are in flight, never what one record sees.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Native Clef-flash BF16 seconds per record measured in the dry run (data_dev, bs=1, GB10). Used only to size units.
SECONDS_PER_RECORD = {"clinc150": 1.6, "banking77": 0.94, "ragtruth": 0.45, "when2call": 0.37, "contractnli": 0.18}
DEFAULT_SPR = 0.16
PRIVATE_SPR = 0.30       # no measurement yet for r1/r2/r4a/r4b at full size; a guess only used to size units
UNIT_TARGET_SECONDS = 600
GPU_GIB_TO_GB = 1.073741824
GPU_TOTAL_GIB = 121.69   # what vLLM sees as total device memory on the GB10 (nvidia-smi / vLLM log)
HEADROOM_GB = 12.0


@dataclass(frozen=True)
class Unit:
    set: str
    start: int
    end: int
    private: bool

    @property
    def id(self) -> str:
        return f"{self.set}__{self.start:06d}-{self.end:06d}"

    @property
    def n(self) -> int:
        return self.end - self.start


def chunk_size(set_name: str, model_scale: float = 1.0, private: bool = False) -> int:
    spr = (PRIVATE_SPR if private else SECONDS_PER_RECORD.get(set_name, DEFAULT_SPR)) * model_scale
    return max(8, int(UNIT_TARGET_SECONDS / spr) // 8 * 8)


def chunk_ranges(n: int, chunk: int) -> list[tuple[int, int]]:
    """Contiguous ranges covering [0, n); every start is a multiple of 8; sizes balanced."""
    if n <= 0:
        return []
    parts = math.ceil(n / chunk)
    size = math.ceil(n / parts / 8) * 8
    out, start = [], 0
    while start < n:
        out.append((start, min(start + size, n)))
        start += size
    return out


def unit_dir(work_root: Path, model: str, config: str, u: Unit) -> Path:
    return work_root / model / config / u.set / f"{u.start:06d}-{u.end:06d}"


def unit_out(work_root: Path, model: str, config: str, u: Unit) -> Path:
    return unit_dir(work_root, model, config, u) / f"{u.set}.jsonl"


# ------------------------------------------------------------------------------------------ claims
def try_claim(claim_dir: Path, unit_id: str, owner: str) -> bool:
    claim_dir.mkdir(parents=True, exist_ok=True)
    try:
        (claim_dir / unit_id).mkdir()      # mkdir is atomic: exactly one claimant wins
    except FileExistsError:
        return False
    (claim_dir / unit_id / "owner").write_text(owner)
    return True


def claimed_ids(claim_dir: Path) -> set[str]:
    return {p.name for p in claim_dir.iterdir()} if claim_dir.is_dir() else set()


def release_owned(claim_dir: Path, owner: str, keep: set[str]) -> None:
    """Drop the claims `owner` holds on units that are not finished (so another worker can take them)."""
    for p in list(claim_dir.iterdir()) if claim_dir.is_dir() else []:
        try:
            who = (p / "owner").read_text()
        except OSError:
            who = owner
        if who == owner and p.name not in keep:
            shutil.rmtree(p, ignore_errors=True)


def clear_claims(claim_dir: Path) -> None:
    shutil.rmtree(claim_dir, ignore_errors=True)


# ------------------------------------------------------------------------------------------ files
def valid_lines(path: Path) -> int | None:
    n = 0
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    json.loads(line)
                    n += 1
    except (ValueError, OSError):
        return None
    return n


def unit_complete(out: Path, expected: int) -> bool:
    return out.is_file() and valid_lines(out) == expected


def merge_units(parts: Sequence[Path], out: Path, records: Path) -> int:
    """Concatenate unit outputs (already in record order) into `out`, verifying ids against the records file.
    Written to a temp name and renamed, so a failure never leaves a plausible-looking partial file."""
    want = (json.loads(l)["id"] for l in open(records, encoding="utf-8") if l.strip())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".merge")
    n = 0
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            for part in parts:
                for line in open(part, encoding="utf-8"):
                    if not line.strip():
                        continue
                    rid = json.loads(line)["id"]
                    expect = next(want, None)
                    if rid != expect:
                        raise ValueError(f"merge order mismatch at row {n}: {rid!r} != {expect!r} in {part.parent.name}")
                    fh.write(line if line.endswith("\n") else line + "\n")
                    n += 1
            if next(want, None) is not None:
                raise ValueError(f"merged {n} rows but the records file has more")
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n


# ------------------------------------------------------------------------------------------ memory
def native_mem_gb(weights_gb: float) -> float:
    # MEASURED: a native worker holds ~12 GB of host memory on top of its GPU-resident weights (unified memory, so both
    # count against the same 121 GB). Four of them at once exhausted the machine and triggered the OOM killer.
    return weights_gb + 12.5


def vllm_mem_gb(util: float) -> float:
    return util * GPU_TOTAL_GIB * GPU_GIB_TO_GB + 6.0   # claimed GPU share + server/engine host RSS


def llama_mem_gb(gguf_gb: float) -> float:
    return gguf_gb * 1.35 + 4.0


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    raise RuntimeError("MemAvailable not found")


# ------------------------------------------------------------------------------------------ launch rule
@dataclass
class TaskState:
    name: str
    cap: int            # max concurrent workers for this task
    unclaimed: int      # units nobody has claimed yet
    mem_gb: float       # estimated unified memory per worker


class _HasMem(Protocol):
    task: str
    mem_gb: float


def next_launch(tasks: Sequence[TaskState], running: Sequence[Any], budget_gb: float, slots: int) -> TaskState | None:
    """First task (priority order) that can take one more worker: below its cap, units left for it, a free global
    slot, and its memory fits the budget on top of what is running."""
    if len(running) >= slots:
        return None
    used = sum(w.mem_gb for w in running)
    for t in tasks:
        mine = sum(1 for w in running if w.task == t.name)
        if t.unclaimed <= mine or mine >= t.cap:
            continue
        if used + t.mem_gb <= budget_gb:
            return t
    return None
