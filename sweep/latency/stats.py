"""Pure-Python latency statistics shared by latency_http.py (host) and latency_native.py (container).

No numpy: the Spark host runs the HTTP client with the system python3. Percentiles use linear
interpolation between order statistics (the numpy default), so they match what the analysis would compute.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

SAMPLE_PER_SET = 300
WARMUP_REQUESTS = 20
SAMPLE_SEED = "clef-quant-latency-v1"
TOKEN_EDGES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def percentile(values: Sequence[float], q: float) -> float:
    """q in [0, 100], linear interpolation."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def sample_indices(n: int, k: int, key: str, seed: str = SAMPLE_SEED) -> list[int]:
    """Fixed sample of k of n record positions (sorted); identical for every config and every run."""
    if n <= k:
        return list(range(n))
    return sorted(random.Random(f"{seed}:{key}").sample(range(n), k))


def bucket_label(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    lo = 0
    for edge in TOKEN_EDGES:
        if tokens <= edge:
            return f"<={edge}" if lo == 0 else f"{lo + 1}-{edge}"
        lo = edge
    return f">{TOKEN_EDGES[-1]}"


def _bucket_order() -> list[str]:
    labels, lo = [], 0
    for edge in TOKEN_EDGES:
        labels.append(f"<={edge}" if lo == 0 else f"{lo + 1}-{edge}")
        lo = edge
    return labels + [f">{TOKEN_EDGES[-1]}", "unknown"]


def summarize(latencies_ms: Sequence[float], errors: int = 0, wall_seconds: float | None = None) -> dict[str, Any]:
    n = len(latencies_ms)
    out: dict[str, Any] = {"n": n, "errors": errors}
    if n:
        out.update(p50_ms=round(percentile(latencies_ms, 50), 3), p95_ms=round(percentile(latencies_ms, 95), 3),
                   p99_ms=round(percentile(latencies_ms, 99), 3), mean_ms=round(sum(latencies_ms) / n, 3),
                   min_ms=round(min(latencies_ms), 3), max_ms=round(max(latencies_ms), 3))
    else:
        out.update(p50_ms=None, p95_ms=None, p99_ms=None, mean_ms=None, min_ms=None, max_ms=None)
    out["decisions_per_sec"] = round(n / wall_seconds, 4) if (n and wall_seconds) else None
    return out


def warmup_order(items_by_set: Mapping[str, Sequence[Any]], count: int = WARMUP_REQUESTS) -> list[Any]:
    """Round-robin across sets (so warm-up sees every task's input shape), at most `count` items."""
    out: list[Any] = []
    depth = 0
    longest = max((len(v) for v in items_by_set.values()), default=0)
    while len(out) < count and depth < longest:
        for items in items_by_set.values():
            if depth < len(items) and len(out) < count:
                out.append(items[depth])
        depth += 1
    return out


def build_report(observations: Sequence[Mapping[str, Any]], wall_seconds: float) -> dict[str, Any]:
    """observations: {"set", "private", "tokens", "ms", "error"} per timed request. Aggregates only:
    no ids, no record content, no error strings leave this function."""
    ok = [o for o in observations if o.get("error") is None and o.get("ms") is not None]
    overall = summarize([o["ms"] for o in ok], errors=len(observations) - len(ok), wall_seconds=wall_seconds)
    overall["wall_seconds"] = round(wall_seconds, 3)

    per_set: dict[str, Any] = {}
    for name in dict.fromkeys(o["set"] for o in observations):
        rows = [o for o in observations if o["set"] == name]
        good = [o for o in rows if o.get("error") is None and o.get("ms") is not None]
        entry = summarize([o["ms"] for o in good], errors=len(rows) - len(good))
        toks = [o["tokens"] for o in good if o.get("tokens") is not None]
        entry["input_tokens"] = ({"mean": round(sum(toks) / len(toks), 1), "p50": percentile(toks, 50),
                                  "max": max(toks)} if toks else None)
        entry["private"] = bool(rows[0]["private"])
        per_set[name] = entry

    buckets: list[dict[str, Any]] = []
    for label in _bucket_order():
        rows = [o for o in ok if bucket_label(o.get("tokens")) == label]
        if not rows:
            continue
        s = summarize([o["ms"] for o in rows])
        buckets.append({"label": label, "n": s["n"], "p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"],
                        "p99_ms": s["p99_ms"], "mean_ms": s["mean_ms"]})
    return {"overall": overall, "per_set": per_set, "input_token_buckets": buckets}
