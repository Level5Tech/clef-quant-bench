#!/usr/bin/env python
"""Quality-run throughput from the sweep log, plus the isolated latency benchmark (`results-latency/`).

Source: `END job=<model>/<config>/<set> exit=0 records=N ... unit_secs=S ...` lines of the sweep log. Only exit=0
lines count; when a (model, config, set) has several END lines the latest one wins. `unit_secs` is the sum of the
unit wall-clock seconds of that set (model/server load time is not in it; `elapsed` is not used because it spans
pauses and restarts of the driver). These are throughputs of the quality runs as they were configured: batched vLLM
(max-num-seqs 16, 16 requests in flight), batch-size-1 native `ref`, batch-size-8 `ref-r2`, single-sequence
llama.cpp. They are not comparable across engines as a latency statement, and they include whatever else shared the
GPU at the time (the result meta files say so in their `latency_note`).

File sizes come from each config's `meta.json` in the results tree; anything missing is reported as missing.

The isolated latency benchmark (`--latency`, default `results-latency/<model>/<config>.json`) is a separate block,
`latency` in speed.json and its own section in speed.md: batch size 1, one request in flight, 300 sampled records per
set, per-request milliseconds. It is not derived from the sweep log. A missing directory is reported in the block and
nothing fails. The `ref-r2` files are aliases of `ref` (same engine path at batch size 1) and carry `alias_of`.

Usage:
  python -m bench.speed --log analysis/inputs/sweep.log --results results --private-results private/results \
      --records data/records --out analysis [--latency results-latency]
writes <out>/speed.json and <out>/speed.md. Private sets appear only as set name, record count and seconds in the
quality-run part, and only as pooled aggregates (never by set) in the latency section of speed.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python bench/speed.py`
    sys.path.append(str(Path(__file__).resolve().parent.parent))

JOB_RE = re.compile(r"^(?P<model>[^/\s]+)/(?P<config>[^/\s]+)/(?P<set>[^/\s]+)$")
SERVER_RE = re.compile(r"^(?P<model>clef-flash|clef)-(?P<config>.+)/server$")
CONFIG_ORDER = ("ref", "q8_0", "q4_k_m", "fp8", "nvfp4", "bf16-srv", "gguf-bf16", "ref-r2",
                "bf16-srv-r2", "fp8-r2", "nvfp4-r2", "gguf-bf16-r2", "ref-fla")
MODEL_ORDER = ("clef-flash", "clef")


def parse_kv(rest: str) -> dict[str, str]:
    out = {}
    for tok in rest.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def parse_log(path: Path) -> dict[str, Any]:
    """END lines of the sweep log: set jobs (latest exit=0 per set), server start lines, ignored lines."""
    sets: dict[tuple[str, str, str], dict[str, Any]] = {}
    servers: dict[tuple[str, str], dict[str, Any]] = {}
    nonzero: list[str] = []
    n_end = 0
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != "END":
            continue
        n_end += 1
        kv = parse_kv(parts[2])
        job = kv.get("job", "")
        m = JOB_RE.match(job)
        if m and "records" in kv:
            key = (m["model"], m["config"], m["set"])
            if kv.get("exit") != "0":
                nonzero.append(f"{job} exit={kv.get('exit')}")
                continue
            sets[key] = {"ts": parts[0], "line": lineno, "records": int(kv["records"]),
                         "expected": int(kv["expected"]) if "expected" in kv else None,
                         "errors": int(kv["errors"]) if "errors" in kv else None,
                         "unit_secs": float(kv["unit_secs"]), "units": int(kv["units"]) if "units" in kv else None,
                         "engine": kv.get("engine")}
            continue
        s = SERVER_RE.match(job)
        if s and kv.get("exit") == "0":
            servers[(s["model"], s["config"])] = {"ts": parts[0], "start_secs": float(kv["secs"]),
                                                  "gpu_memory_utilization": None if kv.get("util") in (None, "None")
                                                  else float(kv["util"])}
    return {"sets": sets, "servers": servers, "nonzero_exit": nonzero, "n_end_lines": n_end}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sidecar_secs(side: dict[str, Any] | None) -> tuple[float | None, str | None]:
    """Seconds recorded in a per-set sidecar: current runners write unit_wall_seconds_sum, the first reference runs
    wrote wall_seconds (both exclude model/server load)."""
    if not side:
        return None, None
    for key in ("unit_wall_seconds_sum", "wall_seconds"):
        if side.get(key) is not None:
            return float(side[key]), key
    return None, None


def list_sets(cfg_dir: Path) -> list[str]:
    return sorted(p.stem for p in cfg_dir.glob("*.jsonl"))


def agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(r["records"] for r in rows)
    secs = sum(r["unit_secs"] for r in rows)
    return {"n_sets": len(rows), "records": n, "unit_secs": round(secs, 1),
            "sec_per_record": (secs / n) if n else None, "records_per_sec": (n / secs) if secs else None}


def build(log: Path, results: Path, private_results: Path | None, public_sets: list[str],
          private_sets: list[str], latency: Path | None = None) -> dict[str, Any]:
    parsed = parse_log(log)
    end = parsed["sets"]
    public, private = set(public_sets), set(private_sets)
    out: dict[str, Any] = {
        "meta": {"source_log": str(log), "n_end_lines": parsed["n_end_lines"], "n_set_end_lines_used": len(end),
                 "nonzero_exit_end_lines": parsed["nonzero_exit"],
                 "note": ("quality-run throughput: batched vLLM (16 in flight), bs1 native ref, bs8 ref-r2, "
                          "single-sequence llama.cpp; these numbers are NOT the isolated latency benchmark, which is in the `latency` block (speed.json) and the section below (speed.md). "
                          "unit_secs excludes model/server load.")},
        "configs": {}, "missing_end_lines": [], "sidecar_check": {}, "ratio_27b_over_9b": {}}

    trees = [(results, "public")]
    if private_results is not None and Path(private_results).is_dir():
        trees.append((Path(private_results), "private"))
    sidecar_diffs: list[float] = []
    for root, kind in trees:
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            for cdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
                model, cfg = mdir.name, cdir.name
                entry = out["configs"].setdefault(model, {}).setdefault(cfg, {"per_set": {}})
                meta = read_json(cdir / "meta.json")
                if meta and kind == "public":
                    entry["meta"] = {k: meta.get(k) for k in (
                        "file_size_gb", "peak_memory_gb", "engine", "engine_version", "batch_size", "max_num_seqs",
                        "client_concurrency", "gdn_kernel", "subset", "repeat_of")}
                for s in list_sets(cdir):
                    side = read_json(cdir / f"{s}.jsonl.meta.json")
                    e = end.get((model, cfg, s))
                    row: dict[str, Any] = {"kind": kind}
                    if e:
                        row.update({"records": e["records"], "unit_secs": e["unit_secs"],
                                    "sec_per_record": e["unit_secs"] / e["records"], "source": "END line",
                                    "log_ts": e["ts"], "errors": e["errors"]})
                        secs, _ = _sidecar_secs(side)
                        if secs is not None:
                            sidecar_diffs.append(abs(secs - e["unit_secs"]))
                    else:
                        row["source"] = "no END line"
                        out["missing_end_lines"].append({"model": model, "config": cfg, "set": s})
                        secs, key = _sidecar_secs(side)
                        if secs is not None and side.get("records"):
                            row["sidecar_records"] = side["records"]
                            row["sidecar_unit_secs"] = secs
                            row["sidecar_sec_per_record"] = secs / side["records"]
                            row["sidecar_key"] = key
                    if side and side.get("peak_memory_gb") is not None:
                        row["peak_memory_gb_sidecar"] = side["peak_memory_gb"]
                    entry["per_set"][s] = row
    # aggregate per config
    for model, cfgs in out["configs"].items():
        for cfg, entry in cfgs.items():
            rows = [{"set": s, **r} for s, r in entry["per_set"].items() if r["source"] == "END line"]
            entry["public"] = agg([r for r in rows if r["set"] in public])
            entry["private"] = agg([r for r in rows if r["set"] in private])
            entry["all"] = agg(rows)
            entry["sets_without_end_line"] = sorted(s for s, r in entry["per_set"].items() if r["source"] != "END line")
            peaks = [r["peak_memory_gb_sidecar"] for r in entry["per_set"].values() if "peak_memory_gb_sidecar" in r]
            entry["peak_memory_gb_max_over_set_sidecars"] = max(peaks) if peaks else None
            if "meta" not in entry:
                entry["meta"] = None
            srv = parsed["servers"].get((model, cfg))
            entry["server_start"] = srv
    out["like_for_like_full_sets"] = _like_for_like(out["configs"].get("clef-flash", {}), public, private)
    out["sidecar_check"] = {"n_sets_compared": len(sidecar_diffs),
                            "max_abs_diff_secs": max(sidecar_diffs) if sidecar_diffs else None,
                            "note": "END unit_secs vs per-set sidecar unit_wall_seconds_sum"}

    # 27B / 9B per-record time ratio on the sets the 27B reference finished
    big = out["configs"].get("clef", {}).get("ref", {}).get("per_set", {})
    small = out["configs"].get("clef-flash", {}).get("ref", {}).get("per_set", {})
    rows = []
    for s, b in sorted(big.items()):
        if b["source"] != "END line":
            continue
        sm = small.get(s)
        r: dict[str, Any] = {"set": s, "kind": b["kind"], "records": b["records"], "sec_per_record_27b": b["sec_per_record"]}
        if sm and sm["source"] == "END line":
            r.update({"sec_per_record_9b": sm["sec_per_record"], "ratio": b["sec_per_record"] / sm["sec_per_record"],
                      "9b_source": "END line"})
        elif sm and "sidecar_sec_per_record" in sm:
            r.update({"sec_per_record_9b": sm["sidecar_sec_per_record"],
                      "ratio": b["sec_per_record"] / sm["sidecar_sec_per_record"],
                      "9b_source": f"per-set sidecar {sm['sidecar_key']} (no END line in the log)"})
        else:
            r.update({"sec_per_record_9b": None, "ratio": None, "9b_source": "9B reference has no timing for this set"})
        rows.append(r)

    def pooled(rs: list[dict[str, Any]]) -> dict[str, Any]:
        rs = [r for r in rs if r["ratio"] is not None]
        if not rs:
            return {"n_sets": 0}
        t27 = sum(r["sec_per_record_27b"] * r["records"] for r in rs)
        t9 = sum(r["sec_per_record_9b"] * r["records"] for r in rs)
        ratios = sorted(r["ratio"] for r in rs)
        return {"n_sets": len(rs), "records": sum(r["records"] for r in rs), "secs_27b": round(t27, 1),
                "secs_9b": round(t9, 1), "pooled_ratio": t27 / t9, "min_ratio": ratios[0], "max_ratio": ratios[-1]}
    out["ratio_27b_over_9b"] = {
        "per_set": rows,
        "pooled_end_lines_only": pooled([r for r in rows if r["9b_source"] == "END line"]),
        "fallback_sets": [r["set"] for r in rows if r["9b_source"].startswith("per-set sidecar")],
        "pooled_with_sidecar_fallback": pooled(rows),
        "pooled_public_with_fallback": pooled([r for r in rows if r["kind"] == "public"]),
        "note": ("ratio = 27B seconds per record / 9B seconds per record, both reference (`ref`, native, batch 1), same "
                 "records; 'pooled' weights by records. The 27B has no clinc150 yet, so the pooled ratio excludes it."),
    }
    out["configs"] = {m: {c: out["configs"][m][c] for c in sorted(out["configs"][m], key=lambda x: CONFIG_ORDER.index(x)
                                                                   if x in CONFIG_ORDER else 99)}
                      for m in sorted(out["configs"], key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)}
    out["file_sizes_missing"] = [{"model": m, "config": c} for m, cfgs in out["configs"].items() for c, e in cfgs.items()
                                 if not e["meta"] or e["meta"].get("file_size_gb") is None]
    out["build_lines_in_log"] = _build_lines(log)
    out["latency"] = build_latency(latency)
    return out


FULL_SET_CONFIGS = ("ref", "q8_0", "q4_k_m", "fp8", "nvfp4", "bf16-srv", "gguf-bf16", "ref-r2")


def _like_for_like(cfgs: dict[str, Any], public: set[str], private: set[str]) -> dict[str, Any]:
    """Full-set configs of one model on the sets where EVERY one of them has an END line, so s/record is comparable
    (the record mix of a set changes s/record a lot: banking77 and clinc150 are slow, mmlu and arc are fast)."""
    full = {c: cfgs[c] for c in FULL_SET_CONFIGS if c in cfgs}
    if not full:
        return {}
    have = [{s for s, r in e["per_set"].items() if r["source"] == "END line"} for e in full.values()]
    common = sorted(set.intersection(*have))
    rows = {}
    for c, e in full.items():
        rs = [{"set": s, **e["per_set"][s]} for s in common]
        a = agg(rs)
        a["public"] = agg([r for r in rs if r["set"] in public])
        a["private"] = agg([r for r in rs if r["set"] in private])
        rows[c] = a
    ref = rows.get("ref")
    for c, a in rows.items():
        a["speed_vs_ref"] = (ref["sec_per_record"] / a["sec_per_record"]) if ref and a["sec_per_record"] else None
    # reference over every set, using the per-set sidecar for sets that have no END line
    full_ref = cfgs.get("ref", {}).get("per_set", {})
    allrows = []
    for s, r in full_ref.items():
        if r["source"] == "END line":
            allrows.append({"records": r["records"], "unit_secs": r["unit_secs"]})
        elif "sidecar_unit_secs" in r:
            allrows.append({"records": r["sidecar_records"], "unit_secs": r["sidecar_unit_secs"]})
    return {"sets": common, "configs": rows,
            "ref_all_sets_with_sidecar_fallback": agg(allrows) if allrows else None,
            "note": ("full-set configs only (ref, q8_0, q4_k_m, fp8, nvfp4, bf16-srv, gguf-bf16, ref-r2); restricted to "
                     "the sets where every one has an END line; speed_vs_ref = ref s/record divided by the config's")}


def _build_lines(log: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(log).read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1] == "END":
            kv = parse_kv(parts[2])
            if kv.get("job", "").startswith("build/") and kv.get("exit") == "0":
                rows.append({"job": kv["job"], "secs": float(kv["secs"]),
                             "size_gb": float(kv["size_gb"]) if "size_gb" in kv else None, "ts": parts[0]})
    return rows


# ------------------------------------------------------------------------------ isolated latency benchmark

LAT_STATS = ("n", "errors", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "decisions_per_sec", "wall_seconds")
POOLED_NOTE = ("Public pooled mean = the n-weighted mean of the per-set mean_ms over the public sets. A pooled p50 "
               "cannot be derived from per-set p50s, so `median_of_set_p50_ms` is the median of the per-set p50s (an "
               "unweighted median over sets, not the p50 of the pooled requests). Private sets are pooled the same way "
               "and appear only as that aggregate.")
RATIO_NOTE = ("`ratio_27b_over_9b[config]` = clef mean_ms / clef-flash mean_ms of the same config (overall mean_ms, all "
              "16 sets). `mean_ms_over_ref` = the config's overall mean_ms / the same model's `ref` overall mean_ms "
              "(above 1 is slower than ref).")


def _median(xs: list[float]) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _round3(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


def _pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """n-weighted mean of the per-set means; median of the per-set p50s (a pooled p50 is not derivable)."""
    n = sum(r["n"] for r in rows)
    return {"n_sets": len(rows), "n": n,
            "mean_ms": round(sum(r["n"] * r["mean_ms"] for r in rows) / n, 3) if n else None,
            "median_of_set_p50_ms": _round3(_median([r["p50_ms"] for r in rows if r.get("p50_ms") is not None]))}


def _latency_entry(d: dict[str, Any]) -> dict[str, Any]:
    per_set = d.get("per_set") or {}
    pub = [r for r in per_set.values() if not r.get("private")]
    priv = [r for r in per_set.values() if r.get("private")]
    private_pool = _pool(priv)
    private_pool.pop("median_of_set_p50_ms")
    overall = d.get("overall") or {}
    return {"engine": d.get("engine"), "engine_version": d.get("engine_version"), "batch_size": d.get("batch_size"),
            "concurrency": d.get("concurrency"), "timing": d.get("timing"), "sample_per_set": d.get("sample_per_set"),
            "measured_at": d.get("measured_at"),
            "overall": {k: overall.get(k) for k in LAT_STATS},
            "public": _pool(pub), "private": private_pool,
            "peak_memory_gb": d.get("peak_memory_gb"), "peak_memory_source": d.get("peak_memory_source"),
            "file_size_gb": d.get("file_size_gb"),
            "other_gpu_processes_at_start": d.get("other_gpu_processes_at_start"),
            "alias_of": d.get("alias_of"), "alias_note": d.get("alias_note")}


def _ratio(num: float | None, den: float | None) -> float | None:
    return (num / den) if num is not None and den else None


def build_latency(latency_dir: Path | None) -> dict[str, Any]:
    """The `latency` block from results-latency/<model>/<config>.json. Never raises on a missing or partial tree."""
    base: dict[str, Any] = {"source_dir": None if latency_dir is None else str(latency_dir)}
    if latency_dir is None or not Path(latency_dir).is_dir():
        return {**base, "available": False,
                "note": f"latency directory not found ({latency_dir}); no isolated latency benchmark in this run"}
    models: dict[str, Any] = {}
    unreadable: list[str] = []
    public_sets: dict[str, set[str]] = {}
    set_means: dict[str, dict[str, dict[str, float]]] = {}
    for mdir in sorted(p for p in Path(latency_dir).iterdir() if p.is_dir()):
        for f in sorted(mdir.glob("*.json")):
            d = read_json(f)
            if not isinstance(d, dict) or "overall" not in d or "per_set" not in d:
                unreadable.append(f"{mdir.name}/{f.name}")
                continue
            cfg = f.stem
            models.setdefault(mdir.name, {})[cfg] = _latency_entry(d)
            for s, r in d["per_set"].items():
                if not r.get("private"):
                    set_means.setdefault(mdir.name, {}).setdefault(s, {})[cfg] = r.get("mean_ms")
    for cfgs in models.values():
        ref = cfgs.get("ref", {}).get("overall", {}).get("mean_ms")
        for e in cfgs.values():
            e["mean_ms_over_ref"] = _ratio(e["overall"]["mean_ms"], ref)
    big, small = models.get("clef", {}), models.get("clef-flash", {})
    ratio = {c: r for c in CONFIG_ORDER if c in big and c in small
             and (r := _ratio(big[c]["overall"]["mean_ms"], small[c]["overall"]["mean_ms"])) is not None}

    def order_cfgs(cfgs: dict[str, Any]) -> dict[str, Any]:
        return {c: cfgs[c] for c in sorted(cfgs, key=lambda x: CONFIG_ORDER.index(x) if x in CONFIG_ORDER else 99)}

    def order_set_table(table: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        return {s: order_cfgs(table[s]) for s in sorted(table)}

    samples = sorted({e["sample_per_set"] for c in models.values() for e in c.values() if e["sample_per_set"] is not None})
    ordered = sorted(models, key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)
    return {**base, "available": True,
            "meta": {"pooled_note": POOLED_NOTE, "ratio_note": RATIO_NOTE,
                     "note": ("isolated latency: batch size 1, one request in flight, "
                              f"{'/'.join(str(v) for v in samples) or '?'} sampled records per set, per-request milliseconds. Timing differs by engine: vLLM and llama.cpp configs are an HTTP "
                              "round trip to localhost (server-side tokenisation included); ref and ref-r2 are "
                              "in-process (encode, collate, forward, cuda sync, no HTTP). ref-r2 is an alias of ref, "
                              "not a second measurement. Peak memory is measured differently per engine (see "
                              "peak_memory_source).")},
            "models": {m: order_cfgs(models[m]) for m in ordered},
            "ratio_27b_over_9b": ratio,
            "per_set_public_mean_ms": {m: order_set_table(set_means[m]) for m in ordered if m in set_means},
            "unreadable_files": unreadable}


# ------------------------------------------------------------------------------ markdown

def _f(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def regime(meta: dict[str, Any] | None) -> str:
    if not meta:
        return "(no meta.json)"
    eng = meta.get("engine") or ""
    if "vllm" in eng:
        return f"vLLM, max-num-seqs {meta.get('max_num_seqs')}, {meta.get('client_concurrency')} in flight"
    if "llama" in eng:
        return "llama.cpp, single sequence"
    kern = {"torch": "PyTorch GDN fallback", "fla": "flash-linear-attention GDN"}.get(meta.get("gdn_kernel"), "")
    return f"native transformers, batch {meta.get('batch_size')}, {kern}".rstrip(", ")


def render_latency_md(lat: dict[str, Any]) -> list[str]:
    """Lines of the latency section. Private sets are shown only as a pooled mean, never by set name."""
    L = ["## Isolated latency benchmark (results-latency)", ""]
    if not lat.get("available"):
        return L + [lat.get("note", "latency block unavailable") + ".", ""]
    L += [lat["meta"]["note"], "", lat["meta"]["pooled_note"], "", lat["meta"]["ratio_note"], ""]
    for model, cfgs in lat["models"].items():
        L += [f"### {model}", ""]
        rows = []
        for cfg, e in cfgs.items():
            o, p, q = e["overall"], e["public"], e["private"]
            gpu = e["other_gpu_processes_at_start"]
            rows.append([cfg + (f" (alias of {e['alias_of']})" if e["alias_of"] else ""), str(e["engine"]),
                         str(o["n"]), str(o["errors"]), _f(o["mean_ms"], 1), _f(o["p50_ms"], 1), _f(o["p95_ms"], 1),
                         _f(o["p99_ms"], 1), _f(p["mean_ms"], 1), _f(p["median_of_set_p50_ms"], 1),
                         _f(q["mean_ms"], 1), _f(e["mean_ms_over_ref"], 2), _f(e["peak_memory_gb"], 2),
                         _f(e["file_size_gb"], 2),
                         "not recorded" if gpu is None else f"{gpu.get('count')} / {gpu.get('total_mib')} MiB"])
        L += [md_table(["config", "engine", "n", "errors", "mean ms", "p50 ms", "p95 ms", "p99 ms",
                        "public pooled mean ms", "public median of set p50s ms", "private pooled mean ms",
                        "mean vs ref (x)", "peak GB", "file GB", "other GPU procs at start (count / MiB)"], rows), ""]
        by_src: dict[str, list[str]] = {}
        for cfg, e in cfgs.items():
            by_src.setdefault(str(e["peak_memory_source"]), []).append(cfg)
        L += ["Peak memory source: " + "; ".join(f"{', '.join(c)}: {src}" for src, c in by_src.items()) + ".", ""]
    if lat["ratio_27b_over_9b"]:
        L += ["### 27B / 9B overall mean_ms by config", "",
              md_table(["config", "27B mean ms", "9B mean ms", "ratio"],
                       [[c, _f(lat["models"]["clef"][c]["overall"]["mean_ms"], 1),
                         _f(lat["models"]["clef-flash"][c]["overall"]["mean_ms"], 1), _f(r, 2)]
                        for c, r in lat["ratio_27b_over_9b"].items()]), ""]
    for model, table in lat["per_set_public_mean_ms"].items():
        cols = list(lat["models"][model])
        L += [f"### {model}: mean ms by public set", "",
              md_table(["set"] + cols, [[s] + [_f(row.get(c), 1) for c in cols] for s, row in table.items()]), ""]
    if lat["unreadable_files"]:
        L += ["Unreadable latency files (skipped): " + ", ".join(lat["unreadable_files"]) + ".", ""]
    return L


def render_md(res: dict[str, Any]) -> str:
    L = ["# Quality-run speed (from the sweep log)", "", res["meta"]["note"], "",
         f"Source: {res['meta']['source_log']}: {res['meta']['n_set_end_lines_used']} set END lines used "
         f"(exit=0, latest per set) of {res['meta']['n_end_lines']} END lines; non-zero exits: "
         f"{len(res['meta']['nonzero_exit_end_lines'])}. END unit_secs vs the per-set sidecar's unit_wall_seconds_sum: "
         f"{res['sidecar_check']['n_sets_compared']} sets compared, max abs difference "
         f"{_f(res['sidecar_check']['max_abs_diff_secs'], 2)} s.", ""]
    for model, cfgs in res["configs"].items():
        L += [f"## {model}", ""]
        rows = []
        for cfg, e in cfgs.items():
            a, p, q = e["all"], e["public"], e["private"]
            meta = e["meta"] or {}
            sub = "subset (first 200 per public task)" if (meta.get("subset") or meta.get("repeat_of")) else "full sets"
            rows.append([cfg, regime(e["meta"]), sub, str(a["n_sets"]), str(a["records"]), _f(a["unit_secs"], 1),
                         _f(a["sec_per_record"], 4), _f(a["records_per_sec"], 3),
                         f"{p['records']} / {_f(p['sec_per_record'], 4)}", f"{q['records']} / {_f(q['sec_per_record'], 4)}",
                         _f(meta.get("file_size_gb"), 2), _f(meta.get("peak_memory_gb") or e["peak_memory_gb_max_over_set_sidecars"], 2),
                         _f((e["server_start"] or {}).get("start_secs"), 1)])
        L += [md_table(["config", "run regime", "scope", "sets", "records", "sum unit_secs", "s/record", "records/s",
                        "public: records / s per record", "private: records / s per record", "file GB", "peak GB",
                        "server start s"], rows), ""]
        miss = [(c, e["sets_without_end_line"]) for c, e in cfgs.items() if e["sets_without_end_line"]]
        for c, s in miss:
            L += [f"- {model}/{c}: no END line for {', '.join(s)}; those sets are excluded from the totals above."]
        if miss:
            L += [""]
        nofile = [c for c, e in cfgs.items() if not e["meta"] or e["meta"].get("file_size_gb") is None]
        if nofile:
            L += [f"- file size not recorded (no config-level meta.json with file_size_gb): {', '.join(nofile)}. "
                  "Peak GB for those, where shown, is the maximum of the per-set sidecars.", ""]
    lfl = res.get("like_for_like_full_sets") or {}
    if lfl:
        L += ["## Like-for-like: full-set clef-flash configs on the same sets", "", lfl["note"] + ".", "",
              "Sets used (" + str(len(lfl["sets"])) + "): " + ", ".join(lfl["sets"]) + ".", ""]
        rows = [[c, str(a["records"]), _f(a["unit_secs"], 1), _f(a["sec_per_record"], 4), _f(a["records_per_sec"], 3),
                 _f(a["speed_vs_ref"], 2)] for c, a in lfl["configs"].items()]
        L += [md_table(["config", "records", "sum unit_secs", "s/record", "records/s", "speed vs ref (x)"], rows), ""]
        ra = lfl.get("ref_all_sets_with_sidecar_fallback")
        if ra:
            L += [f"clef-flash/ref over all {ra['n_sets']} sets with the per-set sidecar standing in for the sets without "
                  f"an END line: {ra['records']} records, {_f(ra['unit_secs'], 1)} s, {_f(ra['sec_per_record'], 4)} s/record.", ""]
    L += ["## Seconds per set (sum of unit_secs; records in brackets)", ""]
    sets = sorted({s for cfgs in res["configs"].values() for e in cfgs.values() for s in e["per_set"]})
    cols = [(m, c) for m, cfgs in res["configs"].items() for c in cfgs]
    rows = []
    for s in sets:
        row = [s]
        for m, c in cols:
            r = res["configs"][m][c]["per_set"].get(s)
            if r is None:
                row.append("-")
            elif r["source"] == "END line":
                row.append(f"{r['unit_secs']:.1f} ({r['records']})")
            else:
                row.append("no END")
        rows.append(row)
    L += [md_table(["set"] + [f"{m}/{c}" for m, c in cols], rows), ""]
    L += ["## Seconds per record by set", ""]
    rows = []
    for s in sets:
        row = [s]
        for m, c in cols:
            r = res["configs"][m][c]["per_set"].get(s)
            row.append("-" if r is None else (f"{r['sec_per_record']:.4f}" if r["source"] == "END line" else "no END"))
        rows.append(row)
    L += [md_table(["set"] + [f"{m}/{c}" for m, c in cols], rows), ""]
    rt = res["ratio_27b_over_9b"]
    L += ["## 27B / 9B per-record time (reference configs, native batch 1)", "", rt["note"], ""]
    rows = [[r["set"], r["kind"], str(r["records"]), _f(r["sec_per_record_27b"], 4), _f(r["sec_per_record_9b"], 4),
             _f(r["ratio"], 2), r["9b_source"]] for r in rt["per_set"]]
    L += [md_table(["set", "kind", "records", "27B s/record", "9B s/record", "ratio", "9B timing source"], rows), ""]
    if rt.get("fallback_sets"):
        L += [f"9B timing for {', '.join(rt['fallback_sets'])} comes from the per-set sidecar (no END line in the log).", ""]
    for key, title in (("pooled_end_lines_only", "END lines only"),
                       ("pooled_with_sidecar_fallback", "with sidecar fallback for sets lacking a 9B END line"),
                       ("pooled_public_with_fallback", "public sets only, with sidecar fallback")):
        p = rt[key]
        if p.get("n_sets"):
            L += [f"- Pooled ({title}): {p['n_sets']} sets, {p['records']} records, ratio {p['pooled_ratio']:.2f} "
                  f"(per-set range {p['min_ratio']:.2f} to {p['max_ratio']:.2f})."]
    L += [""]
    if res["file_sizes_missing"]:
        L += ["## Missing file sizes", "",
              "No config-level `meta.json` with `file_size_gb` for: "
              + ", ".join(f"{r['model']}/{r['config']}" for r in res["file_sizes_missing"]) + ". Nothing is estimated.", ""]
    if res["build_lines_in_log"]:
        L += ["Build lines in the sweep log (quantized 27B builds, not in any results meta.json): "
              + "; ".join(f"{b['job']} size_gb={b['size_gb']} secs={b['secs']}" for b in res["build_lines_in_log"]) + ".", ""]
    if res.get("latency") is not None:
        L += render_latency_md(res["latency"])
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=Path("analysis/inputs/sweep.log"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--private-results", type=Path, default=Path("private/results"))
    ap.add_argument("--records", type=Path, default=Path("data/records"))
    ap.add_argument("--private-records", type=Path, default=Path("private/records"))
    ap.add_argument("--latency", type=Path, default=Path("results-latency"),
                    help="isolated latency benchmark tree <model>/<config>.json; a missing directory is reported, not an error")
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args(argv)
    public_sets = sorted(p.stem for p in args.records.glob("*.jsonl"))
    private_sets = sorted(p.stem for p in args.private_records.glob("*.jsonl") if "hosted_repeat" not in p.stem) \
        if args.private_records.is_dir() else []
    res = build(args.log, args.results, args.private_results, public_sets, private_sets, args.latency)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "speed.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (args.out / "speed.md").write_text(render_md(res), encoding="utf-8")
    print(f"wrote {args.out}/speed.json and speed.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
