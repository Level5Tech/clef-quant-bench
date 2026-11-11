"""Progress and projected finish of the sweep. Runs anywhere that can read the Spark's ~/clef-quant (use ssh).

    ssh spark 'cd ~/clef-quant/bench && python3 -m sweep.status --jobs sweep/jobs.jsonl'

Counts finished work from the OUTPUT FILES (same rule as the driver: full record count and valid JSON), shows the last
log lines, and projects the finish time per stage (quality flash, 27B builds, quality 27B, isolated latency phase).

Projection method (an estimate, not a promise):
* seconds/record per (model, engine family, set) from finished sets: END lines of sweep.log (unit_secs or secs / records),
  final-dir sidecars, and optional --seed-log files (e.g. the dry run). Missing entries borrow, in order: the same model
  and set on native scaled by the median measured server/native ratio; the other model scaled by 3.0 for the 27B
  (assumption: parameter ratio); private sets without a measurement use r4b's rate.
* quality wall time = sum(pending records x seconds/record) / SPEEDUP (default 1.0: measured concurrency gain, see README)
  + one model load / server start per config.
* latency phase = per config 20 warm-up + min(300, n) records per set at that config's rates + a server start.
* 27B quant builds use assumed times (fp8 20 min, nvfp4 60 min).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from sweep import jobs as jobs_mod
from sweep.configs import CONFIG_BY_NAME, REPEAT_SUBSET_DIR
from sweep.driver import Paths, count_lines, output_complete

SCALE_27B = 3.0
LOAD_S = {("clef-flash", "native"): 120, ("clef", "native"): 300, ("clef-flash", "vllm"): 180,
          ("clef", "vllm"): 600, ("clef-flash", "llamacpp"): 60, ("clef", "llamacpp"): 200}
BUILD_S = {"fp8": 1200, "nvfp4": 3600}


def family(config: str) -> str:
    c = CONFIG_BY_NAME[config]
    return "native8" if (c.engine == "native" and c.batch_size > 1) else ("native1" if c.engine == "native" else c.engine)


def parse_log(path: Path) -> list[dict[str, str]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        m = re.match(r"(\S+) (\S+) ?(.*)", line)
        if not m:
            continue
        kv = dict(re.findall(r"(\w+)=(\S*)", m.group(3)))
        rows.append({"ts": m.group(1), "event": m.group(2), **kv})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", default="~/clef-quant")
    ap.add_argument("--log")
    ap.add_argument("--tail", type=int, default=8)
    ap.add_argument("--speedup", type=float, default=1.0, help="measured throughput gain of concurrent workers (1.0 = none)")
    ap.add_argument("--seed-log", action="append", default=[])
    a = ap.parse_args(argv)
    p = Paths.default(a.root)
    log_path = Path(a.log).expanduser() if a.log else p.logs_dir / "sweep.log"
    rows = parse_log(log_path)

    rate: dict[tuple[str, str, str], float] = {}
    for src in [Path(x).expanduser() for x in a.seed_log] + [log_path]:
        for r in parse_log(src):
            if r["event"] != "END" or r.get("job", "").count("/") != 2 or not r.get("records"):
                continue
            model, config, s = r["job"].split("/")
            secs = float(r.get("unit_secs") or r.get("secs") or 0)
            if config in CONFIG_BY_NAME and s not in ("latency", "server") and secs > 0:
                rate[(model, family(config), s)] = secs / max(1, int(r["records"]))
    for side in list(p.results_root.glob("*/*/*.jsonl.meta.json")) + list(p.private_results_root.glob("*/*/*.jsonl.meta.json")):
        try:
            m = json.loads(side.read_text())
            secs = m.get("unit_wall_seconds_sum") or m.get("wall_seconds")
            model, config = side.parent.parent.name, side.parent.name
            if m.get("records") and secs and config in CONFIG_BY_NAME:
                rate[(model, family(config), side.name.split(".")[0])] = secs / m["records"]
        except (OSError, ValueError, KeyError):
            continue

    ratios: dict[str, float] = {}
    for fam_x in ("vllm", "llamacpp"):
        rs = sorted(rate[(m, fam_x, s)] / rate[(m, "native1", s)] for (m, f, s) in list(rate)
                    if f == fam_x and (m, "native1", s) in rate)
        if rs:
            ratios[fam_x] = rs[len(rs) // 2]

    def est0(model: str, config: str, s: str) -> float | None:
        fam = family(config)
        if (model, fam, s) in rate:
            return rate[(model, fam, s)]
        if fam in ratios and (model, "native1", s) in rate:
            return rate[(model, "native1", s)] * ratios[fam]
        other = "clef" if model == "clef-flash" else "clef-flash"
        if fam in ratios and (other, "native1", s) in rate:
            return rate[(other, "native1", s)] * ratios[fam] * (SCALE_27B if model == "clef" else 1 / SCALE_27B)
        for f2 in ("native1", "vllm", "llamacpp", "native8"):
            if (model, f2, s) in rate:
                return rate[(model, f2, s)]
        for f2 in (fam, "native1", "vllm", "llamacpp", "native8"):
            if (other, f2, s) in rate:
                return rate[(other, f2, s)] * (SCALE_27B if model == "clef" else 1 / SCALE_27B)
        return None

    def est(model: str, config: str, s: str) -> float | None:
        r = est0(model, config, s)
        if r is None and s in ("r1", "r2", "r4a"):
            r = est0(model, config, "r4b")
        return r

    st = jobs_mod.stages(jobs_mod.read_jobs(a.jobs))
    unknown = 0
    stage_s = {"quality clef-flash": 0.0, "builds": 0.0, "quality clef": 0.0, "latency": 0.0}
    done_sets = total_sets = 0
    for model, cfgs in st["quality"].items():
        key = f"quality {model}"
        for config, public, private in cfgs:
            cfg = CONFIG_BY_NAME[config]
            pend = 0.0
            any_pending = False
            for s, priv in [(x, False) for x in public] + [(x, True) for x in private]:
                total_sets += 1
                rec = ((p.root / "bench" / REPEAT_SUBSET_DIR / "records") if (cfg.on_subset and not priv)
                       else (p.private_records_dir if priv else p.records_dir)) / f"{s}.jsonl"
                out = p.out_dir(model, config, priv) / f"{s}.jsonl"
                if output_complete(out, rec):
                    done_sets += 1
                    continue
                any_pending = True
                r = est(model, config, s)
                if r is None:
                    unknown += 1
                elif rec.is_file():
                    pend += r * count_lines(rec)
            if any_pending:
                stage_s[key] += pend / a.speedup + LOAD_S[(model, cfg.engine)]
    for b in st["builds"]:
        if not (p.root / f"models/{b['model']}-{b['fmt']}/.sweep_build_ok").is_file():
            stage_s["builds"] += BUILD_S[b["fmt"]]
    lat_done = lat_total = 0
    for model, config, public, private in st["latency"]:
        lat_total += 1
        cfg = CONFIG_BY_NAME[config]
        if p.latency_file(model, config).is_file():
            lat_done += 1
            continue
        if cfg.alias_latency_of:
            continue
        cost = 0.0
        for s, priv in [(x, False) for x in public] + [(x, True) for x in private]:
            rec = (p.private_records_dir if priv else p.records_dir) / f"{s}.jsonl"
            r = est(model, config, s)
            if r is not None and rec.is_file():
                cost += r * min(300, count_lines(rec))
        stage_s["latency"] += cost + LOAD_S[(model, cfg.engine)]

    print(f"sets done {done_sets}/{total_sets}   latency files {lat_done}/{lat_total}")
    active = None
    for r in rows:
        if r["event"] in ("WORKER_START", "SESSION_START"):
            active = r
        elif r["event"] in ("STAGE_START",):
            active = r
    if active:
        print(f"last launch: {active['event']} {active.get('worker') or active.get('session') or ''} at {active['ts']}")
    print("last log lines:")
    for line in (log_path.read_text(errors="ignore").splitlines()[-a.tail:] if (log_path.is_file() and a.tail) else []):
        print("  " + line)
    now = dt.datetime.now().astimezone()
    total = 0.0
    for k, v in stage_s.items():
        total += v
        print(f"  {k:<20} remaining ~ {v / 3600:6.1f} h")
    fin = now + dt.timedelta(seconds=total)
    print(f"estimated remaining {total / 3600:.1f} h -> finish ~ {fin:%Y-%m-%d %H:%M %Z}"
          f"  (speedup assumed {a.speedup}; {unknown} pending sets had no measurement and are not counted)")


if __name__ == "__main__":
    main()
