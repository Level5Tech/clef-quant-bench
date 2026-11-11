"""Resumable sequential job queue for the Clef quantization sweep (runs ON the Spark, under nohup).

    cd ~/clef-quant/bench && nohup setsid python3 -m sweep.driver run --jobs sweep/jobs.jsonl \
        >> ~/clef-quant/logs/sweep-driver.out 2>&1 &

Stages: (1) Clef-flash quality, all configs, concurrent workers; (2) Clef 27B quant builds, one at a time with
nothing else on the GPU; (3) Clef 27B quality; (4) LATENCY for every config, one session at a time on an isolated GPU.
Quality jobs are batch size 1 (ref-r2: 8); concurrency is units (record ranges) in flight, see sched.py/units.py.
A set is complete when its output has exactly the records file's line count and every line parses as JSON. Restart
the driver at any time: finished sets are skipped; interrupted units are re-run.

Control: touch <state>/STOP to stop after the current session (the driver removes it when it exits).
Log: ~/clef-quant/logs/sweep.log (key=value lines; START / END / FAIL / SKIP per job and session).
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sweep import jobs as jobs_mod
from sweep.configs import CONFIG_BY_NAME, REPEAT_SUBSET_DIR


class SessionError(RuntimeError):
    """A session step failed in a way that should be logged and (maybe) retried."""


@dataclass
class Paths:
    root: Path
    records_dir: Path
    private_records_dir: Path
    results_root: Path
    private_results_root: Path
    latency_root: Path
    logs_dir: Path
    state_dir: Path
    work_root: Path | None = None            # shard outputs of public sets (merged into results_root, then removed)
    private_work_root: Path | None = None    # shard outputs of private sets: under the 700 private root only
    pylibs_fla: Path | None = None           # flash-linear-attention install; mounted into fla-kernel native workers only

    def __post_init__(self) -> None:
        if self.work_root is None:
            self.work_root = self.root / "sweep-work"
        if self.private_work_root is None:
            self.private_work_root = self.root / "private/sweep-work"
        if self.pylibs_fla is None:
            self.pylibs_fla = self.root / "pylibs-fla"

    @classmethod
    def default(cls, root: str | Path = "~/clef-quant") -> "Paths":
        r = Path(root).expanduser()
        return cls(root=r, records_dir=r / "bench/data/records", private_records_dir=r / "private/records",
                   results_root=r / "results", private_results_root=r / "private/results",
                   latency_root=r / "results-latency", logs_dir=r / "logs", state_dir=r / "sweep-state")

    @property
    def subset_records_dir(self) -> Path:
        """First REPEAT_SUBSET_N public records per set (repeats and subset configs). Native workers must mount it."""
        return self.root / "bench" / REPEAT_SUBSET_DIR / "records"

    def records_file(self, job: dict[str, Any]) -> Path:
        base = self.private_records_dir if job["private"] else self.records_dir
        return base / f"{job['set']}.jsonl"

    def out_dir(self, model: str, config: str, private: bool) -> Path:
        return (self.private_results_root if private else self.results_root) / model / config

    def out_file(self, job: dict[str, Any]) -> Path:
        return self.out_dir(job["model"], job["config"], job["private"]) / f"{job['set']}.jsonl"

    def latency_file(self, model: str, config: str) -> Path:
        return self.latency_root / model / f"{config}.json"


class Log:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, **kv: Any) -> None:
        ts = dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
        fields = " ".join(f"{k}={v}" for k, v in kv.items())
        line = f"{ts} {event} {fields}".rstrip()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)


def count_lines(path: str | Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def valid_jsonl_lines(path: Path) -> int | None:
    """Number of non-blank lines if every one parses as JSON, else None (truncated or corrupt file)."""
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


def output_complete(out: Path, records: Path) -> bool:
    """Full record count AND valid JSON on every line (a killed run can leave either problem)."""
    return out.is_file() and valid_jsonl_lines(out) == count_lines(records)


def count_errors(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and json.loads(line).get("error") is not None:
                n += 1
    return n


def merge_meta(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """meta.json is rewritten by each runner invocation; keep earlier runs and the larger peak memory."""
    merged = {**old, **{k: v for k, v in new.items() if k != "runs"}}
    peaks = [p for p in (old.get("peak_memory_gb"), new.get("peak_memory_gb")) if p is not None]
    if peaks:
        merged["peak_memory_gb"] = max(peaks)
    key = lambda r: r.get("file") or r.get("records_file")  # noqa: E731
    runs = {key(r): r for r in old.get("runs", [])}
    runs.update({key(r): r for r in new.get("runs", [])})
    merged["runs"] = list(runs.values())
    return merged


class Runner(Protocol):
    def start(self) -> dict[str, Any]: ...
    def quality(self, items: list[dict[str, Any]]) -> None: ...
    def latency(self) -> None: ...
    def stop(self) -> None: ...


def _write_alias_latency(paths: Paths, model: str, config: str, base: str, log: Log) -> bool:
    src = paths.latency_file(model, base)
    if not src.is_file():
        log("SKIP", job=f"{model}/{config}/latency", reason=f"alias base {base} latency missing")
        return False
    data = json.loads(src.read_text())
    data.update(config=config, alias_of=base,
                alias_note=(f"{config} is the same engine path as {base} at batch size 1; the latency benchmark "
                            f"is batch size 1, so this file is a copy of {base}'s measurement, not a second run"))
    dst = paths.latency_file(model, config)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2))
    log("END", job=f"{model}/{config}/latency", exit=0, secs=0, note=f"alias_of={base}")
    return True


def run_session(session: dict[str, Any], runner: Runner, paths: Paths, log: Log, retries: int = 0) -> str:
    """Returns 'ok' | 'skipped' | 'failed'. The runner's stop() runs on every exit path."""
    model, config = session["model"], session["config"]
    cfg = CONFIG_BY_NAME.get(config)
    alias = cfg.alias_latency_of if cfg else None
    result = "ok"
    for attempt in range(retries + 1):
        pend = [j for j in session["jobs"] if j["kind"] == "quality"
                and not output_complete(paths.out_file(j), paths.records_file(j))]
        has_lat = any(j["kind"] == "latency" for j in session["jobs"])
        lat_missing = has_lat and not paths.latency_file(model, config).is_file()
        need_lat = lat_missing and not alias
        if not pend and not need_lat:
            if lat_missing and alias:
                _write_alias_latency(paths, model, config, alias, log)
                return "ok"
            log("SKIP", session=f"{model}/{config}", reason="all outputs complete")
            return "skipped"
        t0 = time.time()
        log("SESSION_START", session=f"{model}/{config}", attempt=attempt + 1,
            pending_sets=len(pend), latency=need_lat)
        try:
            info = runner.start()
            log("SERVER_UP", session=f"{model}/{config}", **{k: v for k, v in (info or {}).items()})
            for private in (False, True):
                group = [j for j in pend if j["private"] == private]
                if group:
                    try:
                        runner.quality(group)
                    except SessionError as exc:
                        log("FAIL", session=f"{model}/{config}", sets=",".join(j["set"] for j in group),
                            error=str(exc).replace("\n", " ")[:300])
                        raise
            if need_lat:
                runner.latency()
            if lat_missing and alias:
                _write_alias_latency(paths, model, config, alias, log)
            result = "ok"
        except SessionError as exc:
            log("FAIL", session=f"{model}/{config}", attempt=attempt + 1, error=str(exc).replace("\n", " ")[:300])
            result = "failed"
        finally:
            try:
                runner.stop()
            except Exception as exc:  # noqa: BLE001 - a stop failure must be loud but must not mask the result
                log("STOP_ERROR", session=f"{model}/{config}", error=repr(exc)[:300])
                result = "failed"
            log("SESSION_END", session=f"{model}/{config}", attempt=attempt + 1, result=result,
                secs=round(time.time() - t0, 1))
        if result == "ok":
            return "ok"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--jobs", required=True)
    run.add_argument("--root", default="~/clef-quant")
    run.add_argument("--records-dir")
    run.add_argument("--private-records-dir")
    run.add_argument("--results-root")
    run.add_argument("--private-results-root")
    run.add_argument("--latency-root")
    run.add_argument("--work-root")
    run.add_argument("--private-work-root")
    run.add_argument("--state-dir")
    run.add_argument("--log", default=None, help="sweep.log path (default <root>/logs/sweep.log)")
    run.add_argument("--retries", type=int, default=1)
    run.add_argument("--slots", type=int, default=None, help="override configs.GLOBAL_SLOTS")
    run.add_argument("--no-gpu-gate", action="store_true", help="do not wait for foreign clefbench-* GPU jobs")
    run.add_argument("--only-stage", choices=["quality-flash", "builds", "quality-clef", "latency"], default=None)
    a = ap.parse_args(argv)

    os.umask(0o077)  # new logs/results are owner-only; private sets must never be group/world readable
    p = Paths.default(a.root)
    for field, val in (("records_dir", a.records_dir), ("private_records_dir", a.private_records_dir),
                       ("results_root", a.results_root), ("private_results_root", a.private_results_root),
                       ("latency_root", a.latency_root), ("state_dir", a.state_dir), ("work_root", a.work_root),
                       ("private_work_root", a.private_work_root)):
        if val:
            setattr(p, field, Path(val).expanduser())
    for d in (p.state_dir, p.logs_dir / "sweep"):
        d.mkdir(parents=True, exist_ok=True)
    os.chmod(p.logs_dir / "sweep", 0o700)  # server/runner logs of private sets live here
    log = Log(a.log or p.logs_dir / "sweep.log")

    lock = open(p.state_dir / "driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another sweep driver holds the lock; exiting", file=sys.stderr)
        return 3

    from sweep import configs, engines, sched  # local imports: they import this module

    def _term(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _term)
    st = jobs_mod.stages(jobs_mod.read_jobs(a.jobs))
    slots = a.slots or configs.GLOBAL_SLOTS
    log("DRIVER_START", jobs=a.jobs, pid=os.getpid(), slots=slots, mode="concurrent")
    engines.cleanup_stale(p, log)
    stop_file = p.state_dir / "STOP"
    jobs_mod.ensure_subset(p.records_dir, p.subset_records_dir, configs.REPEAT_SUBSET_N)
    summary = {"ok": 0, "skipped": 0, "failed": 0}
    failed_builds: set[tuple[str, str]] = set()

    def quality_stage(model: str) -> None:
        tasks = []
        for config, public, private in st["quality"].get(model, []):
            cfg = CONFIG_BY_NAME[config]
            if cfg.engine in ("vllm", "llamacpp") and (model, cfg.fmt) in failed_builds:
                log("SKIP", session=f"{model}/{config}", reason="quant build failed")
                summary["failed"] += 1
                continue
            item = {"model": model, "sets": {"public": public, "private": private}}
            if engines.make_runner(item, cfg, p, log) is None and cfg.engine != "native":
                log("SKIP", session=f"{model}/{config}", reason="model files missing")
                summary["failed"] += 1
                continue
            tasks.append(sched.QualityTask(model, cfg, public, private, p, log))
        if not tasks:
            return
        if not a.no_gpu_gate:
            engines.wait_gpu_idle(p, log)
        res = sched.run_stage(tasks, slots, p, log)
        for name, r in res.items():
            summary["ok" if r == "ok" else "failed"] += 1

    try:
        want = lambda s: a.only_stage in (None, s)  # noqa: E731
        if want("quality-flash") and not stop_file.exists():
            quality_stage("clef-flash")
        if want("builds") and not stop_file.exists():
            for item in st["builds"]:
                if stop_file.exists():
                    break
                if not a.no_gpu_gate:
                    engines.wait_gpu_idle(p, log)
                res = engines.run_build({"type": "build", "id": item["id"], "model": item["model"], "fmt": item["fmt"]}, p, log)
                if res == "failed":
                    failed_builds.add((item["model"], item["fmt"]))
                summary[res] += 1
        if want("quality-clef") and not stop_file.exists():
            for item in st["builds"]:  # a build that finished in an earlier run counts as present
                if not (p.root / f"models/{item['model']}-{item['fmt']}/.sweep_build_ok").is_file():
                    failed_builds.add((item["model"], item["fmt"]))
            quality_stage("clef")
        if want("latency") and not stop_file.exists():
            for model, config, public, private in st["latency"]:
                if stop_file.exists():
                    break
                cfg = CONFIG_BY_NAME[config]
                if cfg.engine in ("vllm", "llamacpp") and (model, cfg.fmt) in failed_builds | {
                        (b["model"], b["fmt"]) for b in st["builds"]
                        if not (p.root / f"models/{b['model']}-{b['fmt']}/.sweep_build_ok").is_file()}:
                    log("SKIP", session=f"{model}/{config}", reason="quant build missing")
                    summary["failed"] += 1
                    continue
                item = {"type": "session", "model": model, "config": config, "sets": {"public": public, "private": private},
                        "jobs": [{"id": f"{model}/{config}/latency", "kind": "latency", "model": model, "config": config}]}
                if not a.no_gpu_gate:
                    engines.wait_gpu_idle(p, log)
                runner = engines.make_runner(item, cfg, p, log)
                if runner is None:
                    log("SKIP", session=f"{model}/{config}", reason="model files missing")
                    summary["failed"] += 1
                    continue
                summary[run_session(item, runner, p, log, a.retries)] += 1
        if stop_file.exists():
            log("DRIVER_STOP", reason="STOP file")
            stop_file.unlink()
    finally:
        log("DRIVER_END", **summary)
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
