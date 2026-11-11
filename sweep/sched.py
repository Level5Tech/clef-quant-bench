"""Concurrent quality scheduler.

A QualityTask is one (model, config). Its work is cut into units (contiguous record ranges, `units.py`); workers (one
model copy each: a native container or a vLLM / llama.cpp server) claim units from the shared plan. The pool starts
workers in priority order while the unified-memory budget, the per-engine caps and the global slot count allow it, and
starts them one at a time (each must be ready before the next launches). Finished sets are merged in record order into
the final results file, verified (full count, valid JSON, ids in order), and the shards are removed.

Every quality job stays at batch size 1 (ref-r2: 8): concurrency changes how many units are in flight, not what one
record sees. The per-row latency_ms in the quality outputs is therefore measured under contention and is NOT the
latency benchmark (that runs alone, last).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from runners.common import build_meta, write_meta
from sweep import engines
from sweep import units as U
from sweep.configs import QUALITY_CONCURRENCY, QUALITY_MAX_NUM_SEQS, REPEAT_SUBSET_DIR, WORKER_CAPS, Config
from sweep.driver import Log, Paths, count_errors, count_lines, output_complete

READY_TIMEOUT_S = 2400
STATUS_EVERY_S = 300
MAX_TASK_FAILURES = 3


class QualityTask:
    def __init__(self, model: str, cfg: Config, public: list[str], private: list[str], paths: Paths, log: Log) -> None:
        self.model, self.cfg, self.paths, self.log = model, cfg, paths, log
        self.name = f"{model}/{cfg.name}"
        self.sets = {"public": public, "private": private}
        self.set_order = [(s, False) for s in public] + [(s, True) for s in private]
        self.scale = 3.0 if model == "clef" else 1.0
        self.units: list[U.Unit] = []
        self.plan: list[dict[str, Any]] = []
        self.done_sets: set[str] = set()
        self.failures = 0
        self.launches = 0
        self.failed = False
        self.workers_used = 0
        self.merged_here: set[bool] = set()   # output roots (private flag) that got a set merged by this process
        self.utils: set[float] = set()
        self.meta_written = False
        tag = f"{model}-{cfg.name}"
        self.dir = paths.state_dir / "tasks" / tag
        self.plan_file = self.dir / "plan.json"
        self.claim_dir = self.dir / "claims"
        util = engines.util_ladder(model, cfg, True)[0]
        self.worker_mem_gb = engines.worker_mem_gb(model, cfg, paths, util)
        self.cap = WORKER_CAPS[(cfg.engine, model)]

    # ---- paths
    def records_file(self, s: str, private: bool) -> Path:
        if self.cfg.on_subset and not private:   # repeats and subset configs run on the first N records of each public task
            return self.paths.subset_records_dir / f"{s}.jsonl"
        return (self.paths.private_records_dir if private else self.paths.records_dir) / f"{s}.jsonl"

    def final_file(self, s: str, private: bool) -> Path:
        return self.paths.out_dir(self.model, self.cfg.name, private) / f"{s}.jsonl"

    def work_root(self, private: bool) -> Path:
        return self.paths.private_work_root if private else self.paths.work_root

    def unit_out(self, u: U.Unit) -> Path:
        return U.unit_out(self.work_root(u.private), self.model, self.cfg.name, u)

    # ---- preparation
    def prepare(self) -> None:
        self.units, self.done_sets = [], set()
        for s, private in self.set_order:
            rec = self.records_file(s, private)
            if output_complete(self.final_file(s, private), rec):
                self.done_sets.add(s)
                continue
            n = count_lines(rec)
            chunk = U.chunk_size(s, self.scale, private)
            self.units += [U.Unit(s, a, b, private) for a, b in U.chunk_ranges(n, chunk)]
        spr = lambda u: (U.PRIVATE_SPR if u.private else U.SECONDS_PER_RECORD.get(u.set, U.DEFAULT_SPR)) * u.n  # noqa: E731
        ordered = sorted(self.units, key=lambda u: -spr(u))   # longest units first: shortest tail at the end
        self.plan = [{"id": u.id, "records": str(self.records_file(u.set, u.private)), "start": u.start, "end": u.end,
                      "out": str(self.unit_out(u))} for u in ordered]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.plan_file.write_text(json.dumps(self.plan))
        U.clear_claims(self.claim_dir)   # no worker of ours is alive at stage start
        for private in (False, True):
            self.work_root(private).mkdir(parents=True, exist_ok=True)

    # ---- state
    def pending_units(self) -> list[dict[str, Any]]:
        return [u for u in self.plan if not U.unit_complete(Path(u["out"]), u["end"] - u["start"])]

    def unclaimed(self) -> int:
        claimed = U.claimed_ids(self.claim_dir)
        return sum(1 for u in self.pending_units() if u["id"] not in claimed)

    def finished(self) -> bool:
        return self.failed or all(s in self.done_sets for s, _ in self.set_order)

    def complete_ids(self) -> set[str]:
        return {u["id"] for u in self.plan if U.unit_complete(Path(u["out"]), u["end"] - u["start"])}

    # ---- finalize: merge finished sets, write sidecars and meta.json
    def finalize(self) -> None:
        if self.failed:
            return
        for s, private in self.set_order:
            if s in self.done_sets:
                continue
            mine = [u for u in self.units if u.set == s]
            if not mine or not all(U.unit_complete(self.unit_out(u), u.n) for u in mine):
                continue
            try:
                self._merge_set(s, private, mine)
            except (ValueError, OSError) as exc:
                self.failed = True
                self.log("FAIL", job=f"{self.name}/{s}", exit=1, error=str(exc).replace("\n", " ")[:300])
                return
            self.done_sets.add(s)
            self.merged_here.add(private)
        if not self.meta_written and all(s in self.done_sets for s, _ in self.set_order):
            self._write_config_meta()
            self.meta_written = True

    def _merge_set(self, s: str, private: bool, mine: list[U.Unit]) -> None:
        rec, final = self.records_file(s, private), self.final_file(s, private)
        outs = [self.unit_out(u) for u in mine]
        n = U.merge_units(outs, final, rec)
        if not output_complete(final, rec):
            raise ValueError(f"merged {s} failed verification")
        metas = []
        for o in outs:
            native = Path(str(o) + ".meta.json")
            metas.append(json.loads(native.read_text()) if native.is_file()
                         else json.loads((o.parent / "meta.json").read_text()) if (o.parent / "meta.json").is_file() else {})
        wall = sum(float(m.get("wall_seconds") or (m.get("runs") or [{}])[0].get("wall_seconds") or 0) for m in metas)
        started = [m["started_at"] for m in metas if "started_at" in m]
        ended = [m["ended_at"] for m in metas if "ended_at" in m]
        peaks = [m["peak_gpu_mem_gib"] * 2**30 / 1e9 for m in metas if m.get("peak_gpu_mem_gib")] + \
                [m["peak_memory_gb"] for m in metas if m.get("peak_memory_gb")]
        first = metas[0] if metas else {}
        errors = count_errors(final)
        side = {
            "records": n, "errors": errors, "units": len(mine), "unit_wall_seconds_sum": round(wall, 2),
            "elapsed_seconds": round(max(ended) - min(started), 2) if started and ended else None,
            "peak_memory_gb": round(max(peaks), 3) if peaks else None,
            "engine": first.get("engine"), "engine_version": first.get("engine_version")
            or (f"transformers {first['transformers']}" if first.get("transformers") else None),
            "batch_size": self.cfg.batch_size, "concurrent_workers_hint": self.workers_used,
            "gpu_memory_utilization": (first.get("server_details") or {}).get("gpu_memory_utilization"),
        }
        Path(str(final) + ".meta.json").write_text(json.dumps(side, indent=2))
        if private:
            engines.hardened(self.paths.private_results_root)
        shutil.rmtree(self.work_root(private) / self.model / self.cfg.name / s, ignore_errors=True)
        self.log("END", job=f"{self.name}/{s}", exit=0, records=n, expected=n, errors=errors,
                 unit_secs=round(wall, 1), elapsed=side["elapsed_seconds"], units=len(mine), engine=self.cfg.engine)

    def _write_config_meta(self) -> None:
        runs: list[tuple[bool, dict[str, Any]]] = []
        peaks, version, engine = [], None, None
        for s, private in self.set_order:
            f = Path(str(self.final_file(s, private)) + ".meta.json")
            if not f.is_file():
                continue
            m = json.loads(f.read_text())
            engine = engine or m.get("engine")
            version = version or m.get("engine_version") or (f"transformers {m['transformers']}" if m.get("transformers") else None)
            peak = m.get("peak_memory_gb") or (m["peak_gpu_mem_gib"] * 2**30 / 1e9 if m.get("peak_gpu_mem_gib") else None)
            if peak:
                peaks.append(peak)
            runs.append((private, {"file": f"{s}.jsonl", "records": m.get("records"), "errors": m.get("errors"),
                                   "unit_wall_seconds_sum": m.get("unit_wall_seconds_sum") or m.get("wall_seconds"),
                                   "elapsed_seconds": m.get("elapsed_seconds")}))
        for private in (False, True):
            sel = [r for p, r in runs if p == private]
            if not sel:
                continue
            out_dir = self.paths.out_dir(self.model, self.cfg.name, private)
            old = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").is_file() else None
            if old is not None and not self.workers_used and private not in self.merged_here:
                # Nothing ran here: a relaunch over a finished config. Rebuilding would overwrite the recorded worker
                # count with this process's 0, which is how the 2026-10-18 and 2026-11-03 relaunches damaged meta.json.
                continue
            old = old or {}
            workers = max(self.workers_used, old.get("concurrent_workers_max") or 0) or None  # None: no worker seen
            utils = sorted(self.utils | set(old.get("gpu_memory_utilization") or [])) or None
            meta = build_meta(engine or self.cfg.engine, version or "unknown",
                              engines.config_file_size_gb(self.model, self.cfg, self.paths),
                              max(peaks) if peaks else None, extra={
                                  "model": self.model, "config": self.cfg.name, "batch_size": self.cfg.batch_size,
                                  "runs": sel, "concurrent_workers_max": workers,
                                  "gpu_memory_utilization": utils,
                                  "max_num_seqs": QUALITY_MAX_NUM_SEQS if self.cfg.engine == "vllm" else None,
                                  "client_concurrency": QUALITY_CONCURRENCY if self.cfg.engine == "vllm" else 1,
                                  "gdn_kernel": self.cfg.gdn_kernel if self.cfg.engine == "native" else None,
                                  "subset": self.cfg.on_subset,
                                  "repeat_of": self.cfg.repeat_of,
                                  "repeat_note": (f"another server instance ({self.cfg.name}) of {self.cfg.repeat_of}, same flags, first "
                                                  "records of each public task: defines the cross-restart noise floor"
                                                  if self.cfg.repeat_of else None),
                                  "latency_note": "latency_ms in the result rows was measured with other workers running "
                                                  "concurrently: use results-latency/ for latency, not these rows",
                                  "memory_note": "peak = per-worker process GPU memory (torch allocator / vLLM plugin weights"
                                                 " + activations / nvidia-smi for llama.cpp); unaffected by other workers",
                              })
            write_meta(out_dir, meta)
        if any(p for _, p in self.set_order):
            engines.hardened(self.paths.private_results_root)
        self.log("CONFIG_DONE", job=self.name, sets=len(self.set_order), workers_max=self.workers_used)


def make_worker(task: QualityTask, wid: int, paths: Paths, log: Log):
    if task.cfg.engine == "native":
        return engines.NativeWorker(task, wid, paths, log)
    return engines.HttpWorker(task, wid, paths, log)


def run_stage(tasks: list[QualityTask], slots: int, paths: Paths, log: Log, poll: float = 5.0,
              budget_gb: float | None = None) -> dict[str, str]:
    """Run quality tasks to completion (or STOP). Returns {task name: ok|failed|stopped}."""
    stop_file = paths.state_dir / "STOP"
    for t in tasks:
        t.prepare()
    budget = budget_gb if budget_gb is not None else U.mem_available_gb() - U.HEADROOM_GB
    log("STAGE_START", tasks=len(tasks), slots=slots, budget_gb=round(budget, 1),
        pending_units=sum(len(t.pending_units()) for t in tasks))
    running: list[Any] = []
    by_name = {t.name: t for t in tasks}
    last_launch: Any = None
    last_launch_t = 0.0
    last_status = time.time()
    wid_next: dict[str, int] = {}
    try:
        while True:
            for w in list(running):
                if w.alive():
                    continue
                running.remove(w)
                t = by_name[w.task]
                engines.unregister_active(paths, w.owner)
                U.release_owned(t.claim_dir, w.owner, keep=t.complete_ids())
                if getattr(w, "util", None):
                    t.utils.add(w.util)
                err = getattr(w, "error", None)
                if (w.returncode or 0) != 0 or err:
                    t.failures += 1
                    log("FAIL", worker=w.owner, exit=w.returncode, error=err or "nonzero exit", failures=t.failures)
                    if t.failures >= MAX_TASK_FAILURES:
                        t.failed = True
                        log("FAIL", job=t.name, exit=1, error="too many worker failures; config abandoned")
                else:
                    log("WORKER_END", worker=w.owner)
            for t in tasks:
                t.finalize()
            live = [t for t in tasks if not t.finished()]
            if not live and not running:
                break
            if stop_file.exists():
                if not running:
                    log("STAGE_STOP", reason="STOP file")
                    return {t.name: "ok" if t.finished() and not t.failed else "stopped" for t in tasks}
                time.sleep(poll)
                continue
            gate_open = last_launch is None or last_launch.ready or not last_launch.alive() \
                or time.time() - last_launch_t > READY_TIMEOUT_S
            if gate_open:
                states = [U.TaskState(t.name, t.cap, t.unclaimed(), t.worker_mem_gb) for t in live if not t.failed]
                pick = U.next_launch(states, running, budget, slots)
                if pick is not None and U.mem_available_gb() >= pick.mem_gb + U.HEADROOM_GB:
                    t = by_name[pick.name]
                    wid = wid_next.get(t.name, 0)
                    wid_next[t.name] = wid + 1
                    t.launches += 1
                    w = make_worker(t, wid, paths, log)
                    w.launch()
                    running.append(w)
                    last_launch, last_launch_t = w, time.time()
                    t.workers_used = max(t.workers_used, sum(1 for x in running if x.task == t.name))
                    log("WORKER_START", worker=w.owner, task=t.name, running=len(running),
                        mem_gb=round(w.mem_gb, 1), avail_gb=round(U.mem_available_gb(), 1))
            if time.time() - last_status > STATUS_EVERY_S:
                last_status = time.time()
                snap = engines.gpu_snapshot()
                log("STATUS", running=",".join(w.owner for w in running), avail_gb=round(U.mem_available_gb(), 1),
                    units_left=sum(len(t.pending_units()) for t in tasks), **snap)
            time.sleep(poll)
    finally:
        for w in running:
            w.kill()
    return {t.name: ("failed" if t.failed else "ok") for t in tasks}
