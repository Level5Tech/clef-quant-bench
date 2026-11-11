"""Determinism proofs and the concurrency benchmark. Runs ON the Spark, GPU otherwise idle. Public data only.

    python3 -m sweep.proof determinism [--out DIR]    # concurrent == serial, bit for bit, per engine
    python3 -m sweep.proof bench --engine native|vllm|llamacpp [--ns 1 2 3 4] [--out DIR]

determinism:
  native   re-run results/clef-flash/ref/anli_r1.jsonl (1000 records) in a worker while 2 other native workers are busy;
           require max |delta logit| == 0.0 against the existing file (written by the serial, single-worker run).
  vllm     fp8: serial run (README flags: --max-num-seqs 16, util 0.25, server alone) vs a run on a quality-worker server
           (--max-num-seqs 1, lean util) while 2 native workers are busy; same 200 records.
  llamacpp q8_0: serial (server alone) vs a second server while 2 native workers are busy; same 200 records.
bench: for each N, N workers of one engine run N copies of a fixed mixed workload at once (start barrier); reports
  records/s, wall time, speedup = N * T(1) / T(N), GPU power / SM clock / utilisation (nvidia-smi, 0.5 s) and memory.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sweep import units as U
from sweep.configs import CONFIG_BY_NAME
from sweep.driver import Log, Paths

PROOF_SET, PROOF_N = "anli_r1", 1000
HTTP_SET, HTTP_N = "when2call", 200
MIX = [("clinc150", 24), ("clinc150", 24), ("banking77", 40), ("banking77", 40), ("ragtruth", 60), ("ragtruth", 60),
       ("when2call", 80), ("when2call", 80), ("anli_r1", 100), ("anli_r1", 100)]   # per copy: ~300 s serial on native
SHORT_MIX = [("anli_r1", 100), ("anli_r2", 100), ("anli_r3", 100), ("arc_challenge", 100), ("boolq", 100),
             ("contractnli", 100), ("mmlu", 100), ("sst5", 100), ("anli_r1", 100), ("mmlu", 100)]   # short inputs only
MIXES = {"full": MIX, "short": SHORT_MIX}
SHORT_MIX = [("anli_r1", 100), ("anli_r2", 100), ("anli_r3", 100), ("arc_challenge", 100), ("boolq", 100),
             ("contractnli", 100), ("mmlu", 100), ("sst5", 100), ("anli_r1", 100), ("mmlu", 100)]   # input-short sets
MIXES = {"full": MIX, "short": SHORT_MIX}


# ------------------------------------------------------------------------------------------ comparison
def compare_exact(a: Path, b: Path) -> dict[str, Any]:
    """Row-by-row exact comparison of two runner outputs (ids, option ids, logits, probs, token counts)."""
    res = {"rows": 0, "id_mismatch": 0, "errors_a": 0, "errors_b": 0, "option_id_mismatch": 0, "token_mismatch": 0,
           "max_abs_dlogit": 0.0, "max_abs_dprob": 0.0, "logits_compared": 0, "rows_with_any_diff": 0,
           "logits_available": True}
    with open(a, encoding="utf-8") as fa, open(b, encoding="utf-8") as fb:
        for la, lb in itertools.zip_longest(fa, fb):
            if la is None or lb is None:
                res["id_mismatch"] += 1
                continue
            ra, rb = json.loads(la), json.loads(lb)
            res["rows"] += 1
            res["id_mismatch"] += ra["id"] != rb["id"]
            res["errors_a"] += ra["error"] is not None
            res["errors_b"] += rb["error"] is not None
            res["token_mismatch"] += ra["input_tokens"] != rb["input_tokens"]
            diff = False
            for qid, x in ra["answers"].items():
                y = rb["answers"].get(qid)
                if y is None or x["option_ids"] != y["option_ids"]:
                    res["option_id_mismatch"] += 1
                    diff = True
                    continue
                dp = max(abs(p - q) for p, q in zip(x["probs"], y["probs"]))
                res["max_abs_dprob"] = max(res["max_abs_dprob"], dp)
                diff |= dp != 0.0
                if x["logits"] is not None and y["logits"] is not None:
                    dl = max(abs(p - q) for p, q in zip(x["logits"], y["logits"]))
                    res["max_abs_dlogit"] = max(res["max_abs_dlogit"], dl)
                    res["logits_compared"] += len(x["logits"])
                    diff |= dl != 0.0
                else:
                    res["logits_available"] = False
            res["rows_with_any_diff"] += diff
    res["identical"] = (res["id_mismatch"] == 0 and res["option_id_mismatch"] == 0 and res["rows_with_any_diff"] == 0
                        and res["max_abs_dprob"] == 0.0 and res["max_abs_dlogit"] == 0.0 and res["rows"] > 0)
    return res


# ------------------------------------------------------------------------------------------ plumbing
def mk_paths(tag: str) -> Paths:
    p = Paths.default()
    base = p.root / "sweep-proof" / tag
    p.state_dir, p.work_root, p.private_work_root = base / "state", base / "work", base / "work"
    for d in (p.state_dir, p.work_root, p.logs_dir / "sweep"):
        d.mkdir(parents=True, exist_ok=True)
    return p


def rec_file(p: Paths, s: str) -> str:
    return str(p.records_dir / f"{s}.jsonl")


def native_task(p: Paths, name: str, plan: list[dict[str, Any]]) -> SimpleNamespace:
    d = p.state_dir / "tasks" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps(plan))
    U.clear_claims(d / "claims")
    return SimpleNamespace(name=name, model="clef-flash", cfg=CONFIG_BY_NAME["ref"], plan_file=d / "plan.json",
                           claim_dir=d / "claims", worker_mem_gb=21.6, sets={"public": [], "private": []})


def filler_plan(p: Paths, tag: str) -> list[dict[str, Any]]:
    """Endless-enough busy work: consecutive clinc150 ranges. Output is throwaway."""
    out = []
    for i in range(80):
        a, b = i * 8, i * 8 + 8
        out.append({"id": f"{tag}-{i}", "records": rec_file(p, "clinc150"), "start": a, "end": b,
                    "out": str(p.work_root / tag / f"{i}.jsonl")})
    return out


class GpuWatch(threading.Thread):
    """nvidia-smi power / clocks / utilisation plus the number of compute processes, every 0.5 s."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.rows: list[tuple[float, float, float, int]] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.sm,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=10).stdout.split()
                w, c, u = (float(x) for x in q.strip().split(","))
                self.rows.append((w, c, u, len(apps)))
            except (ValueError, OSError, subprocess.SubprocessError):
                pass
            self._halt.wait(0.5)

    def stop(self) -> dict[str, Any]:
        self._halt.set()
        self.join(5)
        if not self.rows:
            return {}
        w, c, u, n = zip(*self.rows)
        return {"samples": len(w), "power_w_mean": round(sum(w) / len(w), 1), "power_w_max": max(w),
                "sm_mhz_mean": round(sum(c) / len(c)), "gpu_util_mean": round(sum(u) / len(u), 1),
                "compute_procs_min": min(n), "compute_procs_max": max(n)}


def wait_ready(workers, timeout=1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(w.ready for w in workers):
            return
        if any(not w.alive() and not w.ready for w in workers):
            raise RuntimeError("a worker died before ready")
        time.sleep(2)
    raise RuntimeError("workers not ready in time")


# ------------------------------------------------------------------------------------------ determinism
def http_unit_run(p: Paths, log: Log, out_dir: Path, cfg_name: str, worker: int, quality: bool, tag: str,
                  report: dict[str, Any]) -> Path:
    """Start one server (README flags, or the lean quality-worker flags), run the fixed 200-record unit, stop it."""
    from sweep import engines
    r = engines.ServerRunner("clef-flash", CONFIG_BY_NAME[cfg_name], p, log, {"public": [], "private": []},
                             worker=worker, quality=quality)
    out = out_dir / f"{cfg_name}-{tag}" / f"{HTTP_SET}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if tag == "serial" and U.unit_complete(out, HTTP_N):   # a finished serial reference is reused on a re-run
        report.setdefault("servers", {})[f"{cfg_name}-{tag}"] = {"reused_existing_output": True, "max_num_seqs": 16}
        return out
    try:
        info = r.start()
        report.setdefault("servers", {})[f"{cfg_name}-{tag}"] = info | {"max_num_seqs": 1 if quality else 16}
        rc = r.run_unit({"records": rec_file(p, HTTP_SET), "start": 0, "end": HTTP_N, "out": str(out)})
        if rc != 0:
            raise RuntimeError(f"client exit {rc}")
    finally:
        r.stop()
    return out


def isolate_vllm(out_dir: Path) -> dict[str, Any]:
    """Which of {server instance, server flags, concurrency} moves vLLM logits? GPU otherwise idle, fp8, same 200 records."""
    p = mk_paths("isolate")
    log = Log(p.logs_dir / "sweep-proof.log")
    report: dict[str, Any] = {}
    A = out_dir / "fp8-serial" / f"{HTTP_SET}.jsonl"                     # README flags, alone (from the determinism run)
    C = out_dir / "fp8-concurrent" / f"{HTTP_SET}.jsonl"                 # quality flags, next to 2 busy workers
    B = http_unit_run(p, log, out_dir, "fp8", 0, False, "serial2", report)      # README flags again, alone, new server
    Q = http_unit_run(p, log, out_dir, "fp8", 2, True, "quality-alone", report)  # quality flags, alone
    Q2 = http_unit_run(p, log, out_dir, "fp8", 2, True, "quality-alone2", report)  # quality flags, alone, new server
    report["A_vs_B  same flags, new server instance, both alone"] = compare_exact(A, B)
    report["A_vs_Q  README flags vs quality flags, both alone"] = compare_exact(A, Q)
    report["Q_vs_Q2 quality flags, two server instances, both alone"] = compare_exact(Q, Q2)
    report["Q_vs_C  quality flags alone vs quality flags next to 2 busy workers (pure concurrency)"] = compare_exact(Q, C)
    (out_dir / "isolate-vllm.json").write_text(json.dumps(report, indent=2))
    return report


def determinism(out_dir: Path) -> dict[str, Any]:
    from sweep import engines
    p = mk_paths("determinism")
    log = Log(p.logs_dir / "sweep-proof.log")
    report: dict[str, Any] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = {"public": [], "private": []}

    def http_run(cfg_name: str, worker: int, quality: bool, tag: str) -> Path:
        return http_unit_run(p, log, out_dir, cfg_name, worker, quality, tag, report)

    # serial HTTP references, GPU otherwise idle
    serial_fp8 = http_run("fp8", 0, False, "serial")
    serial_q8 = http_run("q8_0", 0, False, "serial")

    # two native fillers busy for everything below
    fillers = []
    for i in range(2):
        t = native_task(p, f"filler{i}", filler_plan(p, f"f{i}"))
        w = engines.NativeWorker(t, 10 + i, p, log)   # distinct worker id = distinct container name
        w.launch()
        fillers.append(w)
    try:
        wait_ready(fillers)
        watch = GpuWatch()
        watch.start()

        # native: re-run anli_r1 (1000 records) as a third worker next to the two busy fillers
        proof_out = p.work_root / "native-proof" / f"{PROOF_SET}.jsonl"
        plan = [{"id": "proof", "records": rec_file(p, PROOF_SET), "start": 0, "end": PROOF_N, "out": str(proof_out)}]
        pt = native_task(p, "native-proof", plan)
        pw = engines.NativeWorker(pt, 20, p, log)
        pw.launch()
        t0 = time.time()
        while pw.alive():
            time.sleep(2)
        sidecar = json.loads(Path(str(proof_out) + ".meta.json").read_text())
        report["native"] = {
            "reference_file": str(p.results_root / "clef-flash/ref/anli_r1.jsonl"),
            "filler_alive_at_end": [f.alive() for f in fillers],
            "proof_seconds": round(sidecar["wall_seconds"], 1),
            **compare_exact(p.results_root / "clef-flash/ref/anli_r1.jsonl", proof_out)}
        report["native_gpu"] = watch.stop()

        # vLLM and llama.cpp: second run as a worker server next to the two busy fillers
        watch = GpuWatch()
        watch.start()
        conc_fp8 = http_run("fp8", 1, True, "concurrent")
        report["vllm_fp8"] = {"filler_alive_at_end": [f.alive() for f in fillers], **compare_exact(serial_fp8, conc_fp8)}
        report["vllm_gpu"] = watch.stop()
        watch = GpuWatch()
        watch.start()
        conc_q8 = http_run("q8_0", 1, True, "concurrent")
        report["llamacpp_q8_0"] = {"filler_alive_at_end": [f.alive() for f in fillers], **compare_exact(serial_q8, conc_q8)}
        report["llamacpp_gpu"] = watch.stop()
    finally:
        for f in fillers:
            f.kill()
    (out_dir / "determinism.json").write_text(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------------------------------ benchmark
def mix_units(p: Paths, copies: int, tag: str, mix: list[tuple[str, int]] | None = None) -> list[dict[str, Any]]:
    units = []
    for c in range(copies):
        for i, (s, n) in enumerate(mix or MIX):
            a = 8 * (c * 20 + i * 2)   # 8-aligned, < 700 and + n <= 100 stays inside every set (smallest has 1000)
            units.append({"id": f"{tag}-c{c}-{i}", "copy": c, "records": rec_file(p, s), "start": a, "end": a + n,
                          "out": str(p.work_root / tag / f"c{c}-{i}" / f"{s}.jsonl")})
    return units


# Real per-worker footprint of unified memory, MEASURED the hard way (4 native workers exhausted 121 GB and the OOM killer
# took down other services): native = ~19 GB weights + ~12 GB host-side during/after load; servers include process RSS.
WORKER_FOOTPRINT_GB = {"native": 31.0, "vllm": 26.0, "llamacpp": 18.0}   # vllm: 21 GB claim (util 0.16) + ~5 host
SAFETY = 1.5


def mem_guard(engine: str, n: int) -> tuple[bool, str]:
    """Refuse a run that could starve the machine: needs n x footprint x 1.5 + 12 GB headroom available right now."""
    need = n * WORKER_FOOTPRINT_GB[engine] * SAFETY + U.HEADROOM_GB
    have = U.mem_available_gb()
    return have >= need, f"need {need:.0f} GB (n={n}, {WORKER_FOOTPRINT_GB[engine]} GB/worker x {SAFETY} + {U.HEADROOM_GB:.0f}), have {have:.0f} GB"


def bench(engine: str, ns: list[int], out_dir: Path, mix_name: str = "full") -> dict[str, Any]:
    from sweep import engines
    p = mk_paths(f"bench-{engine}")
    log = Log(p.logs_dir / "sweep-proof.log")
    cfg_name = {"native": "ref", "vllm": "fp8", "llamacpp": "q8_0"}[engine]
    cfg = CONFIG_BY_NAME[cfg_name]
    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    mix = MIXES[mix_name]
    total_records = sum(n for _, n in mix)
    servers: list[Any] = []
    try:
        for n in ns:
            ok, why = mem_guard(engine, n)
            if not ok:
                log("BENCH_SKIP", engine=engine, n=n, reason=why)
                print(json.dumps({"engine": engine, "workers": n, "skipped": why}), flush=True)
                continue
            tag = f"{cfg_name}-{mix_name}-n{n}"
            shutil.rmtree(p.work_root / tag, ignore_errors=True)   # a finished unit is skipped by workers: start clean
            units = mix_units(p, n, tag, mix)
            watch = GpuWatch()
            if engine == "native":
                go = p.state_dir / f"go-{tag}"
                go.unlink(missing_ok=True)
                t = native_task(p, tag, units)
                ws = [engines.NativeWorker(t, i, p, log, extra=["--go-file", str(go)]) for i in range(n)]
                for w in ws:
                    w.launch()
                wait_ready(ws)
                watch.start()
                t0 = time.time()
                go.write_text("")
                while any(w.alive() for w in ws):
                    time.sleep(1)
                ends = [json.loads(Path(str(u["out"]) + ".meta.json").read_text())["ended_at"] for u in units]
                wall = max(ends) - t0
                rcs = [w.returncode for w in ws]
                for w in ws:
                    w.kill()
            else:
                while len(servers) < n:   # servers persist across N (start is minutes); start one at a time
                    s = engines.ServerRunner("clef-flash", cfg, p, log, {"public": [], "private": []},
                                             worker=len(servers), quality=True)
                    info = s.start()
                    servers.append(s)
                    log("BENCH_SERVER", n=len(servers), **info)
                t = native_task(p, tag, units)       # reuse plan/claim bookkeeping
                barrier = threading.Barrier(n + 1)
                errs: list[str] = []
                done_at: list[float] = []

                def worker(i: int) -> None:
                    srv = servers[i]
                    barrier.wait()
                    for u in [x for x in units if x["copy"] == i]:
                        Path(u["out"]).parent.mkdir(parents=True, exist_ok=True)
                        rc = srv.run_unit(u)
                        if rc != 0:
                            errs.append(u["id"])
                    done_at.append(time.time())

                th = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
                for x in th:
                    x.start()
                watch.start()
                barrier.wait()
                t0 = time.time()
                for x in th:
                    x.join()
                wall = max(done_at) - t0
                rcs = [1 if errs else 0]
            gpu = watch.stop()
            row = {"engine": engine, "config": cfg_name, "workers": n, "records": total_records * n,
                   "wall_s": round(wall, 1), "records_per_s": round(total_records * n / wall, 3), "exit_codes": rcs,
                   "gpu": gpu}
            rows.append(row)
            log("BENCH_ROW", **{k: v for k, v in row.items() if k != "gpu"}, **gpu)
            print(json.dumps(row), flush=True)
    finally:
        for s in servers:
            try:
                s.stop()
            except Exception as exc:  # noqa: BLE001
                log("STOP_ERROR", error=repr(exc)[:200])
    t1 = next((r["wall_s"] for r in rows if r["workers"] == 1), None)
    for r in rows:
        r["speedup_vs_serial"] = round(r["workers"] * t1 / r["wall_s"], 2) if t1 else None
    rep = {"engine": engine, "mix": mix_name, "mix_records_per_copy": total_records, "rows": rows}
    (out_dir / f"bench-{engine}-{mix_name}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


def instance_test(out_dir: Path) -> dict[str, Any]:
    """Within ONE vLLM server instance: same 200 records alone (X), next to 2 busy native workers (Y), alone again (X2)."""
    from sweep import engines
    p = mk_paths("instance")
    log = Log(p.logs_dir / "sweep-proof.log")
    report: dict[str, Any] = {}
    r = engines.ServerRunner("clef-flash", CONFIG_BY_NAME["fp8"], p, log, {"public": [], "private": []}, worker=2, quality=True)
    fillers: list[Any] = []
    outs = {k: out_dir / f"fp8-instance-{k}" / f"{HTTP_SET}.jsonl" for k in ("X", "Y", "X2")}
    unit = lambda k: {"records": rec_file(p, HTTP_SET), "start": 0, "end": HTTP_N, "out": str(outs[k])}  # noqa: E731
    try:
        report["server"] = r.start()
        for k in ("X",):
            outs[k].parent.mkdir(parents=True, exist_ok=True)
            assert r.run_unit(unit(k)) == 0
        for i in range(2):
            t = native_task(p, f"ifiller{i}", filler_plan(p, f"if{i}"))
            w = engines.NativeWorker(t, 10 + i, p, log)
            w.launch()
            fillers.append(w)
        wait_ready(fillers)
        watch = GpuWatch()
        watch.start()
        outs["Y"].parent.mkdir(parents=True, exist_ok=True)
        assert r.run_unit(unit("Y")) == 0
        report["gpu_during_Y"] = watch.stop() | {"fillers_alive_after": [f.alive() for f in fillers]}
    finally:
        for f in fillers:
            f.kill()
    try:
        outs["X2"].parent.mkdir(parents=True, exist_ok=True)
        assert r.run_unit(unit("X2")) == 0
    finally:
        r.stop()
    report["X_vs_Y  same instance, alone vs next to 2 busy workers (pure concurrency)"] = compare_exact(outs["X"], outs["Y"])
    report["X_vs_X2 same instance, alone twice (repeatability)"] = compare_exact(outs["X"], outs["X2"])
    (out_dir / "instance-vllm.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("determinism")
    d.add_argument("--out", default=str(Path.home() / "clef-quant/sweep-proof/report"))
    sub.add_parser("instance-vllm").add_argument("--out", default=str(Path.home() / "clef-quant/sweep-proof/report"))
    i = sub.add_parser("isolate-vllm")
    i.add_argument("--out", default=str(Path.home() / "clef-quant/sweep-proof/report"))
    b = sub.add_parser("bench")
    b.add_argument("--engine", required=True, choices=["native", "vllm", "llamacpp"])
    b.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3, 4])
    b.add_argument("--mix", choices=sorted(MIXES), default="full")
    b.add_argument("--out", default=str(Path.home() / "clef-quant/sweep-proof/report"))
    for sp in sub.choices.values():
        sp.add_argument("--mps", action="store_true", help="run the workers through a private NVIDIA MPS daemon")
    a = ap.parse_args(argv)
    from sweep import mps
    root = Path.home() / "clef-quant"
    if a.mps:
        mps.start(root, Log(root / "logs/sweep-proof.log"))
    try:
        _dispatch(a)
    finally:
        if a.mps:
            mps.stop(root, Log(root / "logs/sweep-proof.log"))


def _dispatch(a: argparse.Namespace) -> None:
    if a.cmd == "instance-vllm":
        print(json.dumps(instance_test(Path(a.out)), indent=2))
    elif a.cmd == "isolate-vllm":
        print(json.dumps(isolate_vllm(Path(a.out)), indent=2))
    elif a.cmd == "determinism":
        print(json.dumps(determinism(Path(a.out)), indent=2))
    else:
        bench(a.engine, a.ns, Path(a.out) / ("mps" if a.mps else "plain"), a.mix)


if __name__ == "__main__":
    main()
