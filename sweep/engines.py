"""Workers and runners for the sweep driver: native (container), vLLM server (container), llama.cpp server (process).

Quality work is done by WORKERS that claim units (contiguous record ranges) from a shared plan; the latency phase uses
single-session runners. Every subprocess is started in its own session/process group so the driver can stop exactly what
it started. Logs of workers and servers go to <logs>/sweep/ (mode 700): private-set output never reaches sweep.log.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from sweep import units as U
from sweep.configs import BUILD_ARGS, GPU_UTIL, GPU_UTIL_QUALITY, QUALITY_CONCURRENCY, QUALITY_MAX_NUM_SEQS, Config
from sweep.driver import Log, Paths, SessionError

HEALTH_TIMEOUT_S = 1800
START_LOCK = threading.Lock()   # servers start one at a time (memory profiling must not race)


# ----------------------------------------------------------------------------------------------- helpers
def snapshot_host(model: str) -> Path:
    base = Path.home() / f".cache/huggingface/hub/models--Cloudflare--{model}/snapshots"
    snaps = sorted(base.iterdir()) if base.is_dir() else []
    if not snaps:
        raise SessionError(f"no HF snapshot for {model}")
    return snaps[0]


def snapshot_container(model: str) -> str:
    return f"/hf/hub/models--Cloudflare--{model}/snapshots/{snapshot_host(model).name}"


def sh_out(cmd: list[str], timeout: float = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def kill_group(proc: subprocess.Popen, grace: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_logged(cmd: list[str], logfile: Path, cwd: Path | None = None, timeout: float | None = None,
               holder: list | None = None) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "ab") as fh:
        os.chmod(logfile, 0o600)
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, start_new_session=True)
        if holder is not None:
            holder.append(proc)
        try:
            return proc.wait(timeout)
        except BaseException:
            kill_group(proc)
            raise
        finally:
            if holder is not None and proc in holder:
                holder.remove(proc)


def docker_names(all_states: bool = False) -> list[str]:
    cmd = ["docker", "ps", "--format", "{{.Names}}"] + (["-a"] if all_states else [])
    return [n for n in sh_out(cmd).split() if n]


def docker_stop(name: str) -> None:
    sh_out(["docker", "stop", "-t", "30", name], timeout=120)
    if name in docker_names(all_states=True):
        sh_out(["docker", "rm", "-f", name], timeout=60)


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def hardened(path: Path) -> None:
    sh_out(["chmod", "-R", "go-rwx", str(path)])


_ACTIVE_LOCK = threading.Lock()


def _active_file(paths: Paths) -> Path:
    return paths.state_dir / "active.json"


def register_active(paths: Paths, **entry: Any) -> None:
    with _ACTIVE_LOCK:
        f = _active_file(paths)
        cur = json.loads(f.read_text()) if f.is_file() else []
        cur.append(entry)
        f.write_text(json.dumps(cur))


def unregister_active(paths: Paths, name: str) -> None:
    with _ACTIVE_LOCK:
        f = _active_file(paths)
        if f.is_file():
            f.write_text(json.dumps([e for e in json.loads(f.read_text()) if e.get("name") != name]))


def cleanup_stale(paths: Paths, log: Log) -> None:
    """Stop only what an earlier driver instance recorded as started by it."""
    f = _active_file(paths)
    if not f.is_file():
        return
    for e in json.loads(f.read_text()):
        log("CLEANUP_STALE", **{k: v for k, v in e.items()})
        if e.get("container"):
            docker_stop(e["container"])
        pgid = e.get("pgid")
        if pgid:
            try:
                cmdline = Path(f"/proc/{pgid}/cmdline").read_bytes().decode(errors="ignore")
            except OSError:
                continue
            if "serve_" in cmdline or "llama-server" in cmdline:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    f.write_text("[]")


def foreign_gpu_jobs(paths: Paths) -> list[str]:
    names = [n for n in docker_names() if n.startswith("clefbench-")]
    for line in sh_out(["ps", "-eo", "pid=,args="]).splitlines():
        if "llama-server" in line and "clef-quant" in line:
            names.append("llama-server:" + line.split()[0])
    return names


def wait_gpu_idle(paths: Paths, log: Log) -> None:
    n = 0
    while True:
        busy = foreign_gpu_jobs(paths)
        if not busy:
            return
        if n % 10 == 0:
            log("GPU_BUSY_WAIT", other=",".join(busy))
        n += 1
        time.sleep(60)


def gpu_snapshot() -> dict[str, str]:
    out = sh_out(["nvidia-smi", "--query-gpu=power.draw,clocks.sm,utilization.gpu", "--format=csv,noheader,nounits"], 10)
    parts = [p.strip() for p in out.strip().split(",")]
    return dict(zip(("power_w", "sm_mhz", "gpu_util"), parts)) if len(parts) == 3 else {}


def config_file_size_gb(model: str, cfg: Config, paths: Paths) -> float | None:
    from runners.common import weights_size_gb
    try:
        if cfg.engine == "llamacpp":
            return round((paths.root / f"models/gguf/{model}-{cfg.fmt}.gguf").stat().st_size / 1e9, 3)
        if cfg.engine == "vllm" and cfg.fmt != "bf16":
            return round(weights_size_gb(paths.root / f"models/{model}-{cfg.fmt}"), 3)
        return round(weights_size_gb(snapshot_host(model)), 3)
    except (OSError, SessionError):
        return None


def util_ladder(model: str, cfg: Config, quality: bool) -> tuple[float | None, ...]:
    """--gpu-memory-utilization candidates (vLLM only): lean for concurrent quality workers, README values for latency."""
    if cfg.engine != "vllm":
        return (None,)
    return (GPU_UTIL_QUALITY if quality else GPU_UTIL)[(model, cfg.fmt)]


def worker_mem_gb(model: str, cfg: Config, paths: Paths, util: float | None) -> float:
    size = config_file_size_gb(model, cfg, paths) or 20.0
    if cfg.engine == "native":
        return U.native_mem_gb(size)
    if cfg.engine == "vllm":
        return U.vllm_mem_gb(util or 0.25)
    return U.llama_mem_gb(size)


# ------------------------------------------------------------------------------------------- native worker
class NativeWorker:
    """One container running sweep.native_worker over the shared plan."""

    kind = "native"

    def __init__(self, task: Any, wid: int, paths: Paths, log: Log, extra: list[str] | None = None) -> None:
        self.task, self.wid, self.paths, self.log = task.name, wid, paths, log
        self.model, self.cfg = task.model, task.cfg
        self.owner = f"{task.model}-{task.cfg.name}-w{wid}"
        self.container = f"clefbench-sweep-native-{self.owner}"
        self.ready_file = paths.state_dir / "ready" / self.owner
        self.mem_gb = task.worker_mem_gb
        self.proc: subprocess.Popen | None = None
        self.task_obj = task
        self.extra = extra or []

    def launch(self) -> None:
        p = self.paths
        self.ready_file.parent.mkdir(parents=True, exist_ok=True)
        self.ready_file.unlink(missing_ok=True)
        mounts = [f"{p.records_dir}:ro", f"{p.private_records_dir}:ro", str(p.work_root), str(p.private_work_root),
                  str(p.state_dir)]
        if self.cfg.on_subset:
            # Subset plans point at the 200-record files (sched.records_file); the container sees them only if mounted.
            # ref-fla is the only native subset config, so this was never exercised before 2026-10-18.
            mounts.append(f"{p.subset_records_dir}:ro")
        env = dict(os.environ)
        env.pop("CLEF_PYTHONPATH_EXTRA", None)   # the reference never sees flash-linear-attention on its path
        if self.cfg.gdn_kernel == "fla":
            # The NGC image ships no `fla`: identity-mount its install and append it to the container's PYTHONPATH
            # (docker_env.sh reads CLEF_PYTHONPATH_EXTRA). Only fla-kernel workers get either.
            mounts.append(f"{p.pylibs_fla}:ro")
            env["CLEF_PYTHONPATH_EXTRA"] = str(p.pylibs_fla)
        cmd = [str(p.root / "bench/sweep/in_container.sh"), self.container, ",".join(mounts),
               "sweep.native_worker", "--model", snapshot_container(self.model), "--model-name", self.model,
               "--config", self.cfg.name, "--batch-size", str(self.cfg.batch_size),
               "--plan", str(self.task_obj.plan_file), "--claim-dir", str(self.task_obj.claim_dir),
               "--owner", self.owner, "--ready-file", str(self.ready_file),
               "--stop-file", str(p.state_dir / "STOP"), "--gdn-kernel", self.cfg.gdn_kernel, *self.extra]
        logfile = p.logs_dir / "sweep" / f"{self.owner}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "ab") as fh:
            os.chmod(logfile, 0o600)
            self.proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=p.root / "bench",
                                         start_new_session=True, env=env)
        register_active(p, name=self.owner, container=self.container, pgid=self.proc.pid)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def ready(self) -> bool:
        return self.ready_file.is_file()

    @property
    def returncode(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def kill(self) -> None:
        docker_stop(self.container)
        if self.proc is not None:
            kill_group(self.proc, 10.0)
        unregister_active(self.paths, self.owner)

    def info(self) -> dict[str, Any]:
        return {"engine": "native"}


# ---------------------------------------------------------------------------------------------- servers
class ServerRunner:
    """vLLM (container) or llama.cpp (process) server plus the HTTP client calls against it.

    quality=True: lean memory ladder, batched serving (`--max-num-seqs QUALITY_MAX_NUM_SEQS` and QUALITY_CONCURRENCY
    client requests in flight, vLLM only), distinct port/container/pidfile per `worker`.
    quality=False: the verified README flags, used by the isolated latency phase.
    """

    def __init__(self, model: str, cfg: Config, paths: Paths, log: Log, sets: dict[str, list[str]],
                 worker: int = 0, quality: bool = False) -> None:
        self.model, self.cfg, self.paths, self.log, self.sets = model, cfg, paths, log, sets
        self.worker, self.quality = worker, quality
        self.proc: subprocess.Popen | None = None
        self.util: float | None = None
        suffix = f"-w{worker}" if worker else ""
        self.tag = f"{model}-{cfg.name}{suffix}"
        self.container = f"clefbench-serve-{model}-{cfg.fmt}{suffix}" if cfg.engine == "vllm" else None
        self.port = cfg.port + 100 * worker
        self.url = f"http://127.0.0.1:{self.port}"
        self.bench = paths.root / "bench"
        self.model_dir: Path | None = None
        self.model_file: Path | None = None
        if cfg.engine == "llamacpp":
            self.model_file = paths.root / f"models/gguf/{model}-{cfg.fmt}.gguf"
        elif cfg.fmt == "bf16":
            self.model_dir = snapshot_host(model)
        else:
            self.model_dir = paths.root / f"models/{model}-{cfg.fmt}"

    def ladder(self) -> tuple[float | None, ...]:
        return util_ladder(self.model, self.cfg, self.quality)

    def _cmd(self, util: float | None) -> list[str]:
        if self.cfg.engine == "vllm":
            cmd = [str(self.bench / "serve/serve_clef.sh"), self.model, self.cfg.fmt, str(self.port),
                   "--gpu-memory-utilization", str(util)]
            return cmd + (["--max-num-seqs", str(QUALITY_MAX_NUM_SEQS)] if self.quality else [])
        return [str(self.bench / "serve/serve_llamacpp.sh"), self.model, self.cfg.fmt, str(self.port)]

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001 - not up yet
            return False

    def _tail(self, logfile: Path) -> str:
        try:
            lines = [l for l in logfile.read_text(errors="ignore").splitlines() if l.strip()]
        except OSError:
            return ""
        return " | ".join(l[:160] for l in lines[-3:])

    def kill_server(self) -> None:
        if self.container:
            docker_stop(self.container)
        if self.proc is not None:
            kill_group(self.proc, grace=30.0)
            self.proc = None
        unregister_active(self.paths, self.tag)

    def start(self) -> dict[str, Any]:
        if port_in_use(self.port):
            raise SessionError(f"port {self.port} already in use; not touching it")
        last = ""
        env = dict(os.environ)
        if self.container:
            env["CLEF_CONTAINER_NAME"] = self.container
        env["CLEF_PIDFILE"] = str(self.paths.state_dir / f"llama-{self.tag}.pid")
        for util in self.ladder():
            logfile = self.paths.logs_dir / "sweep" / f"serve-{self.tag}.log"
            with START_LOCK:
                t0 = time.time()
                self.log("START", job=f"{self.tag}/server", util=util, port=self.port)
                with open(logfile, "ab") as fh:
                    os.chmod(logfile, 0o600)
                    self.proc = subprocess.Popen(self._cmd(util), stdout=fh, stderr=subprocess.STDOUT, cwd=self.bench,
                                                 start_new_session=True, env=env)
                register_active(self.paths, name=self.tag, container=self.container, pgid=self.proc.pid)
                while time.time() - t0 < HEALTH_TIMEOUT_S:
                    if self._healthy():
                        self.util = util
                        secs = round(time.time() - t0, 1)
                        self.log("END", job=f"{self.tag}/server", exit=0, secs=secs, util=util)
                        return {"gpu_memory_utilization": util, "start_secs": secs}
                    if self.proc.poll() is not None:
                        break
                    time.sleep(5)
                last = self._tail(logfile)
                self.log("FAIL", job=f"{self.tag}/server", util=util, exit=self.proc.poll(), tail=last)
                self.kill_server()
            time.sleep(10)
        raise SessionError(f"server did not become healthy (tried {self.ladder()}): {last}")

    def _client_cmd(self, extra: list[str]) -> list[str]:
        cmd = ["python3", "-m", "runners.http_systemone", "--url", self.url, "--model-name", self.model,
               "--config", self.cfg.name]
        if self.model_file:
            cmd += ["--model-file", str(self.model_file), "--server-pid", str(self.proc.pid)]
        else:
            cmd += ["--model-dir", str(self.model_dir)]
        return cmd + extra

    def run_unit(self, u: dict[str, Any], holder: list | None = None) -> int:
        out = Path(u["out"])
        extra = ["--records", u["records"], "--range", str(u["start"]), str(u["end"]), "--out-dir", str(out.parent)]
        if self.quality and self.cfg.engine == "vllm":   # batched quality runs; llama.cpp serves one sequence at a time
            extra += ["--concurrency", str(QUALITY_CONCURRENCY)]
        cmd = self._client_cmd(extra)
        return run_logged(cmd, self.paths.logs_dir / "sweep" / f"{self.tag}-units.log", cwd=self.bench, holder=holder)

    def latency(self) -> None:
        out = self.paths.latency_file(self.model, self.cfg.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["python3", "-m", "sweep.latency.latency_http", "--url", self.url, "--model-name", self.model,
               "--config", self.cfg.name, "--records-dir", str(self.paths.records_dir),
               "--private-records-dir", str(self.paths.private_records_dir),
               "--public-sets", *self.sets["public"], "--private-sets", *self.sets["private"], "--out", str(out)]
        if self.container:
            cmd += ["--container", self.container]
        else:
            cmd += ["--server-pid", str(self.proc.pid)]
        cmd += (["--model-file", str(self.model_file)] if self.model_file else ["--model-dir", str(self.model_dir)])
        if self.util is not None:
            cmd += ["--gpu-memory-utilization", str(self.util)]
        self.log("START", job=f"{self.model}/{self.cfg.name}/latency")
        t0 = time.time()
        rc = run_logged(cmd, self.paths.logs_dir / "sweep" / f"{self.tag}-latency.log", cwd=self.bench)
        ok = rc == 0 and out.is_file()
        self.log("END" if ok else "FAIL", job=f"{self.model}/{self.cfg.name}/latency", exit=rc,
                 secs=round(time.time() - t0, 1))
        if not ok:
            raise SessionError(f"latency_http exit {rc}")

    def stop(self) -> None:
        self.log("SERVER_STOP", session=self.tag)
        self.kill_server()
        if self.container and self.container in docker_names(all_states=True):
            raise RuntimeError(f"container {self.container} still present after stop")
        if port_in_use(self.port):
            raise RuntimeError(f"port {self.port} still in use after stop")


class HttpWorker(threading.Thread):
    """A server plus a thread that claims units from the shared plan and runs the HTTP client on each."""

    kind = "http"

    def __init__(self, task: Any, wid: int, paths: Paths, log: Log) -> None:
        super().__init__(daemon=True, name=f"http-worker-{task.name}-w{wid}")
        self.task, self.wid, self.paths, self.log = task.name, wid, paths, log
        self.task_obj = task
        self.model, self.cfg = task.model, task.cfg
        self.runner = ServerRunner(task.model, task.cfg, paths, log, task.sets, worker=wid, quality=True)
        self.owner = self.runner.tag
        self.mem_gb = task.worker_mem_gb
        self.ready = False
        self.error: str | None = None
        self.finished = False
        self._holder: list = []
        self._abort = threading.Event()
        self.util: float | None = None

    def launch(self) -> None:
        self.start()

    def alive(self) -> bool:
        return not self.finished

    @property
    def returncode(self) -> int | None:
        return None if not self.finished else (1 if self.error else 0)

    def run(self) -> None:  # thread body
        t = self.task_obj
        stop = self.paths.state_dir / "STOP"
        try:
            info = self.runner.start()
            self.util = self.runner.util
            self.mem_gb = U.vllm_mem_gb(self.util) if self.cfg.engine == "vllm" else self.mem_gb
            self.log("SERVER_UP", session=self.owner, **info)
            self.ready = True
            for u in t.plan:
                if self._abort.is_set() or stop.exists():
                    break
                out, n = Path(u["out"]), u["end"] - u["start"]
                if U.unit_complete(out, n) or not U.try_claim(t.claim_dir, u["id"], self.owner):
                    continue
                rc = self.runner.run_unit(u, self._holder)
                if rc != 0 or not U.unit_complete(out, n):
                    raise SessionError(f"http_systemone exit {rc} on {u['id']}")
        except SessionError as exc:
            self.error = str(exc)[:300].replace("\n", " ")
        except BaseException as exc:  # noqa: BLE001 - a worker thread must report, never die silently
            self.error = repr(exc)[:300]
        finally:
            try:
                self.runner.stop()
            except Exception as exc:  # noqa: BLE001
                self.error = (self.error or "") + f" stop_error={exc!r}"[:200]
            self.finished = True

    def kill(self) -> None:
        self._abort.set()
        for p in list(self._holder):
            kill_group(p, 5.0)
        if not self.finished:
            self.runner.kill_server()

    def info(self) -> dict[str, Any]:
        return {"engine": self.cfg.engine, "gpu_memory_utilization": self.util}


# ------------------------------------------------------------------------------------------ latency runners
class NativeLatencyRunner:
    """Isolated native latency run (the quality work is done by NativeWorker)."""

    def __init__(self, model: str, cfg: Config, paths: Paths, log: Log, sets: dict[str, list[str]]) -> None:
        self.model, self.cfg, self.paths, self.log, self.sets = model, cfg, paths, log, sets
        self.cname = f"clefbench-sweep-nativelat-{model}-{cfg.name}"

    def start(self) -> dict[str, Any]:
        return {"engine": "native"}

    def quality(self, items: list[dict[str, Any]]) -> None:  # pragma: no cover - quality is never done here
        raise SessionError("quality is run by the concurrent scheduler, not the latency runner")

    def latency(self) -> None:
        p = self.paths
        out = p.latency_file(self.model, self.cfg.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        mounts = [f"{p.records_dir}:ro", f"{p.private_records_dir}:ro", str(p.latency_root)]
        cmd = [str(p.root / "bench/sweep/in_container.sh"), self.cname, ",".join(mounts),
               "sweep.latency.latency_native", "--model", snapshot_container(self.model), "--model-name", self.model,
               "--config", self.cfg.name, "--records-dir", str(p.records_dir),
               "--private-records-dir", str(p.private_records_dir),
               "--public-sets", *self.sets["public"], "--private-sets", *self.sets["private"], "--out", str(out)]
        self.log("START", job=f"{self.model}/{self.cfg.name}/latency")
        t0 = time.time()
        try:
            rc = run_logged(cmd, p.logs_dir / "sweep" / f"{self.model}-{self.cfg.name}-latency.log", cwd=p.root / "bench")
        except BaseException:
            docker_stop(self.cname)
            raise
        ok = rc == 0 and out.is_file()
        self.log("END" if ok else "FAIL", job=f"{self.model}/{self.cfg.name}/latency", exit=rc,
                 secs=round(time.time() - t0, 1))
        if not ok:
            raise SessionError(f"latency_native exit {rc}")

    def stop(self) -> None:
        docker_stop(self.cname)


def make_runner(item: dict[str, Any], cfg: Config, paths: Paths, log: Log):
    """Latency-session runner (quality workers are built by the scheduler)."""
    model = item["model"]
    sets = item["sets"]
    try:
        if cfg.engine == "native":
            snapshot_host(model)
            return NativeLatencyRunner(model, cfg, paths, log, sets)
        runner = ServerRunner(model, cfg, paths, log, sets, worker=0, quality=False)
    except SessionError:
        return None
    if runner.model_file is not None and not runner.model_file.is_file():
        return None
    if runner.model_dir is not None and not list(runner.model_dir.glob("**/*.safetensors")):
        return None
    return runner


# -------------------------------------------------------------------------------------------------- builds
def run_build(item: dict[str, Any], paths: Paths, log: Log) -> str:
    model, fmt = item["model"], item["fmt"]
    out = paths.root / f"models/{model}-{fmt}"
    marker = out / ".sweep_build_ok"
    if marker.is_file():
        log("SKIP", job=item["id"], reason="build marker present")
        return "skipped"
    log("START", job=item["id"])
    t0 = time.time()
    rc = run_logged([str(paths.root / "bench/quant/build_quants.sh"), model, fmt, *BUILD_ARGS[fmt]],
                    paths.logs_dir / "sweep" / f"build-{model}-{fmt}.log", cwd=paths.root / "bench")
    secs = round(time.time() - t0, 1)
    ok = rc == 0 and list(out.glob("*.safetensors"))
    if ok:
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    gb = sum(p.stat().st_size for p in out.glob("*.safetensors")) / 1e9 if out.is_dir() else 0
    log("END" if ok else "FAIL", job=item["id"], exit=rc, secs=secs, size_gb=round(gb, 2))
    return "ok" if ok else "failed"
