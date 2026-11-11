"""Command lines the sweep hands to the engines: batched vLLM quality runs, single-request latency, GDN kernel flag."""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sweep import configs, driver, engines  # noqa: E402


@pytest.fixture
def paths(tmp_path):
    return driver.Paths(root=tmp_path / "root", records_dir=tmp_path / "rec", private_records_dir=tmp_path / "prec",
                        results_root=tmp_path / "res", private_results_root=tmp_path / "pres",
                        latency_root=tmp_path / "lat", logs_dir=tmp_path / "logs", state_dir=tmp_path / "state",
                        work_root=tmp_path / "work", private_work_root=tmp_path / "pwork")


def _runner(paths, cfg_name, quality):
    return engines.ServerRunner("clef-flash", configs.CONFIG_BY_NAME[cfg_name], paths, lambda *a, **k: None,
                                {"public": ["anli_r1"], "private": []}, worker=0, quality=quality)


def test_vllm_quality_server_batches_latency_server_does_not(paths):
    q = _runner(paths, "fp8", quality=True)._cmd(0.16)
    assert q[q.index("--max-num-seqs") + 1] == str(configs.QUALITY_MAX_NUM_SEQS)
    assert configs.QUALITY_MAX_NUM_SEQS == 16 and configs.QUALITY_CONCURRENCY == 16
    assert "--max-num-seqs" not in _runner(paths, "fp8", quality=False)._cmd(0.25)


def test_quality_client_concurrency_vllm_only(paths, monkeypatch):
    seen = []
    monkeypatch.setattr(engines, "run_logged", lambda cmd, *a, **k: seen.append(cmd) or 0)
    unit = {"records": "r.jsonl", "start": 0, "end": 10, "out": str(paths.results_root / "x" / "anli_r1.jsonl")}
    _runner(paths, "fp8", quality=True).run_unit(unit)
    llama_runner = _runner(paths, "q8_0", quality=True)
    llama_runner.proc = types.SimpleNamespace(pid=1)   # the client cmd carries the llama-server pid
    llama_runner.run_unit(unit)
    vllm, llama = seen
    assert vllm[vllm.index("--concurrency") + 1] == str(configs.QUALITY_CONCURRENCY)
    assert "--concurrency" not in llama   # llama.cpp's decision server is single-sequence


def test_native_worker_gets_gdn_kernel_flag(paths, monkeypatch):
    launched = []

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(engines.subprocess, "Popen", lambda cmd, **k: launched.append((cmd, k)) or FakeProc())
    monkeypatch.setattr(engines, "register_active", lambda *a, **k: None)
    monkeypatch.setattr(engines, "snapshot_container", lambda m: f"/hf/{m}")
    monkeypatch.setenv("CLEF_PYTHONPATH_EXTRA", "/leaked/from/the/driver/env")   # must not reach a torch worker

    def task(cfg_name):
        class Task:
            name, model, cfg = f"clef-flash/{cfg_name}", "clef-flash", configs.CONFIG_BY_NAME[cfg_name]
            worker_mem_gb = 30.0
            plan_file, claim_dir = paths.state_dir / "plan.json", paths.state_dir / "claims"
        return Task()

    for cfg_name in ("ref", "ref-r2", "ref-fla"):
        engines.NativeWorker(task(cfg_name), 0, paths, lambda *a, **k: None).launch()
    (ref, ref_kw), (ref_r2, ref_r2_kw), (ref_fla, ref_fla_kw) = launched
    assert ref[ref.index("--gdn-kernel") + 1] == "torch"      # the reference stays on transformers' fallback
    assert ref_r2[ref_r2.index("--gdn-kernel") + 1] == "torch"
    assert ref_fla[ref_fla.index("--gdn-kernel") + 1] == "fla"
    assert ref[ref.index("--batch-size") + 1] == "1" and ref_r2[ref_r2.index("--batch-size") + 1] == "8"
    assert configs.CONFIG_BY_NAME["ref-fla"].on_subset and configs.CONFIG_BY_NAME["ref-fla"].repeat_of is None
    # The NGC image has no `fla` (ref-fla failed 2026-10-18 02:15 for exactly that): the fla worker identity-mounts the
    # install and gets it on PYTHONPATH via CLEF_PYTHONPATH_EXTRA; torch workers get neither, even if the driver's env has it.
    fla_dir = str(paths.pylibs_fla)
    assert paths.pylibs_fla == paths.root / "pylibs-fla"
    assert f"{fla_dir}:ro" in ref_fla[2].split(",") and ref_fla_kw["env"]["CLEF_PYTHONPATH_EXTRA"] == fla_dir
    for cmd, kw in ((ref, ref_kw), (ref_r2, ref_r2_kw)):
        assert fla_dir not in cmd[2] and "CLEF_PYTHONPATH_EXTRA" not in kw["env"]
    # A subset plan reads bench/data_r2/records (the same path sched.records_file builds); the worker must mount it.
    # Second ref-fla failure of 2026-10-18 (FileNotFoundError on data_r2/records/clinc150.jsonl inside the container).
    sub = str(paths.subset_records_dir)
    assert sub == str(paths.root / "bench" / configs.REPEAT_SUBSET_DIR / "records")
    assert f"{sub}:ro" in ref_fla[2].split(",")
    assert sub not in ref[2] and sub not in ref_r2[2]   # full-set configs read records_dir / private_records_dir only


def test_docker_env_appends_extra_pythonpath_only_when_set():
    root = Path(__file__).resolve().parents[2]

    def flags(extra):
        env = {k: v for k, v in os.environ.items() if k != "CLEF_PYTHONPATH_EXTRA"}
        if extra is not None:
            env["CLEF_PYTHONPATH_EXTRA"] = extra
        out = subprocess.run(["bash", "-c", "source serve/docker_env.sh && clef_docker_flags"], cwd=root, env=env,
                             capture_output=True, text=True, check=True).stdout.split()
        return [w for w in out if w.startswith("PYTHONPATH=")]

    assert flags(None) == ["PYTHONPATH=/bench:/pylibs"]
    assert flags("") == ["PYTHONPATH=/bench:/pylibs"]
    assert flags("/home/x/clef-quant/pylibs-fla") == ["PYTHONPATH=/bench:/pylibs:/home/x/clef-quant/pylibs-fla"]
