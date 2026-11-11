"""run_stage with fake workers: units get claimed once, merged in order, meta written; failures abandon a config."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sweep import driver, sched, units as U  # noqa: E402
from sweep.configs import CONFIG_BY_NAME  # noqa: E402


def _records(path, n, prefix):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": f"{prefix}{i}"}) + "\n" for i in range(n)))


@pytest.fixture
def env(tmp_path, monkeypatch):
    p = driver.Paths(root=tmp_path / "root", records_dir=tmp_path / "rec", private_records_dir=tmp_path / "prec",
                     results_root=tmp_path / "res", private_results_root=tmp_path / "pres", latency_root=tmp_path / "lat",
                     logs_dir=tmp_path / "logs", state_dir=tmp_path / "state",
                     work_root=tmp_path / "work", private_work_root=tmp_path / "pwork")
    _records(p.records_dir / "mmlu.jsonl", 40, "m")
    _records(p.private_records_dir / "r4b.jsonl", 24, "p")
    p.state_dir.mkdir(parents=True)
    monkeypatch.setattr(U, "mem_available_gb", lambda: 100.0)
    monkeypatch.setattr(sched.engines, "hardened", lambda path: None)
    monkeypatch.setattr(sched.engines, "gpu_snapshot", lambda: {})
    return p, driver.Log(tmp_path / "logs" / "sweep.log")


class FakeWorker:
    """Behaves like a worker: claims units in plan order and writes their outputs, optionally dying first."""
    kind = "fake"
    instances: list = []

    def __init__(self, task, wid, paths, log, die=False):
        self.task, self.owner, self.task_obj = task.name, f"{task.name}-w{wid}", task
        self.mem_gb, self.ready, self.error, self.util = 10.0, True, None, None
        self.die, self._alive = die, True
        self.returncode = None
        FakeWorker.instances.append(self)

    def launch(self):
        t = self.task_obj
        if self.die:
            self.returncode, self.error, self._alive = 1, "boom", False
            return
        for u in t.plan:
            out = Path(u["out"])
            if U.unit_complete(out, u["end"] - u["start"]) or not U.try_claim(t.claim_dir, u["id"], self.owner):
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            recs = [json.loads(l) for l in open(u["records"]) if l.strip()][u["start"]:u["end"]]
            out.write_text("".join(json.dumps({"id": r["id"], "error": None}) + "\n" for r in recs))
            Path(str(out) + ".meta.json").write_text(json.dumps(
                {"records": len(recs), "errors": 0, "wall_seconds": 1.5, "peak_gpu_mem_gib": 2.0,
                 "transformers": "x", "engine": "transformers-native", "started_at": 10.0, "ended_at": 11.0}))
        self.returncode, self._alive = 0, False

    def alive(self):
        return self._alive

    def kill(self):
        self._alive = False


def _task(p, log, cfg="ref"):
    return sched.QualityTask("clef-flash", CONFIG_BY_NAME[cfg], ["mmlu"], ["r4b"], p, log)


def test_stage_merges_in_order_writes_meta_and_cleans_shards(env, monkeypatch):
    p, log = env
    monkeypatch.setattr(sched, "make_worker", lambda t, w, pp, lg: FakeWorker(t, w, pp, lg))
    t = _task(p, log)
    res = sched.run_stage([t], slots=2, paths=p, log=log, poll=0.0, budget_gb=80)
    assert res == {"clef-flash/ref": "ok"}
    pub = p.results_root / "clef-flash/ref"
    assert [json.loads(l)["id"] for l in (pub / "mmlu.jsonl").read_text().splitlines()] == [f"m{i}" for i in range(40)]
    priv = p.private_results_root / "clef-flash/ref"
    assert (priv / "r4b.jsonl").is_file() and not (pub / "r4b.jsonl").exists()   # private only in the private root
    meta = json.loads((pub / "meta.json").read_text())
    assert meta["engine"] == "transformers-native" and meta["peak_memory_gb"] == pytest.approx(2.0 * 2**30 / 1e9, abs=1e-3)
    assert not (p.work_root / "clef-flash").exists() or not any((p.work_root / "clef-flash").rglob("*.jsonl"))
    assert driver.output_complete(pub / "mmlu.jsonl", p.records_dir / "mmlu.jsonl")


def test_second_run_skips_finished_sets(env, monkeypatch):
    p, log = env
    monkeypatch.setattr(sched, "make_worker", lambda t, w, pp, lg: FakeWorker(t, w, pp, lg))
    sched.run_stage([_task(p, log)], 2, p, log, poll=0.0, budget_gb=80)
    FakeWorker.instances.clear()
    t2 = _task(p, log)
    sched.run_stage([t2], 2, p, log, poll=0.0, budget_gb=80)
    assert FakeWorker.instances == [] and t2.plan == []


def test_partial_final_file_is_redone_and_replaced(env, monkeypatch):
    p, log = env
    final = p.results_root / "clef-flash/ref/mmlu.jsonl"
    final.parent.mkdir(parents=True)
    final.write_text('{"id": "m0"}\n{"id": "m1", "x')   # killed mid-write
    monkeypatch.setattr(sched, "make_worker", lambda t, w, pp, lg: FakeWorker(t, w, pp, lg))
    t = _task(p, log)
    sched.run_stage([t], 1, p, log, poll=0.0, budget_gb=80)
    assert driver.output_complete(final, p.records_dir / "mmlu.jsonl")


def test_repeated_worker_failures_abandon_the_config(env, monkeypatch):
    p, log = env
    monkeypatch.setattr(sched, "make_worker", lambda t, w, pp, lg: FakeWorker(t, w, pp, lg, die=True))
    t = _task(p, log)
    res = sched.run_stage([t], 1, p, log, poll=0.0, budget_gb=80)
    assert res == {"clef-flash/ref": "failed"} and t.failures == sched.MAX_TASK_FAILURES
    assert "abandoned" in (p.logs_dir / "sweep.log").read_text()


def test_stop_file_ends_stage_without_new_workers(env, monkeypatch):
    p, log = env
    (p.state_dir / "STOP").write_text("")
    monkeypatch.setattr(sched, "make_worker", lambda t, w, pp, lg: FakeWorker(t, w, pp, lg))
    FakeWorker.instances.clear()
    res = sched.run_stage([_task(p, log)], 2, p, log, poll=0.0, budget_gb=80)
    assert FakeWorker.instances == [] and res == {"clef-flash/ref": "stopped"}


def test_bad_merge_marks_config_failed_and_keeps_shards(env, monkeypatch):
    p, log = env
    class Bad(FakeWorker):
        def launch(self):
            super().launch()
            t = self.task_obj
            u = [x for x in t.plan if x["records"].endswith("mmlu.jsonl")][0]
            Path(u["out"]).write_text("".join(json.dumps({"id": "WRONG%d" % i, "error": None}) + "\n"
                                              for i in range(u["end"] - u["start"])))
    monkeypatch.setattr(sched, "make_worker", lambda t, w, pp, lg: Bad(t, w, pp, lg))
    t = _task(p, log)
    res = sched.run_stage([t], 1, p, log, poll=0.0, budget_gb=80)
    assert res["clef-flash/ref"] == "failed"
    assert not (p.results_root / "clef-flash/ref/mmlu.jsonl").exists()


def test_relaunch_over_finished_config_leaves_meta_byte_identical(env, monkeypatch):
    p, log = env
    monkeypatch.setattr(sched, "make_worker", lambda t, w, pp, lg: FakeWorker(t, w, pp, lg))
    sched.run_stage([_task(p, log)], 2, p, log, poll=0.0, budget_gb=80)
    metas = [root / "clef-flash/ref/meta.json" for root in (p.results_root, p.private_results_root)]
    before = [m.read_bytes() for m in metas]
    assert all(json.loads(b)["concurrent_workers_max"] >= 1 for b in before)
    FakeWorker.instances.clear()
    sched.run_stage([_task(p, log)], 2, p, log, poll=0.0, budget_gb=80)
    assert FakeWorker.instances == [] and [m.read_bytes() for m in metas] == before


def test_relaunch_rebuilds_missing_meta_without_inventing_a_worker_count(env, monkeypatch):
    p, log = env
    monkeypatch.setattr(sched, "make_worker", lambda t, w, pp, lg: FakeWorker(t, w, pp, lg))
    sched.run_stage([_task(p, log)], 2, p, log, poll=0.0, budget_gb=80)
    meta = p.results_root / "clef-flash/ref/meta.json"
    meta.unlink()
    sched.run_stage([_task(p, log)], 2, p, log, poll=0.0, budget_gb=80)
    rebuilt = json.loads(meta.read_text())
    assert rebuilt["concurrent_workers_max"] is None and [r["file"] for r in rebuilt["runs"]] == ["mmlu.jsonl"]


def test_partial_relaunch_keeps_larger_worker_count(env, monkeypatch):
    p, log = env
    monkeypatch.setattr(sched, "make_worker", lambda t, w, pp, lg: FakeWorker(t, w, pp, lg))
    sched.run_stage([_task(p, log)], 2, p, log, poll=0.0, budget_gb=80)
    pub = p.results_root / "clef-flash/ref"
    meta = json.loads((pub / "meta.json").read_text())
    (pub / "meta.json").write_text(json.dumps({**meta, "concurrent_workers_max": 5}))
    (pub / "mmlu.jsonl").unlink()   # one set to redo, so a worker really runs this time
    FakeWorker.instances.clear()
    sched.run_stage([_task(p, log)], 2, p, log, poll=0.0, budget_gb=80)
    assert FakeWorker.instances and json.loads((pub / "meta.json").read_text())["concurrent_workers_max"] == 5
