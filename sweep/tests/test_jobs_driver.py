"""Job generation, session grouping, skip detection, meta merge and the driver session flow (fake engines)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sweep import driver, jobs  # noqa: E402

PUB = ["anli_r1", "banking77"]
PRIV = ["r1", "r4b"]


def test_job_order_flash_then_builds_then_clef():
    js = jobs.make_jobs(PUB, PRIV)
    seq = []
    for j in js:
        key = (j["kind"], j.get("model"), j.get("config") or j.get("fmt"))
        if not seq or seq[-1] != key:
            seq.append(key)
    cfgs = ["ref", "q8_0", "q4_k_m", "fp8", "nvfp4", "bf16-srv", "gguf-bf16", "ref-r2",
            "bf16-srv-r2", "fp8-r2", "nvfp4-r2", "gguf-bf16-r2", "ref-fla"]
    expected = [("session", "clef-flash", c) for c in cfgs] + [("build", "clef", "fp8"), ("build", "clef", "nvfp4")] \
        + [("session", "clef", c) for c in cfgs]
    got = [("session" if k == "quality" or k == "latency" else k, m, c) for k, m, c in seq]
    dedup = []
    for g in got:
        if not dedup or dedup[-1] != g:
            dedup.append(g)
    assert dedup == expected


def test_each_session_has_all_sets_then_latency():
    js = [j for j in jobs.make_jobs(PUB, PRIV) if j.get("model") == "clef-flash" and j["config"] == "ref"]
    assert [j["set"] for j in js if j["kind"] == "quality"] == PUB + PRIV
    assert [j["private"] for j in js if j["kind"] == "quality"] == [False, False, True, True]
    assert js[-1]["kind"] == "latency"


def test_group_sessions_groups_consecutive_same_config():
    js = jobs.make_jobs(PUB, PRIV)
    items = jobs.group(js)
    kinds = [(i["type"], i.get("model"), i.get("config") or i.get("fmt")) for i in items]
    assert kinds[0] == ("session", "clef-flash", "ref")
    assert ("build", "clef", "nvfp4") in kinds
    assert len([k for k in kinds if k[0] == "session"]) == 26     # (8 configs + 4 repeats + 1 subset) x 2 models
    from sweep.configs import CONFIG_BY_NAME
    for i in items:
        if i["type"] == "session":
            subset = CONFIG_BY_NAME[i["config"]].on_subset
            assert len(i["jobs"]) == (len(PUB) if subset else len(PUB) + len(PRIV) + 1)


def test_ref_fla_is_a_subset_config_not_a_noise_floor_repeat():
    import re
    from sweep.configs import CONFIG_BY_NAME
    c = CONFIG_BY_NAME["ref-fla"]
    assert c.on_subset and c.repeat_of is None and c.subset
    assert not re.match(r"^.+-r\d+$", c.name)      # bench.analyze would fold a `<base>-r<N>` name into ref's noise floor
    assert (c.engine, c.fmt, c.batch_size, c.gdn_kernel) == ("native", "bf16", 1, "fla")
    assert CONFIG_BY_NAME["ref"].gdn_kernel == CONFIG_BY_NAME["ref-r2"].gdn_kernel == "torch"
    js = [j for j in jobs.make_jobs(PUB, PRIV) if j.get("config") == "ref-fla"]
    assert [j["set"] for j in js if j["kind"] == "quality"] == PUB + PUB      # both models, public only
    assert not [j for j in js if j["kind"] == "latency"]


def test_jobs_roundtrip(tmp_path):
    js = jobs.make_jobs(PUB, PRIV)
    p = tmp_path / "jobs.jsonl"
    jobs.write_jobs(p, js)
    assert jobs.read_jobs(p) == js


def _write(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(n)))


def test_output_complete_requires_full_count(tmp_path):
    rec, out = tmp_path / "rec.jsonl", tmp_path / "out" / "rec.jsonl"
    _write(rec, 5)
    assert not driver.output_complete(out, rec)
    _write(out, 4)
    assert not driver.output_complete(out, rec)
    _write(out, 5)
    assert driver.output_complete(out, rec)
    _write(out, 6)
    assert not driver.output_complete(out, rec)


def test_merge_meta_keeps_earlier_runs_and_max_peak():
    old = {"peak_memory_gb": 20.0, "runs": [{"file": "a.jsonl", "records": 3}]}
    new = {"peak_memory_gb": 18.0, "runs": [{"file": "b.jsonl", "records": 4}], "engine": "x"}
    m = driver.merge_meta(old, new)
    assert m["peak_memory_gb"] == 20.0 and m["engine"] == "x"
    assert [r["file"] for r in m["runs"]] == ["a.jsonl", "b.jsonl"]
    again = driver.merge_meta(m, {"runs": [{"file": "b.jsonl", "records": 9}]})
    assert [r["records"] for r in again["runs"] if r["file"] == "b.jsonl"] == [9]


class FakeRunner:
    def __init__(self, paths, fail_start=False, fail_set=None):
        self.paths, self.fail_start, self.fail_set = paths, fail_start, fail_set
        self.events = []

    def start(self):
        self.events.append("start")
        if self.fail_start:
            raise driver.SessionError("no health")
        return {"gpu_memory_utilization": 0.25}

    def quality(self, items):
        for it in items:
            self.events.append(("quality", it["set"]))
            if it["set"] == self.fail_set:
                raise driver.SessionError("boom")
            out = self.paths.out_file(it)
            _write(out, driver.count_lines(self.paths.records_file(it)))

    def latency(self):
        self.events.append("latency")
        self.paths.latency_file("m", "c").parent.mkdir(parents=True, exist_ok=True)
        self.paths.latency_file("m", "c").write_text("{}")

    def stop(self):
        self.events.append("stop")


@pytest.fixture
def paths(tmp_path):
    p = driver.Paths(root=tmp_path / "root", records_dir=tmp_path / "rec", private_records_dir=tmp_path / "prec",
                     results_root=tmp_path / "res", private_results_root=tmp_path / "pres",
                     latency_root=tmp_path / "lat", logs_dir=tmp_path / "logs", state_dir=tmp_path / "state")
    for s in PUB:
        _write(p.records_dir / f"{s}.jsonl", 3)
    for s in PRIV:
        _write(p.private_records_dir / f"{s}.jsonl", 2)
    return p


def _session(model="m", config="c"):
    js = jobs._session_jobs(model, config, PUB, PRIV, True)
    return {"type": "session", "model": model, "config": config, "jobs": js}


def test_session_runs_pending_then_stops_server(paths):
    r = FakeRunner(paths)
    logger = driver.Log(paths.logs_dir / "sweep.log")
    res = driver.run_session(_session(), r, paths, logger)
    assert res == "ok"
    assert r.events[0] == "start" and r.events[-1] == "stop" and "latency" in r.events


def test_session_skips_when_everything_done(paths):
    r = FakeRunner(paths)
    logger = driver.Log(paths.logs_dir / "sweep.log")
    driver.run_session(_session(), r, paths, logger)
    r2 = FakeRunner(paths)
    # latency file written by FakeRunner under ("m","c")
    assert driver.run_session(_session(), r2, paths, logger) == "skipped"
    assert r2.events == []


def test_session_stops_server_on_failure_and_logs(paths):
    r = FakeRunner(paths, fail_set="banking77")
    logger = driver.Log(paths.logs_dir / "sweep.log")
    res = driver.run_session(_session(), r, paths, logger)
    assert res == "failed" and r.events[-1] == "stop"
    text = (paths.logs_dir / "sweep.log").read_text()
    assert "FAIL" in text and "banking77" in text


def test_session_start_failure_never_calls_quality_but_stops(paths):
    r = FakeRunner(paths, fail_start=True)
    logger = driver.Log(paths.logs_dir / "sweep.log")
    assert driver.run_session(_session(), r, paths, logger) == "failed"
    assert ("quality", "anli_r1") not in r.events and r.events[-1] == "stop"


def test_partial_resume_only_runs_missing_sets(paths):
    j = _session()
    _write(paths.out_file(j["jobs"][0]), 3)  # anli_r1 already complete
    r = FakeRunner(paths)
    driver.run_session(j, r, paths, driver.Log(paths.logs_dir / "sweep.log"))
    assert ("quality", "anli_r1") not in r.events and ("quality", "banking77") in r.events


def test_private_outputs_go_to_private_root(paths):
    j = _session()
    priv = [x for x in j["jobs"] if x["kind"] == "quality" and x["private"]][0]
    pub = [x for x in j["jobs"] if x["kind"] == "quality" and not x["private"]][0]
    assert str(paths.out_file(priv)).startswith(str(paths.private_results_root))
    assert str(paths.out_file(pub)).startswith(str(paths.results_root))
    assert not str(paths.out_file(priv)).startswith(str(paths.results_root))


def test_output_complete_rejects_invalid_json_and_truncated_line(tmp_path):
    rec, out = tmp_path / "rec.jsonl", tmp_path / "out.jsonl"
    _write(rec, 3)
    out.write_text('{"id": 0}\n{"id": 1}\n{"id": 2, "answ')  # right line count, last line cut mid-write
    assert not driver.output_complete(out, rec)
    out.write_text('{"id": 0}\n{"id": 1}\n{"id": 2}\n')
    assert driver.output_complete(out, rec)


def test_restart_probes_are_third_instances_outside_the_full_sweep():
    from sweep.configs import CONFIG_BY_NAME, CONFIGS, RESTART_PROBES
    assert {c.name for c in RESTART_PROBES} == {"fp8-r3", "nvfp4-r3"}
    assert not {c.name for c in RESTART_PROBES} & {c.name for c in CONFIGS}       # make_jobs' default list is unchanged
    for c in RESTART_PROBES:
        base = CONFIG_BY_NAME[c.repeat_of]
        assert (c.engine, c.fmt, c.port) == (base.engine, base.fmt, base.port)   # same engine + flags as the base
        assert c.name == f"{c.repeat_of}-r3" and c.on_subset                      # `<base>-r<N>`, 200-record subsets
        assert CONFIG_BY_NAME[c.name] is c                                         # the driver and status resolve it


def test_cross_restart_repeat_configs_are_public_subset_only_without_latency():
    from sweep.configs import CONFIG_BY_NAME, CONFIGS
    reps = [c for c in CONFIGS if c.repeat_of]
    assert {c.name for c in reps} == {"bf16-srv-r2", "fp8-r2", "nvfp4-r2", "gguf-bf16-r2"}
    for c in reps:
        base = CONFIG_BY_NAME[c.repeat_of]
        assert (c.engine, c.fmt, c.port) == (base.engine, base.fmt, base.port)   # same engine + flags as the base, new instance
        assert c.name == f"{c.repeat_of}-r2"                                      # bench.analyze reads `<base>-r<N>` as a repeat
    js = jobs.make_jobs(PUB, PRIV)
    r2 = [j for j in js if j.get("config") in {c.name for c in reps}]
    assert r2 and all(not j.get("private") for j in r2 if j["kind"] == "quality")
    assert not [j for j in r2 if j["kind"] == "latency"]
    assert {j["config"] for j in r2 if j["kind"] == "quality"} == {c.name for c in reps}
    # r2 jobs come after the base config of the same model, so the base's server instance is gone before the repeat starts
    for model in ("clef-flash", "clef"):
        order = [j["config"] for j in js if j.get("model") == model and j["kind"] == "quality"]
        order = list(dict.fromkeys(order))
        for c in reps:
            assert order.index(c.name) > order.index(c.repeat_of)


def test_ensure_subset_takes_first_n_records_deterministically(tmp_path):
    src, dst = tmp_path / "records", tmp_path / "sub"
    src.mkdir()
    (src / "a.jsonl").write_text("".join(json.dumps({"id": f"a{i}"}) + "\n" for i in range(10)))
    (src / "a_hosted_repeat.jsonl").write_text("x\n")
    jobs.ensure_subset(src, dst, 4)
    assert [json.loads(l)["id"] for l in (dst / "a.jsonl").read_text().splitlines()] == ["a0", "a1", "a2", "a3"]
    assert not (dst / "a_hosted_repeat.jsonl").exists()
    jobs.ensure_subset(src, dst, 4)   # idempotent
    assert len((dst / "a.jsonl").read_text().splitlines()) == 4
