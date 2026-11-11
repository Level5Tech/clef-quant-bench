"""Tests of bench.speed on a synthetic sweep log and results tree."""

import json
from pathlib import Path

import pytest

from bench import speed as S

LOG = """\
2026-10-16T10:00:00-0700 DRIVER_START jobs=x
2026-10-16T10:00:10-0700 END job=clef-flash/ref/aaa exit=0 records=100 expected=100 errors=0 unit_secs=50.0 elapsed=900.0 units=2 engine=native
2026-10-16T10:00:20-0700 END job=clef-flash/ref/bbb exit=0 records=200 expected=200 errors=0 unit_secs=100.0 elapsed=100.0 units=1 engine=native
2026-10-16T10:00:30-0700 END job=clef-flash/ref/p1 exit=0 records=50 expected=50 errors=0 unit_secs=40.0 elapsed=40.0 units=1 engine=native
2026-10-16T10:01:00-0700 END job=clef-flash/q8_0/aaa exit=0 records=100 expected=100 errors=0 unit_secs=30.0 elapsed=30.0 units=1 engine=llamacpp
2026-10-16T10:01:10-0700 END job=clef-flash/q8_0/bbb exit=1 records=200 expected=200 errors=3 unit_secs=999.0 elapsed=999.0 units=1 engine=llamacpp
2026-10-16T10:01:20-0700 END job=clef-flash/q8_0/bbb exit=0 records=200 expected=200 errors=0 unit_secs=60.0 elapsed=60.0 units=1 engine=llamacpp
2026-10-16T10:01:30-0700 END job=clef-flash/q8_0/bbb exit=0 records=200 expected=200 errors=0 unit_secs=80.0 elapsed=80.0 units=1 engine=llamacpp
2026-10-16T10:01:40-0700 END job=clef-flash/q8_0/p1 exit=0 records=50 expected=50 errors=0 unit_secs=20.0 elapsed=20.0 units=1 engine=llamacpp
2026-10-16T10:02:00-0700 END job=clef/ref/aaa exit=0 records=100 expected=100 errors=0 unit_secs=150.0 elapsed=9000.0 units=3 engine=native
2026-10-16T10:02:10-0700 END job=clef/ref/p1 exit=0 records=50 expected=50 errors=0 unit_secs=120.0 elapsed=120.0 units=1 engine=native
2026-10-16T10:03:00-0700 END job=clef-flash-fp8/server exit=0 secs=170.1 util=0.16
2026-10-16T10:03:10-0700 END job=build/clef/fp8 exit=0 secs=508.8 size_gb=30.39
2026-10-16T10:03:20-0700 STATUS something else
"""


def wjson(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture()
def tree(tmp_path: Path):
    log = tmp_path / "sweep.log"
    log.write_text(LOG)
    res, priv = tmp_path / "results", tmp_path / "private"
    for model, cfg, sets in [("clef-flash", "ref", ["aaa", "bbb"]), ("clef-flash", "q8_0", ["aaa", "bbb"]),
                             ("clef", "ref", ["aaa"])]:
        for s in sets:
            (res / model / cfg).mkdir(parents=True, exist_ok=True)
            (res / model / cfg / f"{s}.jsonl").write_text("{}\n")
    # legacy sidecar (wall_seconds) for a set with no END line, current sidecar elsewhere
    wjson(res / "clef-flash" / "ref" / "aaa.jsonl.meta.json", {"records": 100, "wall_seconds": 50.04, "peak_memory_gb": 20.0})
    wjson(res / "clef-flash" / "ref" / "bbb.jsonl.meta.json", {"records": 200, "unit_wall_seconds_sum": 100.0})
    wjson(res / "clef-flash" / "q8_0" / "aaa.jsonl.meta.json", {"records": 100, "unit_wall_seconds_sum": 30.0})
    wjson(res / "clef-flash" / "q8_0" / "bbb.jsonl.meta.json", {"records": 200, "unit_wall_seconds_sum": 80.0})
    wjson(res / "clef" / "ref" / "aaa.jsonl.meta.json", {"records": 100, "unit_wall_seconds_sum": 150.0, "peak_memory_gb": 55.0})
    wjson(res / "clef-flash" / "ref" / "meta.json", {"file_size_gb": 19.0, "peak_memory_gb": 20.0, "engine": "transformers-native",
                                                      "batch_size": 1, "gdn_kernel": "torch"})
    wjson(res / "clef-flash" / "q8_0" / "meta.json", {"file_size_gb": 9.5, "peak_memory_gb": 12.0, "engine": "llama.cpp"})
    for model, cfg in [("clef-flash", "ref"), ("clef-flash", "q8_0"), ("clef", "ref")]:
        (priv / model / cfg).mkdir(parents=True, exist_ok=True)
        (priv / model / cfg / "p1.jsonl").write_text("{}\n")
    return log, res, priv


def build(tree):
    log, res, priv = tree
    return S.build(log, res, priv, ["aaa", "bbb"], ["p1"])


def test_parse_log_keeps_latest_exit_zero_end_line_per_set(tree):
    parsed = S.parse_log(tree[0])
    e = parsed["sets"][("clef-flash", "q8_0", "bbb")]
    assert e["unit_secs"] == 80.0               # the later exit=0 line, not the exit=1 line (999) or the earlier one (60)
    assert parsed["nonzero_exit"] == ["clef-flash/q8_0/bbb exit=1"]
    assert ("clef-flash", "fp8", "server") not in parsed["sets"]
    assert parsed["servers"][("clef-flash", "fp8")]["start_secs"] == 170.1
    assert parsed["n_end_lines"] == 12     # 8 set lines (incl. the exit=1 one), 1 server, 1 build, 2 for the 27B


def test_per_config_totals_exclude_build_and_server_lines(tree):
    r = build(tree)
    q8 = r["configs"]["clef-flash"]["q8_0"]
    assert q8["public"] == {"n_sets": 2, "records": 300, "unit_secs": 110.0, "sec_per_record": pytest.approx(110 / 300),
                            "records_per_sec": pytest.approx(300 / 110)}
    assert q8["private"]["records"] == 50 and q8["private"]["unit_secs"] == 20.0
    assert q8["all"]["records"] == 350 and q8["all"]["unit_secs"] == 130.0
    assert q8["server_start"] is None             # no server END line was logged for q8_0 in this synthetic log
    assert r["meta"]["n_set_end_lines_used"] == 8


def test_unit_secs_not_elapsed_is_used(tree):
    r = build(tree)
    assert r["configs"]["clef-flash"]["ref"]["per_set"]["aaa"]["unit_secs"] == 50.0   # elapsed was 900
    assert r["configs"]["clef"]["ref"]["per_set"]["aaa"]["unit_secs"] == 150.0        # elapsed was 9000


def test_sets_without_end_line_are_reported_and_excluded_from_totals(tree):
    log, res, priv = tree
    # drop the 9B reference's END line for set aaa
    text = "\n".join(l for l in LOG.splitlines() if "clef-flash/ref/aaa" not in l) + "\n"
    log.write_text(text)
    r = S.build(log, res, priv, ["aaa", "bbb"], ["p1"])
    ref = r["configs"]["clef-flash"]["ref"]
    assert ref["sets_without_end_line"] == ["aaa"]
    assert ref["public"]["records"] == 200
    assert {"model": "clef-flash", "config": "ref", "set": "aaa"} in r["missing_end_lines"]
    # the legacy sidecar key (wall_seconds) stands in only for the 27B/9B ratio, flagged as such
    row = [x for x in r["ratio_27b_over_9b"]["per_set"] if x["set"] == "aaa"][0]
    assert row["sec_per_record_9b"] == pytest.approx(50.04 / 100)
    assert "wall_seconds" in row["9b_source"] and "no END line" in row["9b_source"]
    assert row["ratio"] == pytest.approx(1.5 / 0.5004)
    assert r["ratio_27b_over_9b"]["fallback_sets"] == ["aaa"]


def test_ratio_27b_over_9b_per_set_and_pooled(tree):
    r = build(tree)["ratio_27b_over_9b"]
    by = {x["set"]: x for x in r["per_set"]}
    assert by["aaa"]["ratio"] == pytest.approx(1.5 / 0.5)           # 27B 150s/100 vs 9B 50s/100
    assert by["p1"]["ratio"] == pytest.approx((120 / 50) / (40 / 50))
    assert by["p1"]["kind"] == "private" and by["aaa"]["kind"] == "public"
    pooled = r["pooled_end_lines_only"]
    assert pooled["n_sets"] == 2 and pooled["records"] == 150
    assert pooled["pooled_ratio"] == pytest.approx((150 + 120) / (50 + 40))
    assert r["pooled_public_with_fallback"]["n_sets"] == 1
    assert "bbb" not in by    # the 27B has no result for bbb


def test_like_for_like_uses_only_sets_where_every_full_config_has_an_end_line(tree):
    log, res, priv = tree
    log.write_text("\n".join(l for l in LOG.splitlines() if "clef-flash/ref/bbb" not in l) + "\n")
    r = S.build(log, res, priv, ["aaa", "bbb"], ["p1"])
    lfl = r["like_for_like_full_sets"]
    assert lfl["sets"] == ["aaa", "p1"]                              # bbb has no 9B ref END line
    assert lfl["configs"]["ref"]["records"] == 150 and lfl["configs"]["q8_0"]["records"] == 150
    assert lfl["configs"]["ref"]["speed_vs_ref"] == pytest.approx(1.0)
    # ref: 90 s / 150 records; q8_0: 50 s / 150 records
    assert lfl["configs"]["q8_0"]["speed_vs_ref"] == pytest.approx(90 / 50)
    # ref over every set with the sidecar standing in for bbb (sidecar says 100 s for 200 records)
    assert lfl["ref_all_sets_with_sidecar_fallback"]["records"] == 350
    assert lfl["ref_all_sets_with_sidecar_fallback"]["unit_secs"] == pytest.approx(190.0)


def test_file_sizes_come_from_meta_json_and_missing_ones_are_reported(tree):
    r = build(tree)
    assert r["configs"]["clef-flash"]["ref"]["meta"]["file_size_gb"] == 19.0
    assert r["configs"]["clef-flash"]["q8_0"]["meta"]["file_size_gb"] == 9.5
    assert r["configs"]["clef"]["ref"]["meta"] is None
    assert {"model": "clef", "config": "ref"} in r["file_sizes_missing"]
    assert r["configs"]["clef"]["ref"]["peak_memory_gb_max_over_set_sidecars"] == 55.0
    assert r["build_lines_in_log"][0]["size_gb"] == 30.39


def test_sidecar_cross_check_and_markdown(tree):
    r = build(tree)
    assert r["sidecar_check"]["n_sets_compared"] >= 4
    assert r["sidecar_check"]["max_abs_diff_secs"] == pytest.approx(0.04)
    md = S.render_md(r)
    assert "Like-for-like" in md and "27B / 9B" in md and "Missing file sizes" in md


def test_cli_writes_files(tree, tmp_path):
    log, res, priv = tree
    recs, prec = tmp_path / "recs", tmp_path / "prec"
    recs.mkdir()
    prec.mkdir()
    for s in ("aaa", "bbb"):
        (recs / f"{s}.jsonl").write_text("")
    (prec / "p1.jsonl").write_text("")
    (prec / "p1_hosted_repeat.jsonl").write_text("")
    out = tmp_path / "out"
    assert S.main(["--log", str(log), "--results", str(res), "--private-results", str(priv), "--records", str(recs),
                   "--private-records", str(prec), "--out", str(out)]) == 0
    data = json.loads((out / "speed.json").read_text())
    assert data["configs"]["clef-flash"]["ref"]["private"]["n_sets"] == 1       # p1 counted as private, repeat file ignored
    assert (out / "speed.md").exists()


# ------------------------------------------------------------------------------ latency block

PUB = ("pa", "pb")
PRIV = ("r1", "r4b")


def lat_file(mean_scale: float, *, engine="vllm", alias_of=None, other_gpu=None, peak=10.0, size=9.0,
             source="src-x", with_gpu_key=True) -> dict:
    """Synthetic latency file: public set pa has n=100 mean 10*scale p50 8*scale, pb n=300 mean 20*scale p50 18*scale;
    private sets r1 (n=50, mean 100*scale) and r4b (n=150, mean 200*scale)."""
    def st(n, mean, p50, private):
        return {"n": n, "errors": 0, "p50_ms": p50 * mean_scale, "p95_ms": 2 * mean * mean_scale,
                "p99_ms": 3 * mean * mean_scale, "mean_ms": mean * mean_scale, "private": private,
                "input_tokens": {"mean": 5.0, "p50": 5.0, "max": 9}}
    d = {"engine": engine, "batch_size": 1, "concurrency": 1, "timing": "t", "sample_per_set": 300,
         "overall": {"n": 600, "errors": 0, "p50_ms": 15.0 * mean_scale, "p95_ms": 90.0 * mean_scale,
                     "p99_ms": 99.0 * mean_scale, "mean_ms": 77.0 * mean_scale, "decisions_per_sec": 1.5,
                     "wall_seconds": 1000.0},
         "per_set": {"pa": st(100, 10.0, 8.0, False), "pb": st(300, 20.0, 18.0, False),
                     "r1": st(50, 100.0, 90.0, True), "r4b": st(150, 200.0, 190.0, True)},
         "input_token_buckets": [], "peak_memory_gb": peak, "peak_memory_source": source, "file_size_gb": size,
         "measured_at": "2026-10-23T00:00:00+0000"}
    if with_gpu_key:
        d["other_gpu_processes_at_start"] = other_gpu if other_gpu is not None else {"count": 1, "total_mib": 4742}
    if alias_of:
        d.update({"alias_of": alias_of, "alias_note": "copy of ref"})
    return d


@pytest.fixture()
def lat_dir(tmp_path: Path) -> Path:
    root = tmp_path / "results-latency"
    wjson(root / "clef-flash" / "ref.json", lat_file(1.0, engine="transformers-native", with_gpu_key=False))
    wjson(root / "clef-flash" / "ref-r2.json", lat_file(1.0, engine="transformers-native", with_gpu_key=False,
                                                        alias_of="ref"))
    wjson(root / "clef-flash" / "q4_k_m.json", lat_file(0.5, engine="llama.cpp", other_gpu={"count": 2, "total_mib": 10448},
                                                         peak=5.0, size=4.0, source="src-y"))
    wjson(root / "clef" / "ref.json", lat_file(3.0, engine="transformers-native", with_gpu_key=False, peak=30.0, size=27.0))
    wjson(root / "clef" / "q4_k_m.json", lat_file(2.0, engine="llama.cpp", peak=15.0, size=12.0, source="src-y"))
    return root


def test_latency_missing_dir_says_so_and_does_not_fail(tmp_path):
    lat = S.build_latency(tmp_path / "nope")
    assert lat["available"] is False and "not found" in lat["note"]
    assert "models" not in lat or lat["models"] == {}
    assert S.build_latency(None)["available"] is False


def test_latency_overall_public_and_private_pooling(lat_dir):
    lat = S.build_latency(lat_dir)
    assert lat["available"] is True
    e = lat["models"]["clef-flash"]["q4_k_m"]
    assert e["overall"] == {"n": 600, "errors": 0, "p50_ms": 7.5, "p95_ms": 45.0, "p99_ms": 49.5, "mean_ms": 38.5,
                            "decisions_per_sec": 1.5, "wall_seconds": 1000.0}
    # pooled public mean is n-weighted over the per-set means: (100*5 + 300*10) / 400
    assert e["public"]["n_sets"] == 2 and e["public"]["n"] == 400
    assert e["public"]["mean_ms"] == pytest.approx(8.75)
    # median of the per-set p50s (4.0 and 9.0), not a pooled p50
    assert e["public"]["median_of_set_p50_ms"] == pytest.approx(6.5)
    # private: only pooled, (50*50 + 150*100) / 200
    assert e["private"]["n_sets"] == 2 and e["private"]["n"] == 200 and e["private"]["mean_ms"] == pytest.approx(87.5)
    assert "per_set" not in e and "sets" not in e["private"]
    assert e["peak_memory_gb"] == 5.0 and e["peak_memory_source"] == "src-y" and e["file_size_gb"] == 4.0
    assert e["other_gpu_processes_at_start"] == {"count": 2, "total_mib": 10448}
    assert e["alias_of"] is None
    assert "p50" in lat["meta"]["pooled_note"] and "median of the per-set p50" in lat["meta"]["pooled_note"]


def test_latency_alias_and_missing_gpu_field(lat_dir):
    lat = S.build_latency(lat_dir)
    r2 = lat["models"]["clef-flash"]["ref-r2"]
    assert r2["alias_of"] == "ref" and r2["alias_note"] == "copy of ref"
    ref = lat["models"]["clef-flash"]["ref"]
    assert ref["alias_of"] is None and ref["other_gpu_processes_at_start"] is None   # not recorded, not guessed


def test_latency_ratios(lat_dir):
    lat = S.build_latency(lat_dir)
    assert lat["models"]["clef-flash"]["q4_k_m"]["mean_ms_over_ref"] == pytest.approx(0.5)
    assert lat["models"]["clef"]["q4_k_m"]["mean_ms_over_ref"] == pytest.approx(2.0 / 3.0)
    assert lat["models"]["clef"]["ref"]["mean_ms_over_ref"] == pytest.approx(1.0)
    r = lat["ratio_27b_over_9b"]
    assert r["ref"] == pytest.approx(3.0) and r["q4_k_m"] == pytest.approx(4.0)
    assert "ref-r2" not in r                      # the 27B has no ref-r2 file here
    assert "mean_ms" in lat["meta"]["ratio_note"]


def test_latency_per_set_public_table_has_no_private_sets(lat_dir):
    lat = S.build_latency(lat_dir)
    t = lat["per_set_public_mean_ms"]["clef-flash"]
    assert set(t) == {"pa", "pb"}
    assert t["pa"] == {"ref": 10.0, "ref-r2": 10.0, "q4_k_m": 5.0}
    assert t["pb"]["q4_k_m"] == 10.0
    assert "r1" not in json.dumps(lat["per_set_public_mean_ms"])


def test_latency_config_without_ref_has_no_ratio_and_bad_file_is_reported(lat_dir):
    (lat_dir / "clef" / "ref.json").unlink()
    (lat_dir / "clef" / "q8_0.json").write_text("{not json")
    lat = S.build_latency(lat_dir)
    assert lat["models"]["clef"]["q4_k_m"]["mean_ms_over_ref"] is None
    assert lat["ratio_27b_over_9b"] == {"q4_k_m": pytest.approx(4.0)}     # no 27B ref file, so no ratio for ref
    assert lat["unreadable_files"] == ["clef/q8_0.json"]
    assert "q8_0" not in lat["models"]["clef"]


def test_latency_markdown_never_names_private_sets(lat_dir):
    lat = S.build_latency(lat_dir)
    md = "\n".join(S.render_latency_md(lat))
    assert "Isolated latency" in md and "pooled" in md and "median of the per-set p50" in md
    for name in PRIV:
        assert name not in md.replace("private", "")
    assert "pa" in md and "q4_k_m" in md
    assert "not found" in "\n".join(S.render_latency_md(S.build_latency(lat_dir / "nope")))


def test_cli_latency_flag_and_default_dir_missing(tree, tmp_path, lat_dir, monkeypatch):
    log, res, priv = tree
    recs, prec = tmp_path / "recs", tmp_path / "prec"
    recs.mkdir()
    prec.mkdir()
    for s in ("aaa", "bbb"):
        (recs / f"{s}.jsonl").write_text("")
    (prec / "p1.jsonl").write_text("")
    base = ["--log", str(log), "--results", str(res), "--private-results", str(priv), "--records", str(recs),
            "--private-records", str(prec)]
    out = tmp_path / "out"
    assert S.main(base + ["--out", str(out), "--latency", str(lat_dir)]) == 0
    data = json.loads((out / "speed.json").read_text())
    assert data["latency"]["models"]["clef"]["ref"]["overall"]["mean_ms"] == pytest.approx(231.0)
    assert "Isolated latency" in (out / "speed.md").read_text()
    out2 = tmp_path / "out2"
    monkeypatch.chdir(tmp_path / "recs")             # default --latency is relative and does not exist here
    assert S.main(base + ["--out", str(out2)]) == 0
    assert json.loads((out2 / "speed.json").read_text())["latency"]["available"] is False
    assert "not found" in (out2 / "speed.md").read_text()
