"""Tests of bench.restart_probes on small synthetic trees (modelled on tests/test_decompose.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

from bench import restart_probes as R

UNIT = 20
BASE = np.array([2.0, 1.0, 0.0])
FLIPPED = np.array([0.0, 2.0, 1.0])
MODEL = "clef"


def softmax(z):
    e = np.exp(np.asarray(z, float) - np.max(z))
    return e / e.sum()


def make_records(root: Path) -> Path:
    rec = root / "records"
    rec.mkdir(parents=True)
    for task in ("toy_a", "toy_b"):
        rows = [{"id": f"{task}-{i}", "task": task,
                 "request": {"model": MODEL, "state": {"t": i},
                             "questions": {"q": {"type": "choice", "instructions": "pick",
                                                 "criteria": {"c": "3", "a": "1", "b": "2"}}}},
                 "gold": {"q": "a"}} for i in range(UNIT + 5)]
        (rec / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rec


def write_cfg(root: Path, cfg: str, flipped=frozenset(), nudged=frozenset(), nudge=0.0):
    """Unit indices 0..19 are toy_a, 20..39 toy_b. `nudged` records get a tiny logit change of size `nudge`."""
    for ti, task in enumerate(("toy_a", "toy_b")):
        rows = []
        for i in range(UNIT):
            ui = ti * UNIT + i
            lg = (FLIPPED if ui in flipped else BASE).copy()
            if ui in nudged:
                lg = lg + np.array([nudge, 0.0, 0.0])
            ans = {"option_ids": ["a", "b", "c"], "logits": lg.tolist(), "probs": softmax(lg).tolist()}
            rows.append({"id": f"{task}-{i}", "task": task, "model": MODEL, "config": cfg, "answers": {"q": ans},
                         "engine": "toy", "engine_version": "1", "error": None})
        d = root / "results" / MODEL / cfg
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def build(tmp_path, spec):
    rec = make_records(tmp_path)
    for cfg, kw in spec.items():
        write_cfg(tmp_path, cfg, **kw)
    return rec, tmp_path / "results"


def run(rec, results, formats=("fp8",)):
    return R.probe(rec, results, MODEL, unit_n=UNIT, n_boot=200, formats=formats)


def test_two_instances_one_pair(tmp_path, capsys):
    rec, results = build(tmp_path, {"ref": {}, "fp8": {"flipped": {0, 1, 2}},
                                    "fp8-r2": {"flipped": {0, 1, 2, 5}}})
    e = run(rec, results)["formats"]["fp8"]
    assert e["instances"] == ["fp8", "fp8-r2"] and e["missing"] == ["fp8-r3"]
    assert "fp8-r3" in capsys.readouterr().err
    assert list(e["pairs"]) == ["fp8|fp8-r2"]
    assert e["pairs"]["fp8|fp8-r2"]["flips"] == 1
    assert e["pairs"]["fp8|fp8-r2"]["n_records"] == 2 * UNIT
    assert e["vs_ref"]["fp8"]["flips"] == 3 and e["vs_ref"]["fp8-r2"]["flips"] == 4
    assert (e["vs_ref_flips_min"], e["vs_ref_flips_max"]) == (3, 4)
    assert "r3_vs_base_and_r2" not in e["discreteness"]
    assert e["discreteness"]["base_vs_r2_context"]["identical"]["n"] == 2 * UNIT - 1


def test_three_instances_all_pairs(tmp_path):
    rec, results = build(tmp_path, {"ref": {}, "fp8": {"flipped": {0, 1, 2}},
                                    "fp8-r2": {"flipped": {0, 1, 2, 5}},
                                    "fp8-r3": {"flipped": {0, 1, 7, 8}}})
    e = run(rec, results)["formats"]["fp8"]
    assert e["missing"] == []
    assert list(e["pairs"]) == ["fp8|fp8-r2", "fp8|fp8-r3", "fp8-r2|fp8-r3"]
    assert [e["pairs"][k]["flips"] for k in e["pairs"]] == [1, 3, 4]
    assert [e["vs_ref"][n]["flips"] for n in e["vs_ref"]] == [3, 4, 4]
    assert (e["vs_ref_flips_min"], e["vs_ref_flips_max"]) == (3, 4)


def test_fewer_than_two_instances_is_an_error(tmp_path):
    rec, results = build(tmp_path, {"ref": {}, "fp8": {}})
    with pytest.raises(SystemExit, match="at least 2"):
        run(rec, results)


def test_discreteness_four_way_split(tmp_path):
    # base: records 0-3 nudged (a). r2: 0,1 nudged same as base?  Build explicit equality classes:
    #  rec 0,1: r3 == base only   (base & r3 share nudge 0.5, r2 nudge 0.25)
    #  rec 2,3: r3 == r2 only     (r2 & r3 share nudge 0.25, base nudge 0.5)
    #  rec 4,5,6: all three equal (no nudge anywhere)  -- plus every other record, see below
    #  rec 7:   all three differ
    spec = {"ref": {}}
    base_n, r2_n, r3_n = {}, {}, {}
    for i, (b, r2, r3) in {0: (.5, .25, .5), 1: (.5, .25, .5), 2: (.5, .25, .25), 3: (.5, .25, .25),
                           7: (.1, .2, .3)}.items():
        base_n[i], r2_n[i], r3_n[i] = b, r2, r3

    def mk(d):  # per-record nudge sizes; write_cfg takes one size, so write then patch
        return set(d)
    for name, d in (("fp8", base_n), ("fp8-r2", r2_n), ("fp8-r3", r3_n)):
        spec[name] = {"nudged": mk(d)}
    rec, results = build(tmp_path, spec)
    # patch per-record nudge sizes into the files
    for name, d in (("fp8", base_n), ("fp8-r2", r2_n), ("fp8-r3", r3_n)):
        p = results / MODEL / name / "toy_a.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        for i, size in d.items():
            lg = (BASE + np.array([size, 0, 0])).tolist()
            rows[i]["answers"]["q"]["logits"] = lg
            rows[i]["answers"]["q"]["probs"] = softmax(lg).tolist()
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    e = run(rec, results)["formats"]["fp8"]
    t = e["discreteness"]["r3_vs_base_and_r2"]
    n = 2 * UNIT
    assert t["n_records"] == n
    assert t["equals_base_only"]["n"] == 2
    assert t["equals_r2_only"]["n"] == 2
    assert t["equals_neither"]["n"] == 1
    assert t["equals_both"]["n"] == n - 5
    assert t["equals_both"]["share"] == pytest.approx((n - 5) / n)
    assert sum(t[k]["n"] for k in t if k != "n_records") == n
    # base vs r2 context: records 0,1,2,3,7 differ
    assert e["discreteness"]["base_vs_r2_context"]["different"]["n"] == 5


def test_missing_r2_only_base_and_r3_still_runs(tmp_path):
    rec, results = build(tmp_path, {"ref": {}, "fp8": {}, "fp8-r3": {"flipped": {3}}})
    e = run(rec, results)["formats"]["fp8"]
    assert e["instances"] == ["fp8", "fp8-r3"] and e["missing"] == ["fp8-r2"]
    assert e["pairs"]["fp8|fp8-r3"]["flips"] == 1
    assert e["discreteness"]["base_vs_r2_context"] is None


def test_cli_writes_json_and_md(tmp_path):
    rec, results = build(tmp_path, {"ref": {}, "fp8": {}, "fp8-r2": {}, "nvfp4": {"flipped": {1}}, "nvfp4-r2": {}})
    out = tmp_path / "out"
    rc = R.main(["--records-dir", str(rec), "--results-dir", str(results), "--model", MODEL,
                 "--out", str(out), "--unit-n", str(UNIT), "--n-boot", "100"])
    assert rc == 0
    md = (out / "restart_probes.md").read_text()
    assert md.splitlines()[2].startswith("Exploratory (not pre-registered). Third vLLM instance per format")
    assert "fp8 vs fp8-r2" in md and "nvfp4 vs nvfp4-r2" in md
    assert json.loads((out / "restart_probes.json").read_text())["exploratory_not_preregistered"] is True


def test_position_split_counts_each_task_separately():
    tasks = ["a"] * 4 + ["b"] * 4
    mask = np.array([1, 0, 1, 1, 0, 0, 0, 1], bool)
    assert R.position_split(tasks, mask) == {"n": 4, "first_half": 1, "second_half": 3}


def test_position_split_odd_task_puts_the_middle_in_the_second_half():
    assert R.position_split(["a"] * 3, np.array([0, 1, 0], bool)) == {"n": 1, "first_half": 0, "second_half": 1}
