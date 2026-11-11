"""Schema validation for records, plus the ordering / reproducibility guarantees of datasets/build.py.

The data-dependent tests run against the files produced by `datasets/build.py` and are skipped when
those files have not been built yet.
"""

import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from bench.schema import option_ids, validate_record

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RECORDS = sorted((DATA / "records").glob("*.jsonl")) if (DATA / "records").is_dir() else []

_spec = importlib.util.spec_from_file_location("clef_build", ROOT / "datasets" / "build.py")
build = importlib.util.module_from_spec(_spec)
sys.modules["clef_build"] = build  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(build)


def rec(task="toy", n="0", qtype="choice", gold="a", **over):
    q = {"choice": {"type": "choice", "instructions": "pick", "criteria": {"b": "B", "a": "A"}},
         "noul": {"type": "noul", "instructions": "is it so"},
         "score": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]}}[qtype]
    r = {"id": f"{task}-{n}", "task": task,
         "request": {"model": "clef", "state": {"x": 1}, "questions": {"answer": q}}, "gold": {"answer": gold}}
    r.update(over)
    return r


# ------------------------------------------------------------------- validator itself

def test_valid_records_of_each_type_pass():
    assert validate_record(rec(qtype="choice", gold="b")) == []
    assert validate_record(rec(qtype="noul", gold="false")) == []
    assert validate_record(rec(qtype="score", gold="2")) == []
    assert validate_record(rec(n="cal-3")) == []


def test_option_ids_follow_the_reference_encoder():
    assert option_ids({"type": "choice", "criteria": {"b": "", "a": "", "C": ""}}) == ["C", "a", "b"]  # plain sort
    assert option_ids({"type": "noul"}) == ["true", "false"]
    assert option_ids({"type": "score", "criteria": ["x", "y", "z"]}) == ["0", "1", "2"]


@pytest.mark.parametrize("bad,expect", [
    (rec(gold="zzz"), "not an allowed option"),
    (rec(qtype="score", gold="3"), "not an allowed option"),
    (rec(qtype="noul", gold="yes"), "not an allowed option"),
    (rec(id="other-0"), "does not start with task"),
    (rec(id="toy_0"), "does not match"),
    ({k: v for k, v in rec().items() if k != "gold"}, "missing key 'gold'"),
])
def test_invalid_records_are_rejected(bad, expect):
    assert any(expect in e for e in validate_record(bad)), validate_record(bad)


def test_other_structural_errors_are_rejected():
    r = rec()
    r["request"]["questions"]["answer"]["instructions"] = " "
    assert validate_record(r)
    r = rec()
    r["request"]["questions"]["answer"]["criteria"] = {"only": "one"}
    assert validate_record(r)
    r = rec(qtype="score")
    r["request"]["questions"]["answer"]["criteria"] = {"0": "x", "1": "y"}
    assert validate_record(r)
    r = rec(qtype="noul")
    r["request"]["questions"]["answer"]["criteria"] = {"maybe": "?"}
    assert validate_record(r)
    r = rec()
    r["gold"] = {"other": "a"}
    assert validate_record(r)
    r = rec()
    r["request"]["questions"]["answer"]["type"] = "ranking"
    assert validate_record(r)


# ----------------------------------------------------------------------- ordering

def test_stratified_order_is_a_permutation_and_prefix_stratified():
    labels = ["a"] * 60 + ["b"] * 30 + ["c"] * 10
    order = build.stratified_order(labels, random.Random(1))
    assert sorted(order) == list(range(100))
    share = {"a": 0.6, "b": 0.3, "c": 0.1}
    for k in (1, 7, 10, 25, 50, 99, 100):
        got = Counter(labels[i] for i in order[:k])
        for lab, s in share.items():
            assert abs(got[lab] - s * k) <= 1.0 + 1e-9, (k, lab, got[lab], s * k)


def test_stratified_order_is_deterministic_and_limit_is_a_prefix():
    labels = [random.Random(5).choice("abcd") for _ in range(200)]
    a = build.stratified_order(labels, random.Random("20261015:t:s"))
    b = build.stratified_order(labels, random.Random("20261015:t:s"))
    c = build.stratified_order(labels, random.Random("20261015:t:s"), limit=37)
    assert a == b and c == a[:37]
    assert a != build.stratified_order(labels, random.Random("other"))


def test_text_key_ignores_subject_and_formatting():
    a = build.text_key({"question": "What  is\nX?", "subject": "college physics"})
    b = build.text_key({"question": "what is x?", "subject": "something else"})
    assert a == b


def test_pinned_revisions_are_full_commit_hashes():
    for key, src in build.SOURCES.items():
        assert re.fullmatch(r"[0-9a-f]{40}", src.revision), key
    assert build.SEED == 20261015 and build.CAP == 5000


# ------------------------------------------------------------------- built data

needs_data = pytest.mark.skipif(not RECORDS, reason="run datasets/build.py first")


def _load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@needs_data
def test_every_generated_record_validates_and_follows_conventions():
    seen = set()
    for path in RECORDS:
        task = path.stem
        rows = _load(path)
        assert 0 < len(rows) <= build.CAP, task
        assert [r["id"] for r in rows] == [f"{task}-{i}" for i in range(len(rows))], f"{task}: ids not sequential"
        for r in rows:
            assert r["task"] == task
            assert validate_record(r) == [], (r["id"], validate_record(r))
            assert r["request"]["model"] == "clef"
            assert r["id"] not in seen
            seen.add(r["id"])
        # instruction wording is identical across a task's records (the same file feeds every engine)
        assert len({r["request"]["questions"]["answer"]["instructions"] for r in rows}) == 1, task


@needs_data
def test_calibration_is_valid_train_only_and_disjoint_from_scored():
    calib = _load(DATA / "calibration.jsonl")
    assert 700 <= len(calib) <= 800
    scored_ids, scored_keys = set(), set()
    for path in RECORDS:
        for r in _load(path):
            scored_ids.add(r["id"])
            scored_keys.add(build.text_key(r["request"]["state"]))
    keys = []
    for r in calib:
        assert validate_record(r) == [], r["id"]
        assert "-cal-" in r["id"]
        assert r["id"] not in scored_ids
        keys.append(build.text_key(r["request"]["state"]))
    assert not set(keys) & scored_keys, "calibration text overlaps a scored record"
    assert len(set(keys)) == len(keys)
    assert len({r["task"] for r in calib}) == len(RECORDS), "calibration should cover every task"


@needs_data
def test_record_ids_cover_every_record_once_with_source_splits():
    ids = _load(DATA / "record_ids.jsonl")
    all_ids = [r["id"] for path in RECORDS for r in _load(path)] + [r["id"] for r in _load(DATA / "calibration.jsonl")]
    assert sorted(x["id"] for x in ids) == sorted(all_ids)
    by_task_split = defaultdict(set)
    for x in ids:
        assert {"task", "kind", "dataset", "revision", "source_split", "source_index", "release"} <= set(x)
        assert re.fullmatch(r"[0-9a-f]{40}", x["revision"])
        assert isinstance(x["source_index"], int)
        by_task_split[(x["task"], x["kind"])].add(x["source_split"])
        if x["kind"] == "calibration":
            assert "train" in x["source_split"], x["id"]
        else:
            assert "train" not in x["source_split"], x["id"]
    # a source row appears at most once per (task, kind)
    pairs = Counter((x["task"], x["kind"], x["source_split"], x["source_index"]) for x in ids)
    assert max(pairs.values()) == 1


@needs_data
def test_manifest_hashes_match_files():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["seed"] == build.SEED and manifest["cap"] == build.CAP
    for rel, digest in manifest["sha256"].items():
        assert hashlib.sha256((DATA / rel).read_bytes()).hexdigest() == digest, rel


@needs_data
def test_dev_subset_is_a_prefix_of_the_full_build_when_present():
    dev = ROOT / "data_dev" / "records"
    if not dev.is_dir():
        pytest.skip("no dev build")
    for path in sorted(dev.glob("*.jsonl")):
        full = (DATA / "records" / path.name).read_text(encoding="utf-8").splitlines()
        sub = path.read_text(encoding="utf-8").splitlines()
        assert sub == full[: len(sub)], path.name
