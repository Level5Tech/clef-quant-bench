"""Sharding, atomic claims, ordered merge, memory estimates and the launch rule (pure Python)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sweep import units  # noqa: E402


def test_chunk_ranges_cover_exactly_and_align_to_batch_multiple_of_8():
    for n in (1, 7, 8, 200, 1000, 3080, 5000, 7461):
        for chunk in (8, 64, 376, 1000):
            ranges = units.chunk_ranges(n, chunk)
            assert ranges[0][0] == 0 and ranges[-1][1] == n
            assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
            assert all(r[0] % 8 == 0 for r in ranges)  # ref-r2 batches of 8 must match the serial grouping


def test_chunk_ranges_balanced():
    ranges = units.chunk_ranges(5000, 376)
    sizes = [b - a for a, b in ranges]
    assert max(sizes) - min(sizes) <= 8 or sizes[-1] <= sizes[0]
    assert len(ranges) == 14


def test_chunk_size_scales_with_cost():
    heavy = units.chunk_size("clinc150", 1.0)
    light = units.chunk_size("mmlu", 1.0)
    assert heavy < light and heavy % 8 == 0 and light % 8 == 0
    assert units.chunk_size("clinc150", 3.0) < heavy  # 27B: slower records -> smaller units


def test_unit_id_and_paths_are_stable(tmp_path):
    u = units.Unit(set="banking77", start=0, end=376, private=False)
    assert u.id == "banking77__000000-000376"
    d = units.unit_dir(tmp_path, "clef-flash", "ref", u)
    assert d == tmp_path / "clef-flash" / "ref" / "banking77" / "000000-000376"
    assert units.unit_out(tmp_path, "clef-flash", "ref", u) == d / "banking77.jsonl"


def test_claim_is_exclusive_and_releasable(tmp_path):
    assert units.try_claim(tmp_path / "claims", "u1", "w0")
    assert not units.try_claim(tmp_path / "claims", "u1", "w1")
    assert units.try_claim(tmp_path / "claims", "u2", "w1")
    assert units.claimed_ids(tmp_path / "claims") == {"u1", "u2"}
    units.release_owned(tmp_path / "claims", "w0", keep=set())
    assert units.claimed_ids(tmp_path / "claims") == {"u2"}
    units.release_owned(tmp_path / "claims", "w1", keep={"u2"})
    assert units.claimed_ids(tmp_path / "claims") == {"u2"}
    units.clear_claims(tmp_path / "claims")
    assert units.claimed_ids(tmp_path / "claims") == set()


def _rows(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": i, "error": None}) + "\n" for i in ids))


def test_unit_complete_requires_count_and_valid_json(tmp_path):
    f = tmp_path / "u.jsonl"
    assert not units.unit_complete(f, 3)
    _rows(f, ["a", "b"])
    assert not units.unit_complete(f, 3)
    _rows(f, ["a", "b", "c"])
    assert units.unit_complete(f, 3)
    f.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "x')
    assert not units.unit_complete(f, 3)


def test_merge_concatenates_in_record_order_and_checks_ids(tmp_path):
    rec = tmp_path / "rec.jsonl"
    ids = [f"r{i}" for i in range(10)]
    rec.write_text("".join(json.dumps({"id": i}) + "\n" for i in ids))
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _rows(a, ids[:4]); _rows(b, ids[4:8]); _rows(c, ids[8:])
    out = tmp_path / "final" / "rec.jsonl"
    n = units.merge_units([a, b, c], out, rec)
    assert n == 10 and [json.loads(l)["id"] for l in out.read_text().splitlines()] == ids
    _rows(b, ids[4:7] + ["WRONG"])
    out2 = tmp_path / "final" / "bad.jsonl"
    with pytest.raises(ValueError):
        units.merge_units([a, b, c], out2, rec)
    assert not out2.exists()  # nothing half-written is left at the final path


def test_mem_estimates():
    assert units.native_mem_gb(19.06) == pytest.approx(31.56, abs=0.1)   # 19 GB weights + ~12 GB host, measured
    assert units.vllm_mem_gb(0.25) == pytest.approx(0.25 * 130.66 + 6.0, abs=0.1)
    assert units.llama_mem_gb(9.66) == pytest.approx(9.66 * 1.35 + 4.0, abs=0.1)
    # three native workers must never fit the free memory of this machine
    assert 3 * units.native_mem_gb(19.06) + units.HEADROOM_GB > 106


class W:
    def __init__(self, task, mem):
        self.task, self.mem_gb = task, mem


def test_next_launch_respects_priority_caps_slots_and_memory():
    t1 = units.TaskState("t1", cap=2, unclaimed=5, mem_gb=20.0)
    t2 = units.TaskState("t2", cap=2, unclaimed=5, mem_gb=30.0)
    t3 = units.TaskState("t3", cap=1, unclaimed=0, mem_gb=10.0)
    # empty pool: highest priority first
    assert units.next_launch([t1, t2, t3], [], budget_gb=100, slots=4).name == "t1"
    # t1 at cap -> next task
    assert units.next_launch([t1, t2, t3], [W("t1", 20), W("t1", 20)], 100, 4).name == "t2"
    # slots exhausted
    assert units.next_launch([t1, t2, t3], [W("t1", 20)], 100, 1) is None
    # memory: 20 used + 30 needed > 45 budget -> t2 refused; t1 (20) fits
    assert units.next_launch([t1, t2], [W("t1", 20)], 45, 4).name == "t1"
    assert units.next_launch([t2, t1], [W("t1", 20)], 45, 4).name == "t1"
    # nothing unclaimed -> nothing to launch
    t1.unclaimed = 0; t2.unclaimed = 0
    assert units.next_launch([t1, t2, t3], [], 100, 4) is None
    # never launch more workers than there are unclaimed units
    t1.unclaimed = 1
    assert units.next_launch([t1], [W("t1", 20)], 100, 4) is None
