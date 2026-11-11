import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sweep import proof  # noqa: E402


def _row(i, logits, probs=None, tokens=10, err=None):
    return {"id": f"r{i}", "error": err, "input_tokens": tokens,
            "answers": {"q": {"option_ids": ["a", "b"], "logits": logits, "probs": probs or [0.5, 0.5]}}}


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_identical_files_are_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    rows = [_row(i, [0.1 * i, -0.3]) for i in range(5)]
    _write(a, rows); _write(b, rows)
    r = proof.compare_exact(a, b)
    assert r["identical"] and r["max_abs_dlogit"] == 0.0 and r["rows"] == 5 and r["logits_compared"] == 10


def test_one_ulp_difference_is_detected_and_reported(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, [_row(0, [0.1, 0.2])])
    _write(b, [_row(0, [0.1, 0.2 + 1e-9])])
    r = proof.compare_exact(a, b)
    assert not r["identical"] and r["rows_with_any_diff"] == 1 and 0 < r["max_abs_dlogit"] < 1e-8


def test_probs_only_engines_compare_on_probs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, [_row(0, None, [0.7, 0.3])]); _write(b, [_row(0, None, [0.7, 0.3])])
    r = proof.compare_exact(a, b)
    assert r["identical"] and r["logits_available"] is False
    _write(b, [_row(0, None, [0.7000001, 0.2999999])])
    assert not proof.compare_exact(a, b)["identical"]


def test_id_or_length_mismatch_is_not_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, [_row(0, [0, 0]), _row(1, [0, 0])]); _write(b, [_row(0, [0, 0])])
    r = proof.compare_exact(a, b)
    assert not r["identical"] and r["id_mismatch"] == 1


def test_mix_units_stay_inside_smallest_set_and_are_8_aligned(tmp_path):
    from sweep.driver import Paths
    p = Paths.default(tmp_path)
    us = proof.mix_units(p, 4, "t")
    assert len(us) == 4 * len(proof.MIX)
    assert all(u["start"] % 8 == 0 and u["end"] <= 1000 for u in us)
    assert len({u["id"] for u in us}) == len(us)


def test_mem_guard_refuses_what_would_exhaust_memory(monkeypatch):
    from sweep import units as U
    monkeypatch.setattr(U, "mem_available_gb", lambda: 106.0)
    assert proof.mem_guard("native", 1)[0]
    assert proof.mem_guard("native", 2)[0]
    assert not proof.mem_guard("native", 3)[0]      # 3 x 31 x 1.5 + 12 = 151 GB: this is the run that took the box down
    assert not proof.mem_guard("native", 4)[0]
    assert proof.mem_guard("vllm", 2)[0] and proof.mem_guard("llamacpp", 3)[0]
