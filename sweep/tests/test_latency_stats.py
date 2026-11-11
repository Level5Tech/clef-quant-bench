"""Pure-Python latency statistics: percentiles, sampling, buckets, report shape (no private ids)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sweep.latency import stats  # noqa: E402


def test_percentile_linear_interpolation():
    vals = [10.0, 20.0, 30.0, 40.0]
    assert stats.percentile(vals, 0) == 10.0
    assert stats.percentile(vals, 100) == 40.0
    assert stats.percentile(vals, 50) == pytest.approx(25.0)
    assert stats.percentile([5.0], 99) == 5.0
    with pytest.raises(ValueError):
        stats.percentile([], 50)


def test_percentile_matches_numpy_convention():
    vals = [float(x) for x in range(1, 101)]
    assert stats.percentile(vals, 95) == pytest.approx(95.05)
    assert stats.percentile(vals, 99) == pytest.approx(99.01)


def test_sample_indices_deterministic_sorted_and_capped():
    a = stats.sample_indices(1000, 300, "banking77")
    b = stats.sample_indices(1000, 300, "banking77")
    assert a == b and a == sorted(a) and len(set(a)) == 300
    assert stats.sample_indices(1000, 300, "clinc150") != a
    assert stats.sample_indices(200, 300, "x") == list(range(200))  # smaller than k: everything


def test_bucket_labels():
    assert stats.bucket_label(1) == "<=128"
    assert stats.bucket_label(128) == "<=128"
    assert stats.bucket_label(129) == "129-256"
    assert stats.bucket_label(4096) == "2049-4096"
    assert stats.bucket_label(16384) == "8193-16384"
    assert stats.bucket_label(99999) == ">16384"
    assert stats.bucket_label(None) == "unknown"


def test_summarize_counts_errors_and_decisions_per_sec():
    s = stats.summarize([100.0, 200.0, 300.0], errors=1, wall_seconds=1.5)
    assert s["n"] == 3 and s["errors"] == 1
    assert s["p50_ms"] == 200.0
    assert s["decisions_per_sec"] == pytest.approx(2.0)
    assert s["mean_ms"] == pytest.approx(200.0)
    empty = stats.summarize([], errors=2, wall_seconds=1.0)
    assert empty["n"] == 0 and empty["p50_ms"] is None and empty["decisions_per_sec"] is None


def test_warmup_order_round_robin_across_sets():
    items = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"], "c": ["c1"]}
    assert stats.warmup_order(items, 5) == ["a1", "b1", "c1", "a2", "b2"]
    assert len(stats.warmup_order(items, 20)) == 6  # cannot exceed what exists


def _obs(set_name, private, tokens, ms, err=None):
    return {"set": set_name, "private": private, "tokens": tokens, "ms": ms, "error": err}


def test_report_has_overall_per_set_and_buckets_without_ids():
    obs = [_obs("anli_r1", False, 100, 10.0), _obs("anli_r1", False, 300, 30.0),
           _obs("r1", True, 600, 60.0), _obs("r1", True, None, None, "boom")]
    rep = stats.build_report(obs, wall_seconds=2.0)
    assert rep["overall"]["n"] == 3 and rep["overall"]["errors"] == 1
    assert rep["per_set"]["anli_r1"]["n"] == 2 and rep["per_set"]["anli_r1"]["private"] is False
    assert rep["per_set"]["r1"]["private"] is True
    labels = {b["label"]: b for b in rep["input_token_buckets"]}
    assert labels["<=128"]["n"] == 1 and labels["257-512"]["n"] == 1 and labels["513-1024"]["n"] == 1
    assert rep["per_set"]["anli_r1"]["input_tokens"]["max"] == 300
    text = json.dumps(rep)
    assert "boom" not in text  # error strings never enter the aggregate report
