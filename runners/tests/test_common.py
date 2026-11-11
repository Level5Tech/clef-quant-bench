"""Unit tests for the runner-output interface (PLAN.md 'Shared interface'). No GPU needed."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runners import common, compare  # noqa: E402

SMOKE = Path(__file__).resolve().parents[2] / "data" / "smoke" / "smoke.jsonl"


def test_softmax_sums_to_one_and_is_stable():
    p = common.softmax([1000.0, 1001.0, 999.0])
    assert math.isclose(sum(p), 1.0, rel_tol=1e-12)
    assert p[1] > p[0] > p[2]


def test_softmax_full_precision_not_rounded():
    p = common.softmax([0.1234567, 0.0])
    assert p[0] != round(p[0], 4)


def test_read_records_smoke_schema():
    recs = list(common.read_records(SMOKE))
    assert len(recs) >= 5
    for r in recs:
        assert {"id", "task", "request", "gold"} <= set(r)
        assert "questions" in r["request"]


def test_read_records_limit():
    assert len(list(common.read_records(SMOKE, limit=2))) == 2


def test_make_row_has_plan_fields():
    option_ids = {"q": ["a", "b"]}
    answers = common.answers_from_logits(option_ids, {"q": [0.5, -0.5]})
    row = common.make_row(
        record_id="t-0", task="t", model="clef-flash", config="ref", answers=answers,
        latency_ms=12.5, input_tokens=100, engine="transformers-native", engine_version="5.16.1",
    )
    for key in ("id", "task", "model", "config", "answers", "latency_ms", "input_tokens",
                "engine", "engine_version", "error"):
        assert key in row
    assert row["answers"]["q"]["option_ids"] == ["a", "b"]
    assert row["answers"]["q"]["logits"] == [0.5, -0.5]
    assert math.isclose(sum(row["answers"]["q"]["probs"]), 1.0, rel_tol=1e-12)
    json.dumps(row)  # must be serialisable
    common.validate_row(row)


def test_answers_without_logits_keep_probs_and_mark_null():
    answers = common.answers_from_probs({"q": ["a", "b"]}, {"q": [0.7, 0.3]})
    assert answers["q"]["logits"] is None
    assert answers["q"]["probs"] == [0.7, 0.3]


def test_validate_row_rejects_length_mismatch():
    row = common.make_row("t-0", "t", "m", "c", {"q": {"option_ids": ["a"], "logits": None, "probs": [0.5, 0.5]}},
                          1.0, 1, "e", "1")
    with pytest.raises(ValueError):
        common.validate_row(row)


def _row(rid, logits):
    ans = common.answers_from_logits({"q": ["a", "b", "c"]}, {"q": logits})
    return common.make_row(rid, "t", "m", "c", ans, 1.0, 10, "e", "1")


def test_compare_identical_runs_are_deterministic():
    a = [_row("r0", [1.0, 2.0, 0.0]), _row("r1", [0.0, 0.0, 1.0])]
    b = [_row("r0", [1.0, 2.0, 0.0]), _row("r1", [0.0, 0.0, 1.0])]
    out = compare.compare(a, b)
    assert out["max_abs_logit_diff"] == 0.0
    assert out["max_abs_prob_diff"] == 0.0
    assert out["top1_agreement"] == 1.0
    assert out["questions"] == 2


def test_compare_detects_flip_and_differences():
    a = [_row("r0", [1.0, 2.0, 0.0])]
    b = [_row("r0", [2.5, 2.0, 0.0])]
    out = compare.compare(a, b)
    assert out["max_abs_logit_diff"] == pytest.approx(1.5)
    assert out["top1_agreement"] == 0.0
    assert out["max_abs_prob_diff"] > 0.1


def test_compare_logit_diff_none_when_logits_missing():
    a = [_row("r0", [1.0, 2.0, 0.0])]
    row = _row("r0", [1.0, 2.0, 0.0])
    row["answers"]["q"]["logits"] = None
    out = compare.compare(a, [row])
    assert out["max_abs_logit_diff"] is None
    assert out["max_abs_prob_diff"] == pytest.approx(0.0)


def test_compare_skips_errored_rows_and_reports_them():
    a = [_row("r0", [1.0, 2.0, 0.0])]
    bad = common.make_row("r0", "t", "m", "c", {}, 1.0, 10, "e", "1", error="boom")
    out = compare.compare(a, [bad])
    assert out["questions"] == 0
    assert out["errors"] == 1


def test_plan_outputs_single_file_and_dir():
    assert common.plan_outputs(["a/x.jsonl"], out="o/x.jsonl", out_dir=None) == [("a/x.jsonl", Path("o/x.jsonl"))]
    plan = common.plan_outputs(["a/x.jsonl", "b/y.jsonl"], out=None, out_dir="o")
    assert plan == [("a/x.jsonl", Path("o/x.jsonl")), ("b/y.jsonl", Path("o/y.jsonl"))]


def test_plan_outputs_rejects_ambiguous():
    with pytest.raises(ValueError):
        common.plan_outputs(["a.jsonl", "b.jsonl"], out="o.jsonl", out_dir=None)
    with pytest.raises(ValueError):
        common.plan_outputs(["a.jsonl"], out=None, out_dir=None)
    with pytest.raises(ValueError):
        common.plan_outputs(["a.jsonl"], out="o.jsonl", out_dir="o")


def test_compare_dirs_matches_files_by_name(tmp_path):
    from runners import compare_dirs

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    for d, logits in ((a, [1.0, 2.0, 0.0]), (b, [1.0, 2.0, 0.5])):
        (d / "t1.jsonl").write_text(json.dumps(_row("r0", logits)) + "\n")
    (a / "only_a.jsonl").write_text(json.dumps(_row("r1", [0.0, 1.0, 2.0])) + "\n")
    out = compare_dirs.compare_dirs(a, b)
    assert list(out["per_task"]) == ["t1"]
    assert out["overall"]["questions"] == 1 and out["overall"]["top1_agreement"] == 1.0
    assert out["overall"]["max_abs_logit_diff"] == pytest.approx(0.5)


def test_read_record_range_is_a_contiguous_slice(tmp_path):
    import json as _json
    p = tmp_path / "r.jsonl"
    p.write_text("".join(_json.dumps({"id": i}) + "\n" for i in range(10)))
    assert [r["id"] for r in common.read_record_range(p, 3, 7)] == [3, 4, 5, 6]
    assert [r["id"] for r in common.read_record_range(p, 8, 99)] == [8, 9]
    assert list(common.read_record_range(p, 5, 5)) == []
