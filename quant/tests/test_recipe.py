"""Unit tests for quant/recipe.py (no GPU, no llm-compressor)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant import recipe  # noqa: E402


def test_ignore_patterns_cover_the_documented_set():
    names = {
        "lm_head": True,
        "model.visual.blocks.0.attn.qkv": True,
        "model.visual.merger.linear_fc1": True,
        "model.language_model.layers.0.linear_attn.in_proj_a": True,
        "model.language_model.layers.5.linear_attn.in_proj_b": True,
        # must stay quantized
        "model.language_model.layers.0.linear_attn.in_proj_qkv": False,
        "model.language_model.layers.0.linear_attn.in_proj_z": False,
        "model.language_model.layers.0.linear_attn.out_proj": False,
        "model.language_model.layers.3.self_attn.q_proj": False,
        "model.language_model.layers.3.mlp.down_proj": False,
    }
    for name, expected in names.items():
        assert recipe.is_ignored(name) is expected, name


def test_fused_pair_registered_for_gdn_qkvz():
    assert ("in_proj_qkv", "in_proj_z") in recipe.EXTRA_FUSED_LAYER_NAMES


def test_schemes_share_one_ignore_list():
    assert recipe.SCHEMES == {"FP8_DYNAMIC": False, "NVFP4": True}  # name -> needs calibration


def test_select_calibration_prefers_real_file(tmp_path):
    real = tmp_path / "calibration.jsonl"
    real.write_text(json.dumps({"id": "c-0", "request": {"state": "x", "questions": {}}}) + "\n")
    smoke = tmp_path / "smoke.jsonl"
    smoke.write_text("")
    path, is_fallback = recipe.select_calibration(real, smoke)
    assert path == real and not is_fallback


def test_select_calibration_falls_back_to_smoke_loudly(tmp_path, capsys):
    smoke = tmp_path / "smoke.jsonl"
    smoke.write_text(json.dumps({"id": "s-0", "request": {"state": "x", "questions": {}}}) + "\n")
    path, is_fallback = recipe.select_calibration(tmp_path / "missing.jsonl", smoke)
    assert path == smoke and is_fallback
    assert "FALLBACK" in capsys.readouterr().err


def test_select_calibration_errors_when_nothing_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        recipe.select_calibration(tmp_path / "a.jsonl", tmp_path / "b.jsonl")


def test_load_requests_reads_request_field(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps({"id": f"c-{i}", "request": {"state": i, "questions": {"q": {}}}})
                           for i in range(5)) + "\n")
    reqs = recipe.load_requests(p, num_samples=3)
    assert len(reqs) == 3 and reqs[0]["id"] == "c-0" and "questions" in reqs[0]


def test_load_requests_caps_at_available(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": "c-0", "request": {"state": 1, "questions": {}}}) + "\n")
    assert len(recipe.load_requests(p, num_samples=512)) == 1


def test_load_requests_is_task_balanced_round_robin(tmp_path):
    p = tmp_path / "c.jsonl"
    rows = [{"id": f"{t}-{i}", "task": t, "request": {"state": i, "questions": {}}}
            for t in ("a", "b", "c") for i in range(10)]  # sorted by task: a first
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = recipe.load_requests(p, num_samples=9)
    tasks = [r["id"].split("-")[0] for r in got]
    assert sorted(set(tasks)) == ["a", "b", "c"]
    assert all(tasks.count(t) == 3 for t in "abc")
    assert got == recipe.load_requests(p, num_samples=9)  # deterministic
