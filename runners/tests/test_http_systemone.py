"""Unit tests for the pure parts of runners/http_systemone.py (response parsing, rounding detection)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runners import common, http_systemone as hs  # noqa: E402

REQUEST = {
    "model": "clef",
    "state": "x",
    "questions": {
        "dept": {"type": "choice", "instructions": "?", "criteria": {"b": "B", "a": "A", "c": "C"}},
        "sc": {"type": "score", "criteria": ["lo", "mid", "hi"]},
        "yn": {"type": "noul"},
    },
}


def test_option_ids_follow_cloudflare_order():
    ids = hs.question_option_ids(REQUEST)
    assert ids == {"dept": ["a", "b", "c"], "sc": ["0", "1", "2"], "yn": ["true", "false"]}


def test_raw_block_gives_logits_and_full_precision_probs():
    resp = {"answers": {}, "raw": {
        "dept": {"option_ids": ["a", "b", "c"], "logits": [0.1234567, -1.0, 2.0]},
        "sc": {"option_ids": ["0", "1", "2"], "logits": [0.0, 0.0, 1.0]},
        "yn": {"option_ids": ["true", "false"], "logits": [3.0, 1.0]},
    }, "usage": {"input_tokens": 50}}
    answers, rounded = hs.parse_response(REQUEST, resp)
    assert rounded is False
    assert answers["dept"]["logits"] == [0.1234567, -1.0, 2.0]
    assert answers["dept"]["probs"] == common.softmax([0.1234567, -1.0, 2.0])


def test_probabilities_only_response_has_null_logits():
    resp = {"answers": {
        "dept": {"type": "choice", "choice": "a", "confidence": 0.5,
                 "probabilities": {"c": 0.2, "a": 0.5123456789, "b": 0.2876543211}},
        "sc": {"type": "score", "score": 1.0, "confidence": 0.5, "legend": {},
               "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25}},
        "yn": {"type": "noul", "noul": 0.9123456789},
    }}
    answers, rounded = hs.parse_response(REQUEST, resp)
    assert answers["dept"]["logits"] is None
    assert answers["dept"]["option_ids"] == ["a", "b", "c"]
    assert answers["dept"]["probs"] == [0.5123456789, 0.2876543211, 0.2]
    assert answers["yn"]["probs"] == pytest.approx([0.9123456789, 1 - 0.9123456789])
    assert rounded is False  # more than 4 decimals present somewhere


def test_rounded_probabilities_are_flagged():
    resp = {"answers": {
        "dept": {"probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}},
        "sc": {"probabilities": {"0": 0.25, "1": 0.5, "2": 0.25}},
        "yn": {"noul": 0.9123},
    }}
    _, rounded = hs.parse_response(REQUEST, resp)
    assert rounded is True


def test_missing_question_in_response_raises():
    with pytest.raises(ValueError):
        hs.parse_response(REQUEST, {"answers": {}})


def test_meta_has_orchestrator_keys():
    meta = hs.build_meta(engine="vllm", engine_version="0.29", file_size_gb=1.5, peak_memory_gb=None,
                         extra={"records": 3})
    for key in ("file_size_gb", "peak_memory_gb", "engine", "engine_version"):
        assert key in meta
    assert meta["records"] == 3


def test_dir_size_gb(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"0" * 1000)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.safetensors").write_bytes(b"0" * 500)
    (tmp_path / "note.md").write_bytes(b"0" * 5000)
    assert hs.weights_size_gb(tmp_path) == pytest.approx(1500 / 1e9)


def test_parse_nvidia_smi_apps():
    text = "2428386, 6430 MiB\n2670073, 4142 MiB\n"
    assert hs.parse_nvidia_smi_apps(text) == {2428386: 6430 * 2**20, 2670073: 4142 * 2**20}
    assert hs.parse_nvidia_smi_apps("") == {}
    assert hs.parse_nvidia_smi_apps("2428386, [N/A]\n") == {}
