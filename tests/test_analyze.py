"""End-to-end tests of bench.analyze on small synthetic result trees with hand-checkable numbers."""

import json
import math
from pathlib import Path

import pytest

from bench import analyze as A
from bench import metrics as M

REF_RIGHT = [0.9, 0.05, 0.05]
WRONG_B = [0.2, 0.6, 0.2]


def choice_record(i: str, gold: str = "a", text: str = "toy text", tokens_marker: int = 0) -> dict:
    return {
        "id": f"toychoice-{i}", "task": "toychoice",
        "request": {"model": "clef", "state": {"text": f"{text} {i}"},
                    "questions": {"answer": {"type": "choice", "instructions": "pick one",
                                             "criteria": {"c": "third", "a": "first", "b": "second"}}}},
        "gold": {"answer": gold},
    }


def score_record(i: str, gold: str = "1") -> dict:
    return {
        "id": f"toyscore-{i}", "task": "toyscore",
        "request": {"model": "clef", "state": {"text": f"s {i}"},
                    "questions": {"answer": {"type": "score", "instructions": "rate",
                                             "criteria": ["low", "mid", "high"]}}},
        "gold": {"answer": gold},
    }


def result_line(rec: dict, probs, model="clef", config="ref", tokens=10, latency=100.0, logits=None) -> dict:
    qid = "answer"
    q = rec["request"]["questions"][qid]
    opts = sorted(q["criteria"]) if q["type"] == "choice" else [str(k) for k in range(len(q["criteria"]))]
    return {"id": rec["id"], "task": rec["task"], "model": model, "config": config,
            "answers": {qid: {"option_ids": opts, "logits": logits, "probs": list(probs)}},
            "latency_ms": latency, "input_tokens": tokens, "engine": "test", "engine_version": "0", "error": None}


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# Probabilities in option order [a, b, c].
def ref_probs(i: int):
    return REF_RIGHT if i < 8 else WRONG_B


def cfg_probs(i: int):
    if i == 7:
        return [0.3, 0.5, 0.2]  # right -> wrong flip
    if i == 8:
        return [0.6, 0.2, 0.2]  # wrong -> right flip
    return ref_probs(i)


@pytest.fixture()
def tree(tmp_path: Path):
    records = tmp_path / "records"
    results = tmp_path / "results"
    recs = [choice_record(str(i)) for i in range(10)]
    write_jsonl(records / "toychoice.jsonl", recs)
    srecs = [score_record(str(i)) for i in range(4)]
    write_jsonl(records / "toyscore.jsonl", srecs)

    # model clef: ref, ref-r2 (tiny perturbation, no flips), q8 (two flips), q4 (choice task only)
    for cfg, fn, tok in [("ref", ref_probs, 10), ("ref-r2", None, 10), ("q8", cfg_probs, 10), ("q4", cfg_probs, 10)]:
        rows = []
        for i, r in enumerate(recs):
            if cfg == "ref-r2":
                p = list(ref_probs(i))
                if i == 0:
                    p = [0.89, 0.06, 0.05]
            else:
                p = fn(i)
            rows.append(result_line(r, p, config=cfg, tokens=i + 1, latency=100.0 + i))
        write_jsonl(results / "clef" / cfg / "toychoice.jsonl", rows)

    # score task: ref expects mid (E=1.0); q8 shifts toward the high end (E=1.5)
    for cfg, p in [("ref", [0.0, 1.0, 0.0]), ("q8", [0.0, 0.5, 0.5])]:
        write_jsonl(results / "clef" / cfg / "toyscore.jsonl",
                    [result_line(r, p, config=cfg) for r in srecs])

    # model clef-flash: no ref at all
    write_jsonl(results / "clef-flash" / "q8" / "toychoice.jsonl",
                [result_line(r, ref_probs(i), model="clef-flash", config="q8") for i, r in enumerate(recs)])

    # sizes for equal-memory pairing
    (results / "clef" / "ref" / "meta.json").write_text(json.dumps({"file_size_gb": 54.0}))
    (results / "clef" / "q4" / "meta.json").write_text(json.dumps({"file_size_gb": 17.0}))
    (results / "clef" / "q8" / "meta.json").write_text(json.dumps({"file_size_gb": 28.0}))
    (results / "clef-flash" / "q8" / "meta.json").write_text(json.dumps({"file_size_gb": 17.5}))
    return records, results


def run(tree, **kw):
    records, results = tree
    return A.analyze(records, results, n_boot=200, **kw)


def test_ref_absolute_metrics_hand_computed(tree):
    s = run(tree)
    ref = s["models"]["clef"]["configs"]["ref"]["tasks"]["toychoice"]
    assert ref["n_scored"] == 10
    assert ref["accuracy"] == pytest.approx(0.8)
    # brier: right items (0.9,0.05,0.05)->0.01+0.0025+0.0025=0.015 ; wrong items (0.2,0.6,0.2)->0.64+0.36+0.04=1.04
    assert ref["brier"] == pytest.approx((8 * 0.015 + 2 * 1.04) / 10)
    assert "vs_ref" not in ref


def test_flips_mcnemar_kl_tvd_vs_ref(tree):
    s = run(tree)
    q8 = s["models"]["clef"]["configs"]["q8"]["tasks"]["toychoice"]
    v = q8["vs_ref"]
    assert v["n_paired"] == 10
    assert v["agreement"] == pytest.approx(0.8)
    assert v["n_flips"] == 2
    assert v["delta_accuracy"]["delta"] == pytest.approx(0.0)
    assert (v["mcnemar"]["b"], v["mcnemar"]["c"]) == (1, 1)
    assert v["mcnemar"]["p"] == pytest.approx(1.0)
    kl7 = M.kl_div(REF_RIGHT, [0.3, 0.5, 0.2])
    kl8 = M.kl_div(WRONG_B, [0.6, 0.2, 0.2])
    assert v["kl_mean"] == pytest.approx((kl7 + kl8) / 10)
    tv7 = 0.5 * (0.6 + 0.45 + 0.15)
    tv8 = 0.5 * (0.4 + 0.4 + 0.0)
    assert v["tvd_mean"] == pytest.approx((tv7 + tv8) / 10)


def test_flip_rate_by_reference_confidence_bucket(tree):
    v = run(tree)["models"]["clef"]["configs"]["q8"]["tasks"]["toychoice"]["vs_ref"]
    by = {b["bucket"]: b for b in v["flip_by_ref_confidence"]}
    # ref confidences: items 0-7 -> 0.9 (bucket [0.9,0.99)), items 8-9 -> 0.6 ([0.5,0.7)); flips: item 7 and 8
    assert by["[0.9,0.99)"]["n"] == 8 and by["[0.9,0.99)"]["flips"] == 1
    assert by["[0.5,0.7)"]["n"] == 2 and by["[0.5,0.7)"]["flips"] == 1


def test_noise_floor_from_repeat_run(tree):
    s = run(tree)
    nf = s["models"]["clef"]["noise_floor"]["ref"]["ref-r2"]["tasks"]["toychoice"]
    assert nf["n_flips"] == 0 and nf["agreement"] == 1.0
    assert nf["kl_mean"] == pytest.approx(M.kl_div(REF_RIGHT, [0.89, 0.06, 0.05]) / 10)
    # the repeat is not listed as an ordinary config comparison
    assert "ref-r2" not in s["models"]["clef"]["configs"]
    # and q8 is compared against that noise floor
    v = s["models"]["clef"]["configs"]["q8"]["tasks"]["toychoice"]["vs_ref"]
    assert v["noise_floor_max_kl"] == pytest.approx(nf["kl_mean"])
    assert v["exceeds_noise_floor"] is True


def test_missing_config_task_is_reported_not_fabricated(tree):
    s = run(tree)
    q4 = s["models"]["clef"]["configs"]["q4"]
    assert "toyscore" not in q4["tasks"]
    assert {"model": "clef", "config": "q4", "task": "toyscore", "reason": "no results file"} in s["absent"]["missing"]


def test_model_without_ref_has_absolute_metrics_only(tree):
    s = run(tree)
    assert "clef-flash" in s["absent"]["models_without_ref"]
    blk = s["models"]["clef-flash"]["configs"]["q8"]["tasks"]["toychoice"]
    assert blk["accuracy"] == pytest.approx(0.8)
    assert "vs_ref" not in blk


def test_pooled_block_combines_tasks(tree):
    s = run(tree)
    pooled = s["models"]["clef"]["configs"]["q8"]["pooled"]
    assert pooled["n_scored"] == 14
    v = pooled["vs_ref"]
    assert v["n_paired"] == 14
    # 10 choice items with 2 flips; the 4 score items shift probabilities but top-1 flips (mid -> tie a b) none
    assert v["n_flips"] == 2 + 0
    lo, hi = v["delta_accuracy"]["ci95"]
    assert lo <= v["delta_accuracy"]["delta"] <= hi


def test_score_task_signed_shift(tree):
    sh = run(tree)["models"]["clef"]["configs"]["q8"]["tasks"]["toyscore"]["score_shift"]
    assert sh["mean_signed_shift"] == pytest.approx(0.5)
    assert sh["center_pull"] == pytest.approx(0.5)  # ref sits in the middle, q8 moves toward an end


def test_length_quartiles_partition_the_paired_items(tree):
    v = run(tree)["models"]["clef"]["configs"]["q8"]["by_length_quartile"]["toychoice"]
    assert [b["quartile"] for b in v["buckets"]] == ["Q1", "Q2", "Q3", "Q4"]
    assert sum(b["n"] for b in v["buckets"]) == 10
    assert len(v["token_edges"]) == 3


def test_question_type_breakdown(tree):
    bq = run(tree)["models"]["clef"]["configs"]["q8"]["by_question_type"]
    assert set(bq) == {"choice", "score"}
    assert bq["choice"]["n_paired"] == 10 and bq["score"]["n_paired"] == 4


def test_cascade_simulation(tree):
    rows = run(tree)["models"]["clef"]["configs"]["q8"]["cascade"]["toychoice"]
    by = {r["threshold"]: r for r in rows}
    assert by[0.5]["escalation_rate"] == 0.0
    # q8 confidences: items 0-6 0.9, item 7 0.5, item 8 0.6, item 9 0.6; at t=0.95 everything escalates
    assert by[0.95]["escalation_rate"] == 1.0
    assert by[0.95]["agreement_with_ref"] == 1.0
    assert by[0.95]["accuracy"] == pytest.approx(0.8)


def test_latency_block(tree):
    lat = run(tree)["models"]["clef"]["configs"]["ref"]["tasks"]["toychoice"]["latency"]
    assert lat["p50_ms"] == pytest.approx(104.5)


def test_equal_memory_pairs_across_models_by_file_size(tree):
    em = run(tree, mem_tolerance=0.10)["equal_memory"]
    pairs = {(p["a"]["model"], p["a"]["config"], p["b"]["model"], p["b"]["config"]) for p in em["pairs"]}
    # clef/q4 is 17.0 GB, clef-flash/q8 is 17.5 GB (2.9% apart); clef/q8 (28 GB) pairs with nothing
    assert ("clef", "q4", "clef-flash", "q8") in pairs
    assert not any("q8" == p[1] and p[0] == "clef" for p in pairs)
    assert [r["file_size_gb"] for r in em["by_size"]] == sorted(r["file_size_gb"] for r in em["by_size"])


def test_equal_memory_reports_missing_sizes(tree):
    records, results = tree
    (results / "clef" / "q4" / "meta.json").unlink()
    em = A.analyze(records, results, n_boot=50)["equal_memory"]
    assert {"model": "clef", "config": "q4"} in em["sizes_absent"]


def test_analysis_is_deterministic(tree):
    a = json.dumps(run(tree), sort_keys=True)
    b = json.dumps(run(tree), sort_keys=True)
    assert a == b


def test_logits_are_used_when_present(tree, tmp_path):
    records, results = tree
    recs = [json.loads(l) for l in (records / "toychoice.jsonl").read_text().splitlines()]
    rows = [result_line(r, [0.3, 0.3, 0.4], config="q8", logits=[math.log(0.7), math.log(0.2), math.log(0.1)])
            for r in recs]
    write_jsonl(results / "clef" / "q8" / "toychoice.jsonl", rows)
    blk = A.analyze(records, results, n_boot=50)["models"]["clef"]["configs"]["q8"]["tasks"]["toychoice"]
    assert blk["accuracy"] == pytest.approx(1.0)  # logits say 'a'; the (ignored) probs would say 'c'


def test_bad_result_lines_are_counted_as_errors(tree):
    records, results = tree
    path = results / "clef" / "q8" / "toychoice.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    lines[0]["error"] = "boom"
    lines[1]["answers"] = {}
    write_jsonl(path, lines[:-1] + [])  # also drop the last record -> missing
    blk = A.analyze(records, results, n_boot=50)["models"]["clef"]["configs"]["q8"]["tasks"]["toychoice"]
    assert blk["n_errors"] == 2
    assert blk["n_missing"] == 1
    assert blk["n_scored"] == 7


# ------------------------------------------------------------------- private mode

SECRETS = ["SECRET-TEXT-123", "OPT-DESC-XYZ", "privrec-7", "gold-label-QQ"]


def test_private_summary_has_only_aggregates(tmp_path: Path):
    root = tmp_path / "private"
    recs = []
    for i in range(12):
        rec = choice_record(str(i), text="SECRET-TEXT-123")
        rec["id"], rec["task"] = f"privrec-{i}", "privrec"
        rec["request"]["questions"]["answer"]["criteria"] = {"a": "OPT-DESC-XYZ", "b": "other", "c": "gold-label-QQ"}
        recs.append(rec)
    write_jsonl(root / "records" / "privrec.jsonl", recs)
    for cfg, flip in [("ref", False), ("q4", True)]:
        rows = [result_line(r, ref_probs(i) if not (flip and i == 3) else WRONG_B, config=cfg)
                for i, r in enumerate(recs)]
        write_jsonl(root / "results" / "clef" / cfg / "privrec.jsonl", rows)
    out = tmp_path / "out"
    A.run_private(root, out, n_boot=50)
    blob = (out / "private_summary.json").read_text() + (out / "private_tables.md").read_text()
    for secret in SECRETS:
        assert secret not in blob
    assert "privrec" in blob  # keyed by set name
    summary = json.loads((out / "private_summary.json").read_text())
    assert summary["models"]["clef"]["configs"]["q4"]["tasks"]["privrec"]["vs_ref"]["n_flips"] == 1
    # no option ids / record ids anywhere: every string value must be a known name, label or number-like key
    def strings(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield str(k)
                yield from strings(v)
        elif isinstance(o, list):
            for v in o:
                yield from strings(v)
        elif isinstance(o, str):
            yield o
    assert not [s for s in strings(summary) if s.startswith("privrec-")]


def test_scrub_rejects_unexpected_strings():
    with pytest.raises(ValueError):
        A.scrub_private({"models": {"m": {"note": "contains a record id privrec-7"}}}, allowed={"models", "m", "note"})


# ------------------------------------------------------------------- --models filter

def test_models_filter_leaves_other_models_out_entirely(tree):
    records, results = tree
    s = A.analyze(records, results, n_boot=50, models=["clef-flash"])
    assert set(s["models"]) == {"clef-flash"}
    # the filtered-out model contributes nothing to the cross-model sections either
    assert all(r["model"] == "clef-flash" for r in s["equal_memory"]["by_size"])
    assert not any(r["model"] == "clef" for r in s["absent"]["missing"])
    assert s["equal_memory"]["pairs"] == []


def test_models_filter_none_keeps_everything_and_unknown_name_fails_loudly(tree):
    records, results = tree
    assert set(A.analyze(records, results, n_boot=50)["models"]) == {"clef", "clef-flash"}
    with pytest.raises(ValueError, match="not found"):
        A.analyze(records, results, n_boot=50, models=["clef-flashh"])


def test_models_filter_cli_writes_only_that_model(tree, tmp_path):
    records, results = tree
    out = tmp_path / "out"
    assert A.main(["--records", str(records), "--results", str(results), "--out", str(out),
                   "--n-boot", "50", "--models", "clef"]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert set(s["models"]) == {"clef"}


# ------------------------------------------------------- subset configs pair by record id on their own records

def test_subset_config_is_paired_by_record_id_on_its_records_only(tree):
    """A subset config (ref-fla style: not a -rN repeat) holds the first half of the records, in shuffled file order."""
    records, results = tree
    recs = [json.loads(l) for l in (records / "toychoice.jsonl").read_text().splitlines()]
    first = recs[:5]
    rows = []
    for i in [3, 0, 4, 1, 2]:  # shuffled on purpose: pairing must go by id, not by line position
        r = first[i]
        p = [0.3, 0.5, 0.2] if i == 3 else ref_probs(i)   # record 3 flips (right -> wrong)
        rows.append(result_line(r, p, config="ref-sub"))
    write_jsonl(results / "clef" / "ref-sub" / "toychoice.jsonl", rows)
    s = A.analyze(records, results, n_boot=50)
    cfg = s["models"]["clef"]["configs"]["ref-sub"]            # reported as a config, not as a noise-floor repeat
    assert "ref-sub" not in s["models"]["clef"]["noise_floor"].get("ref", {})
    v = cfg["tasks"]["toychoice"]["vs_ref"]
    assert v["n_paired"] == 5 and v["n_flips"] == 1 and v["agreement"] == pytest.approx(0.8)
    assert cfg["pooled"]["vs_ref"]["n_paired"] == 5
    assert {"model": "clef", "config": "ref-sub", "task": "toychoice", "coverage": 0.5} in s["absent"]["partial"]
    assert cfg["tasks"]["toychoice"]["n_missing"] == 5


def test_repeat_with_a_subset_of_records_defines_its_noise_floor_on_those_records(tree):
    records, results = tree
    recs = [json.loads(l) for l in (records / "toychoice.jsonl").read_text().splitlines()]
    rows = [result_line(r, [0.89, 0.06, 0.05] if i == 0 else ref_probs(i), config="ref-r3") for i, r in enumerate(recs[:4])]
    write_jsonl(results / "clef" / "ref-r3" / "toychoice.jsonl", rows)
    nf = A.analyze(records, results, n_boot=50)["models"]["clef"]["noise_floor"]["ref"]["ref-r3"]
    assert nf["tasks"]["toychoice"]["n_paired"] == 4 and nf["pooled"]["n_paired"] == 4
    assert nf["tasks"]["toychoice"]["n_flips"] == 0
