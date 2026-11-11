"""Tests of bench.private_agg on synthetic private trees: hand-checkable numbers and no-leak checks."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from bench import metrics as M
from bench import private_agg as P

SECRET_TEXT = "SECRET-RECORD-TEXT-91"
SECRET_ID = "secretid"
SECRET_OPT = "zz_"            # every private option id starts with this
SECRET_QID = "secret_q"
OPTS3 = ["zz_alpha", "zz_beta", "zz_gamma"]       # sorted = option order
N = 80
N_REPEAT = 50


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


# ---- synthetic r4a-like set: choice5 (3-way choice, 45/32/3), score5 (5-bucket score) --------------------------------

def choice5_gold(i: int) -> str:
    return OPTS3[0] if i < 45 else (OPTS3[1] if i < 77 else OPTS3[2])


def score5_gold(i: int) -> str:
    return str((i * 7) % 5)


def r4a_records():
    rows = []
    for i in range(N):
        rows.append({"id": f"{SECRET_ID}-{i}", "task": "r4a",
                     "request": {"model": "clef", "state": {"text": f"{SECRET_TEXT} {i}"},
                                 "questions": {
                                     "choice5": {"type": "choice", "instructions": SECRET_TEXT,
                                                "criteria": {o: f"{SECRET_TEXT} desc" for o in OPTS3}},
                                     "score5": {"type": "score", "instructions": SECRET_TEXT,
                                                "criteria": [f"{SECRET_TEXT} {k}" for k in range(5)]}}},
                     "gold": {"choice5": choice5_gold(i), "score5": score5_gold(i)}})
    return rows


# predictions per config: dict (i, qid) -> predicted option; default = gold except `choice5_wrong` / `score5_wrong`
CHOICE5_WRONG = {"ref": {0, 1, 50, 78}, "ref-r2": {0, 1, 50, 78}, "q4_k_m": {0, 1, 2, 3, 4, 50, 51, 78, 79}}
SCORE5_SHIFT = {"ref": {2, 3}, "ref-r2": {2, 3}, "q4_k_m": {2, 3, 4, 5, 6, 7}}   # score5 predicted = gold + 1 (mod 5)
# the 27B differs from the 9B on purpose: its own ref is not the 9B ref, and its q4_k_m is one step from its own ref
BIG_CHOICE5_WRONG = {"ref": {1, 2, 50, 60, 70}, "ref-r2": {1, 2, 50, 60, 70}, "q4_k_m": {1, 2, 3, 4, 50, 60, 70}}
BIG_SCORE5_SHIFT = {"ref": {2, 9}, "ref-r2": {2, 9}, "q4_k_m": {2, 9, 11, 12, 13}}
FLASH, BIG = "clef-flash", "clef"


def choice5_pred(cfg: str, i: int, model: str = FLASH) -> str:
    g = choice5_gold(i)
    if i in (CHOICE5_WRONG if model == FLASH else BIG_CHOICE5_WRONG).get(cfg, set()):
        return OPTS3[(OPTS3.index(g) + 1) % 3]
    return g


def score5_pred(cfg: str, i: int, model: str = FLASH) -> str:
    g = int(score5_gold(i))
    return str((g + 1) % 5) if i in (SCORE5_SHIFT if model == FLASH else BIG_SCORE5_SHIFT).get(cfg, set()) else str(g)


def ans(opts, logits) -> dict:
    return {"option_ids": list(opts), "logits": np.asarray(logits, dtype=float).tolist(),
            "probs": softmax(logits).tolist()}


def one_hot_logits(k: int, n: int, strength: float = 3.0, alt: int | None = None):
    lg = np.zeros(n)
    lg[k] = strength
    if alt is not None:
        lg[alt] = strength - 0.5
    return lg


def r4a_result_rows(cfg: str, model: str = "clef-flash"):
    rows = []
    for i in range(N):
        sp = OPTS3.index(choice5_pred(cfg, i, model))
        hp = int(score5_pred(cfg, i, model))
        rows.append({"id": f"{SECRET_ID}-{i}", "task": "r4a", "model": model, "config": cfg,
                     "answers": {
                         "choice5": ans(OPTS3, one_hot_logits(sp, 3)),
                         "score5": ans([str(k) for k in range(5)], one_hot_logits(hp, 5))},
                     "latency_ms": 1.0, "input_tokens": (i % 10) + 1 if model == FLASH else 10 - (i % 10),
                     "engine": "toy", "engine_version": "1", "error": None})
    return rows


def hosted_repeat_rows():
    """Repeat on the first N_REPEAT records: run a == recorded label; run b differs on 4 choice5 / 12 score5 items."""
    rows = []
    for i in range(N_REPEAT):
        sig_b = OPTS3[(OPTS3.index(choice5_gold(i)) + 1) % 3] if i in {0, 10, 20, 49} else choice5_gold(i)
        hb = (int(score5_gold(i)) + 1) % 5 if i % 4 == 0 and i < 48 else int(score5_gold(i))
        rows.append({"id": f"{SECRET_ID}-{i}", "choice5_a": choice5_gold(i), "choice5_b": sig_b,
                     "score5_a": int(score5_gold(i)), "score5_b": hb})
    return rows


# ---- synthetic r1-like set: no gold ------------------------------------------------------------------------------

R1_OPTS = ["zz_k1", "zz_k2", "zz_k3", "zz_k4"]
R1_N = 40


def r1_records():
    rows = []
    for i in range(R1_N):
        rows.append({"id": f"{SECRET_ID}-r1-{i}", "task": "r1",
                     "request": {"model": "clef", "state": {"text": SECRET_TEXT},
                                 "questions": {
                                     f"{SECRET_QID}_effort": {"type": "score", "instructions": SECRET_TEXT,
                                                              "criteria": [f"{SECRET_TEXT}{k}" for k in range(6)]},
                                     f"{SECRET_QID}_kind": {"type": "choice", "instructions": SECRET_TEXT,
                                                            "criteria": {o: SECRET_TEXT for o in R1_OPTS}},
                                     f"{SECRET_QID}_flag": {"type": "noul", "instructions": SECRET_TEXT}}},
                     "gold": {}})
    return rows


R1_FLIPS = {"ref": set(), "ref-r2": {5}, "bf16-srv": {1, 2}, "q4_k_m": set(range(0, 12))}   # flips of the choice question
# 27B: its q4_k_m flips records 28..39, which are the SHORTEST inputs by the 27B's own token counts (R1_N - i) but the
# longest by the 9B's (i + 1), so the quartile split only comes out right if each model uses its own ref's tokens
BIG_R1_FLIPS = {"ref": set(), "ref-r2": {7, 8, 9}, "q4_k_m": set(range(28, 40))}


def r1_result_rows(cfg: str, model: str = FLASH):
    rows = []
    for i in range(R1_N):
        flipped = i in (R1_FLIPS if model == FLASH else BIG_R1_FLIPS).get(cfg, set())
        kind = np.array([2.0, 0.0, 0.0, 0.0])
        if flipped:
            kind = np.array([0.0, 2.0, 0.0, 0.0])
        score = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
        if cfg == "q4_k_m":                      # moves the expected score toward the end of the scale
            score = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 1.0 if model == FLASH else 2.0])
        rows.append({"id": f"{SECRET_ID}-r1-{i}", "task": "r1", "model": model, "config": cfg,
                     "answers": {
                         f"{SECRET_QID}_effort": ans([str(k) for k in range(6)], score),
                         f"{SECRET_QID}_kind": ans(R1_OPTS, kind),
                         f"{SECRET_QID}_flag": ans(["true", "false"], np.array([1.0, 0.0]))},
                     "latency_ms": 1.0, "input_tokens": i + 1 if model == FLASH else R1_N - i,
                     "engine": "toy", "engine_version": "1", "error": None})
    return rows


def make_tree(root: Path, big_r4a: tuple[str, ...], big_r1: tuple[str, ...]):
    """Synthetic private tree; the 9B part is fixed, the 27B configs on r4a / r1 are the arguments."""
    rec, res = root / "records", root / "results"
    write_jsonl(rec / "r4a.jsonl", r4a_records())
    write_jsonl(rec / "r4a_hosted_repeat.jsonl", hosted_repeat_rows())
    write_jsonl(rec / "r1.jsonl", r1_records())
    for cfg in ("ref", "ref-r2", "q4_k_m"):
        write_jsonl(res / "clef-flash" / cfg / "r4a.jsonl", r4a_result_rows(cfg))
    for cfg in ("ref", "ref-r2", "bf16-srv", "q4_k_m"):
        write_jsonl(res / "clef-flash" / cfg / "r1.jsonl", r1_result_rows(cfg))
    for cfg in big_r4a:
        write_jsonl(res / "clef" / cfg / "r4a.jsonl", r4a_result_rows(cfg, model="clef"))
    for cfg in big_r1:
        write_jsonl(res / "clef" / cfg / "r1.jsonl", r1_result_rows(cfg, model="clef"))
    return rec, res


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    # 27B: three configs on r4a; on r1 only q4_k_m (no 27B ref there, as in the real data today)
    return make_tree(tmp_path_factory.mktemp("priv"), ("ref", "ref-r2", "q4_k_m"), ("q4_k_m",))


@pytest.fixture(scope="module")
def tree_full(tmp_path_factory):
    return make_tree(tmp_path_factory.mktemp("priv_full"), ("ref", "ref-r2", "q4_k_m"), ("ref", "ref-r2", "q4_k_m"))


@pytest.fixture(scope="module")
def tree_ref_only(tmp_path_factory):
    """The original fixture: the 27B has only its ref, on r4a."""
    return make_tree(tmp_path_factory.mktemp("priv_ref_only"), ("ref",), ())


@pytest.fixture(scope="module")
def out(tree):
    rec, res = tree
    return P.build(rec, res, n_boot=300, sets=("r4a", "r1"))


@pytest.fixture(scope="module")
def out_full(tree_full):
    rec, res = tree_full
    return P.build(rec, res, n_boot=300, sets=("r4a", "r1"))


# ------------------------------------------------------------------------------------ labels

def test_question_labels_are_named_only_where_plan_names_them(tree):
    rec, res = tree
    sd = P.SetData("r1", rec, res, {"clef-flash": ("ref",)})
    assert sorted(set(sd.qlabel)) == ["choice4", "noul", "score6"]
    assert not any(SECRET_QID in l for l in sd.qlabel)
    sd4 = P.SetData("r4a", rec, res, {"clef-flash": ("ref",)})
    assert sorted(set(sd4.qlabel)) == ["choice5", "score5"]


def test_duplicate_question_shapes_get_numbered_labels(tmp_path):
    rows = []
    for i in range(10):
        qs = {f"q{k}": {"type": "noul", "instructions": "x"} for k in "ab"}
        rows.append({"id": f"x-{i}", "task": "rx", "request": {"model": "clef", "state": {}, "questions": qs}, "gold": {}})
    write_jsonl(tmp_path / "rx.jsonl", rows)
    from bench import decompose as D
    u = D.build_unit(tmp_path, None, tasks=["rx"])
    assert sorted(set(P.question_labels(u, "rx"))) == ["noul_1", "noul_2"]


def test_class_ranks_are_by_support():
    gold = np.array([2, 2, 2, 0, 0, 1])
    assert P.class_ranks(gold, 4) == {2: "c1", 0: "c2", 1: "c3"}  # class 3 has no support and is omitted


# ----------------------------------------------------------------------------- hosted repeat

def test_hosted_self_consistency_counts(out):
    h = out["hosted_repeat"]["r4a"]
    assert h["choice5"]["n_pairs"] == N_REPEAT and h["choice5"]["disagreement_n"] == 4
    assert h["choice5"]["disagreement"] == pytest.approx(4 / N_REPEAT)
    hb = sum(1 for i in range(N_REPEAT) if i % 4 == 0 and i < 48)
    assert h["score5"]["disagreement_n"] == hb
    assert h["choice5"]["recorded_label_equals_run_a"] == 1.0
    lo, hi = h["choice5"]["disagreement_ci95"]
    assert lo <= 4 / N_REPEAT <= hi
    # the PLAN.md quote needs 5,319 pairs; the synthetic set has 50, so it must not claim to reproduce it
    assert h["choice5"]["reproduces_plan_quote"] is False
    assert "r1" not in out["hosted_repeat"]


def test_hosted_repeat_with_recorded_label_mismatch_fails_loudly(tree, tmp_path):
    rec, res = tree
    bad = tmp_path / "records"
    bad.mkdir()
    write_jsonl(bad / "r4a.jsonl", r4a_records())
    rows = hosted_repeat_rows()
    rows[0]["choice5_a"] = OPTS3[2] if rows[0]["choice5_a"] != OPTS3[2] else OPTS3[1]
    write_jsonl(bad / "r4a_hosted_repeat.jsonl", rows)
    with pytest.raises(ValueError, match="recorded label differs"):
        P.build(bad, res, n_boot=50, sets=("r4a",))


# ------------------------------------------------------------------------ label agreement

def test_choice5_agreement_hand_computed(out):
    sig = out["label_agreement"]["r4a"]["choice5"]
    assert sig["n"] == N and sig["class_support"] == {"c1": 45, "c2": 32, "c3": 3}
    ref = sig["models"]["clef-flash"]["configs"]["ref"]
    wrong = CHOICE5_WRONG["ref"]
    assert ref["accuracy"] == pytest.approx((N - len(wrong)) / N)
    gold = np.array([choice5_gold(i) for i in range(N)])
    pred = np.array([choice5_pred("ref", i) for i in range(N)])
    assert ref["macro_f1"] == pytest.approx(M.macro_f1(gold, pred))
    # recall of the most frequent class: items 0..44 are class c1 and indices 0 and 1 are wrong
    assert ref["recall"]["c1"]["support"] == 45 and ref["recall"]["c1"]["recall"] == pytest.approx(43 / 45)
    lo, hi = ref["accuracy_ci95"]
    assert lo <= ref["accuracy"] <= hi


def test_majority_baseline_and_core_macro_f1(out):
    sig = out["label_agreement"]["r4a"]["choice5"]
    mb = sig["majority_baseline"]
    assert mb["accuracy"] == pytest.approx(45 / N)
    assert mb["recall"]["c1"]["recall"] == 1.0 and mb["recall"]["c2"]["recall"] == 0.0
    # two classes have support >= 30 (c1: 45, c2: 32); c3 (3 items) is outside the core set
    assert mb["core_classes"] == 2
    # always predicting c1: F1(c1) = 2*45 / (45 gold + 80 predicted), F1(c2) = 0
    assert mb["macro_f1_core"] == pytest.approx((2 * 45 / (45 + N) + 0.0) / 2)
    ref = sig["models"]["clef-flash"]["configs"]["ref"]
    gold = np.array([OPTS3.index(choice5_gold(i)) for i in range(N)])
    pred = np.array([OPTS3.index(choice5_pred("ref", i)) for i in range(N)])
    assert ref["macro_f1_core"] == pytest.approx(P.fixed_macro_f1(gold, pred, [0, 1]))
    assert ref["core_classes"] == 2


def test_fixed_macro_f1_equals_union_macro_f1_when_all_classes_present_and_counts_absent_class_as_zero():
    gold = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    assert P.fixed_macro_f1(gold, pred, [0, 1, 2]) == pytest.approx(M.macro_f1(gold, pred))
    assert P.fixed_macro_f1(gold, pred, [0, 1, 2, 3]) == pytest.approx(M.macro_f1(gold, pred) * 3 / 4)
    idx = np.arange(6)[None, :]
    assert P.boot_fixed_macro_f1(gold, pred, idx, [0, 1, 2])[0] == pytest.approx(M.macro_f1(gold, pred))


def test_delta_vs_ref_is_paired_and_zero_for_identical_predictions(out):
    sig = out["label_agreement"]["r4a"]["choice5"]["models"]["clef-flash"]["configs"]
    d_same = sig["ref-r2"]["delta_vs_ref"]
    assert d_same["delta_accuracy"]["delta"] == 0.0 and d_same["delta_accuracy"]["ci95"] == [0.0, 0.0]
    assert d_same["delta_macro_f1"]["delta"] == 0.0
    d = sig["q4_k_m"]["delta_vs_ref"]
    assert d["delta_accuracy"]["delta"] == pytest.approx(sig["q4_k_m"]["accuracy"] - sig["ref"]["accuracy"])
    lo, hi = d["delta_accuracy"]["ci95"]
    assert lo <= d["delta_accuracy"]["delta"] <= hi
    assert sig["q4_k_m"]["agreement_with_ref"] == pytest.approx(
        np.mean([choice5_pred("q4_k_m", i) == choice5_pred("ref", i) for i in range(N)]))


def test_score_question_reports_bucket_error(out):
    h = out["label_agreement"]["r4a"]["score5"]["models"]["clef-flash"]["configs"]["ref"]
    exp = np.mean([abs(int(score5_pred("ref", i)) - int(score5_gold(i))) for i in range(N)])
    # (g + 1) % 5 wraps 4 -> 0, which is a bucket error of 4, so compute it from the same definition
    assert h["mean_abs_bucket_error"] == pytest.approx(exp)
    assert 0.0 <= h["within_one_bucket"] <= 1.0
    assert "mean_abs_bucket_error" not in out["label_agreement"]["r4a"]["choice5"]["models"]["clef-flash"]["configs"]["ref"]


def test_hosted_subset_compares_model_with_the_repeat_on_the_same_records(out):
    sub = out["label_agreement"]["r4a"]["choice5"]["hosted_repeat_subset"]
    assert sub["n"] == N_REPEAT
    hb = sub["hosted_run_b_vs_recorded"]
    assert hb["accuracy"] == pytest.approx(1 - 4 / N_REPEAT)
    ref = sub["models"]["clef-flash"]["configs"]["ref"]
    wrong_in_subset = len([i for i in CHOICE5_WRONG["ref"] if i < N_REPEAT])
    assert ref["accuracy"] == pytest.approx(1 - wrong_in_subset / N_REPEAT)
    assert ref["disagreement_with_recorded"] == pytest.approx(wrong_in_subset / N_REPEAT)
    d = ref["delta_vs_hosted_run_b"]["delta_accuracy"]["delta"]
    assert d == pytest.approx(ref["accuracy"] - hb["accuracy"])


def test_big_model_ref_has_no_delta_and_non_ref_configs_are_scored_against_it(out):
    cfgs = out["label_agreement"]["r4a"]["choice5"]["models"]["clef"]["configs"]
    assert set(cfgs) == {"ref", "ref-r2", "q4_k_m"}
    ref = cfgs["ref"]
    assert "delta_vs_ref" not in ref and "agreement_with_ref" not in ref
    assert ref["accuracy"] == pytest.approx(1 - len(BIG_CHOICE5_WRONG["ref"]) / N)
    for cfg in ("ref-r2", "q4_k_m"):
        assert "delta_vs_ref" in cfgs[cfg] and "agreement_with_ref" in cfgs[cfg]
    # the 9B blocks carry neither the 27B-only field nor anything new
    for cfg, blk in out["label_agreement"]["r4a"]["choice5"]["models"]["clef-flash"]["configs"].items():
        assert "agreement_with_flash_ref" not in blk
        assert ("delta_vs_ref" in blk) == (cfg != "ref") and ("agreement_with_ref" in blk) == (cfg != "ref")


def test_big_delta_vs_ref_is_against_the_27b_ref_and_zero_for_identical_predictions(out):
    sig = out["label_agreement"]["r4a"]["choice5"]["models"]["clef"]["configs"]
    d_same = sig["ref-r2"]["delta_vs_ref"]
    assert d_same["delta_accuracy"]["delta"] == 0.0 and d_same["delta_accuracy"]["ci95"] == [0.0, 0.0]
    assert d_same["delta_macro_f1"]["delta"] == 0.0
    assert sig["ref-r2"]["agreement_with_ref"] == 1.0
    # 27B ref has 5 wrong, q4_k_m has 7 wrong: accuracy 75/80 -> 73/80, delta -2/80. Against the 9B ref (4 wrong, 76/80)
    # it would be -3/80.
    d = sig["q4_k_m"]["delta_vs_ref"]
    assert sig["ref"]["accuracy"] == pytest.approx(75 / 80) and sig["q4_k_m"]["accuracy"] == pytest.approx(73 / 80)
    assert d["delta_accuracy"]["delta"] == pytest.approx(-2 / 80)
    lo, hi = d["delta_accuracy"]["ci95"]
    assert lo <= d["delta_accuracy"]["delta"] <= hi
    gold = np.array([choice5_gold(i) for i in range(N)])
    f1 = lambda cfg: M.macro_f1(gold, np.array([choice5_pred(cfg, i, BIG) for i in range(N)]))  # noqa: E731
    assert d["delta_macro_f1"]["delta"] == pytest.approx(f1("q4_k_m") - f1("ref"))
    # q4_k_m differs from the 27B ref on records 3 and 4 only: agreement 78/80; against the 9B ref it differs on 7
    assert sig["q4_k_m"]["agreement_with_ref"] == pytest.approx(78 / 80)
    assert sig["q4_k_m"]["agreement_with_flash_ref"] == pytest.approx(73 / 80)
    # the score5 score question goes through the same path: ref shifts {2, 9}, q4_k_m {2, 9, 11, 12, 13}
    hq = out["label_agreement"]["r4a"]["score5"]["models"]["clef"]["configs"]["q4_k_m"]
    assert hq["agreement_with_ref"] == pytest.approx(77 / 80) and "delta_vs_ref" in hq


def test_agreement_with_flash_ref_on_the_27b_ref_is_the_hand_computed_top1_agreement(out):
    models = out["label_agreement"]["r4a"]["choice5"]["models"]
    ref = models["clef"]["configs"]["ref"]
    # 9B ref wrong on {0, 1, 50, 78}, 27B ref wrong on {1, 2, 50, 60, 70}; the wrong answer is the same cyclic step,
    # so they disagree exactly on the symmetric difference {0, 78, 2, 60, 70}: 5 of 80
    assert ref["agreement_with_flash_ref"] == pytest.approx(75 / 80)
    expected = np.mean([choice5_pred("ref", i, BIG) == choice5_pred("ref", i, FLASH) for i in range(N)])
    assert ref["agreement_with_flash_ref"] == pytest.approx(expected)
    assert models["clef"]["configs"]["ref-r2"]["agreement_with_flash_ref"] == pytest.approx(75 / 80)
    h = out["label_agreement"]["r4a"]["score5"]["models"]["clef"]["configs"]["ref"]
    # score5 shifts {2, 3} (9B) vs {2, 9} (27B): disagree on records 3 and 9
    assert h["agreement_with_flash_ref"] == pytest.approx(78 / 80)


def test_27b_config_without_its_own_ref_gets_no_delta_but_still_the_flash_ref_agreement(tmp_path):
    rec, res = make_tree(tmp_path, ("q4_k_m",), ())
    o = P.build(rec, res, n_boot=50, sets=("r4a",))
    blk = o["label_agreement"]["r4a"]["choice5"]["models"]["clef"]["configs"]["q4_k_m"]
    assert "delta_vs_ref" not in blk and "agreement_with_ref" not in blk
    assert blk["agreement_with_flash_ref"] == pytest.approx(73 / 80)
    # the 9B is untouched by the missing 27B ref
    assert "delta_vs_ref" in o["label_agreement"]["r4a"]["choice5"]["models"]["clef-flash"]["configs"]["q4_k_m"]


# --------------------------------------------------------------------------- label free

def test_label_free_flip_counts_without_any_gold(out):
    lf = out["label_free"]["r1"]
    assert "r1" not in out["label_agreement"]
    kind = lf["choice4"]["pairs"]["ref|q4_k_m"]
    assert kind["flips"] == 12 and kind["flip_rate"] == pytest.approx(12 / R1_N)
    assert lf["choice4"]["pairs"]["ref|bf16-srv"]["flips"] == 2
    assert lf["choice4"]["pairs"]["ref|ref-r2"]["flips"] == 1
    assert lf["noul"]["pairs"]["ref|q4_k_m"]["flips"] == 0
    pooled = lf["pooled"]["pairs"]["ref|q4_k_m"]
    assert pooled["flips"] == 12 and pooled["flip_rate"] == pytest.approx(12 / (3 * R1_N))
    lo, hi = kind["flip_ci95"]
    assert lo <= kind["flip_rate"] <= hi


def test_label_free_score_shift_has_ci_and_center_pull_sign(out):
    sh = out["label_free"]["r1"]["score6"]["pairs"]["ref|q4_k_m"]["score_shift"]
    e_ref = float(np.sum(np.arange(6) * softmax([0, 0, 3, 0, 0, 0])))
    e_q = float(np.sum(np.arange(6) * softmax([0, 0, 3, 0, 0, 1])))
    assert sh["mean_signed_shift"] == pytest.approx(e_q - e_ref, abs=1e-9)
    assert sh["center_pull"] == pytest.approx(abs(e_q - 2.5) - abs(e_ref - 2.5), abs=1e-9)  # 2.5 is the middle of 0..5
    assert sh["center_pull_ci95"] is not None and sh["mean_signed_shift_ci95"] is not None


def test_length_quartiles_and_reference_confidence_buckets(out):
    r = out["label_free"]["r1"]["choice4"]["pairs"]["ref|q4_k_m"]
    q = r["flip_by_length_quartile"]
    assert sum(q[k]["n"] for k in ("Q1", "Q2", "Q3", "Q4")) == R1_N
    assert sum(q[k]["flips"] for k in ("Q1", "Q2", "Q3", "Q4")) == 12
    # the 12 flips are records 0..11, i.e. the shortest inputs: Q4 minus Q1 is negative
    assert r["flip_q4_minus_q1"]["delta"] < 0
    cb = r["flip_by_ref_confidence"]
    assert sum(b["n"] for b in cb.values()) == R1_N and sum(b["flips"] for b in cb.values()) == 12


def test_label_free_big_flip_counts_are_hand_computed_against_the_27b_ref(out_full):
    lf = out_full["label_free_big"]
    # r4a: the 27B q4_k_m differs from the 27B ref on choice5 records 3, 4 (2 flips) and score5 records 11, 12, 13 (3 flips);
    # the 9B q4_k_m would give 5 choice5 flips (records 2, 3, 4, 51, 79), so these are not the 9B pairs
    r4a = lf["r4a"]
    assert r4a["choice5"]["pairs"]["ref|q4_k_m"]["flips"] == 2
    assert r4a["choice5"]["pairs"]["ref|q4_k_m"]["flip_rate"] == pytest.approx(2 / N)
    assert r4a["score5"]["pairs"]["ref|q4_k_m"]["flips"] == 3
    assert r4a["pooled"]["pairs"]["ref|q4_k_m"]["flips"] == 5
    assert r4a["pooled"]["pairs"]["ref|q4_k_m"]["flip_rate"] == pytest.approx(5 / (2 * N))
    assert r4a["pooled"]["pairs"]["ref|ref-r2"]["flips"] == 0
    assert out_full["label_free"]["r4a"]["choice5"]["pairs"]["ref|q4_k_m"]["flips"] == 5
    # r1: 12 choice flips for q4_k_m, 3 for ref-r2, none on the other questions
    r1 = lf["r1"]
    assert r1["choice4"]["pairs"]["ref|q4_k_m"]["flips"] == 12 and r1["choice4"]["pairs"]["ref|ref-r2"]["flips"] == 3
    assert r1["noul"]["pairs"]["ref|q4_k_m"]["flips"] == 0 and r1["score6"]["pairs"]["ref|q4_k_m"]["flips"] == 0
    assert r1["pooled"]["pairs"]["ref|q4_k_m"]["flips"] == 12
    # pairs the 27B does not have are not invented (no bf16-srv / gguf-bf16 on the 27B here)
    assert "ref|bf16-srv" not in r1["pooled"]["pairs"] and "ref|ref-r2" in r1["pooled"]["pairs"]
    lo, hi = r1["choice4"]["pairs"]["ref|q4_k_m"]["flip_ci95"]
    assert lo <= 12 / R1_N <= hi


def test_label_free_big_length_quartiles_use_the_27b_own_ref_tokens(out_full):
    r = out_full["label_free_big"]["r1"]["choice4"]["pairs"]["ref|q4_k_m"]
    q = r["flip_by_length_quartile"]
    # 27B tokens are R1_N - i = 1..40; edges 10.75 / 20.5 / 30.25. Flipped records 28..39 have tokens 12..1:
    # Q1 = tokens 1..10 (records 30..39) holds 10 flips, Q2 = tokens 11..20 (records 20..29) holds records 28, 29
    assert [q[k]["n"] for k in ("Q1", "Q2", "Q3", "Q4")] == [10, 10, 10, 10]
    assert [q[k]["flips"] for k in ("Q1", "Q2", "Q3", "Q4")] == [10, 2, 0, 0]
    assert r["flip_q4_minus_q1"]["delta"] == pytest.approx(-1.0)
    # the 9B (tokens i + 1) keeps its own split: its flips are records 0..11, tokens 1..12 -> Q1 10, Q2 2 of 12 flips
    q9 = out_full["label_free"]["r1"]["choice4"]["pairs"]["ref|q4_k_m"]["flip_by_length_quartile"]
    assert [q9[k]["flips"] for k in ("Q1", "Q2", "Q3", "Q4")] == [10, 2, 0, 0]


def test_label_free_big_score_shift_uses_27b_configs(out_full):
    sh = out_full["label_free_big"]["r1"]["score6"]["pairs"]["ref|q4_k_m"]["score_shift"]
    e_ref = float(np.sum(np.arange(6) * softmax([0, 0, 3, 0, 0, 0])))
    e_q = float(np.sum(np.arange(6) * softmax([0, 0, 3, 0, 0, 2])))
    assert sh["mean_signed_shift"] == pytest.approx(e_q - e_ref, abs=1e-9)
    e9 = float(np.sum(np.arange(6) * softmax([0, 0, 3, 0, 0, 1])))
    assert sh["mean_signed_shift"] != pytest.approx(e9 - e_ref, abs=1e-6)  # not the 9B's shift


def test_label_free_big_without_27b_configs_has_no_pairs_and_the_9b_label_free_is_unchanged_by_27b_configs(
        out, out_full, tree_ref_only):
    rec, res = tree_ref_only
    base = P.build(rec, res, n_boot=300, sets=("r4a", "r1"))
    # only a 27B ref on r4a, none on r1: no pairs at all
    assert base["label_free_big"]["r4a"]["pooled"]["pairs"] == {} and base["label_free_big"]["r1"]["pooled"]["pairs"] == {}
    assert out["label_free_big"]["r1"]["pooled"]["pairs"] == {}   # q4_k_m without a 27B ref on r1 pairs with nothing
    dump = lambda o: json.dumps(o["label_free"], sort_keys=True)  # noqa: E731
    assert dump(out) == dump(base) == dump(out_full)
    # the 9B label-agreement blocks do not move either (the 27B sits in its own `models` entry)
    flash = lambda o: json.dumps({s: {q: b["models"]["clef-flash"] for q, b in qs.items()}  # noqa: E731
                                  for s, qs in o["label_agreement"].items()}, sort_keys=True)
    assert flash(out) == flash(base)


def test_meta_lists_both_config_sets_and_label_free_big_is_scrubbed(out_full):
    assert out_full["meta"]["big_configs"] == list(P.BIG_CONFIGS) == out_full["meta"]["flash_configs"]
    P.scrub(json.loads(json.dumps(out_full)))
    bad = json.loads(json.dumps(out_full))
    bad["label_free_big"]["r1"]["choice4"]["note"] = "x"
    with pytest.raises(ValueError, match="unexpected key"):
        P.scrub(bad)


def test_markdown_renders_a_change_table_per_model_and_the_27b_label_free_section(out_full):
    md = P.render_md(out_full)
    assert "Change against the `ref` of Clef-flash (9B)" in md and "Change against the `ref` of Clef (27B)" in md
    assert "## Label-free comparison against ref (every set)" in md
    assert "## Label-free comparison against the 27B ref: Clef (27B)" in md and "### r1 (Clef 27B)" in md
    assert md.index("## Label-free comparison against ref (every set)") < md.index("## Label-free comparison against the 27B ref")


def test_wilson_interval():
    lo, hi = P.wilson(5, 100)
    assert lo == pytest.approx(0.02154, abs=1e-4) and hi == pytest.approx(0.11175, abs=1e-4)
    assert P.wilson(0, 0) is None


# ------------------------------------------------------------------------------- privacy

def blob(res_dir_out: Path) -> str:
    return (res_dir_out / "private_aggregates.json").read_text() + (res_dir_out / "private_aggregates.md").read_text()


def test_outputs_contain_no_record_text_ids_options_or_question_names(tree, tmp_path):
    rec, res = tree
    assert P.main(["--records", str(rec), "--results", str(res), "--out", str(tmp_path), "--n-boot", "100",
                   "--sets", "r4a", "r1"]) == 0
    text = blob(tmp_path)
    for secret in (SECRET_TEXT, SECRET_ID, SECRET_OPT, SECRET_QID, "alpha", "beta", "gamma"):
        assert secret not in text, secret
    data = json.loads((tmp_path / "private_aggregates.json").read_text())
    assert set(data["label_agreement"]["r4a"]["choice5"]["class_support"]) == {"c1", "c2", "c3"}


def test_scrub_rejects_unknown_keys_and_strings(out):
    P.scrub(json.loads(json.dumps(out)))  # the real output passes
    bad = json.loads(json.dumps(out))
    bad["label_free"]["r1"]["choice4"]["note"] = "x"
    with pytest.raises(ValueError, match="unexpected key"):
        P.scrub(bad)
    bad = json.loads(json.dumps(out))
    bad["absent"].append({"set": "r1", "model": "clef-flash", "config": f"{SECRET_ID}-3"})
    with pytest.raises(ValueError, match="unexpected string"):
        P.scrub(bad)


def test_missing_configs_are_listed_not_imputed(out):
    absent = {(a["set"], a["model"], a["config"]) for a in out["absent"]}
    assert ("r4a", "clef-flash", "fp8") in absent and ("r1", "clef", "ref") in absent
    assert ("r4a", "clef-flash", "ref") not in absent
    assert ("r4a", "clef", "fp8") in absent and ("r4a", "clef", "ref") not in absent and ("r1", "clef", "q4_k_m") not in absent


def test_build_is_deterministic(tree):
    rec, res = tree
    a = P.build(rec, res, n_boot=80, sets=("r4a",))
    b = P.build(rec, res, n_boot=80, sets=("r4a",))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ------------------------------------------------------------------------------- another operator's sets

def custom_tree(tree, root: Path) -> tuple[Path, Path]:
    """The r1 fixture under a new set name, plus a third ref instance."""
    rec, res = root / "rec", root / "res"
    shutil.copytree(tree[0], rec)
    shutil.copytree(tree[1], res)
    (rec / "r1.jsonl").rename(rec / "set_a.jsonl")
    for f in res.rglob("r1.jsonl"):
        f.rename(f.with_name("set_a.jsonl"))
    shutil.copytree(res / "clef-flash" / "ref-r2", res / "clef-flash" / "ref-r3")
    return rec, res


CUSTOM = ["--sets", "set_a", "--configs", "ref", "ref-r2", "ref-r3", "--pairs", "ref-r2|ref-r3", "--n-boot", "100"]


def test_custom_sets_configs_and_pairs_pass_scrub_without_leaks(tree, tmp_path):
    rec, res = custom_tree(tree, tmp_path)
    out = tmp_path / "out"
    assert P.main(["--records", str(rec), "--results", str(res), "--out", str(out), *CUSTOM]) == 0
    data = json.loads((out / "private_aggregates.json").read_text())
    assert data["meta"]["sets"] == ["set_a"] and data["meta"]["flash_configs"] == ["ref", "ref-r2", "ref-r3"]
    assert data["meta"]["label_free_pairs"] == ["ref|ref-r2", "ref|ref-r3", "ref-r2|ref-r3"]
    assert "ref-r2|ref-r3" in data["label_free"]["set_a"]["pooled"]["pairs"]
    for secret in (SECRET_TEXT, SECRET_ID, SECRET_OPT, SECRET_QID):
        assert secret not in blob(out), secret


def test_names_not_supplied_for_this_run_still_fail_the_scrub(tree, tmp_path):
    rec, res = custom_tree(tree, tmp_path)
    out = tmp_path / "out"
    assert P.main(["--records", str(rec), "--results", str(res), "--out", str(out), *CUSTOM]) == 0
    data = json.loads((out / "private_aggregates.json").read_text())
    with pytest.raises(ValueError):
        P.scrub(data)   # the default vocabulary doesn't know set_a, and set names aren't taken from the output


@pytest.mark.parametrize("args", [
    ["--sets", "../up"], ["--sets", "a b"], ["--sets", ".hidden"], ["--sets", "a/b"], ["--sets", "x" * 49],
    ["--configs", "ref", "../up"], ["--big-configs", "a b"], ["--pairs", "ref"], ["--pairs", "ref|a b"],
    ["--pairs", "ref|not_listed"],
])
def test_unsafe_names_are_refused_at_the_command_line(args, tmp_path):
    with pytest.raises(SystemExit):
        P.main(["--records", str(tmp_path), "--results", str(tmp_path), "--out", str(tmp_path), *args])
    assert not (tmp_path / "private_aggregates.json").exists()


def test_default_pairs_reproduce_the_frozen_list():
    assert P.default_pairs(P.FLASH_CONFIGS) == P.LABEL_FREE_PAIRS
