"""Tests of bench.hypotheses on a synthetic public tree with hand-checkable numbers."""

import json
from pathlib import Path

import numpy as np
import pytest

from bench import hypotheses as H
from bench import metrics as M

N_C = 60      # clinc150-like task: options out_of_scope, a, b (gold: first 20 are OOS)
N_S = 40      # sst5-like score task, 5 options
N_B = 30      # boolean task


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


COPTS = ["a", "b", "out_of_scope"]      # sorted option order of the clinc150-like task
SOPTS = [str(k) for k in range(5)]
BOPTS = ["true", "false"]


def c_gold(i):
    return "out_of_scope" if i < 20 else ("a" if i < 40 else "b")


def s_gold(i):
    return str(i % 5)


def b_gold(i):
    return "true" if i % 3 else "false"


def tokens(i):
    return i + 1


# config -> {task: {index: predicted option}} deviations from the gold prediction
DEV = {
    "ref": {"clinc150": {0: "a", 25: "b"}, "toyb": {1: "false"}},
    "ref-r2": {"clinc150": {0: "a", 25: "b"}, "toyb": {1: "false"}},
    "bf16-srv": {"clinc150": {0: "a", 25: "b"}, "toyb": {1: "false"}},
    "gguf-bf16": {"clinc150": {0: "a", 25: "b"}, "toyb": {1: "false"}},
    "q8_0": {"clinc150": {0: "a", 25: "b"}, "toyb": {1: "false"}},
    "fp8": {"clinc150": {0: "a", 1: "a", 25: "b"}, "toyb": {1: "false"}},                    # loses one more OOS item
    "nvfp4": {"clinc150": {0: "a", 1: "a", 2: "a", 3: "a", 25: "b", 41: "a"}, "toyb": {1: "false", 2: "false"}},
    "q4_k_m": {"clinc150": {0: "a", 1: "a", 25: "b", 26: "a", 27: "a"}, "toyb": {1: "false"}},
}


def probs_for(opts, pick, conf):
    lg = np.zeros(len(opts))
    lg[opts.index(pick)] = conf
    return lg


def rows_for(cfg: str, task: str, model="clef-flash", conf_by_cfg=None):
    dev = DEV[cfg].get(task, {})
    rows = []
    n = {"clinc150": N_C, "sst5": N_S, "toyb": N_B}[task]
    for i in range(n):
        if task == "clinc150":
            opts, gold = COPTS, c_gold(i)
        elif task == "sst5":
            opts, gold = SOPTS, s_gold(i)
        else:
            opts, gold = BOPTS, b_gold(i)
        pick = dev.get(i, gold)
        conf = 2.0 if i % 4 else 0.5
        lg = probs_for(opts, pick, conf)
        if task == "sst5":
            # nvfp4 pulls the expected score toward the middle: mass from option 4 and 0 moves to option 2
            if cfg == "nvfp4":
                lg = lg.copy()
                lg[2] += 1.0
        rows.append({"id": f"{task}-{i}", "task": task, "model": model, "config": cfg,
                     "answers": {"answer": {"option_ids": opts, "logits": lg.tolist(), "probs": softmax(lg).tolist()}},
                     "latency_ms": 1.0, "input_tokens": tokens(i), "engine": "toy", "engine_version": "1", "error": None})
    return rows


def records_for(task: str):
    rows = []
    n = {"clinc150": N_C, "sst5": N_S, "toyb": N_B}[task]
    for i in range(n):
        if task == "clinc150":
            q = {"type": "choice", "instructions": "x", "criteria": {o: "d" for o in COPTS}}
            gold = c_gold(i)
        elif task == "sst5":
            q = {"type": "score", "instructions": "x", "criteria": ["d"] * 5}
            gold = s_gold(i)
        else:
            q = {"type": "noul", "instructions": "x"}
            gold = b_gold(i)
        rows.append({"id": f"{task}-{i}", "task": task,
                     "request": {"model": "clef-flash", "state": {}, "questions": {"answer": q}}, "gold": {"answer": gold}})
    return rows


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    root = tmp_path_factory.mktemp("hyp")
    rec, res = root / "records", root / "results"
    for task in ("clinc150", "sst5", "toyb"):
        write_jsonl(rec / f"{task}.jsonl", records_for(task))
    for cfg in H.CONFIGS:
        for task in ("clinc150", "sst5", "toyb"):
            write_jsonl(res / "clef-flash" / cfg / f"{task}.jsonl", rows_for(cfg, task))
    # a 27B reference that has only finished two of the three tasks, and is wrong on a few more items
    for task in ("clinc150", "toyb"):
        rows = rows_for("ref", task, model="clef")
        write_jsonl(res / "clef" / "ref" / f"{task}.jsonl", rows)
    return rec, res


@pytest.fixture(scope="module")
def data(trees):
    rec, res = trees
    return H.Data(rec, res)


@pytest.fixture(scope="module")
def idx(data):
    return H.make_idx(data.n, 400, 5)


# ------------------------------------------------------------------------------ bootstrap

def test_boot_ece_matches_metrics_ece_on_the_identity_resample(data):
    ident = np.arange(data.n, dtype=np.int32)[None, :]
    for c in ("ref", "nvfp4"):
        assert H.boot_ece(data.conf[c], data.correct[c], ident)[0] == pytest.approx(M.ece(data.conf[c], data.correct[c]))


def test_boot_ece_chunking_does_not_change_results(data):
    big = H.make_idx(data.n, 230, 3)           # > one chunk of 100
    whole = H.boot_ece(data.conf["ref"], data.correct["ref"], big)
    parts = np.concatenate([H.boot_ece(data.conf["ref"], data.correct["ref"], big[s:s + 50]) for s in range(0, 230, 50)])
    np.testing.assert_allclose(whole, parts)


def test_est_flags_whether_the_interval_excludes_zero():
    assert H.est(0.1, np.array([0.05, 0.15]))["excludes_zero"] is True
    assert H.est(0.0, np.array([-0.05, 0.05]))["excludes_zero"] is False


# ------------------------------------------------------------------------- calibration

def test_calibration_point_estimates_and_deltas(data, idx):
    cal = H.calibration(data, idx)
    ref, nv = cal["per_config"]["ref"], cal["per_config"]["nvfp4"]
    assert ref["accuracy"] == pytest.approx(data.correct["ref"].mean())
    assert nv["delta_vs_ref"]["accuracy"]["delta"] == pytest.approx(data.correct["nvfp4"].mean() - data.correct["ref"].mean())
    assert nv["delta_vs_ref"]["brier"]["delta"] == pytest.approx(data.brier["nvfp4"].mean() - data.brier["ref"].mean())
    assert nv["delta_vs_ref"]["ece"]["delta"] == pytest.approx(M.ece(data.conf["nvfp4"], data.correct["nvfp4"])
                                                               - M.ece(data.conf["ref"], data.correct["ref"]))
    same = cal["per_config"]["bf16-srv"]["delta_vs_ref"]["accuracy"]
    assert same["delta"] == 0.0 and same["ci95"] == [0.0, 0.0] and same["excludes_zero"] is False
    assert set(cal["within_engine"]) == {"bf16-srv|fp8", "bf16-srv|nvfp4", "gguf-bf16|q8_0", "gguf-bf16|q4_k_m"}
    fp8 = cal["within_engine"]["bf16-srv|fp8"]["accuracy"]["delta"]
    assert fp8 == pytest.approx(data.correct["fp8"].mean() - data.correct["bf16-srv"].mean())


# ------------------------------------------------------------------------------- oos

def test_oos_recall_and_in_scope_accuracy_by_gold_class(data):
    r = H.oos(data, 300, 9)
    assert (r["n"], r["n_oos"], r["n_in_scope"]) == (N_C, 20, 40)
    ref = r["per_config"]["ref"]
    assert ref["oos_recall"] == pytest.approx(19 / 20)          # item 0 is the only OOS item the reference misses
    assert ref["in_scope_acc"] == pytest.approx(39 / 40)        # item 25 is the only in-scope miss
    fp8 = r["per_config"]["fp8"]
    assert fp8["oos_recall"] == pytest.approx(18 / 20)
    d = fp8["delta_vs_ref"]
    assert d["oos_recall"]["delta"] == pytest.approx(-1 / 20)
    assert d["in_scope_acc"]["delta"] == 0.0
    assert d["oos_recall_minus_in_scope_acc"]["delta"] == pytest.approx(-1 / 20)
    nv = r["per_config"]["nvfp4"]
    assert nv["delta_vs_ref"]["oos_recall"]["delta"] == pytest.approx(-3 / 20)
    assert nv["delta_vs_ref"]["in_scope_acc"]["delta"] == pytest.approx(-1 / 40)
    assert nv["flip_rate_oos_items"] == pytest.approx(3 / 20) and nv["flip_rate_in_scope_items"] == pytest.approx(1 / 40)
    lo, hi = nv["delta_vs_ref"]["oos_recall"]["ci95"]
    assert lo <= -3 / 20 <= hi


# ----------------------------------------------------------------------------- length

def test_length_quartiles_use_within_task_input_tokens(data):
    r = H.length(data, 300, 4)
    nv = r["per_config"]["nvfp4"]
    assert sum(nv["by_quartile"][q]["n"] for q in ("Q1", "Q2", "Q3", "Q4")) == data.n
    assert nv["overall_flip_rate"] == pytest.approx(np.mean(data.pred["nvfp4"] != data.pred["ref"]))
    # tokens equal the within-task record index, so the quartile of an item is its position in the task
    assert nv["by_quartile"]["Q1"]["n"] > 0
    assert "ref" not in r["per_config"]
    lo, hi = nv["q4_minus_q1"]["ci95"]
    assert lo <= nv["q4_minus_q1"]["delta"] <= hi


# ----------------------------------------------------------------------- confidence

def test_confidence_concentration_of_flips(data):
    r = H.confidence(data)["per_config"]["nvfp4"]
    flip = data.pred["nvfp4"] != data.pred["ref"]
    assert r["flips"] == int(flip.sum())
    assert r["max_ref_confidence_among_flips"] == pytest.approx(data.conf["ref"][flip].max())
    low = data.conf["ref"] < 0.7
    assert r["flip_rate_ref_conf_below_0_7"] == pytest.approx(flip[low].mean() if low.any() else 0.0)
    assert r["items_with_ref_conf_at_least_0_9"] == int((data.conf["ref"] >= 0.9).sum())


# ----------------------------------------------------------------------------- q4 pair

def test_q4_pair_flips_between_the_two_4bit_builds(data, idx):
    r = H.q4_pair(data, idx)
    assert r["flips_between"] == int((data.pred["q4_k_m"] != data.pred["nvfp4"]).sum())
    assert r["delta_accuracy_nvfp4_minus_q4_k_m"]["delta"] == pytest.approx(
        data.correct["nvfp4"].mean() - data.correct["q4_k_m"].mean())
    assert set(r["per_task"]) == {"clinc150", "sst5", "toyb"}
    assert r["tasks_where_nvfp4_more_accurate"] + r["tasks_where_q4_k_m_more_accurate"] <= 3


# --------------------------------------------------------------------------- score shift

def test_score_shift_center_pull_sign(data):
    r = H.score_shift(data, 300, 2)
    nv = r["per_config"]["nvfp4"]
    # nvfp4 adds mass to the middle option, so for items whose ref expected score is off-centre the pull is toward it
    assert nv["center_pull"]["delta"] < 0
    assert r["per_config"]["bf16-srv"]["center_pull"]["delta"] == 0.0
    assert r["per_config"]["bf16-srv"]["mean_signed_shift"]["ci95"] == [0.0, 0.0]


# ------------------------------------------------------------------------- 27B vs 9B

def test_big_vs_small_uses_only_finished_tasks_and_pairs_by_record(trees):
    rec, res = trees
    r = H.big_vs_small(rec, res, 200, 1)
    assert r["tasks_finished"] == ["clinc150", "toyb"] and r["tasks_missing"] == ["sst5"]
    t = r["per_task"]["clinc150"]
    assert t["acc_27b"] == pytest.approx(t["acc_9b"]) and t["top1_agreement_27b_vs_9b"] == 1.0
    assert t["delta_accuracy_27b_minus_9b"]["delta"] == 0.0
    assert r["pooled"]["n"] == N_C + N_B


def test_run_writes_all_blocks_and_cli(trees, tmp_path):
    rec, res = trees
    assert H.main(["--records", str(rec), "--results", str(res), "--out", str(tmp_path), "--n-boot", "50"]) == 0
    out = json.loads((tmp_path / "hypotheses.json").read_text())
    assert set(out) == {"meta", "calibration", "oos", "length", "confidence", "q4_pair", "score_shift", "big_vs_small",
                        "cross_size_4bit_vs_8bit"}
    assert out["meta"]["n_questions"] == N_C + N_S + N_B
    # this tree has no 27B 4-bit results at all: every pair is listed as absent, none is imputed
    cross = out["cross_size_4bit_vs_8bit"]
    assert cross["pairs"] == {} and len(cross["absent"]) == 4


# --------------------------------------------------------------- 27B 4-bit vs Clef-flash 8-bit

X_N = 10
# number of wrong answers (gold is "true" everywhere) per model/config and task; four tasks of 10 questions each
X_WRONG = {
    ("clef-flash", "ref"): {"t_a": 1, "t_b": 3, "t_c": 2, "t_d": 2},       # ref accuracy .9 .7 .8 .8: t_c and t_d tie
    ("clef-flash", "fp8"): {"t_a": 1, "t_b": 3, "t_c": 2, "t_d": 3},       # accuracy .9 .7 .8 .7
    ("clef-flash", "q8_0"): {"t_a": 1, "t_b": 3, "t_c": 3, "t_d": 2},      # accuracy .9 .7 .7 .8
    ("clef", "nvfp4"): {"t_a": 0, "t_b": 1, "t_c": 2, "t_d": 1},           # accuracy 1. .9 .8 .9
    ("clef", "q4_k_m"): {"t_a": 2, "t_b": 2, "t_c": 1, "t_d": 2},          # accuracy .8 .8 .9 .8
    ("clef", "ref"): {"t_a": 0, "t_b": 3, "t_c": 0, "t_d": 3},             # accuracy 1. .7 1. .7 (only the sensitivity splits read it)
}
X_OFFSET = {("clef-flash", "ref"): 0, ("clef-flash", "fp8"): 0, ("clef-flash", "q8_0"): 2,
            ("clef", "nvfp4"): 5, ("clef", "q4_k_m"): 7, ("clef", "ref"): 3}      # which questions are wrong: (offset + j) % 10 for j < wrong


def build_cross(root: Path, skip=(), wrong_by_cfg=None):
    rec, res = root / "records", root / "results"
    wrong_by_cfg = {**X_WRONG, **(wrong_by_cfg or {})}
    for t in ("t_a", "t_b", "t_c", "t_d"):
        write_jsonl(rec / f"{t}.jsonl", [
            {"id": f"{t}-{i}", "task": t,
             "request": {"model": "clef-flash", "state": {}, "questions": {"answer": {"type": "noul", "instructions": "x"}}},
             "gold": {"answer": "true"}} for i in range(X_N)])
    for (model, cfg), wrong in wrong_by_cfg.items():
        if (model, cfg) in skip:
            continue
        for t, k in wrong.items():
            bad = {(X_OFFSET[(model, cfg)] + j) % X_N for j in range(k)}
            rows = []
            for i in range(X_N):
                pick = "false" if i in bad else "true"
                lg = probs_for(BOPTS, pick, 2.0)
                rows.append({"id": f"{t}-{i}", "task": t, "model": model, "config": cfg,
                             "answers": {"answer": {"option_ids": BOPTS, "logits": lg.tolist(), "probs": softmax(lg).tolist()}},
                             "latency_ms": 1.0, "input_tokens": i + 1, "engine": "toy", "engine_version": "1", "error": None})
            write_jsonl(res / model / cfg / f"{t}.jsonl", rows)
    return rec, res


@pytest.fixture(scope="module")
def cross(tmp_path_factory):
    rec, res = build_cross(tmp_path_factory.mktemp("cross"))
    return H.cross_size_4bit_vs_8bit(rec, res, 300, 11)


def test_split_difficulty_sorts_ascending_with_ties_by_task_name_and_hard_is_the_lower_half():
    hard, easy = H.split_difficulty({"t_d": 0.8, "t_a": 0.9, "t_c": 0.8, "t_b": 0.7})
    assert (hard, easy) == (["t_b", "t_c"], ["t_d", "t_a"])          # t_c before t_d on the tie at the median
    assert H.split_difficulty({"b": 0.5, "a": 0.5, "c": 0.5, "d": 0.5}) == (["a", "b"], ["c", "d"])
    hard, easy = H.split_difficulty({"x": 0.3, "y": 0.2, "z": 0.1})   # odd count: len // 2 = 1 hard, the rest easy
    assert (hard, easy) == (["z"], ["y", "x"])


def test_cross_size_difficulty_is_clef_flash_ref_accuracy_median_split(cross):
    d = cross["difficulty"]
    assert d["basis"] == "clef-flash ref accuracy, median split"
    assert d["per_task"] == pytest.approx({"t_a": 0.9, "t_b": 0.7, "t_c": 0.8, "t_d": 0.8})
    assert d["hard"] == ["t_b", "t_c"] and d["easy"] == ["t_d", "t_a"]


def test_cross_size_pairs_are_keyed_big_over_small_and_nothing_is_absent(cross):
    assert sorted(cross["pairs"]) == ["clef/nvfp4|clef-flash/fp8", "clef/nvfp4|clef-flash/q8_0",
                                      "clef/q4_k_m|clef-flash/fp8", "clef/q4_k_m|clef-flash/q8_0"]
    assert cross["absent"] == []


# hand-computed (big minus small) per task t_a..t_d, then pooled over hard = [t_b, t_c] and easy = [t_d, t_a]
CROSS_EXPECT = {
    "clef/nvfp4|clef-flash/fp8": ({"t_a": .1, "t_b": .2, "t_c": 0.0, "t_d": .2}, (.85, .75), (.95, .80)),
    "clef/nvfp4|clef-flash/q8_0": ({"t_a": .1, "t_b": .2, "t_c": .1, "t_d": .1}, (.85, .70), (.95, .85)),
    "clef/q4_k_m|clef-flash/fp8": ({"t_a": -.1, "t_b": .1, "t_c": .1, "t_d": .1}, (.85, .75), (.80, .80)),
    "clef/q4_k_m|clef-flash/q8_0": ({"t_a": -.1, "t_b": .1, "t_c": .2, "t_d": 0.0}, (.85, .70), (.80, .85)),
}


@pytest.mark.parametrize("key", sorted(CROSS_EXPECT))
def test_cross_size_per_task_and_pooled_deltas_match_hand_computation(cross, key):
    per_task, (hb, hs), (eb, es) = CROSS_EXPECT[key]
    pair = cross["pairs"][key]
    assert set(pair) == {"per_task", "pooled_hard", "pooled_easy", "hard_minus_easy"}
    for t, delta in per_task.items():
        row = pair["per_task"][t]
        assert row["n"] == X_N
        assert row["delta_accuracy_big_minus_small"]["delta"] == pytest.approx(delta)
        assert row["acc_big"] - row["acc_small"] == pytest.approx(delta)
        lo, hi = row["delta_accuracy_big_minus_small"]["ci95"]
        assert lo <= delta + 1e-12 and delta - 1e-12 <= hi
    for name, (ab, asm) in (("pooled_hard", (hb, hs)), ("pooled_easy", (eb, es))):
        g = pair[name]
        assert g["n"] == 2 * X_N
        assert g["acc_big"] == pytest.approx(ab) and g["acc_small"] == pytest.approx(asm)
        assert g["delta_accuracy_big_minus_small"]["delta"] == pytest.approx(ab - asm)
    hme = pair["hard_minus_easy"]
    assert hme["delta"] == pytest.approx((hb - hs) - (eb - es))
    assert hme["ci95"][0] <= hme["delta"] <= hme["ci95"][1]
    assert hme["excludes_zero"] in (True, False)


def test_cross_size_first_pair_hard_minus_easy_is_minus_five_points(cross):
    assert cross["pairs"]["clef/nvfp4|clef-flash/fp8"]["hard_minus_easy"]["delta"] == pytest.approx(-0.05)
    assert cross["pairs"]["clef/q4_k_m|clef-flash/q8_0"]["hard_minus_easy"]["delta"] == pytest.approx(0.20)


def test_cross_size_per_task_bootstrap_uses_the_big_vs_small_seeding_pattern(cross):
    # task t_b, pair nvfp4 / fp8: rebuild the paired bootstrap from the documented seed (seed + sum of ordinals of the task name)
    big = np.ones(X_N); big[[(5 + j) % X_N for j in range(1)]] = 0.0
    small = np.ones(X_N); small[[(0 + j) % X_N for j in range(3)]] = 0.0
    idx = H.make_idx(X_N, 300, (11 + sum(map(ord, "t_b"))) % (2**32))
    want = H.est(float(big.mean() - small.mean()), H.boot_mean(big, idx) - H.boot_mean(small, idx))
    assert cross["pairs"]["clef/nvfp4|clef-flash/fp8"]["per_task"]["t_b"]["delta_accuracy_big_minus_small"] == want


def test_cross_size_is_deterministic_and_pooled_groups_use_distinct_seeds(tmp_path_factory, cross):
    rec, res = build_cross(tmp_path_factory.mktemp("cross2"))
    again = H.cross_size_4bit_vs_8bit(rec, res, 300, 11)
    assert again == cross
    # hard and easy of one pair pool different questions with different resamples: their intervals are not identical
    p = cross["pairs"]["clef/nvfp4|clef-flash/q8_0"]
    assert p["pooled_hard"]["delta_accuracy_big_minus_small"]["ci95"] != p["pooled_easy"]["delta_accuracy_big_minus_small"]["ci95"]


def test_cross_size_lists_a_pair_with_a_missing_result_dir_as_absent_and_does_not_impute_it(tmp_path):
    rec, res = build_cross(tmp_path, skip={("clef", "q4_k_m"), ("clef-flash", "q8_0")})
    out = H.cross_size_4bit_vs_8bit(rec, res, 100, 3)
    assert sorted(out["pairs"]) == ["clef/nvfp4|clef-flash/fp8"]
    assert out["absent"] == ["clef/nvfp4|clef-flash/q8_0", "clef/q4_k_m|clef-flash/fp8", "clef/q4_k_m|clef-flash/q8_0"]
    assert out["difficulty"]["hard"] == ["t_b", "t_c"]           # difficulty comes from clef-flash ref alone


def test_cross_size_with_no_clef_4bit_results_returns_only_absent(tmp_path):
    rec, res = build_cross(tmp_path, skip={("clef", "q4_k_m"), ("clef", "nvfp4")})
    out = H.cross_size_4bit_vs_8bit(rec, res, 100, 3)
    assert out["pairs"] == {} and len(out["absent"]) == 4
    assert out["difficulty"]["easy"] == ["t_d", "t_a"]


# --------------------------------------------------------------- difficulty sensitivity (alternate hard/easy splits)

SENS = "difficulty_sensitivity"


def sens_seed(seed: int, name: str) -> int:
    return (seed + sum(map(ord, name))) % (2**32)


def test_sensitivity_has_the_three_alternate_splits_and_leaves_the_primary_split_alone(cross):
    sens = cross[SENS]
    assert list(sens) == ["threshold_80", "median_mean_refs", "median_ex_boundary"]
    assert cross["difficulty"]["hard"] == ["t_b", "t_c"] and cross["difficulty"]["easy"] == ["t_d", "t_a"]
    for entry in sens.values():
        assert entry["basis"] and isinstance(entry["basis"], str)
        assert sorted(entry["hard"] + entry["easy"]) == ["t_a", "t_b", "t_c", "t_d"]
        assert sorted(entry["pairs"]) == sorted(cross["pairs"])


def test_threshold_80_hard_is_ref_accuracy_strictly_below_point_eight(cross):
    e = cross[SENS]["threshold_80"]
    assert e["hard"] == ["t_b"]                       # .7 only: t_c and t_d sit exactly on .80 and are not hard
    assert sorted(e["easy"]) == ["t_a", "t_c", "t_d"]
    assert "same_as" not in e


def test_median_mean_refs_splits_on_the_mean_of_the_flash_and_27b_ref_accuracies(cross):
    e = cross[SENS]["median_mean_refs"]
    # mean refs: t_a .95, t_b .70, t_c .90, t_d .75
    assert e["per_task"] == pytest.approx({"t_a": .95, "t_b": .70, "t_c": .90, "t_d": .75})
    assert e["hard"] == ["t_b", "t_d"] and e["easy"] == ["t_c", "t_a"]
    assert "same_as" not in e


def test_median_ex_boundary_swaps_the_hardest_easy_task_with_the_easiest_hard_task(cross):
    e = cross[SENS]["median_ex_boundary"]
    # primary: hard [t_b, t_c], easy [t_d, t_a]; t_c (easiest hard) goes to easy, t_d (hardest easy) goes to hard
    assert e["hard"] == ["t_b", "t_d"] and e["easy"] == ["t_c", "t_a"]


def test_identical_splits_are_both_reported_and_the_later_one_says_same_as(cross):
    sens = cross[SENS]
    assert (sens["median_ex_boundary"]["hard"], sens["median_ex_boundary"]["easy"]) == (
        sens["median_mean_refs"]["hard"], sens["median_mean_refs"]["easy"])
    assert sens["median_ex_boundary"]["same_as"] == "median_mean_refs"
    assert "same_as" not in sens["median_mean_refs"]
    assert sens["median_ex_boundary"]["pairs"].keys() == sens["median_mean_refs"]["pairs"].keys()   # still fully reported


def test_a_split_equal_to_the_primary_split_says_same_as_primary(tmp_path):
    # clef ref accuracy 1. .6 .8 1.: mean refs .95 .65 .80 .90, so the median split is [t_b, t_c] / [t_d, t_a] like the primary
    rec, res = build_cross(tmp_path, wrong_by_cfg={("clef", "ref"): {"t_a": 0, "t_b": 4, "t_c": 2, "t_d": 0}})
    out = H.cross_size_4bit_vs_8bit(rec, res, 100, 3)
    e = out[SENS]["median_mean_refs"]
    assert e["hard"] == ["t_b", "t_c"] and e["easy"] == ["t_d", "t_a"]
    assert e["same_as"] == "primary"


@pytest.mark.parametrize("split", ["threshold_80", "median_mean_refs", "median_ex_boundary"])
@pytest.mark.parametrize("key", sorted(CROSS_EXPECT))
def test_sensitivity_pooled_deltas_are_the_mean_of_the_hand_computed_per_task_deltas(cross, split, key):
    per_task = CROSS_EXPECT[key][0]
    e = cross[SENS][split]
    row = e["pairs"][key]
    assert set(row) == {"pooled_hard", "pooled_easy", "hard_minus_easy"}
    want = {}
    for name in ("hard", "easy"):
        ts = e[name]
        g = row[f"pooled_{name}"]
        want[name] = sum(per_task[t] for t in ts) / len(ts)          # every task has the same n, so pooled = mean of tasks
        assert g["n"] == X_N * len(ts)
        assert g["delta_accuracy_big_minus_small"]["delta"] == pytest.approx(want[name])
        assert g["acc_big"] - g["acc_small"] == pytest.approx(want[name])
        lo, hi = g["delta_accuracy_big_minus_small"]["ci95"]
        assert lo <= want[name] + 1e-12 and want[name] - 1e-12 <= hi
    assert row["hard_minus_easy"]["delta"] == pytest.approx(want["hard"] - want["easy"])


def test_sensitivity_pooled_bootstrap_uses_a_per_split_seed_name(cross):
    # threshold_80, hard = [t_b] only; pair nvfp4 / fp8: big wrong at {5}, small wrong at {0, 1, 2}
    big = np.ones(X_N); big[[5]] = 0.0
    small = np.ones(X_N); small[[0, 1, 2]] = 0.0
    idx = H.make_idx(X_N, 300, sens_seed(11, "pooled-threshold_80-hard"))
    want = H.est(float(big.mean() - small.mean()), H.boot_mean(big, idx) - H.boot_mean(small, idx))
    got = cross[SENS]["threshold_80"]["pairs"]["clef/nvfp4|clef-flash/fp8"]["pooled_hard"]["delta_accuracy_big_minus_small"]
    assert got == want


def test_sensitivity_is_deterministic_and_splits_use_distinct_resamples(tmp_path_factory, cross):
    rec, res = build_cross(tmp_path_factory.mktemp("cross3"))
    assert H.cross_size_4bit_vs_8bit(rec, res, 300, 11)[SENS] == cross[SENS]
    a = cross[SENS]["median_mean_refs"]["pairs"]["clef/nvfp4|clef-flash/q8_0"]["pooled_hard"]["delta_accuracy_big_minus_small"]
    b = cross[SENS]["median_ex_boundary"]["pairs"]["clef/nvfp4|clef-flash/q8_0"]["pooled_hard"]["delta_accuracy_big_minus_small"]
    assert a["delta"] == b["delta"] and a["ci95"] != b["ci95"]          # same sets, different seed names


def test_median_mean_refs_is_skipped_when_there_is_no_clef_ref(tmp_path):
    rec, res = build_cross(tmp_path, skip={("clef", "ref")})
    out = H.cross_size_4bit_vs_8bit(rec, res, 100, 3)
    assert list(out[SENS]) == ["threshold_80", "median_ex_boundary"]
    assert "same_as" not in out[SENS]["median_ex_boundary"]
    assert sorted(out["pairs"]) == sorted(out[SENS]["threshold_80"]["pairs"])


def test_median_mean_refs_is_skipped_when_the_clef_ref_misses_a_task(tmp_path):
    rec, res = build_cross(tmp_path)
    (res / "clef" / "ref" / "t_d.jsonl").unlink()
    out = H.cross_size_4bit_vs_8bit(rec, res, 100, 3)
    assert list(out[SENS]) == ["threshold_80", "median_ex_boundary"]


def test_sensitivity_without_present_pairs_keeps_the_sets_and_has_no_pair_estimates(tmp_path):
    rec, res = build_cross(tmp_path, skip={("clef", "q4_k_m"), ("clef", "nvfp4")})
    out = H.cross_size_4bit_vs_8bit(rec, res, 100, 3)
    assert out["pairs"] == {}
    assert out[SENS]["threshold_80"]["hard"] == ["t_b"]
    assert all(e["pairs"] == {} for e in out[SENS].values())


def test_threshold_split_with_an_empty_group_fails_loudly(tmp_path):
    # every flash ref accuracy is at or above .80: no hard task under the threshold
    rec, res = build_cross(tmp_path, wrong_by_cfg={("clef-flash", "ref"): {"t_a": 0, "t_b": 1, "t_c": 2, "t_d": 2}})
    with pytest.raises(ValueError, match="threshold_80"):
        H.cross_size_4bit_vs_8bit(rec, res, 100, 3)
