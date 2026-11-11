"""Metric tests on tiny synthetic inputs. Every expected value below is computed by hand."""

import math

import numpy as np
import pytest
from scipy import stats

from bench import metrics as M


# ------------------------------------------------------------------ distributions

def test_dist_from_logits_is_softmax():
    p = M.dist(logits=[0.0, 0.0])
    assert p == pytest.approx([0.5, 0.5])
    p = M.dist(logits=[math.log(3), 0.0])  # softmax -> 0.75 / 0.25
    assert p == pytest.approx([0.75, 0.25])


def test_dist_prefers_logits_over_probs_when_both_present():
    p = M.dist(logits=[0.0, 0.0], probs=[0.9, 0.1])
    assert p == pytest.approx([0.5, 0.5])


def test_dist_from_probs_clips_and_renormalises():
    p = M.dist(probs=[1.0, 0.0])
    assert p[1] == pytest.approx(M.EPS / (1.0 + M.EPS))
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)


# ------------------------------------------------------------------------ KL / TVD

def test_kl_hand_computed():
    # 0.5 ln(0.5/0.9) + 0.5 ln(0.5/0.1) = 0.5 ln(25/9)
    assert M.kl_div([0.5, 0.5], [0.9, 0.1]) == pytest.approx(0.5 * math.log(25 / 9))


def test_kl_identical_is_zero_and_direction_matters():
    assert M.kl_div([0.3, 0.7], [0.3, 0.7]) == pytest.approx(0.0, abs=1e-12)
    # KL(ref || cfg) != KL(cfg || ref)
    assert M.kl_div([0.9, 0.1], [0.5, 0.5]) == pytest.approx(0.9 * math.log(1.8) + 0.1 * math.log(0.2))
    assert M.kl_div([0.9, 0.1], [0.5, 0.5]) != pytest.approx(M.kl_div([0.5, 0.5], [0.9, 0.1]))


def test_kl_epsilon_clipping_keeps_zero_probabilities_finite():
    value = M.kl_div([0.5, 0.5], [1.0, 0.0])
    expected = 0.5 * math.log(0.5) + 0.5 * math.log(0.5 / M.EPS)
    assert math.isfinite(value)
    assert value == pytest.approx(expected, rel=1e-6)


def test_tvd_hand_computed():
    assert M.tvd([0.5, 0.5], [0.9, 0.1]) == pytest.approx(0.4)
    assert M.tvd([1.0, 0.0, 0.0], [0.0, 0.5, 0.5]) == pytest.approx(1.0, abs=1e-9)
    assert M.tvd([0.2, 0.8], [0.2, 0.8]) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------- Brier / ECE / F1

def test_brier_hand_computed():
    # (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01
    assert M.brier([0.7, 0.2, 0.1], 0) == pytest.approx(0.14)
    # gold index 2: 0.7^2 + 0.2^2 + (0.1-1)^2 = 0.49 + 0.04 + 0.81
    assert M.brier([0.7, 0.2, 0.1], 2) == pytest.approx(1.34)


def test_ece_hand_computed_two_bins():
    # 0.9 -> bin 13 (acc 0.5), 0.3 -> bin 4 (acc 1.0): 0.5*|0.5-0.9| + 0.5*|1.0-0.3| = 0.55
    assert M.ece([0.9, 0.9, 0.3, 0.3], [1, 0, 1, 1]) == pytest.approx(0.55)


def test_ece_confidence_one_lands_in_last_bin_and_perfect_is_zero():
    assert M.ece([1.0, 1.0], [1, 1]) == pytest.approx(0.0)
    assert M.ece([1.0, 1.0], [1, 0]) == pytest.approx(0.5)  # one bin: |0.5 - 1.0|


def test_macro_f1_hand_computed():
    gold = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 0])
    # class0: tp1 fp1 fn1 -> 0.5 ; class1: tp2 fp1 fn0 -> 0.8 ; class2: tp0 -> 0
    assert M.macro_f1(gold, pred) == pytest.approx((0.5 + 0.8 + 0.0) / 3)


def test_macro_f1_uses_union_of_gold_and_predicted_labels():
    gold = np.array([0, 0])
    pred = np.array([0, 1])  # class 1 only predicted: F1 0; class 0: tp1 fn1 -> 2/3
    assert M.macro_f1(gold, pred) == pytest.approx((2 / 3 + 0.0) / 2)


# --------------------------------------------------------------------- McNemar

def test_mcnemar_exact_hand_computed():
    assert M.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert M.mcnemar_exact(9, 1) == pytest.approx(22 / 1024)
    assert M.mcnemar_exact(0, 4) == pytest.approx(2 / 16)
    assert M.mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert M.mcnemar_exact(0, 0) == pytest.approx(1.0)


def test_mcnemar_exact_handles_large_discordant_counts_without_overflow():
    assert M.mcnemar_exact(1000, 1000) == pytest.approx(1.0)
    assert M.mcnemar_exact(0, 3000) == 0.0  # underflows to zero rather than raising
    assert M.mcnemar_exact(400, 450) == pytest.approx(stats.binomtest(400, 850, 0.5).pvalue, rel=1e-6)


def test_mcnemar_exact_matches_scipy_binomtest():
    for b, c in [(3, 11), (7, 8), (0, 12), (20, 4)]:
        assert M.mcnemar_exact(b, c) == pytest.approx(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


# ------------------------------------------------------------------- bootstrap

def test_bootstrap_degenerate_all_flipped_gives_point_interval():
    gold = np.zeros(20, dtype=int)
    ref = np.zeros(20, dtype=int)  # all correct
    cfg = np.ones(20, dtype=int)  # all wrong
    out = M.bootstrap_paired(gold, ref, cfg, n_boot=200, seed=1)
    assert out["accuracy"]["delta"] == pytest.approx(-1.0)
    assert out["accuracy"]["ci95"] == pytest.approx([-1.0, -1.0])


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 3, 60)
    ref = np.where(rng.random(60) < 0.8, gold, (gold + 1) % 3)
    cfg = np.where(rng.random(60) < 0.7, gold, (gold + 2) % 3)
    a = M.bootstrap_paired(gold, ref, cfg, n_boot=300, seed=7)
    b = M.bootstrap_paired(gold, ref, cfg, n_boot=300, seed=7)
    c = M.bootstrap_paired(gold, ref, cfg, n_boot=300, seed=8)
    assert a == b
    assert a["accuracy"]["ci95"] != c["accuracy"]["ci95"]


def test_bootstrap_interval_brackets_point_estimate():
    gold = np.zeros(10, dtype=int)
    ref = np.zeros(10, dtype=int)
    cfg = np.zeros(10, dtype=int)
    cfg[0] = 1  # 9/10 correct vs 10/10
    out = M.bootstrap_paired(gold, ref, cfg, n_boot=2000, seed=20261015)["accuracy"]
    assert out["delta"] == pytest.approx(-0.1)
    lo, hi = out["ci95"]
    # each resample's delta is -(wrong draws)/10 <= 0, and P(no wrong draw) = 0.9**10 = 0.35
    assert hi == pytest.approx(0.0)
    assert -0.5 <= lo <= -0.1


# -------------------------------------------------------------------- flip buckets

def test_flip_buckets_hand_computed():
    conf = [0.1, 0.5, 0.69, 0.7, 0.89, 0.9, 0.98, 0.99, 1.0]
    flipped = [1, 0, 1, 0, 0, 1, 0, 0, 1]
    got = M.flip_buckets(conf, flipped)
    assert [b["bucket"] for b in got] == ["[0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,0.99)", "[0.99,1]"]
    assert [b["n"] for b in got] == [1, 2, 2, 2, 2]
    assert [b["flips"] for b in got] == [1, 1, 0, 1, 1]
    assert [b["flip_rate"] for b in got] == pytest.approx([1.0, 0.5, 0.0, 0.5, 0.5])


def test_flip_buckets_empty_bucket_has_no_rate():
    got = M.flip_buckets([0.995], [0])
    assert got[0]["n"] == 0 and got[0]["flip_rate"] is None
    assert got[4]["n"] == 1 and got[4]["flip_rate"] == 0.0


# ------------------------------------------------------------------- score shift

def test_expected_score_and_shift():
    ref = [np.array([0.1, 0.2, 0.7])]  # E = 0.2 + 1.4 = 1.6
    cfg = [np.array([0.2, 0.3, 0.5])]  # E = 0.3 + 1.0 = 1.3
    assert M.expected_score(ref[0]) == pytest.approx(1.6)
    out = M.score_shift(ref, cfg)
    assert out["mean_signed_shift"] == pytest.approx(-0.3)
    assert out["mean_abs_shift"] == pytest.approx(0.3)
    # middle of a 3-level scale is 1.0: |1.3-1| - |1.6-1| = -0.3 -> toward the middle
    assert out["center_pull"] == pytest.approx(-0.3)
    assert out["shift_when_ref_above_middle"] == pytest.approx(-0.3)
    assert out["shift_when_ref_below_middle"] is None


def test_score_shift_toward_the_ends_is_positive_center_pull():
    ref = [np.array([0.0, 1.0, 0.0])]  # E = 1.0 (middle)
    cfg = [np.array([0.0, 0.5, 0.5])]  # E = 1.5
    assert M.score_shift(ref, cfg)["center_pull"] == pytest.approx(0.5)


# ----------------------------------------------------------------------- cascade

def test_cascade_hand_computed():
    gold = np.array([0, 1, 1, 0])
    ref_pred = np.array([0, 1, 0, 0])
    cfg_pred = np.array([0, 0, 1, 1])
    cfg_conf = np.array([0.95, 0.6, 0.8, 0.55])
    rows = M.cascade(gold, ref_pred, cfg_pred, cfg_conf, thresholds=[0.5, 0.7, 0.9])
    # t=0.5: nobody escalates -> predictions = cfg_pred
    assert rows[0]["escalation_rate"] == 0.0
    assert rows[0]["agreement_with_ref"] == pytest.approx(0.25)  # only item 0 agrees
    assert rows[0]["accuracy"] == pytest.approx(0.5)  # items 0 and 2 correct
    # t=0.7: items 1 (0.6) and 3 (0.55) escalate -> preds [0, 1, 1, 0]
    assert rows[1]["escalation_rate"] == pytest.approx(0.5)
    assert rows[1]["agreement_with_ref"] == pytest.approx(0.75)
    assert rows[1]["accuracy"] == pytest.approx(1.0)
    # t=0.9: only item 0 stays -> preds [0, 1, 0, 0] == ref
    assert rows[2]["escalation_rate"] == pytest.approx(0.75)
    assert rows[2]["agreement_with_ref"] == pytest.approx(1.0)
    assert rows[2]["accuracy"] == pytest.approx(0.75)


def test_cascade_escalates_exactly_below_threshold():
    rows = M.cascade(np.array([0]), np.array([0]), np.array([0]), np.array([0.7]), thresholds=[0.7])
    assert rows[0]["escalation_rate"] == 0.0  # conf == t stays on the cheap model


# ------------------------------------------------------------------------ latency

def test_latency_stats_hand_computed():
    out = M.latency_stats([10, 20, 30, 40, 50])
    assert out["p50_ms"] == pytest.approx(30.0)
    assert out["p95_ms"] == pytest.approx(48.0)  # 40 + 0.8 * 10 (linear interpolation)
    assert out["decisions_per_sec"] == pytest.approx(5 / 0.150)


def test_latency_stats_handles_empty_input():
    out = M.latency_stats([])
    assert out == {"n": 0, "p50_ms": None, "p95_ms": None, "decisions_per_sec": None}
