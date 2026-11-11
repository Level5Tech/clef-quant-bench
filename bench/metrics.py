"""Pure metric functions for the Clef quantization benchmark.

Conventions (all documented here so the published numbers are reproducible):

* Distributions. For one question, `dist(logits, probs)` returns a float64 probability vector over
  the question's options. Logits are used when present (stable softmax); otherwise the runner's
  probabilities are used. Either way the vector is clipped at EPS = 1e-12 and renormalised, so
  KL(ref || cfg) is always finite. The clip changes a probability by at most ~1e-12.
* KL(ref || cfg) = sum_k p_ref[k] * (ln p_ref[k] - ln p_cfg[k]), natural log, per question.
* TVD = 0.5 * sum_k |p_ref[k] - p_cfg[k]|.
* Brier = sum_k (p[k] - 1[k == gold])^2 per question (multi-class form, range 0..2).
* ECE: 15 equal-width confidence bins on [0, 1] (confidence = max probability, 1.0 falls in the
  last bin); ECE = sum_b (n_b / N) * |accuracy_b - mean confidence_b|.
* Macro-F1 averages per-class F1 over the union of gold and predicted labels.
* Ties in the top-1 prediction break toward the first option in the question's option order.
* McNemar: exact two-sided binomial test on discordant correctness pairs (b: ref right, cfg wrong;
  c: ref wrong, cfg right).
* Bootstrap: paired, percentile 95% interval on (cfg - ref); the same resampled item indices are
  applied to both configs.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12
ECE_BINS = 15
BOOTSTRAP_RESAMPLES = 2000
SEED = 20261015
FLIP_EDGES = (0.5, 0.7, 0.9, 0.99)
FLIP_LABELS = ("[0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,0.99)", "[0.99,1]")
CASCADE_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
_BOOT_CHUNK = 250


# ------------------------------------------------------------------ distributions

def dist(logits: Sequence[float] | None = None, probs: Sequence[float] | None = None) -> np.ndarray:
    """Probability vector (float64) from logits if given, else from probs; clipped at EPS, renormalised."""
    if logits is not None:
        z = np.asarray(logits, dtype=np.float64)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
    elif probs is not None:
        p = np.asarray(probs, dtype=np.float64)
    else:
        raise ValueError("need logits or probs")
    p = np.clip(p, EPS, None)
    return p / p.sum()


def kl_div(p_ref: Sequence[float], p_cfg: Sequence[float]) -> float:
    """KL(ref || cfg) in nats, with EPS clipping applied to both inputs."""
    p = _clip_norm(p_ref)
    q = _clip_norm(p_cfg)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def tvd(p: Sequence[float], q: Sequence[float]) -> float:
    return float(0.5 * np.abs(_clip_norm(p) - _clip_norm(q)).sum())


def _clip_norm(p: Sequence[float]) -> np.ndarray:
    a = np.clip(np.asarray(p, dtype=np.float64), EPS, None)
    return a / a.sum()


# -------------------------------------------------------------- gold-based metrics

def brier(p: Sequence[float], gold_idx: int) -> float:
    a = np.asarray(p, dtype=np.float64)
    onehot = np.zeros_like(a)
    onehot[gold_idx] = 1.0
    return float(np.sum((a - onehot) ** 2))


def ece(confidence: Iterable[float], correct: Iterable[float], n_bins: int = ECE_BINS) -> float:
    conf = np.asarray(list(confidence), dtype=np.float64)
    corr = np.asarray(list(correct), dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(corr[mask].mean() - conf[mask].mean())
    return float(total)


def macro_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    labels = np.union1d(gold, pred)
    f1s = []
    for lab in labels:
        tp = np.sum((gold == lab) & (pred == lab))
        denom = np.sum(gold == lab) + np.sum(pred == lab)
        f1s.append(2.0 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def _macro_f1_batch(gold: np.ndarray, pred: np.ndarray, idx: np.ndarray, n_labels: int) -> np.ndarray:
    """Macro-F1 for each row of resample indices `idx` (shape B x n). Labels are ints in [0, n_labels)."""
    g = gold[idx]
    p = pred[idx]
    b = idx.shape[0]
    offset = (np.arange(b) * n_labels)[:, None]
    size = b * n_labels
    gold_cnt = np.bincount((g + offset).ravel(), minlength=size).reshape(b, n_labels)
    pred_cnt = np.bincount((p + offset).ravel(), minlength=size).reshape(b, n_labels)
    hit = g == p
    tp = np.bincount((g + offset)[hit], minlength=size).reshape(b, n_labels)
    denom = gold_cnt + pred_cnt
    present = denom > 0
    f1 = np.where(present, 2.0 * tp / np.where(present, denom, 1), 0.0)
    return f1.sum(axis=1) / np.maximum(present.sum(axis=1), 1)


# ---------------------------------------------------------------------- McNemar

def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Exact integer arithmetic: 2**n overflows a float for n > 1023 (pooled flips easily exceed that).
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(1.0, (2 * tail) / (1 << n)))


# --------------------------------------------------------------------- bootstrap

def _encode(gold: np.ndarray, *preds: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], int]:
    labels = np.unique(np.concatenate([gold, *preds]))
    lut = {v: i for i, v in enumerate(labels.tolist())}
    enc = lambda a: np.array([lut[v] for v in a.tolist()], dtype=np.int64)  # noqa: E731
    return enc(gold), [enc(p) for p in preds], len(labels)


def bootstrap_arrays(gold, ref_pred, cfg_pred, n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> dict[str, np.ndarray]:
    """Per-resample paired deltas (cfg - ref): accuracy, macro-F1 and the raw count of correct answers."""
    g, (r, c), n_labels = _encode(np.asarray(gold), np.asarray(ref_pred), np.asarray(cfg_pred))
    n = len(g)
    rng = np.random.default_rng(seed)
    d_acc, d_f1, d_cnt = [], [], []
    done = 0
    while done < n_boot:
        bsz = min(_BOOT_CHUNK, n_boot - done)
        idx = rng.integers(0, n, size=(bsz, n))
        ref_ok = (r[idx] == g[idx]).sum(axis=1)
        cfg_ok = (c[idx] == g[idx]).sum(axis=1)
        d_cnt.append(cfg_ok - ref_ok)
        d_acc.append((cfg_ok - ref_ok) / n)
        d_f1.append(_macro_f1_batch(g, c, idx, n_labels) - _macro_f1_batch(g, r, idx, n_labels))
        done += bsz
    return {"d_acc": np.concatenate(d_acc), "d_f1": np.concatenate(d_f1), "d_correct": np.concatenate(d_cnt)}


def ci95(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_paired(gold, ref_pred, cfg_pred, n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> dict[str, dict]:
    gold, ref_pred, cfg_pred = np.asarray(gold), np.asarray(ref_pred), np.asarray(cfg_pred)
    arrays = bootstrap_arrays(gold, ref_pred, cfg_pred, n_boot, seed)
    return {
        "accuracy": {"delta": float(np.mean(cfg_pred == gold) - np.mean(ref_pred == gold)), "ci95": ci95(arrays["d_acc"])},
        "macro_f1": {"delta": macro_f1(gold, cfg_pred) - macro_f1(gold, ref_pred), "ci95": ci95(arrays["d_f1"])},
        "n_boot": n_boot, "seed": seed,
    }


# ------------------------------------------------------------------ flip buckets

def flip_buckets(ref_conf: Iterable[float], flipped: Iterable[float]) -> list[dict]:
    conf = np.asarray(list(ref_conf), dtype=np.float64)
    flip = np.asarray(list(flipped), dtype=np.float64)
    idx = np.digitize(conf, FLIP_EDGES) if conf.size else np.array([], dtype=int)
    out = []
    for b, label in enumerate(FLIP_LABELS):
        mask = idx == b
        n = int(mask.sum())
        flips = int(flip[mask].sum()) if n else 0
        out.append({"bucket": label, "n": n, "flips": flips, "flip_rate": (flips / n) if n else None})
    return out


# --------------------------------------------------------------------- score tasks

def expected_score(p: Sequence[float]) -> float:
    a = np.asarray(p, dtype=np.float64)
    return float(np.sum(np.arange(len(a)) * a))


def score_shift(ref_ps: Sequence[Sequence[float]], cfg_ps: Sequence[Sequence[float]]) -> dict:
    """How quantization moves the expected score. center_pull < 0 means toward the middle of the scale."""
    if not len(ref_ps):
        return {"n": 0, "mean_signed_shift": None, "mean_abs_shift": None, "center_pull": None,
                "shift_when_ref_above_middle": None, "shift_when_ref_below_middle": None}
    e_ref = np.array([expected_score(p) for p in ref_ps])
    e_cfg = np.array([expected_score(p) for p in cfg_ps])
    mid = np.array([(len(p) - 1) / 2.0 for p in ref_ps])
    shift = e_cfg - e_ref
    above, below = e_ref > mid, e_ref < mid
    return {
        "n": int(len(shift)),
        "mean_signed_shift": float(shift.mean()),
        "mean_abs_shift": float(np.abs(shift).mean()),
        "center_pull": float((np.abs(e_cfg - mid) - np.abs(e_ref - mid)).mean()),
        "shift_when_ref_above_middle": float(shift[above].mean()) if above.any() else None,
        "shift_when_ref_below_middle": float(shift[below].mean()) if below.any() else None,
    }


# ------------------------------------------------------------------------ cascade

def cascade(gold, ref_pred, cfg_pred, cfg_conf, thresholds: Sequence[float] = CASCADE_THRESHOLDS) -> list[dict]:
    """Use cfg unless its top probability is < t, then escalate to ref."""
    gold, ref_pred, cfg_pred = np.asarray(gold), np.asarray(ref_pred), np.asarray(cfg_pred)
    conf = np.asarray(cfg_conf, dtype=np.float64)
    rows = []
    for t in thresholds:
        esc = conf < t
        pred = np.where(esc, ref_pred, cfg_pred)
        rows.append({"threshold": float(t), "escalation_rate": float(esc.mean()),
                     "agreement_with_ref": float(np.mean(pred == ref_pred)),
                     "accuracy": float(np.mean(pred == gold))})
    return rows


# ------------------------------------------------------------------------ latency

def latency_stats(latencies_ms: Sequence[float]) -> dict:
    """p50/p95 per request; decisions/sec assumes requests ran one after another (n / sum of latencies)."""
    a = np.asarray([x for x in latencies_ms if x is not None], dtype=np.float64)
    if a.size == 0 or a.sum() <= 0:
        return {"n": int(a.size), "p50_ms": None, "p95_ms": None, "decisions_per_sec": None}
    return {"n": int(a.size), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "decisions_per_sec": float(a.size / (a.sum() / 1000.0))}
