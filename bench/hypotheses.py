#!/usr/bin/env python
"""Numbers for the pre-registered hypotheses H3-H6, H9 and H11 that bench.analyze does not produce, on PUBLIC data.

All of it is computed on the full public sets of Clef-flash (31,275 questions) unless stated, with 2000 paired
bootstrap resamples over questions (one shared index matrix, seed metrics.SEED), so every difference between two
configs is paired on the same questions.

  calibration  per config vs ref: delta accuracy, delta ECE(15), delta Brier with 95% CIs (H4), pooled over all tasks.
  oos          clinc150 split by gold class: out-of-scope recall vs in-scope accuracy, change vs ref, and the
               difference of the two changes with a CI (H5).
  length       flip rate vs ref by input-length quartile (quartiles inside each task, pooled), Q4 minus Q1 with CI (H6).
  confidence   where the flips are: share of flips with ref confidence < 0.7, highest ref confidence of any flip (H3).
  q4_pair      Q4_K_M vs NVFP4 directly: flip rate, delta accuracy / ECE / Brier with CIs, per-task accuracy (H11).
  score_shift  sst5 expected-score shift and center pull vs ref with CIs (H9).
  big_vs_small the 27B reference against the 9B reference on the public sets the 27B has finished (absolute only).
  cross_size_4bit_vs_8bit  27B 4-bit (nvfp4, q4_k_m) against Clef-flash 8-bit (fp8, q8_0), paired on the same questions: accuracy
               difference per task and pooled over the hard and the easy half of the tasks (hard = lowest Clef-flash ref
               accuracy), and the hard-minus-easy difference. Supplementary to H8, which it does not pre-register; the split is
               chosen from Clef-flash ref alone.

Usage:
  python -m bench.hypotheses --records data/records --results results --out analysis
writes <out>/hypotheses.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):  # allow `python bench/hypotheses.py`
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from bench import decompose as D  # noqa: E402
from bench import metrics as M  # noqa: E402

FLASH = "clef-flash"
BIG = "clef"
CROSS_BIG = ("nvfp4", "q4_k_m")
CROSS_SMALL = ("fp8", "q8_0")
CONFIGS = ("ref", "bf16-srv", "fp8", "nvfp4", "gguf-bf16", "q8_0", "q4_k_m", "ref-r2")
OOS_LABEL = "out_of_scope"
CHUNK = 100


# --------------------------------------------------------------------------- data

class Data:
    def __init__(self, records_dir: Path, results_dir: Path, configs: Sequence[str] = CONFIGS, model: str = FLASH):
        self.unit = D.build_unit(Path(records_dir), None)
        u = self.unit
        if any(g is None for g in u.gold):
            raise ValueError("public records without gold")
        self.n = u.n_questions
        self.task = np.array(u.task_of)
        self.cfgs = {c: D.load_cfg(u, Path(results_dir), model, c) for c in configs}
        self.gold_idx = np.array([u.opts[i].index(u.gold[i]) for i in range(self.n)])
        self.pred = {c: np.array([int(np.argmax(p)) for p in cfg.p]) for c, cfg in self.cfgs.items()}
        self.conf = {c: np.array([float(np.max(p)) for p in cfg.p]) for c, cfg in self.cfgs.items()}
        self.correct = {c: (self.pred[c] == self.gold_idx).astype(float) for c in self.cfgs}
        self.brier = {c: np.array([M.brier(p, g) for p, g in zip(cfg.p, self.gold_idx)]) for c, cfg in self.cfgs.items()}
        tok = {}
        for task in u.tasks:
            for line in D.read_jsonl(Path(results_dir) / model / "ref" / f"{task}.jsonl"):
                tok[(task, line["id"])] = line.get("input_tokens")
        self.tokens = np.array([np.nan if tok.get((u.task_of[i], u.qkeys[i][0])) is None
                                else float(tok[(u.task_of[i], u.qkeys[i][0])]) for i in range(self.n)])


# --------------------------------------------------------------------- bootstrap

def make_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n), dtype=np.int32)


def boot_mean(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.empty(idx.shape[0])
    for s in range(0, idx.shape[0], CHUNK):
        out[s:s + CHUNK] = x[idx[s:s + CHUNK]].mean(axis=1)
    return out


def boot_ece(conf: np.ndarray, correct: np.ndarray, idx: np.ndarray, n_bins: int = M.ECE_BINS) -> np.ndarray:
    bins = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    out = np.empty(idx.shape[0])
    for s in range(0, idx.shape[0], CHUNK):
        ii = idx[s:s + CHUNK]
        b = ii.shape[0]
        key = (bins[ii] + (np.arange(b) * n_bins)[:, None]).ravel()
        cnt = np.bincount(key, minlength=b * n_bins).reshape(b, n_bins).astype(float)
        sc = np.bincount(key, weights=conf[ii].ravel(), minlength=b * n_bins).reshape(b, n_bins)
        sk = np.bincount(key, weights=correct[ii].ravel(), minlength=b * n_bins).reshape(b, n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.where(cnt > 0, np.abs(sk / cnt - sc / cnt), 0.0)
        out[s:s + CHUNK] = (cnt / ii.shape[1] * gap).sum(axis=1)
    return out


def ci(v: np.ndarray) -> list[float]:
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def est(point: float, boots: np.ndarray) -> dict[str, Any]:
    return {"delta": float(point), "ci95": ci(boots), "excludes_zero": bool(ci(boots)[0] > 0 or ci(boots)[1] < 0)}


def point_ece(conf: np.ndarray, correct: np.ndarray) -> float:
    return M.ece(conf, correct)


# ----------------------------------------------------------------------- analyses

def calibration(d: Data, idx: np.ndarray) -> dict[str, Any]:
    boot = {c: {"acc": boot_mean(d.correct[c], idx), "ece": boot_ece(d.conf[c], d.correct[c], idx),
                "brier": boot_mean(d.brier[c], idx)} for c in d.cfgs}
    point = {c: {"acc": float(d.correct[c].mean()), "ece": point_ece(d.conf[c], d.correct[c]),
                 "brier": float(d.brier[c].mean()), "mean_confidence": float(d.conf[c].mean())} for c in d.cfgs}
    out: dict[str, Any] = {"n": d.n, "per_config": {}}
    for c in d.cfgs:
        row = {"accuracy": point[c]["acc"], "ece": point[c]["ece"], "brier": point[c]["brier"],
               "mean_confidence": point[c]["mean_confidence"],
               "ece_ci95": ci(boot[c]["ece"]), "brier_ci95": ci(boot[c]["brier"])}
        if c != "ref":
            row["delta_vs_ref"] = {
                "accuracy": est(point[c]["acc"] - point["ref"]["acc"], boot[c]["acc"] - boot["ref"]["acc"]),
                "ece": est(point[c]["ece"] - point["ref"]["ece"], boot[c]["ece"] - boot["ref"]["ece"]),
                "brier": est(point[c]["brier"] - point["ref"]["brier"], boot[c]["brier"] - boot["ref"]["brier"]),
                "mean_confidence": {"delta": point[c]["mean_confidence"] - point["ref"]["mean_confidence"]}}
        out["per_config"][c] = row
    # engine-matched: quantized build against the BF16 build in the same engine (isolates quantization from engine)
    out["within_engine"] = {}
    for base, c in (("bf16-srv", "fp8"), ("bf16-srv", "nvfp4"), ("gguf-bf16", "q8_0"), ("gguf-bf16", "q4_k_m")):
        if base in d.cfgs and c in d.cfgs:
            out["within_engine"][f"{base}|{c}"] = {
                "accuracy": est(point[c]["acc"] - point[base]["acc"], boot[c]["acc"] - boot[base]["acc"]),
                "ece": est(point[c]["ece"] - point[base]["ece"], boot[c]["ece"] - boot[base]["ece"]),
                "brier": est(point[c]["brier"] - point[base]["brier"], boot[c]["brier"] - boot[base]["brier"])}
    return out


def oos(d: Data, n_boot: int, seed: int) -> dict[str, Any]:
    mask = (d.task == "clinc150")
    items = np.flatnonzero(mask)
    gold_oos = np.array([d.unit.gold[i] == OOS_LABEL for i in items])
    n = len(items)
    idx = make_idx(n, n_boot, seed)
    oos_pred = {}
    for c, cfg in d.cfgs.items():
        oos_pred[c] = np.array([d.unit.opts[i][d.pred[c][i]] == OOS_LABEL for i in items])
    correct = {c: d.correct[c][items] for c in d.cfgs}
    g_o, g_i = gold_oos.astype(float), (~gold_oos).astype(float)

    def metrics(c: str) -> dict[str, np.ndarray]:
        tp_o = (gold_oos & oos_pred[c]).astype(float)
        ok_i = correct[c] * g_i
        return {"oos_recall": (tp_o, g_o), "in_scope_acc": (ok_i, g_i),
                "predicted_oos_rate": (oos_pred[c].astype(float), np.ones(n))}

    def boots(c: str) -> dict[str, np.ndarray]:
        out = {}
        for name, (num, den) in metrics(c).items():
            out[name] = boot_mean(num, idx) / boot_mean(den, idx)
        return out
    point = {}
    for c in d.cfgs:
        m = metrics(c)
        point[c] = {k: float(num.sum() / den.sum()) for k, (num, den) in m.items()}
    bt = {c: boots(c) for c in d.cfgs}
    out: dict[str, Any] = {"n": n, "n_oos": int(gold_oos.sum()), "n_in_scope": int((~gold_oos).sum()), "per_config": {}}
    for c in d.cfgs:
        row = dict(point[c])
        if c != "ref":
            d_oos = bt[c]["oos_recall"] - bt["ref"]["oos_recall"]
            d_in = bt[c]["in_scope_acc"] - bt["ref"]["in_scope_acc"]
            row["delta_vs_ref"] = {
                "oos_recall": est(point[c]["oos_recall"] - point["ref"]["oos_recall"], d_oos),
                "in_scope_acc": est(point[c]["in_scope_acc"] - point["ref"]["in_scope_acc"], d_in),
                "predicted_oos_rate": est(point[c]["predicted_oos_rate"] - point["ref"]["predicted_oos_rate"],
                                          bt[c]["predicted_oos_rate"] - bt["ref"]["predicted_oos_rate"]),
                "oos_recall_minus_in_scope_acc": est(
                    (point[c]["oos_recall"] - point["ref"]["oos_recall"])
                    - (point[c]["in_scope_acc"] - point["ref"]["in_scope_acc"]), d_oos - d_in)}
            flips = np.array([d.pred[c][i] != d.pred["ref"][i] for i in items])
            row["flip_rate_oos_items"] = float(flips[gold_oos].mean())
            row["flip_rate_in_scope_items"] = float(flips[~gold_oos].mean())
        out["per_config"][c] = row
    return out


def length(d: Data, n_boot: int, seed: int) -> dict[str, Any]:
    quart = np.full(d.n, -1)
    for t in np.unique(d.task):
        m = np.flatnonzero(d.task == t)
        tok = d.tokens[m]
        if np.isnan(tok).any():
            continue
        edges = np.quantile(tok, [0.25, 0.5, 0.75])
        quart[m] = np.searchsorted(edges, tok, side="left")
    ok = quart >= 0
    items = np.flatnonzero(ok)
    q = quart[items]
    n = len(items)
    idx = make_idx(n, n_boot, seed)
    out: dict[str, Any] = {"n": n, "per_config": {}}
    for c in d.cfgs:
        if c == "ref":
            continue
        flip = (d.pred[c][items] != d.pred["ref"][items]).astype(float)
        rows = {}
        for k in range(4):
            m = (q == k).astype(float)
            rows[f"Q{k + 1}"] = {"n": int(m.sum()), "flip_rate": float((flip * m).sum() / m.sum())}
        q1, q4 = (q == 0).astype(float), (q == 3).astype(float)
        diff_b = boot_mean(flip * q4, idx) / boot_mean(q4, idx) - boot_mean(flip * q1, idx) / boot_mean(q1, idx)
        diff = rows["Q4"]["flip_rate"] - rows["Q1"]["flip_rate"]
        # rank correlation of flip with within-task token percentile, for a trend rather than two end points
        pct = np.empty(n)
        for t in np.unique(d.task[items]):
            mm = np.flatnonzero(d.task[items] == t)
            order = d.tokens[items][mm].argsort().argsort()
            pct[mm] = order / max(len(mm) - 1, 1)
        corr = float(np.corrcoef(pct, flip)[0, 1]) if flip.std() > 0 else None
        out["per_config"][c] = {"by_quartile": rows, "q4_minus_q1": est(diff, diff_b),
                                "corr_flip_with_within_task_token_percentile": corr,
                                "overall_flip_rate": float(flip.mean())}
    return out


def confidence(d: Data) -> dict[str, Any]:
    out: dict[str, Any] = {"per_config": {}}
    ref_conf = d.conf["ref"]
    for c in d.cfgs:
        if c == "ref":
            continue
        flip = d.pred[c] != d.pred["ref"]
        nf = int(flip.sum())
        low = int((flip & (ref_conf < 0.7)).sum())
        out["per_config"][c] = {
            "flips": nf, "share_of_flips_with_ref_conf_below_0_7": (low / nf) if nf else None,
            "share_of_items_with_ref_conf_below_0_7": float((ref_conf < 0.7).mean()),
            "max_ref_confidence_among_flips": float(ref_conf[flip].max()) if nf else None,
            "flip_rate_ref_conf_below_0_7": float(flip[ref_conf < 0.7].mean()),
            "flip_rate_ref_conf_at_least_0_7": float(flip[ref_conf >= 0.7].mean()),
            "flips_with_ref_conf_at_least_0_9": int((flip & (ref_conf >= 0.9)).sum()),
            "items_with_ref_conf_at_least_0_9": int((ref_conf >= 0.9).sum())}
    return out


def q4_pair(d: Data, idx: np.ndarray) -> dict[str, Any]:
    a, b = "q4_k_m", "nvfp4"
    flip = (d.pred[a] != d.pred[b]).astype(float)
    acc_a, acc_b = boot_mean(d.correct[a], idx), boot_mean(d.correct[b], idx)
    ece_a, ece_b = boot_ece(d.conf[a], d.correct[a], idx), boot_ece(d.conf[b], d.correct[b], idx)
    br_a, br_b = boot_mean(d.brier[a], idx), boot_mean(d.brier[b], idx)
    flip_ref = {c: float((d.pred[c] != d.pred["ref"]).mean()) for c in (a, b)}
    both_wrong_vs_ref = float(((d.pred[a] != d.pred["ref"]) & (d.pred[b] != d.pred["ref"])).mean())
    both_flip_same_way = float(((d.pred[a] != d.pred["ref"]) & (d.pred[b] != d.pred["ref"]) & (d.pred[a] == d.pred[b])).mean())
    per_task = {}
    for t in np.unique(d.task):
        m = d.task == t
        per_task[str(t)] = {"n": int(m.sum()), "acc_q4_k_m": float(d.correct[a][m].mean()),
                            "acc_nvfp4": float(d.correct[b][m].mean()),
                            "flip_rate_between": float(flip[m].mean())}
    return {
        "n": d.n, "flips_between": int(flip.sum()), "flip_rate_between": float(flip.mean()),
        "flip_rate_between_ci95": ci(boot_mean(flip, idx)),
        "flip_rate_vs_ref": flip_ref,
        "share_both_differ_from_ref": both_wrong_vs_ref, "share_both_differ_from_ref_and_agree_with_each_other": both_flip_same_way,
        "delta_accuracy_nvfp4_minus_q4_k_m": est(float(d.correct[b].mean() - d.correct[a].mean()), acc_b - acc_a),
        "delta_ece_nvfp4_minus_q4_k_m": est(point_ece(d.conf[b], d.correct[b]) - point_ece(d.conf[a], d.correct[a]), ece_b - ece_a),
        "delta_brier_nvfp4_minus_q4_k_m": est(float(d.brier[b].mean() - d.brier[a].mean()), br_b - br_a),
        "tasks_where_nvfp4_more_accurate": sum(1 for v in per_task.values() if v["acc_nvfp4"] > v["acc_q4_k_m"]),
        "tasks_where_q4_k_m_more_accurate": sum(1 for v in per_task.values() if v["acc_q4_k_m"] > v["acc_nvfp4"]),
        "per_task": per_task}


def score_shift(d: Data, n_boot: int, seed: int, task: str = "sst5") -> dict[str, Any]:
    items = np.flatnonzero(d.task == task)
    n = len(items)
    idx = make_idx(n, n_boot, seed)
    e = {c: np.array([M.expected_score(d.cfgs[c].p[i]) for i in items]) for c in d.cfgs}
    mid = np.array([(len(d.unit.opts[i]) - 1) / 2.0 for i in items])
    out: dict[str, Any] = {"task": task, "n": n, "per_config": {}}
    for c in d.cfgs:
        if c == "ref":
            continue
        shift = e[c] - e["ref"]
        pull = np.abs(e[c] - mid) - np.abs(e["ref"] - mid)
        out["per_config"][c] = {
            "mean_signed_shift": est(float(shift.mean()), boot_mean(shift, idx)),
            "mean_abs_shift": float(np.abs(shift).mean()),
            "center_pull": est(float(pull.mean()), boot_mean(pull, idx)),
            "top1_flip_rate": float((d.pred[c][items] != d.pred["ref"][items]).mean())}
    return out


def big_vs_small(records_dir: Path, results_dir: Path, n_boot: int, seed: int) -> dict[str, Any]:
    big_dir = Path(results_dir) / "clef" / "ref"
    tasks = sorted(p.stem for p in big_dir.glob("*.jsonl"))
    all_tasks = sorted(p.stem for p in Path(records_dir).glob("*.jsonl"))
    out: dict[str, Any] = {"tasks_finished": tasks, "tasks_missing": sorted(set(all_tasks) - set(tasks)), "per_task": {}}
    pooled_c27: list[np.ndarray] = []
    pooled_c9: list[np.ndarray] = []
    pooled_flip: list[np.ndarray] = []
    for t in tasks:
        u = D.build_unit(Path(records_dir), None, tasks=[t])
        c27 = D.load_cfg(u, Path(results_dir), "clef", "ref")
        c9 = D.load_cfg(u, Path(results_dir), FLASH, "ref")
        g = np.array([u.opts[i].index(u.gold[i]) for i in range(u.n_questions)])
        p27 = np.array([int(np.argmax(p)) for p in c27.p])
        p9 = np.array([int(np.argmax(p)) for p in c9.p])
        k27, k9 = (p27 == g).astype(float), (p9 == g).astype(float)
        n = u.n_questions
        idx = make_idx(n, n_boot, (seed + sum(map(ord, t))) % (2**32))
        labels = {"27b": p27, "9b": p9}
        f1 = {k: M.macro_f1(np.array([u.opts[i][g[i]] for i in range(n)]), np.array([u.opts[i][v[i]] for i in range(n)]))
              for k, v in labels.items()}
        out["per_task"][t] = {
            "n": n, "acc_27b": float(k27.mean()), "acc_9b": float(k9.mean()),
            "delta_accuracy_27b_minus_9b": est(float(k27.mean() - k9.mean()), boot_mean(k27, idx) - boot_mean(k9, idx)),
            "macro_f1_27b": f1["27b"], "macro_f1_9b": f1["9b"],
            "top1_agreement_27b_vs_9b": float(np.mean(p27 == p9)),
            "mean_confidence_27b": float(np.mean([np.max(p) for p in c27.p])),
            "mean_confidence_9b": float(np.mean([np.max(p) for p in c9.p]))}
        pooled_c27.append(k27)
        pooled_c9.append(k9)
        pooled_flip.append((p27 != p9).astype(float))
    if pooled_c27:
        a27, a9 = np.concatenate(pooled_c27), np.concatenate(pooled_c9)
        # pooled paired bootstrap over questions (stratification by task is lost; per-task rows above carry it)
        idx = make_idx(len(a27), n_boot, seed)
        out["pooled"] = {"n": int(len(a27)), "acc_27b": float(a27.mean()), "acc_9b": float(a9.mean()),
                         "delta_accuracy_27b_minus_9b": est(float(a27.mean() - a9.mean()), boot_mean(a27, idx) - boot_mean(a9, idx)),
                         "top1_agreement_27b_vs_9b": 1.0 - float(np.concatenate(pooled_flip).mean())}
    return out


def split_difficulty(acc: dict[str, float]) -> tuple[list[str], list[str]]:
    """Tasks ascending by accuracy (ties by task name); the lowest len // 2 are hard, the rest easy."""
    order = sorted(acc, key=lambda t: (acc[t], t))
    k = len(order) // 2
    return order[:k], order[k:]


def _paired_accuracy(kb: np.ndarray, ks: np.ndarray, idx: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Accuracy of the big and the small config on the same questions, with a paired bootstrap of the difference."""
    boot = boot_mean(kb, idx) - boot_mean(ks, idx)
    row = {"n": int(len(kb)), "acc_big": float(kb.mean()), "acc_small": float(ks.mean()),
           "delta_accuracy_big_minus_small": est(float(kb.mean() - ks.mean()), boot)}
    return row, boot


def cross_size_4bit_vs_8bit(records_dir: Path, results_dir: Path, n_boot: int, seed: int) -> dict[str, Any]:
    results_dir = Path(results_dir)
    all_tasks = sorted(p.stem for p in Path(records_dir).glob("*.jsonl"))
    if len(all_tasks) < 2:
        raise ValueError(f"need at least two public tasks for a hard/easy split, found {len(all_tasks)}")
    key = lambda b, s: f"{BIG}/{b}|{FLASH}/{s}"  # noqa: E731
    pairs = [(b, s) for b in CROSS_BIG for s in CROSS_SMALL]
    present = [(b, s) for b, s in pairs if (results_dir / BIG / b).is_dir() and (results_dir / FLASH / s).is_dir()]
    need = sorted({(BIG, b) for b, _ in present} | {(FLASH, s) for _, s in present})
    correct: dict[tuple[str, str], dict[str, np.ndarray]] = {m: {} for m in need}
    ref_acc: dict[str, float] = {}
    big_ref_acc: dict[str, float] = {}
    # the 27B ref feeds only the mean-of-refs split, and only when it covers every public task (a partial ref is never imputed)
    has_big_ref = all((results_dir / BIG / "ref" / f"{t}.jsonl").exists() for t in all_tasks)
    n_of: dict[str, int] = {}
    for t in all_tasks:
        u = D.build_unit(Path(records_dir), None, tasks=[t])
        n_of[t] = u.n_questions
        if any(g is None for g in u.gold):
            raise ValueError(f"{t}: public records without gold")
        g = np.array([u.opts[i].index(u.gold[i]) for i in range(u.n_questions)])

        def right(model: str, cfg: str) -> np.ndarray:
            return (np.array([int(np.argmax(p)) for p in D.load_cfg(u, results_dir, model, cfg).p]) == g).astype(float)
        ref_acc[t] = float(right(FLASH, "ref").mean())
        if has_big_ref:
            big_ref_acc[t] = float(right(BIG, "ref").mean())
        for m in need:
            correct[m][t] = right(*m)
    hard, easy = split_difficulty(ref_acc)
    out: dict[str, Any] = {
        "difficulty": {"basis": "clef-flash ref accuracy, median split", "per_task": ref_acc, "hard": hard, "easy": easy},
        "pairs": {}, "absent": [key(b, s) for b, s in pairs if (b, s) not in present]}
    seed_of = lambda name: (seed + sum(map(ord, name))) % (2**32)  # noqa: E731
    task_idx = {t: make_idx(n_of[t], n_boot, seed_of(t)) for t in all_tasks} if present else {}

    def pooled(cb: dict[str, np.ndarray], cs: dict[str, np.ndarray], hard_ts: list[str], easy_ts: list[str], tag: str) -> dict[str, Any]:
        """Pooled hard and easy estimates and their difference; the bootstrap index of each group is seeded by its own name."""
        row: dict[str, Any] = {}
        boots = {}
        for name, ts in (("hard", hard_ts), ("easy", easy_ts)):
            idx = make_idx(sum(n_of[t] for t in ts), n_boot, seed_of(f"pooled-{tag}{name}"))
            row[f"pooled_{name}"], boots[name] = _paired_accuracy(np.concatenate([cb[t] for t in ts]), np.concatenate([cs[t] for t in ts]), idx)
        row["hard_minus_easy"] = est(row["pooled_hard"]["delta_accuracy_big_minus_small"]["delta"]
                                     - row["pooled_easy"]["delta_accuracy_big_minus_small"]["delta"], boots["hard"] - boots["easy"])
        return row

    for b, s in present:
        cb, cs = correct[(BIG, b)], correct[(FLASH, s)]
        out["pairs"][key(b, s)] = {"per_task": {t: _paired_accuracy(cb[t], cs[t], task_idx[t])[0] for t in all_tasks},
                                   **pooled(cb, cs, hard, easy, "")}
    # sensitivity of the post-hoc split: alternate hard/easy sets, estimated exactly like the primary split
    order = sorted(ref_acc, key=lambda t: (ref_acc[t], t))
    k = len(hard)
    splits: dict[str, dict[str, Any]] = {}
    thr_hard = [t for t in order if ref_acc[t] < 0.80]
    if present and (not thr_hard or len(thr_hard) == len(order)):
        raise ValueError(f"threshold_80: clef-flash ref accuracy < 0.80 puts {len(thr_hard)} of {len(order)} tasks in hard; "
                         "both groups must be non-empty")
    splits["threshold_80"] = {"basis": "hard = clef-flash ref accuracy below 80%, easy = the rest",
                              "hard": thr_hard, "easy": [t for t in order if ref_acc[t] >= 0.80]}
    if has_big_ref:
        mean_ref = {t: (ref_acc[t] + big_ref_acc[t]) / 2 for t in all_tasks}
        mh, me = split_difficulty(mean_ref)
        splits["median_mean_refs"] = {"basis": "median split on the mean of the clef-flash ref and clef (27B) ref accuracies",
                                      "per_task": mean_ref, "hard": mh, "easy": me}
    splits["median_ex_boundary"] = {
        "basis": "primary median split with the two tasks nearest the median boundary swapped (hardest easy task to hard, easiest hard task to easy)",
        "hard": order[:k - 1] + [order[k]], "easy": [order[k - 1]] + order[k + 1:]}
    seen: dict[str, tuple[set[str], set[str]]] = {"primary": (set(hard), set(easy))}
    sens: dict[str, Any] = {}
    for name, sp in splits.items():
        entry = {"basis": sp["basis"], **({"per_task": sp["per_task"]} if "per_task" in sp else {}), "hard": sp["hard"], "easy": sp["easy"]}
        same = next((o for o, sets in seen.items() if sets == (set(sp["hard"]), set(sp["easy"]))), None)
        if same is not None:
            entry["same_as"] = same
        seen[name] = (set(sp["hard"]), set(sp["easy"]))
        entry["pairs"] = {key(b, s): pooled(correct[(BIG, b)], correct[(FLASH, s)], sp["hard"], sp["easy"], f"{name}-") for b, s in present}
        sens[name] = entry
    out["difficulty_sensitivity"] = sens
    return out


# -------------------------------------------------------------------------- driver

def run(records_dir: Path, results_dir: Path, n_boot: int = M.BOOTSTRAP_RESAMPLES, seed: int = M.SEED) -> dict[str, Any]:
    d = Data(records_dir, results_dir)
    idx = make_idx(d.n, n_boot, seed)
    return {
        "meta": {"model": FLASH, "n_questions": d.n, "n_boot": n_boot, "seed": seed, "configs": list(CONFIGS),
                 "note": "full public sets, simple paired bootstrap over questions"},
        "calibration": calibration(d, idx),
        "oos": oos(d, n_boot, seed),
        "length": length(d, n_boot, seed),
        "confidence": confidence(d),
        "q4_pair": q4_pair(d, idx),
        "score_shift": score_shift(d, n_boot, seed),
        "big_vs_small": big_vs_small(records_dir, results_dir, n_boot, seed),
        "cross_size_4bit_vs_8bit": cross_size_4bit_vs_8bit(records_dir, results_dir, n_boot, seed),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=Path("data/records"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    ap.add_argument("--n-boot", type=int, default=M.BOOTSTRAP_RESAMPLES)
    args = ap.parse_args(argv)
    res = run(args.records, args.results, args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "hypotheses.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}/hypotheses.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
