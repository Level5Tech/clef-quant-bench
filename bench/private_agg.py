#!/usr/bin/env python
"""Aggregate-only analysis of the private (owner's own data) sets: R1, R2, R4a, R4b.

PRIVACY: this module reads private records and results but writes only aggregate numbers keyed by set name, model name,
config name, an anonymised question label and an anonymised class label (c1 = the most frequent class of that question,
c2 = the next, ...). It never writes a record id, record text, gold label, option id or option description. Every
output goes through `scrub()`, which rejects any key or string that is not in a fixed vocabulary of structural names.

What it adds to bench.analyze (which pairs on gold labels, so it produces nothing label-free for R1/R2 and blends the
two R4a questions into one macro-F1):
  1. Per question (not blended): accuracy, macro-F1 and per-class recall of every Clef-flash config and every Clef
     (27B) config against the recorded hosted label, and of each non-ref config against its own model's ref (the 27B
     against the 27B ref, the 9B against the 9B ref; the 27B ref is also compared with the 9B ref), with bootstrap 95%
     CIs over records, set against the majority-class baseline and against the hosted model's own repeat on the same
     records.
  2. The hosted model's repeat disagreement from the *_hosted_repeat files (reproduces PLAN.md's 7.7% / 28.1%).
  3. Label-free comparison against `ref` for every set, including R1/R2, for both models (each against its own ref,
     input-length quartiles from its own ref's token counts): flip rate, mean TVD, mean KL, confidence shift,
     score-question shift, flip rate by input-length quartile, with the non-quantization sources that exist on the
     private sets (ref-r2, a rerun at batch 8) and the engine changes.

Bootstrap: 2000 resamples (metrics.BOOTSTRAP_RESAMPLES), per-question seed (metrics.SEED + crc32(set/question)); the
same resampled items are used for every config of a question, so differences are paired.

Usage:
  python -m bench.private_agg --records private/records --results private/results --out analysis

Another operator's sets and configs (no code change needed; records and results use the same layout):
  python -m bench.private_agg --records R --results S --out O --sets set_a set_b \\
      --configs ref fp8 nvfp4 ref-r2 fp8-r2 --pairs "fp8|fp8-r2"
--configs covers both models unless --big-configs is given. `ref` is paired with every config, the format pairs are
added where both builds are listed, and --pairs adds more. Each name joins the scrub vocabulary for that run only, so
each must match NAME_RE: operator-chosen labels, never anything read from the records.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
import json
import math
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):  # allow `python bench/private_agg.py`
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from bench import decompose as D  # noqa: E402
from bench import metrics as M  # noqa: E402

SETS = ("r1", "r2", "r4a", "r4b")
HOSTED_REPEAT_FILES = {"r4a": "r4a_hosted_repeat", "r4b": "r4b_hosted_repeat"}
FLASH, BIG = "clef-flash", "clef"
FLASH_CONFIGS = ("ref", "bf16-srv", "fp8", "nvfp4", "gguf-bf16", "q8_0", "q4_k_m", "ref-r2")
BIG_CONFIGS = FLASH_CONFIGS  # same names today; a separate name so the two lists can diverge
MODEL_LABEL = {FLASH: "Clef-flash (9B)", BIG: "Clef (27B)"}
# question ids that PLAN.md itself names; every other question gets a generic type-based label
NAMED_QUESTIONS = {"r4a": {"choice5": "choice5", "score5": "score5"}}
PLAN_QUOTE = {"n_pairs": 5319, "choice5": 0.077, "score5": 0.281}
FORMAT_PAIRS: tuple[tuple[str, str], ...] = (
    ("bf16-srv", "fp8"), ("bf16-srv", "nvfp4"), ("gguf-bf16", "q8_0"), ("gguf-bf16", "q4_k_m"),
)
LABEL_FREE_PAIRS: tuple[tuple[str, str], ...] = (*[("ref", c) for c in FLASH_CONFIGS if c != "ref"], *FORMAT_PAIRS)
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,47}")
QUARTILES = ("Q1", "Q2", "Q3", "Q4")
CHUNK = 250


@dataclass(frozen=True)
class Spec:
    """What one run covers. Every name here joins the scrub vocabulary, so all of them come from the operator."""
    sets: tuple[str, ...] = SETS
    flash_configs: tuple[str, ...] = FLASH_CONFIGS
    big_configs: tuple[str, ...] = BIG_CONFIGS
    pairs: tuple[tuple[str, str], ...] = LABEL_FREE_PAIRS


DEFAULT_SPEC = Spec()


def default_pairs(configs: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """`ref` against every config, then the format pairs whose two builds are both listed."""
    return (*[("ref", c) for c in configs if c != "ref"], *[p for p in FORMAT_PAIRS if p[0] in configs and p[1] in configs])


# ------------------------------------------------------------------------------ helpers

def qseed(set_name: str, qlabel: str) -> int:
    return (M.SEED + zlib.crc32(f"{set_name}/{qlabel}".encode("utf-8"))) % (2**32)


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def ci_of(v: np.ndarray) -> list[float] | None:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def boot_mean(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.empty(idx.shape[0])
    for s in range(0, idx.shape[0], CHUNK):
        out[s:s + CHUNK] = x[idx[s:s + CHUNK]].mean(axis=1)
    return out


def boot_macro_f1(gold: np.ndarray, pred: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(idx.shape[0])
    for s in range(0, idx.shape[0], CHUNK):
        out[s:s + CHUNK] = M._macro_f1_batch(gold, pred, idx[s:s + CHUNK].astype(np.int64), k)
    return out


CORE_MIN_SUPPORT = 30


def fixed_macro_f1(gold: np.ndarray, pred: np.ndarray, classes: Sequence[int]) -> float:
    """Macro-F1 over a FIXED class list (an absent class counts as F1 = 0)."""
    f1 = []
    for c in classes:
        tp = np.sum((gold == c) & (pred == c))
        den = np.sum(gold == c) + np.sum(pred == c)
        f1.append(2.0 * tp / den if den else 0.0)
    return float(np.mean(f1)) if f1 else float("nan")


def boot_fixed_macro_f1(gold: np.ndarray, pred: np.ndarray, idx: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    out = np.zeros(idx.shape[0])
    for c in classes:
        g, p = (gold == c).astype(float), (pred == c).astype(float)
        tp = boot_mean(g * p, idx)
        den = boot_mean(g, idx) + boot_mean(p, idx)
        with np.errstate(invalid="ignore", divide="ignore"):
            out += np.where(den > 0, 2.0 * tp / np.where(den > 0, den, 1), 0.0)
    return out / max(len(classes), 1)


def boot_ratio(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.empty(idx.shape[0])
    for s in range(0, idx.shape[0], CHUNK):
        ii = idx[s:s + CHUNK]
        d = den[ii].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:s + CHUNK] = np.where(d > 0, num[ii].sum(axis=1) / np.where(d > 0, d, 1), np.nan)
    return out


def num(x: Any) -> Any:
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return num(o)


# ----------------------------------------------------------------------------- loading

class SetData:
    """One private set: aligned items (one per record x question) and every config's distributions."""

    def __init__(self, set_name: str, rec_dir: Path, res_dir: Path, models: dict[str, Sequence[str]]):
        self.name = set_name
        self.unit = D.build_unit(rec_dir, None, tasks=[set_name])
        u = self.unit
        self.qlabel = question_labels(u, set_name)
        self.cfgs: dict[tuple[str, str], D.Cfg] = {}
        for model, names in models.items():
            for c in names:
                if (Path(res_dir) / model / c / f"{set_name}.jsonl").exists():
                    self.cfgs[(model, c)] = D.load_cfg(u, res_dir, model, c)
        self.tokens: dict[str, np.ndarray] = {}  # per model: input tokens of its own ref run, NaN where unknown
        for model in models:
            ref = Path(res_dir) / model / "ref" / f"{set_name}.jsonl"
            tok = {line["id"]: line.get("input_tokens") for line in D.read_jsonl(ref)} if ref.exists() else {}
            self.tokens[model] = np.array([np.nan if tok.get(r) is None else float(tok[r]) for r, _ in u.qkeys])
        self.pos = {k: i for i, k in enumerate(u.qkeys)}


def question_labels(unit: D.Unit, set_name: str) -> list[str]:
    named = NAMED_QUESTIONS.get(set_name, {})
    qids = sorted({q for _, q in unit.qkeys})
    base: dict[str, str] = {}
    for qid in qids:
        j = next(i for i, (_, q) in enumerate(unit.qkeys) if q == qid)
        t = unit.qtype[j]
        k = len(unit.opts[j]) if t != "noul" else 0
        base[qid] = named.get(qid) or f"{t}{k or ''}"
    counts = Counter(base.values())
    seen: Counter = Counter()
    label = {}
    for qid in qids:
        b = base[qid]
        if counts[b] > 1:
            seen[b] += 1
            label[qid] = f"{b}_{seen[b]}"
        else:
            label[qid] = b
    labels = [label[q] for _, q in unit.qkeys]
    for lab in set(labels):  # one option list per question label, or class indices would not line up
        opts = {unit.opts[i] for i, l in enumerate(labels) if l == lab}
        if len(opts) != 1:
            raise ValueError(f"{set_name}/{lab}: option lists differ between records")
    return labels


def enc(opts: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    pos = {o: i for i, o in enumerate(opts)}
    return np.array([pos[x] for x in labels], dtype=np.int64)


def load_hosted(rec_dir: Path, sd: SetData) -> dict[int, tuple[str, str]]:
    """item index -> (run a label, run b label) from the hosted-repeat file of the set, {} when there is none."""
    name = HOSTED_REPEAT_FILES.get(sd.name)
    path = Path(rec_dir) / f"{name}.jsonl" if name else None
    if not path or not path.exists():
        return {}
    out: dict[int, tuple[str, str]] = {}
    for row in D.read_jsonl(path):
        rid = row["id"]
        if sd.name == "r4a":
            pairs = {"choice5": (str(row["choice5_a"]), str(row["choice5_b"])),
                     "score5": (str(row["score5_a"]), str(row["score5_b"]))}
        else:
            qs = [q for (r, q) in sd.unit.qkeys if r == rid]
            pairs = {qs[0]: (str(row["label_a"]), str(row["label_b"]))} if qs else {}
        for qid, ab in pairs.items():
            i = sd.pos.get((rid, qid))
            if i is not None:
                out[i] = ab
    return out


# ------------------------------------------------------------------------ label metrics

def class_ranks(gold: np.ndarray, k: int) -> dict[int, str]:
    counts = np.bincount(gold, minlength=k)
    order = sorted(range(k), key=lambda i: (-counts[i], i))
    return {i: f"c{r + 1}" for r, i in enumerate(order) if counts[i] > 0}


class Boots:
    """Bootstrap arrays of one predictor on one set of items (resample matrix `idx` shared by every predictor)."""

    def __init__(self, gold: np.ndarray, k: int, idx: np.ndarray):
        self.gold, self.k, self.idx = gold, k, idx
        self.sup = {c: boot_mean((gold == c).astype(float), idx) for c in np.unique(gold)}
        counts = np.bincount(gold, minlength=k)
        self.core = [c for c in range(k) if counts[c] >= CORE_MIN_SUPPORT]

    def of(self, pred: np.ndarray) -> dict[str, Any]:
        correct = (pred == self.gold).astype(float)
        rec = {}
        for c, sup in self.sup.items():
            tp = boot_mean(((pred == c) & (self.gold == c)).astype(float), self.idx)
            with np.errstate(invalid="ignore", divide="ignore"):
                rec[int(c)] = np.where(sup > 0, tp / np.where(sup > 0, sup, 1), np.nan)
        out = {"acc": boot_mean(correct, self.idx), "f1": boot_macro_f1(self.gold, pred, self.idx, self.k), "recall": rec}
        if len(self.core) >= 2:
            out["f1_core"] = boot_fixed_macro_f1(self.gold, pred, self.idx, self.core)
        return out


def label_block(gold: np.ndarray, pred: np.ndarray, k: int, ranks: dict[int, str], boot: dict[str, Any],
                score: bool) -> dict[str, Any]:
    n = len(gold)
    out: dict[str, Any] = {
        "n": n, "accuracy": float(np.mean(pred == gold)), "accuracy_ci95": ci_of(boot["acc"]),
        "macro_f1": M.macro_f1(gold, pred), "macro_f1_ci95": ci_of(boot["f1"]),
        "recall": {}}
    for c, lab in ranks.items():
        sup = int(np.sum(gold == c))
        out["recall"][lab] = {"support": sup, "recall": float(np.sum((pred == c) & (gold == c)) / sup),
                              "ci95": ci_of(boot["recall"][c]),
                              "predicted_share": float(np.mean(pred == c))}
    core = [c for c in ranks if int(np.sum(gold == c)) >= CORE_MIN_SUPPORT]
    if len(core) >= 2 and "f1_core" in boot:
        out["macro_f1_core"] = fixed_macro_f1(gold, pred, core)
        out["macro_f1_core_ci95"] = ci_of(boot["f1_core"])
        out["core_classes"] = len(core)
    if score:
        out["mean_abs_bucket_error"] = float(np.mean(np.abs(pred - gold)))
        out["within_one_bucket"] = float(np.mean(np.abs(pred - gold) <= 1))
    return out


def delta_block(a_boot: dict[str, Any], b_boot: dict[str, Any], gold: np.ndarray, a_pred: np.ndarray,
                b_pred: np.ndarray) -> dict[str, Any]:
    """b minus a, paired on the same resamples."""
    d_acc = b_boot["acc"] - a_boot["acc"]
    d_f1 = b_boot["f1"] - a_boot["f1"]
    out = {"delta_accuracy": {"delta": float(np.mean(b_pred == gold) - np.mean(a_pred == gold)), "ci95": ci_of(d_acc)},
           "delta_macro_f1": {"delta": M.macro_f1(gold, b_pred) - M.macro_f1(gold, a_pred), "ci95": ci_of(d_f1)}}
    if "f1_core" in a_boot and "f1_core" in b_boot:
        counts = np.bincount(gold, minlength=int(max(gold.max(), a_pred.max(), b_pred.max())) + 1)
        core = [c for c in range(len(counts)) if counts[c] >= CORE_MIN_SUPPORT]
        out["delta_macro_f1_core"] = {"delta": fixed_macro_f1(gold, b_pred, core) - fixed_macro_f1(gold, a_pred, core),
                                      "ci95": ci_of(b_boot["f1_core"] - a_boot["f1_core"])}
    return out


def label_agreement(sd: SetData, hosted: dict[int, tuple[str, str]], n_boot: int) -> dict[str, Any]:
    u = sd.unit
    res: dict[str, Any] = {}
    for qlab in sorted(set(sd.qlabel)):
        items = np.array([i for i, l in enumerate(sd.qlabel) if l == qlab and u.gold[i] is not None], dtype=int)
        if items.size == 0:
            continue
        opts = u.opts[items[0]]
        k = len(opts)
        score = u.qtype[items[0]] == "score"
        gold = enc(opts, [u.gold[i] for i in items])
        ranks = class_ranks(gold, k)
        n = len(items)
        idx = np.random.default_rng(qseed(sd.name, qlab)).integers(0, n, size=(n_boot, n), dtype=np.int32)
        boots = Boots(gold, k, idx)
        preds = {key: np.array([int(np.argmax(cfg.p[i])) for i in items]) for key, cfg in sd.cfgs.items()}
        maj = np.full(n, int(np.argmax(np.bincount(gold, minlength=k))))
        block: dict[str, Any] = {
            "n": n, "n_classes": len(ranks), "class_support": {lab: int(np.sum(gold == c)) for c, lab in ranks.items()},
            "majority_baseline": label_block(gold, maj, k, ranks, boots.of(maj), score), "models": {}}
        cache = {key: boots.of(p) for key, p in preds.items()}
        flash_ref = (FLASH, "ref")
        for key, pred in preds.items():
            model, cfg = key
            own_ref = (model, "ref")  # each model is compared with its own ref
            blk = label_block(gold, pred, k, ranks, cache[key], score)
            if cfg != "ref" and own_ref in preds:
                blk["delta_vs_ref"] = delta_block(cache[own_ref], cache[key], gold, preds[own_ref], pred)
                blk["agreement_with_ref"] = float(np.mean(pred == preds[own_ref]))
            if model == BIG and flash_ref in preds:
                blk["agreement_with_flash_ref"] = float(np.mean(pred == preds[flash_ref]))
            block["models"].setdefault(model, {"configs": {}})["configs"][cfg] = blk
        # the hosted model's own repeat on the same records
        sub = [j for j, i in enumerate(items) if int(i) in hosted]
        if sub:
            sub_items = items[sub]
            a_lab = [hosted[int(i)][0] for i in sub_items]
            b_lab = [hosted[int(i)][1] for i in sub_items]
            gold_s = gold[sub]
            ns = len(sub)
            if [opts[x] for x in gold_s] != a_lab:
                raise ValueError(f"{sd.name}/{qlab}: recorded label differs from hosted run a on the repeat subset")
            ok = set(opts)
            if not set(b_lab) <= ok:
                raise ValueError(f"{sd.name}/{qlab}: hosted run b has a label outside the option set")
            b_pred = enc(opts, b_lab)
            idx_s = np.random.default_rng(qseed(sd.name, qlab + "/hosted-subset")).integers(
                0, ns, size=(n_boot, ns), dtype=np.int32)
            bs = Boots(gold_s, k, idx_s)
            ranks_s = class_ranks(gold_s, k)
            hb = bs.of(b_pred)
            sub_block: dict[str, Any] = {
                "n": ns,
                "majority_baseline": label_block(gold_s, np.full(ns, int(np.argmax(np.bincount(gold_s, minlength=k)))), k,
                                                 ranks_s, bs.of(np.full(ns, int(np.argmax(np.bincount(gold_s, minlength=k))))), score),
                "hosted_run_b_vs_recorded": label_block(gold_s, b_pred, k, ranks_s, hb, score),
                "models": {}}
            for key, pred in preds.items():
                p_s = pred[sub]
                cb = bs.of(p_s)
                blk = label_block(gold_s, p_s, k, ranks_s, cb, score)
                blk["delta_vs_hosted_run_b"] = delta_block(hb, cb, gold_s, b_pred, p_s)
                blk["disagreement_with_recorded"] = 1.0 - blk["accuracy"]
                sub_block["models"].setdefault(key[0], {"configs": {}})["configs"][key[1]] = blk
            block["hosted_repeat_subset"] = sub_block
        res[qlab] = block
    return res


def hosted_self_consistency(sd: SetData, hosted: dict[int, tuple[str, str]], n_boot: int) -> dict[str, Any]:
    u = sd.unit
    out: dict[str, Any] = {}
    for qlab in sorted(set(sd.qlabel)):
        items = [i for i in hosted if sd.qlabel[i] == qlab]
        if not items:
            continue
        items.sort()
        opts = u.opts[items[0]]
        k = len(opts)
        a = enc(opts, [hosted[i][0] for i in items])
        b = enc(opts, [hosted[i][1] for i in items])
        n = len(items)
        dis = (a != b).astype(float)
        idx = np.random.default_rng(qseed(sd.name, qlab + "/hosted-self")).integers(0, n, size=(n_boot, n), dtype=np.int32)
        ranks = class_ranks(a, k)
        boot = Boots(a, k, idx).of(b)
        blk = label_block(a, b, k, ranks, boot, u.qtype[items[0]] == "score")
        gold_match = float(np.mean([u.gold[i] == hosted[i][0] for i in items]))
        row = {"n_pairs": n, "disagreement": float(dis.mean()), "disagreement_n": int(dis.sum()),
               "disagreement_ci95": ci_of(boot_mean(dis, idx)),
               "recorded_label_equals_run_a": gold_match,
               "run_b_vs_run_a": blk, "n_classes_used_by_run_b": int(len(np.unique(b)))}
        if sd.name == "r4a" and qlab in PLAN_QUOTE:
            row["plan_quote"] = PLAN_QUOTE[qlab]
            row["plan_quote_n_pairs"] = PLAN_QUOTE["n_pairs"]
            row["reproduces_plan_quote"] = bool(round(row["disagreement"], 3) == PLAN_QUOTE[qlab]
                                                and n == PLAN_QUOTE["n_pairs"])
        out[qlab] = row
    return out


# ------------------------------------------------------------------------ label-free

def conf_buckets(ref_conf: np.ndarray, flips: np.ndarray) -> dict[str, Any]:
    """Flip rate by the reference's top probability (metrics.FLIP_LABELS buckets)."""
    return {b["bucket"]: {"n": b["n"], "flips": b["flips"], "flip_rate": b["flip_rate"]}
            for b in M.flip_buckets(ref_conf, flips.astype(float))}


def q4_minus_q1(flips: np.ndarray, bucket: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    q1, q4 = (bucket == 0).astype(float), (bucket == 3).astype(float)
    d1, d4 = q1.sum(), q4.sum()
    if d1 == 0 or d4 == 0:
        return {"delta": None, "ci95": None}
    point = float((flips * q4).sum() / d4 - (flips * q1).sum() / d1)
    diff = boot_ratio(flips * q4, q4, idx) - boot_ratio(flips * q1, q1, idx)
    return {"delta": point, "ci95": ci_of(diff)}


def label_free(sd: SetData, n_boot: int, model: str = FLASH,
               label_free_pairs: Sequence[tuple[str, str]] = LABEL_FREE_PAIRS) -> dict[str, Any]:
    u = sd.unit
    boot = D.Boot(u, n_boot, qseed(sd.name, "pooled"))
    out: dict[str, Any] = {"pooled": {"n_records": u.n_records, "n_questions": u.n_questions, "pairs": {}}}
    pairs = {}
    for a, b in label_free_pairs:
        ka, kb = (model, a), (model, b)
        if ka in sd.cfgs and kb in sd.cfgs:
            arr = D.pair_arrays(sd.cfgs[ka], sd.cfgs[kb])
            kl = np.array([M.kl_div(sd.cfgs[ka].p[i], sd.cfgs[kb].p[i]) for i in range(u.n_questions)])
            conf_a = np.array([float(np.max(p)) for p in sd.cfgs[ka].p])
            conf_b = np.array([float(np.max(p)) for p in sd.cfgs[kb].p])
            pairs[f"{a}|{b}"] = (arr, kl, conf_a, conf_b)
    for key, (arr, kl, ca, cb) in pairs.items():
        out["pooled"]["pairs"][key] = {
            "flip_rate": float(arr.flip.mean()), "flips": int(arr.flip.sum()),
            "flip_ci95": ci_of(boot.ratio(arr.flip)), "tvd_mean": float(arr.tvd.mean()),
            "tvd_ci95": ci_of(boot.ratio(arr.tvd)), "kl_mean": float(kl.mean()),
            "n_identical_records": int((boot.rec_sum(~arr.identical) == 0).sum()),
            "identical_basis": arr.basis}
        if key.startswith("ref|"):
            out["pooled"]["pairs"][key]["flip_by_ref_confidence"] = conf_buckets(ca, arr.flip)
    tokens = sd.tokens.get(model, np.full(u.n_questions, np.nan))
    for qlab in sorted(set(sd.qlabel)):
        items = np.array([i for i, l in enumerate(sd.qlabel) if l == qlab], dtype=int)
        n = len(items)
        idx = np.random.default_rng(qseed(sd.name, qlab + "/free")).integers(0, n, size=(n_boot, n), dtype=np.int32)
        qtype = u.qtype[items[0]]
        tok = tokens[items]
        q_out: dict[str, Any] = {"n": n, "qtype": qtype, "pairs": {}}
        if not np.isnan(tok).any() and n >= 8:
            edges = np.quantile(tok, [0.25, 0.5, 0.75])
            bucket = np.searchsorted(edges, tok, side="left")
        else:
            bucket = None
        for key, (arr, kl, ca, cb) in pairs.items():
            flips = arr.flip[items].astype(float)
            tvd = arr.tvd[items]
            row: dict[str, Any] = {
                "flips": int(flips.sum()), "flip_rate": float(flips.mean()), "agreement": 1.0 - float(flips.mean()),
                "flip_ci95": ci_of(boot_mean(flips, idx)), "tvd_mean": float(tvd.mean()),
                "tvd_ci95": ci_of(boot_mean(tvd, idx)), "kl_mean": float(kl[items].mean()),
                "mean_confidence_a": float(ca[items].mean()), "mean_confidence_b": float(cb[items].mean()),
                "confidence_shift": float(cb[items].mean() - ca[items].mean()),
                "n_identical": int(arr.identical[items].sum())}
            if key.startswith("ref|"):
                row["flip_by_ref_confidence"] = conf_buckets(ca[items], arr.flip[items])
            if bucket is not None:
                row["flip_q4_minus_q1"] = q4_minus_q1(flips, bucket, idx)
                row["flip_by_length_quartile"] = {
                    QUARTILES[q]: {"n": int((bucket == q).sum()), "flips": int(flips[bucket == q].sum()),
                                   "flip_rate": float(flips[bucket == q].mean()) if (bucket == q).any() else None,
                                   "wilson_ci95": wilson(int(flips[bucket == q].sum()), int((bucket == q).sum()))}
                    for q in range(4)}
            if qtype == "score":
                a_name, b_name = key.split("|")
                pa = [sd.cfgs[(model, a_name)].p[i] for i in items]
                pb = [sd.cfgs[(model, b_name)].p[i] for i in items]
                sh = clean(M.score_shift(pa, pb))
                e_a = np.array([M.expected_score(p) for p in pa])
                e_b = np.array([M.expected_score(p) for p in pb])
                mid = np.array([(len(p) - 1) / 2.0 for p in pa])
                sh["center_pull_ci95"] = ci_of(boot_mean(np.abs(e_b - mid) - np.abs(e_a - mid), idx))
                sh["mean_signed_shift_ci95"] = ci_of(boot_mean(e_b - e_a, idx))
                row["score_shift"] = sh
            q_out["pairs"][key] = row
        out[qlab] = q_out
    return out


# ------------------------------------------------------------------------------- build

def build(rec_dir: Path, res_dir: Path, n_boot: int = M.BOOTSTRAP_RESAMPLES,
          sets: Sequence[str] = SETS, spec: Spec | None = None) -> dict[str, Any]:
    spec = spec or Spec(sets=tuple(sets))
    out: dict[str, Any] = {
        "meta": {"n_boot": n_boot, "seed": M.SEED, "sets": list(spec.sets), "flash_configs": list(spec.flash_configs),
                 "big_configs": list(spec.big_configs),
                 "label_free_pairs": [f"{a}|{b}" for a, b in spec.pairs]},
        "set_sizes": {}, "hosted_repeat": {}, "label_agreement": {}, "label_free": {}, "label_free_big": {},
        "absent": []}
    for s in spec.sets:
        sd = SetData(s, rec_dir, res_dir, {FLASH: spec.flash_configs, BIG: spec.big_configs})
        for model, cfgs in ((FLASH, spec.flash_configs), (BIG, spec.big_configs)):
            for c in cfgs:
                if (model, c) not in sd.cfgs:
                    out["absent"].append({"set": s, "model": model, "config": c})
        out["set_sizes"][s] = {"n_records": sd.unit.n_records, "n_questions": sd.unit.n_questions,
                               "questions": sorted(set(sd.qlabel))}
        hosted = load_hosted(rec_dir, sd)
        if hosted:
            out["hosted_repeat"][s] = hosted_self_consistency(sd, hosted, n_boot)
        gold_labels = [q for i, q in enumerate(sd.qlabel) if sd.unit.gold[i] is not None]
        if gold_labels:
            out["label_agreement"][s] = label_agreement(sd, hosted, n_boot)
        out["label_free"][s] = label_free(sd, n_boot, FLASH, spec.pairs)
        out["label_free_big"][s] = label_free(sd, n_boot, BIG, spec.pairs)
    return clean(out)


# ------------------------------------------------------------------------------- scrub

def vocabulary(res: dict[str, Any], spec: Spec = DEFAULT_SPEC) -> set[str]:
    names: set[str] = set(spec.sets) | {FLASH, BIG} | set(spec.flash_configs) | set(spec.big_configs) | set(QUARTILES)
    for a, b in spec.pairs:
        names.add(f"{a}|{b}")
    for s, info in res["set_sizes"].items():
        names.update(info["questions"])
    for lab in range(1, 12):
        names.add(f"c{lab}")
    names |= {"logits", "probs", "noul", "choice", "score"}
    names |= set(M.FLIP_LABELS)
    return names


def collect_fields(obj: Any, vocab: set[str], acc: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) not in vocab:
                acc.add(str(k))
            collect_fields(v, vocab, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_fields(v, vocab, acc)


def scrub(res: dict[str, Any], spec: Spec = DEFAULT_SPEC) -> dict[str, Any]:
    """Reject any dict key or string value outside the structural vocabulary or the frozen field-name list.

    The set and config names in `spec` come from the operator, never from `res`. Question labels do come from `res`
    (see `vocabulary`), and they are safe only because `question_labels` anonymises them first.
    """
    vocab = vocabulary(res, spec)
    allowed = vocab | FIELD_NAMES

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k) not in allowed:
                    raise ValueError(f"unexpected key in private aggregate: {str(k)[:40]!r}")
                walk(v, f"{path}/{k}")
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, path)
        elif isinstance(o, str) and o not in allowed:
            raise ValueError(f"unexpected string in private aggregate: {o[:40]!r}")
    walk(res, "")
    return res


FIELD_NAMES = {
    "meta", "n_boot", "seed", "sets", "flash_configs", "big_configs", "label_free_pairs", "label_free_big",
    "agreement_with_flash_ref", "set_sizes", "hosted_repeat",
    "label_agreement", "label_free", "absent", "set", "model", "config", "n_records", "n_questions", "questions",
    "n_pairs", "disagreement", "disagreement_n", "disagreement_ci95", "recorded_label_equals_run_a", "run_b_vs_run_a",
    "n_classes_used_by_run_b", "plan_quote", "plan_quote_n_pairs", "reproduces_plan_quote",
    "n", "accuracy", "accuracy_ci95", "macro_f1", "macro_f1_ci95", "recall", "support", "ci95", "predicted_share",
    "macro_f1_core", "macro_f1_core_ci95", "core_classes", "delta_macro_f1_core",
    "mean_abs_bucket_error", "within_one_bucket", "delta_accuracy", "delta_macro_f1", "delta",
    "n_classes", "class_support", "majority_baseline", "models", "configs", "delta_vs_ref", "agreement_with_ref",
    "hosted_repeat_subset", "hosted_run_b_vs_recorded", "delta_vs_hosted_run_b", "disagreement_with_recorded",
    "pooled", "pairs", "flip_rate", "flips", "flip_ci95", "tvd_mean", "tvd_ci95", "kl_mean", "n_identical_records",
    "identical_basis", "qtype", "agreement", "mean_confidence_a", "mean_confidence_b", "confidence_shift",
    "n_identical", "flip_by_length_quartile", "wilson_ci95", "score_shift", "mean_signed_shift", "mean_abs_shift",
    "center_pull", "shift_when_ref_above_middle", "shift_when_ref_below_middle",
    "flip_by_ref_confidence", "flip_q4_minus_q1", "center_pull_ci95", "mean_signed_shift_ci95",
}


# ------------------------------------------------------------------------------ markdown

def _p(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def _ci(c: Sequence[float] | None, nd: int = 1, scale: float = 100.0, suffix: str = "") -> str:
    return "-" if not c else f"[{scale * c[0]:.{nd}f}{suffix}, {scale * c[1]:.{nd}f}{suffix}]"


def _f(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _row(name: str, blk: dict[str, Any], classes: list[str]) -> list[str]:
    cells = [name, f"{_p(blk['accuracy'])} {_ci(blk['accuracy_ci95'])}", f"{_f(blk['macro_f1'])} {_ci(blk['macro_f1_ci95'], 3, 1.0)}",
             f"{_f(blk.get('macro_f1_core'))} {_ci(blk.get('macro_f1_core_ci95'), 3, 1.0)}" if "macro_f1_core" in blk else "-"]
    for c in classes:
        r = blk["recall"].get(c)
        cells.append("-" if r is None else f"{_p(r['recall'])} (n={r['support']})")
    if "mean_abs_bucket_error" in blk:
        cells.append(_f(blk["mean_abs_bucket_error"], 3))
    return cells


def _label_free_md(label_free: dict[str, Any], suffix: str = "") -> list[str]:
    L: list[str] = []
    for s, d in label_free.items():
        rows = []
        pooled = d["pooled"]
        for key, r in pooled["pairs"].items():
            rows.append(["all questions", key, f"{_p(r['flip_rate'], 2)} {_ci(r['flip_ci95'], 2)}",
                         f"{_f(r['tvd_mean'], 5)} {_ci(r['tvd_ci95'], 5, 1.0)}", _f(r["kl_mean"], 5), str(r["flips"])])
        for q, qd in d.items():
            if q == "pooled":
                continue
            for key, r in qd["pairs"].items():
                rows.append([f"{q} (n={qd['n']})", key, f"{_p(r['flip_rate'], 2)} {_ci(r['flip_ci95'], 2)}",
                             f"{_f(r['tvd_mean'], 5)} {_ci(r['tvd_ci95'], 5, 1.0)}", _f(r["kl_mean"], 5), str(r["flips"])])
        L += [f"### {s}{suffix}", "", md_table(["question", "pair", "flip rate [95% CI]", "mean TVD [95% CI]", "mean KL", "flips"], rows), ""]
        rows = []
        for q, qd in d.items():
            if q == "pooled":
                continue
            for key, r in qd["pairs"].items():
                if key.startswith("ref|") and "flip_by_length_quartile" in r:
                    dq = r.get("flip_q4_minus_q1") or {}
                    rows.append([q, key] + [f"{_p(r['flip_by_length_quartile'][qq]['flip_rate'], 1)} (n={r['flip_by_length_quartile'][qq]['n']})"
                                            for qq in QUARTILES]
                                + [("-" if dq.get("delta") is None else f"{100 * dq['delta']:+.2f} {_ci(dq['ci95'], 2)}")])
        if rows:
            L += [f"Flip rate against ref by input-length quartile ({s}); last column Q4 minus Q1 in percentage points:", "",
                  md_table(["question", "pair", *QUARTILES, "Q4 - Q1 pp [95% CI]"], rows), ""]
        rows = []
        for key, r in pooled["pairs"].items():
            if "flip_by_ref_confidence" in r:
                rows.append([key] + [f"{_p(b['flip_rate'], 1)} (n={b['n']})" for b in r["flip_by_ref_confidence"].values()])
        if rows:
            L += [f"Flip rate against ref by the reference's top probability, all questions of {s}:", "",
                  md_table(["pair", *M.FLIP_LABELS], rows), ""]
        rows = []
        for q, qd in d.items():
            if q == "pooled":
                continue
            for key, r in qd["pairs"].items():
                if "score_shift" in r and key.startswith("ref|"):
                    sh = r["score_shift"]
                    rows.append([q, key, f"{_f(sh['mean_signed_shift'], 4)} {_ci(sh.get('mean_signed_shift_ci95'), 4, 1.0)}",
                                 _f(sh["mean_abs_shift"], 4), f"{_f(sh['center_pull'], 4)} {_ci(sh.get('center_pull_ci95'), 4, 1.0)}"])
        if rows:
            L += [f"Score questions: shift of the expected score vs ref ({s}); center pull < 0 means toward the middle:", "",
                  md_table(["question", "pair", "mean signed shift [95% CI]", "mean abs shift", "center pull [95% CI]"], rows), ""]
    return L


def render_md(res: dict[str, Any]) -> str:
    m = res["meta"]
    L = ["# Private sets: aggregate numbers (no record content)", "",
         f"Aggregates only, keyed by set / model / config / anonymised question and class labels (c1 = most frequent "
         f"class of that question). Bootstrap: {m['n_boot']} resamples over records, seed {m['seed']} + crc32(set/question), "
         "percentile 95% CIs; one resample matrix per question, shared by every config, so differences are paired. "
         "`hosted label` = the hosted model's recorded label (run a). Accuracy is never shown without macro-F1 and "
         "per-class recall. All-class macro-F1 intervals are approximate when a class has very few items (a resample "
         "that drops the class averages over fewer classes), so a second macro-F1 over a fixed set of classes with "
         "support >= 30 is given next to it.", ""]
    L += ["## Hosted model against its own repeat", ""]
    rows = []
    for s, qs in res["hosted_repeat"].items():
        for q, r in qs.items():
            rb = r["run_b_vs_run_a"]
            rows.append([s, q, str(r["n_pairs"]), f"{_p(r['disagreement'], 2)} {_ci(r['disagreement_ci95'], 2)}",
                         str(r["disagreement_n"]), _f(rb["macro_f1"]), _p(r["recorded_label_equals_run_a"], 1),
                         ("yes" if r.get("reproduces_plan_quote") else "no") if "reproduces_plan_quote" in r else "-"])
    L += [md_table(["set", "question", "pairs", "run b disagrees with run a", "count", "macro-F1 (b vs a)",
                    "recorded label = run a", "reproduces PLAN.md quote"], rows), ""]
    for s, qs in res["label_agreement"].items():
        for q, blk in qs.items():
            classes = [c for c in blk["class_support"]]
            score = "mean_abs_bucket_error" in blk["majority_baseline"]
            hdr = ["predictor", "agreement with hosted label [95% CI]", "macro-F1, all classes [95% CI]",
                   "macro-F1, classes with support >= 30 [95% CI]"] + \
                  [f"recall {c} (support)" for c in classes] + (["mean |bucket error|"] if score else [])
            L += [f"## {s} / {q}: agreement with the hosted label (n = {blk['n']}, {blk['n_classes']} classes; "
                  f"class support {', '.join(f'{c}={v}' for c, v in blk['class_support'].items())})", ""]
            rows = [_row("majority-class baseline", blk["majority_baseline"], classes)]
            for model, mm in blk["models"].items():
                for cfg, b in mm["configs"].items():
                    rows.append(_row(f"{model}/{cfg}", b, classes))
            L += [md_table(hdr, rows), ""]
            for model, mm in blk["models"].items():
                rows = []
                for cfg, b in mm["configs"].items():
                    if "delta_vs_ref" in b:
                        d = b["delta_vs_ref"]
                        rows.append([cfg, f"{100 * d['delta_accuracy']['delta']:+.2f} {_ci(d['delta_accuracy']['ci95'], 2)}",
                                     f"{100 * d['delta_macro_f1']['delta']:+.2f} {_ci(d['delta_macro_f1']['ci95'], 2)}",
                                     (f"{100 * d['delta_macro_f1_core']['delta']:+.2f} {_ci(d['delta_macro_f1_core']['ci95'], 2)}"
                                      if "delta_macro_f1_core" in d else "-"),
                                     _p(b.get("agreement_with_ref"), 2)])
                if rows:
                    L += [f"Change against the `ref` of {MODEL_LABEL.get(model, model)} (percentage points, paired; each config "
                          f"against its own model's ref):", "",
                          md_table(["config", "d agreement pp [95% CI]", "d macro-F1 (all classes) pp [95% CI]",
                                    "d macro-F1 (support >= 30) pp [95% CI]", "top-1 agreement with ref"], rows), ""]
            hs = blk.get("hosted_repeat_subset")
            if hs:
                L += [f"### {s} / {q}: on the {hs['n']} records that have a hosted repeat", ""]
                rows = [_row("majority-class baseline", hs["majority_baseline"], classes),
                        _row("hosted run b vs recorded label", hs["hosted_run_b_vs_recorded"], classes)]
                for model, mm in hs["models"].items():
                    for cfg, b in mm["configs"].items():
                        rows.append(_row(f"{model}/{cfg}", b, classes))
                L += [md_table(hdr, rows), ""]
                rows = []
                for model, mm in hs["models"].items():
                    for cfg, b in mm["configs"].items():
                        d = b["delta_vs_hosted_run_b"]
                        rows.append([f"{model}/{cfg}", _p(b["disagreement_with_recorded"], 2),
                                     f"{100 * d['delta_accuracy']['delta']:+.2f} {_ci(d['delta_accuracy']['ci95'], 2)}",
                                     f"{100 * d['delta_macro_f1']['delta']:+.2f} {_ci(d['delta_macro_f1']['ci95'], 2)}",
                                     (f"{100 * d['delta_macro_f1_core']['delta']:+.2f} {_ci(d['delta_macro_f1_core']['ci95'], 2)}"
                                      if "delta_macro_f1_core" in d else "-")])
                L += ["Model minus the hosted repeat (paired, percentage points):", "",
                      md_table(["predictor", "disagreement with recorded", "d agreement pp [95% CI]",
                                "d macro-F1 (all classes) pp [95% CI]", "d macro-F1 (support >= 30) pp [95% CI]"], rows), ""]
    L += ["## Label-free comparison against ref (every set)", "",
          "flip = top-1 differs; TVD = mean total-variation distance; ref-r2 is a rerun at batch 8 (rerun + batch-size change). "
          "The private sets have no vLLM or llama.cpp restart repeats, so ref-r2 is the only measured non-quantization "
          "floor besides the two engine changes (ref -> bf16-srv, ref -> gguf-bf16).", ""]
    L += _label_free_md(res["label_free"])
    big = {s: d for s, d in res.get("label_free_big", {}).items() if d["pooled"]["pairs"]}
    if big:
        L += ["## Label-free comparison against the 27B ref: Clef (27B), every set with 27B runs", "",
              "Same quantities as the section above, but for Clef (27B): each 27B config is paired with the 27B `ref`, not "
              "with the Clef-flash `ref`, and the input-length quartiles use the 27B ref's own token counts. The 27B ref-r2 "
              "is also a rerun at batch 8 (rerun + batch-size change).", ""]
        L += _label_free_md(big, " (Clef 27B)")
    if res["absent"]:
        L += ["## Absent", "", ", ".join(f"{a['set']}:{a['model']}/{a['config']}" for a in res["absent"]), ""]
    return "\n".join(L) + "\n"


def safe_name(text: str) -> str:
    if not NAME_RE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"not a safe set or config name: {text[:48]!r} (letters, digits, . _ -; 48 max)")
    return text


def safe_pair(text: str) -> tuple[str, str]:
    a, sep, b = text.partition("|")
    if not sep:
        raise argparse.ArgumentTypeError(f"a pair is two config names joined by |: {text[:48]!r}")
    return safe_name(a), safe_name(b)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=Path("private/records"))
    ap.add_argument("--results", type=Path, default=Path("private/results"))
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    ap.add_argument("--n-boot", type=int, default=M.BOOTSTRAP_RESAMPLES)
    ap.add_argument("--sets", nargs="+", type=safe_name, default=list(SETS))
    ap.add_argument("--configs", nargs="+", type=safe_name, default=list(FLASH_CONFIGS),
                    help="both models, or the 9B when --big-configs is given")
    ap.add_argument("--big-configs", nargs="+", type=safe_name, default=None,
                    help="27B configs, when they differ from --configs")
    ap.add_argument("--pairs", nargs="+", type=safe_pair, default=[], help='extra label-free pairs, e.g. "fp8|fp8-r2"')
    ap.add_argument("--print-new-fields", action="store_true", help="development: list field names not yet frozen")
    args = ap.parse_args(argv)
    big = args.big_configs if args.big_configs is not None else args.configs
    listed = [*args.configs, *[c for c in big if c not in args.configs]]
    for a, b in args.pairs:
        if a not in listed or b not in listed:
            ap.error(f"--pairs {a}|{b}: both configs must be listed in --configs or --big-configs")
    defaults = default_pairs(listed)
    spec = Spec(tuple(args.sets), tuple(args.configs), tuple(big),
                (*defaults, *[p for p in args.pairs if p not in defaults]))
    res = build(args.records, args.results, args.n_boot, spec=spec)
    if args.print_new_fields:
        acc: set[str] = set()
        collect_fields(res, vocabulary(res, spec) | FIELD_NAMES, acc)
        print(sorted(acc), file=sys.stderr)
    scrub(res, spec)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "private_aggregates.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (args.out / "private_aggregates.md").write_text(render_md(res), encoding="utf-8")
    print(f"wrote {args.out}/private_aggregates.json and private_aggregates.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
