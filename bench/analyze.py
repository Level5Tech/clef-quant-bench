#!/usr/bin/env python
"""Analyse benchmark runs: results/<model>/<config>/<task>.jsonl against the records' gold labels.

Layout read (see PLAN.md, "Shared interface"):
  data/records/<task>.jsonl                          records with gold labels
  results/<model>/<config>/<task>.jsonl              runner output, one line per record
  results/<model>/<config>/meta.json  (optional)     {"file_size_gb": 54.0 | "file_size_bytes": ..., "peak_memory_gb": ...}
Configs named `<base>-r<N>` (e.g. `ref-r2`) are repeat runs: they are not reported as configs, they
define the noise floor of `<base>` (same metrics, repeat vs base).

Per (model, config, task) and pooled:
  accuracy, macro-F1, Brier, ECE(15), mean confidence, latency p50/p95, decisions/sec, and, against the
  model's `ref` config: paired 95% bootstrap CI on the accuracy / macro-F1 delta (2000 resamples, fixed
  seed), top-1 agreement, exact McNemar, mean KL(ref||cfg), mean TVD, flip rate by ref-confidence bucket,
  and the noise-floor annotation. Per model/config there are also: input-length quartile and
  question-type breakdowns (point estimates), score-task expected-score shift, a cascade simulation
  (use cfg unless its top probability < t, otherwise escalate to ref) and an equal-memory table.
Pooled accuracy is micro (all items); pooled macro-F1 is the mean of the per-task macro-F1s.
Absent configs/tasks are listed under `absent`; nothing is imputed.

Usage:
  python -m bench.analyze --records data/records --results results --out analysis
  python -m bench.analyze --records data/records --results results --out analysis/flash_public --models clef-flash
  python -m bench.analyze --private /path/to/private_root --out analysis   # aggregates only
    --models restricts the run to the named model directories (e.g. to leave out a model whose sweep is unfinished).
    private_root/records/<set>.jsonl and private_root/results/<model>/<config>/<set>.jsonl;
    writes analysis/private_summary.json + private_tables.md containing only aggregate numbers keyed by
    set / model / config name: never record text, ids, gold labels or option descriptions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):  # allow `python bench/analyze.py`
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from bench import metrics as M  # noqa: E402
from bench.schema import option_ids  # noqa: E402

REPEAT_RE = re.compile(r"^(?P<base>.+)-r(?P<n>\d+)$")
REF = "ref"
QUARTILES = ("Q1", "Q2", "Q3", "Q4")
MIN_FOR_QUARTILES = 8


# ----------------------------------------------------------------------------- loading

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_records(records_dir: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(Path(records_dir).glob("*.jsonl")):
        out[path.stem] = {r["id"]: r for r in read_jsonl(path)}
    return out


def scan_results(results_root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    """model -> config -> task -> path (plus the config's meta.json under task key '__meta__')."""
    tree: dict[str, dict[str, dict[str, Path]]] = {}
    root = Path(results_root)
    if not root.is_dir():
        return tree
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            files = {p.stem: p for p in sorted(cdir.glob("*.jsonl"))}
            if (cdir / "meta.json").exists():
                files["__meta__"] = cdir / "meta.json"
            tree.setdefault(mdir.name, {})[cdir.name] = files
    return tree


@dataclass
class Item:
    task: str
    key: tuple[str, str]  # (record id, question id)
    qtype: str
    opts: tuple[str, ...]
    p: np.ndarray
    gold: str | None
    tokens: int | None
    latency: float | None

    @property
    def pred(self) -> str:
        return self.opts[int(np.argmax(self.p))]

    @property
    def conf(self) -> float:
        return float(np.max(self.p))


def build_items(task: str, recs: dict[str, dict], lines: list[dict]) -> tuple[dict[tuple[str, str], Item], dict[str, int]]:
    counts = {"n_expected": sum(len(r["request"]["questions"]) for r in recs.values()),
              "n_errors": 0, "n_missing": 0, "n_unknown_ids": 0, "n_duplicate_ids": 0}
    by_id: dict[str, dict] = {}
    for line in lines:
        rid = line.get("id")
        if rid not in recs:
            counts["n_unknown_ids"] += 1
        elif rid in by_id:
            counts["n_duplicate_ids"] += 1  # first occurrence wins
        else:
            by_id[rid] = line
    items: dict[tuple[str, str], Item] = {}
    for rid, rec in recs.items():
        line = by_id.get(rid)
        questions = rec["request"]["questions"]
        if line is None:
            counts["n_missing"] += len(questions)
            continue
        for qid, q in questions.items():
            opts = tuple(option_ids(q))
            ans = None if line.get("error") else (line.get("answers") or {}).get(qid)
            p = _answer_dist(ans, opts)
            if p is None:
                counts["n_errors"] += 1
                continue
            gold = (rec.get("gold") or {}).get(qid)
            items[(rid, qid)] = Item(task, (rid, qid), q["type"], opts, p, gold if gold in opts else None,
                                     line.get("input_tokens"), line.get("latency_ms"))
    return items, counts


def _answer_dist(ans: Any, opts: tuple[str, ...]) -> np.ndarray | None:
    """Distribution over `opts` (record order) from a runner answer, or None when unusable."""
    if not isinstance(ans, dict):
        return None
    ids = [str(x) for x in ans.get("option_ids") or []]
    if sorted(ids) != sorted(opts):
        return None
    logits, probs = ans.get("logits"), ans.get("probs")
    try:
        if logits is not None and len(logits) == len(ids):
            vec = M.dist(logits=logits)
        elif probs is not None and len(probs) == len(ids):
            vec = M.dist(probs=probs)
        else:
            return None
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(vec)):
        return None
    pos = {o: i for i, o in enumerate(ids)}
    return vec[[pos[o] for o in opts]]


# ---------------------------------------------------------------------- absolute blocks

def _num(x: Any) -> Any:
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def abs_block(items_by_task: dict[str, list[Item]]) -> dict[str, Any]:
    """Accuracy / macro-F1 / Brier / ECE over items that have a valid gold label."""
    all_items = [it for its in items_by_task.values() for it in its]
    scored = [it for it in all_items if it.gold is not None]
    block: dict[str, Any] = {"n_scored": len(all_items), "n_with_gold": len(scored)}
    if not scored:
        return {**block, "accuracy": None, "macro_f1": None, "brier": None, "ece": None, "mean_confidence": None}
    correct = np.array([it.pred == it.gold for it in scored], dtype=float)
    f1s = []
    for its in items_by_task.values():
        g = np.array([it.gold for it in its if it.gold is not None])
        if g.size:
            f1s.append(M.macro_f1(g, np.array([it.pred for it in its if it.gold is not None])))
    block.update({
        "accuracy": _num(correct.mean()),
        "macro_f1": _num(np.mean(f1s)),
        "brier": _num(np.mean([M.brier(it.p, it.opts.index(it.gold)) for it in scored])),
        "ece": _num(M.ece([it.conf for it in scored], correct)),
        "mean_confidence": _num(np.mean([it.conf for it in scored])),
    })
    return block


# ---------------------------------------------------------------------- paired objects

def task_seed(task: str) -> int:
    return (M.SEED + zlib.crc32(task.encode("utf-8"))) % (2**32)


@dataclass
class Pair:
    """Items present (with valid gold) in both ref and cfg, in a fixed order."""
    task: str
    gold: np.ndarray
    ref_pred: np.ndarray
    cfg_pred: np.ndarray
    ref_conf: np.ndarray
    cfg_conf: np.ndarray
    kl: np.ndarray
    tv: np.ndarray
    tokens: np.ndarray  # float, nan when unknown
    qtype: np.ndarray
    ref_p: list[np.ndarray]
    cfg_p: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.gold)

    def subset(self, mask: np.ndarray) -> "Pair":
        idx = np.flatnonzero(mask)
        return Pair(self.task, self.gold[idx], self.ref_pred[idx], self.cfg_pred[idx], self.ref_conf[idx],
                    self.cfg_conf[idx], self.kl[idx], self.tv[idx], self.tokens[idx], self.qtype[idx],
                    [self.ref_p[i] for i in idx], [self.cfg_p[i] for i in idx])


def make_pair(task: str, ref: dict[tuple[str, str], Item], cfg: dict[tuple[str, str], Item]) -> Pair | None:
    keys = sorted(k for k in ref.keys() & cfg.keys() if ref[k].gold is not None and ref[k].opts == cfg[k].opts)
    if not keys:
        return None
    r = [ref[k] for k in keys]
    c = [cfg[k] for k in keys]
    tokens = [(a.tokens if a.tokens is not None else b.tokens) for a, b in zip(r, c)]
    return Pair(
        task, np.array([a.gold for a in r]), np.array([a.pred for a in r]), np.array([b.pred for b in c]),
        np.array([a.conf for a in r]), np.array([b.conf for b in c]),
        np.array([M.kl_div(a.p, b.p) for a, b in zip(r, c)]), np.array([M.tvd(a.p, b.p) for a, b in zip(r, c)]),
        np.array([np.nan if t is None else float(t) for t in tokens]), np.array([a.qtype for a in r]),
        [a.p for a in r], [b.p for b in c])


def paired_block(pairs: dict[str, Pair], n_boot: int, with_ci: bool = True) -> dict[str, Any]:
    """Comparison of cfg vs ref over one or several tasks (pooled = several)."""
    pairs = {t: p for t, p in pairs.items() if len(p)}
    n = sum(len(p) for p in pairs.values())
    if not n:
        return {"n_paired": 0}
    ref_ok = np.concatenate([p.ref_pred == p.gold for p in pairs.values()])
    cfg_ok = np.concatenate([p.cfg_pred == p.gold for p in pairs.values()])
    flipped = np.concatenate([p.ref_pred != p.cfg_pred for p in pairs.values()])
    ref_conf = np.concatenate([p.ref_conf for p in pairs.values()])
    b = int(np.sum(ref_ok & ~cfg_ok))
    c = int(np.sum(~ref_ok & cfg_ok))
    f1_ref = np.mean([M.macro_f1(p.gold, p.ref_pred) for p in pairs.values()])
    f1_cfg = np.mean([M.macro_f1(p.gold, p.cfg_pred) for p in pairs.values()])
    out: dict[str, Any] = {
        "n_paired": n,
        "ref_accuracy": _num(ref_ok.mean()),
        "ref_macro_f1": _num(f1_ref),
        "agreement": _num(1.0 - flipped.mean()),
        "n_flips": int(flipped.sum()),
        "mcnemar": {"b_ref_right_cfg_wrong": b, "c_ref_wrong_cfg_right": c, "p": _num(M.mcnemar_exact(b, c)),
                    "b": b, "c": c},
        "kl_mean": _num(np.concatenate([p.kl for p in pairs.values()]).mean()),
        "tvd_mean": _num(np.concatenate([p.tv for p in pairs.values()]).mean()),
        "flip_by_ref_confidence": M.flip_buckets(ref_conf, flipped),
        "delta_accuracy": {"delta": _num(cfg_ok.mean() - ref_ok.mean()), "ci95": None},
        "delta_macro_f1": {"delta": _num(f1_cfg - f1_ref), "ci95": None},
    }
    if with_ci:
        boots = [M.bootstrap_arrays(p.gold, p.ref_pred, p.cfg_pred, n_boot, task_seed(t)) for t, p in pairs.items()]
        d_acc = np.sum([bt["d_correct"] for bt in boots], axis=0) / n
        d_f1 = np.mean([bt["d_f1"] for bt in boots], axis=0)
        out["delta_accuracy"]["ci95"] = M.ci95(d_acc)
        out["delta_macro_f1"]["ci95"] = M.ci95(d_f1)
        out["bootstrap"] = {"n_boot": n_boot, "seed_base": M.SEED, "paired": True}
    score = [(rp, cp) for p in pairs.values() for rp, cp, qt in zip(p.ref_p, p.cfg_p, p.qtype) if qt == "score"]
    if score:
        out["score_shift"] = _clean(M.score_shift([s[0] for s in score], [s[1] for s in score]))
    return out


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return _num(obj)


# ------------------------------------------------------------------- derived breakdowns

def length_quartiles(pairs: dict[str, Pair]) -> dict[str, Any]:
    """Per task (and pooled by quartile index): paired metrics in input_tokens quartiles."""
    out: dict[str, Any] = {}
    pooled: dict[int, list[Pair]] = {i: [] for i in range(4)}
    for task, p in pairs.items():
        tok = p.tokens
        if len(p) < MIN_FOR_QUARTILES or np.isnan(tok).any():
            out[task] = {"skipped": "too few items or input_tokens unknown"}
            continue
        edges = [float(x) for x in np.quantile(tok, [0.25, 0.5, 0.75])]
        bucket = np.searchsorted(np.array(edges), tok, side="left")
        buckets = []
        for q in range(4):
            sub = p.subset(bucket == q)
            pooled[q].append(sub)
            buckets.append({"quartile": QUARTILES[q], **_light(sub)})
        out[task] = {"token_edges": edges, "buckets": buckets}
    out["pooled"] = {"buckets": [{"quartile": QUARTILES[q], **_light_multi(pooled[q])} for q in range(4)]}
    return out


def _light(p: Pair) -> dict[str, Any]:
    return _light_multi([p])


def _light_multi(ps: list[Pair]) -> dict[str, Any]:
    ps = [p for p in ps if len(p)]
    if not ps:
        return {"n": 0}
    blk = paired_block({f"{i}": p for i, p in enumerate(ps)}, 0, with_ci=False)
    return {"n": blk["n_paired"], "ref_accuracy": blk["ref_accuracy"],
            "accuracy": _num(np.mean(np.concatenate([p.cfg_pred == p.gold for p in ps]))),
            "delta_accuracy": blk["delta_accuracy"]["delta"], "agreement": blk["agreement"],
            "kl_mean": blk["kl_mean"], "tvd_mean": blk["tvd_mean"]}


def by_question_type(pairs: dict[str, Pair]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qt in ("noul", "choice", "score"):
        sub = {t: p.subset(p.qtype == qt) for t, p in pairs.items()}
        sub = {t: p for t, p in sub.items() if len(p)}
        if sub:
            blk = paired_block(sub, 0, with_ci=False)
            blk["accuracy"] = _num(np.mean(np.concatenate([p.cfg_pred == p.gold for p in sub.values()])))
            blk["tasks"] = sorted(sub)
            out[qt] = blk
    return out


def cascade_rows(pairs: dict[str, Pair]) -> list[dict]:
    ps = [p for p in pairs.values() if len(p)]
    if not ps:
        return []
    return [_clean(r) for r in M.cascade(np.concatenate([p.gold for p in ps]), np.concatenate([p.ref_pred for p in ps]),
                                         np.concatenate([p.cfg_pred for p in ps]),
                                         np.concatenate([p.cfg_conf for p in ps]))]


# ------------------------------------------------------------------------ main analysis

def _read_meta(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    size = raw.get("file_size_gb")
    if size is None and raw.get("file_size_bytes") is not None:
        size = raw["file_size_bytes"] / 1e9
    peak = raw.get("peak_memory_gb")
    if peak is None and raw.get("peak_memory_bytes") is not None:
        peak = raw["peak_memory_bytes"] / 1e9
    return {"file_size_gb": size, "peak_memory_gb": peak}


def analyze(records_dir: Path, results_root: Path, n_boot: int = M.BOOTSTRAP_RESAMPLES,
            mem_tolerance: float = 0.15, models: Sequence[str] | None = None) -> dict[str, Any]:
    records = load_records(Path(records_dir))
    tree = scan_results(Path(results_root))
    if models is not None:
        wanted = set(models)
        unknown = sorted(wanted - set(tree))
        if unknown:
            raise ValueError(f"--models names not found under {results_root}: {unknown}")
        tree = {m: c for m, c in tree.items() if m in wanted}
    summary: dict[str, Any] = {
        "meta": {"n_boot": n_boot, "seed": M.SEED, "eps": M.EPS, "ece_bins": M.ECE_BINS,
                 "flip_buckets": list(M.FLIP_LABELS), "cascade_thresholds": list(M.CASCADE_THRESHOLDS),
                 "tasks_with_records": sorted(records), "mem_tolerance": mem_tolerance},
        "absent": {"missing": [], "models_without_ref": [], "tasks_without_records": [], "partial": []},
        "models": {}, "equal_memory": {},
    }
    compact: dict[tuple[str, str], dict[str, dict]] = {}
    sizes: dict[tuple[str, str], dict] = {}
    pooled_ok: dict[tuple[str, str], dict] = {}

    for model, configs in tree.items():
        repeats = {c: REPEAT_RE.match(c).group("base") for c in configs if REPEAT_RE.match(c)
                   and REPEAT_RE.match(c).group("base") in configs}
        names = sorted((c for c in configs if c not in repeats), key=lambda c: (c != REF, c))  # ref first
        task_names = sorted({t for c in configs.values() for t in c if t != "__meta__"})
        for t in task_names:
            if t not in records and t not in summary["absent"]["tasks_without_records"]:
                summary["absent"]["tasks_without_records"].append(t)

        loaded: dict[str, dict[str, tuple[dict, dict]]] = {}
        for cfg_name, files in configs.items():
            loaded[cfg_name] = {}
            for task in sorted(set(records) & set(files)):
                loaded[cfg_name][task] = build_items(task, records[task], read_jsonl(files[task]))
        has_ref = REF in configs
        if not has_ref:
            summary["absent"]["models_without_ref"].append(model)

        model_out: dict[str, Any] = {"configs": {}, "noise_floor": {}}
        ref_items = {t: v[0] for t, v in loaded.get(REF, {}).items()}

        # noise floors: repeat vs base
        noise_by_task: dict[str, list[dict]] = {}
        for rep, base in sorted(repeats.items()):
            tasks_out, pairs = {}, {}
            for task, (items, _) in loaded[rep].items():
                if task in loaded[base]:
                    pr = make_pair(task, loaded[base][task][0], items)
                    if pr is not None:
                        pairs[task] = pr
                        tasks_out[task] = paired_block({task: pr}, n_boot)
                        noise_by_task.setdefault(task, []).append(tasks_out[task])
            model_out["noise_floor"].setdefault(base, {})[rep] = {
                "tasks": tasks_out, "pooled": paired_block(pairs, n_boot) if pairs else {"n_paired": 0}}

        for cfg_name in names:
            files = configs[cfg_name]
            meta = _read_meta(files.get("__meta__"))
            sizes[(model, cfg_name)] = meta
            cfg_out: dict[str, Any] = {"file": meta, "tasks": {}}
            tasks_items: dict[str, list[Item]] = {}
            pairs: dict[str, Pair] = {}
            compact[(model, cfg_name)] = {}
            for task in sorted(records):
                if task not in loaded[cfg_name]:
                    summary["absent"]["missing"].append(
                        {"model": model, "config": cfg_name, "task": task, "reason": "no results file"})
                    continue
                items, counts = loaded[cfg_name][task]
                lst = list(items.values())
                tasks_items[task] = lst
                compact[(model, cfg_name)][task] = {k: (it.pred, it.gold) for k, it in items.items()}
                blk = {**counts, **abs_block({task: lst}),
                       "coverage": _num(len(lst) / counts["n_expected"]) if counts["n_expected"] else None,
                       "latency": M.latency_stats([it.latency for it in lst])}
                if blk["coverage"] is not None and blk["coverage"] < 1.0:
                    summary["absent"]["partial"].append({"model": model, "config": cfg_name, "task": task,
                                                         "coverage": blk["coverage"]})
                if has_ref and cfg_name != REF:
                    if task in ref_items:
                        pr = make_pair(task, ref_items[task], items)
                        if pr is not None:
                            pairs[task] = pr
                            blk["vs_ref"] = _annotate_noise(paired_block({task: pr}, n_boot),
                                                            noise_by_task.get(task, []))
                            if "score_shift" in blk["vs_ref"]:
                                blk["score_shift"] = blk["vs_ref"].pop("score_shift")
                    else:
                        summary["absent"]["missing"].append(
                            {"model": model, "config": cfg_name, "task": task, "reason": "ref has no results for task"})
                if any(it.qtype == "score" for it in lst) and "vs_ref" not in blk:
                    blk["mean_expected_score"] = _num(np.mean([M.expected_score(it.p) for it in lst]))
                cfg_out["tasks"][task] = blk

            all_lat = [it.latency for lst in tasks_items.values() for it in lst]
            pooled = {**abs_block(tasks_items), "latency": M.latency_stats(all_lat)}
            if pairs:
                pooled["vs_ref"] = _annotate_noise(paired_block(pairs, n_boot),
                                                   [b for blks in noise_by_task.values() for b in blks])
                if "score_shift" in pooled["vs_ref"]:
                    pooled["score_shift"] = pooled["vs_ref"].pop("score_shift")
                cfg_out["by_length_quartile"] = length_quartiles(pairs)
                cfg_out["by_question_type"] = by_question_type(pairs)
                casc = {t: cascade_rows({t: p}) for t, p in pairs.items()}
                casc["pooled"] = cascade_rows(pairs)
                cfg_out["cascade"] = casc
            cfg_out["pooled"] = pooled
            pooled_ok[(model, cfg_name)] = pooled
            model_out["configs"][cfg_name] = cfg_out
        summary["models"][model] = model_out

    summary["equal_memory"] = equal_memory(compact, sizes, pooled_ok, mem_tolerance, n_boot)
    summary["absent"]["tasks_without_results"] = sorted(
        set(records) - {t for c in compact.values() for t in c})
    return _clean(summary)


def _annotate_noise(block: dict[str, Any], noise_blocks: list[dict]) -> dict[str, Any]:
    flips = [1.0 - b["agreement"] for b in noise_blocks if b.get("n_paired")]
    kls = [b["kl_mean"] for b in noise_blocks if b.get("n_paired")]
    if not flips or not block.get("n_paired"):
        block["noise_floor_max_flip_rate"] = None
        block["noise_floor_max_kl"] = None
        block["exceeds_noise_floor"] = None
        return block
    block["noise_floor_max_flip_rate"] = max(flips)
    block["noise_floor_max_kl"] = max(kls)
    block["exceeds_noise_floor"] = bool((1.0 - block["agreement"]) > max(flips) and block["kl_mean"] > max(kls))
    return block


def equal_memory(compact, sizes, pooled_ok, tol: float, n_boot: int) -> dict[str, Any]:
    by_size, absent = [], []
    for (model, cfg), meta in sorted(sizes.items()):
        if meta.get("file_size_gb") is None:
            absent.append({"model": model, "config": cfg})
            continue
        pooled = pooled_ok.get((model, cfg), {})
        by_size.append({"model": model, "config": cfg, "file_size_gb": meta["file_size_gb"],
                        "peak_memory_gb": meta.get("peak_memory_gb"), "n_scored": pooled.get("n_scored"),
                        "accuracy": pooled.get("accuracy"), "macro_f1": pooled.get("macro_f1")})
    by_size.sort(key=lambda r: (r["file_size_gb"], r["model"], r["config"]))
    pairs = []
    for i, a in enumerate(by_size):
        for b in by_size[i + 1:]:
            if a["model"] == b["model"] or max(a["file_size_gb"], b["file_size_gb"]) > (1 + tol) * min(a["file_size_gb"], b["file_size_gb"]):
                continue
            x, y = (a, b) if a["model"] < b["model"] else (b, a)
            ca, cb = compact.get((x["model"], x["config"]), {}), compact.get((y["model"], y["config"]), {})
            gold, pa, pb, tasks = [], [], [], []
            for task in sorted(set(ca) & set(cb)):
                keys = sorted(k for k in ca[task].keys() & cb[task].keys() if ca[task][k][1] is not None)
                if keys:
                    tasks.append(task)
                    gold += [ca[task][k][1] for k in keys]
                    pa += [ca[task][k][0] for k in keys]
                    pb += [cb[task][k][0] for k in keys]
            if not gold:
                continue
            g, ra, rb = np.array(gold), np.array(pa), np.array(pb)
            arr = M.bootstrap_arrays(g, ra, rb, n_boot, task_seed("equal-memory"))
            pairs.append({
                "a": {k: x[k] for k in ("model", "config", "file_size_gb")},
                "b": {k: y[k] for k in ("model", "config", "file_size_gb")},
                "size_ratio": max(x["file_size_gb"], y["file_size_gb"]) / min(x["file_size_gb"], y["file_size_gb"]),
                "n_items": len(g), "tasks": tasks,
                "accuracy_a": float(np.mean(ra == g)), "accuracy_b": float(np.mean(rb == g)),
                "delta_accuracy_b_minus_a": {"delta": float(np.mean(rb == g) - np.mean(ra == g)),
                                             "ci95": M.ci95(arr["d_acc"])},
            })
    return {"tolerance": tol, "by_size": by_size, "pairs": pairs, "sizes_absent": absent}


# --------------------------------------------------------------------------- tables

def _f(x: Any, nd: int = 4, pct: bool = False) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if pct:
        return f"{100 * x:.1f}%"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def _ci(d: dict | None, scale: float = 100.0) -> str:
    if not d or d.get("delta") is None:
        return "-"
    if d.get("ci95") is None:
        return f"{scale * d['delta']:+.2f}"
    lo, hi = d["ci95"]
    return f"{scale * d['delta']:+.2f} [{scale * lo:+.2f}, {scale * hi:+.2f}]"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_tables(summary: dict[str, Any], title: str = "Clef quantization benchmark: analysis") -> str:
    L = [f"# {title}", "",
         f"Bootstrap: {summary['meta']['n_boot']} paired resamples, seed {summary['meta']['seed']}. "
         "Accuracy and macro-F1 deltas are in percentage points (cfg minus ref) with 95% CI. "
         "KL is KL(ref || cfg) in nats. Pooled accuracy is micro; pooled macro-F1 averages the per-task values.", ""]
    for model, m in summary["models"].items():
        L += [f"## {model}", "", "### Pooled over all tasks", ""]
        rows = []
        for cfg, c in m["configs"].items():
            p, v, lat = c["pooled"], c["pooled"].get("vs_ref"), c["pooled"].get("latency", {})
            rows.append([cfg, _f(p.get("n_scored")), _f(p.get("accuracy"), pct=True), _f(p.get("macro_f1"), 4),
                         _ci(v["delta_accuracy"]) if v else "-", _ci(v["delta_macro_f1"]) if v else "-",
                         _f(v["agreement"], pct=True) if v else "-", _f(v["mcnemar"]["p"], 4) if v else "-",
                         _f(v["kl_mean"], 5) if v else "-", _f(v["tvd_mean"], 5) if v else "-",
                         _f(p.get("brier")), _f(p.get("ece")),
                         _f(v.get("exceeds_noise_floor")) if v else "-",
                         _f(lat.get("p50_ms"), 1), _f(lat.get("p95_ms"), 1), _f(lat.get("decisions_per_sec"), 2),
                         _f(c["file"].get("file_size_gb"), 2)])
        L += [md_table(["config", "n", "acc", "macro-F1", "dAcc pp [95% CI]", "dF1 pp [95% CI]", "agree vs ref",
                        "McNemar p", "KL", "TVD", "Brier", "ECE", "> noise", "p50 ms", "p95 ms", "dec/s", "GB"], rows), ""]

        L += ["### Accuracy delta vs ref by task (pp [95% CI]; agreement in brackets)", ""]
        tasks = sorted({t for c in m["configs"].values() for t in c["tasks"]})
        rows = []
        for t in tasks:
            row = [t]
            for cfg, c in m["configs"].items():
                b = c["tasks"].get(t)
                if b is None:
                    row.append("absent")
                elif "vs_ref" in b:
                    row.append(f"{_ci(b['vs_ref']['delta_accuracy'])} ({_f(b['vs_ref']['agreement'], pct=True)})")
                else:
                    row.append(f"acc {_f(b.get('accuracy'), pct=True)}")
            rows.append(row)
        L += [md_table(["task"] + list(m["configs"]), rows), ""]

        if m["noise_floor"]:
            L += ["### Noise floor (repeat run vs its base config)", ""]
            rows = []
            for base, reps in m["noise_floor"].items():
                for rep, nf in reps.items():
                    p = nf["pooled"]
                    if p.get("n_paired"):
                        rows.append([base, rep, _f(p["n_paired"]), _f(p["agreement"], pct=True), _f(p["n_flips"]),
                                     _ci(p["delta_accuracy"]), _f(p["kl_mean"], 6), _f(p["tvd_mean"], 6)])
            L += [md_table(["base", "repeat", "n", "agreement", "flips", "dAcc pp [95% CI]", "KL", "TVD"], rows), ""]

        L += ["### Flip rate by ref confidence (pooled)", ""]
        rows = []
        for cfg, c in m["configs"].items():
            v = c["pooled"].get("vs_ref")
            if v:
                rows.append([cfg] + [f"{_f(b['flip_rate'], pct=True)} (n={b['n']})" for b in v["flip_by_ref_confidence"]])
        if rows:
            L += [md_table(["config"] + summary["meta"]["flip_buckets"], rows), ""]

        L += ["### By input length quartile (pooled; accuracy delta pp, agreement)", ""]
        rows = []
        for cfg, c in m["configs"].items():
            q = c.get("by_length_quartile")
            if q:
                rows.append([cfg] + [f"{_f(b.get('delta_accuracy') and 100 * b['delta_accuracy'], 2)} / "
                                     f"{_f(b.get('agreement'), pct=True)} (n={b['n']})" for b in q["pooled"]["buckets"]])
        if rows:
            L += [md_table(["config"] + list(QUARTILES), rows), "Quartiles are of input_tokens within each task.", ""]

        L += ["### By question type (accuracy delta pp, agreement, KL)", ""]
        rows = []
        for cfg, c in m["configs"].items():
            for qt, b in (c.get("by_question_type") or {}).items():
                rows.append([cfg, qt, _f(b["n_paired"]), _f(100 * b["delta_accuracy"]["delta"], 2),
                             _f(b["agreement"], pct=True), _f(b["kl_mean"], 5)])
        if rows:
            L += [md_table(["config", "type", "n", "dAcc pp", "agreement", "KL"], rows), ""]

        L += ["### Score tasks: shift of expected score vs ref", ""]
        rows = []
        for cfg, c in m["configs"].items():
            for t, b in c["tasks"].items():
                s = b.get("score_shift")
                if s:
                    rows.append([cfg, t, _f(s["n"]), _f(s["mean_signed_shift"]), _f(s["mean_abs_shift"]),
                                 _f(s["center_pull"]), _f(s["shift_when_ref_above_middle"]),
                                 _f(s["shift_when_ref_below_middle"])])
        if rows:
            L += [md_table(["config", "task", "n", "mean signed shift", "mean abs shift", "center pull",
                            "shift (ref above mid)", "shift (ref below mid)"], rows),
                  "center pull < 0: expected scores move toward the middle of the scale; > 0: toward the ends.", ""]

        L += ["### Cascade: use config, escalate to ref when top probability < t (pooled)", ""]
        thresholds = summary["meta"]["cascade_thresholds"]
        for cfg, c in m["configs"].items():
            casc = (c.get("cascade") or {}).get("pooled")
            if casc:
                rows = [[f"{r['threshold']:.2f}", _f(r["escalation_rate"], pct=True), _f(r["agreement_with_ref"], pct=True),
                         _f(r["accuracy"], pct=True)] for r in casc]
                L += [f"**{cfg}** (thresholds {thresholds[0]}..{thresholds[-1]})", "",
                      md_table(["t", "escalation rate", "agreement with ref", "accuracy"], rows), ""]

    em = summary.get("equal_memory") or {}
    if em.get("by_size"):
        L += ["## Equal-memory comparison", "",
              md_table(["model", "config", "file GB", "peak GB", "n", "accuracy", "macro-F1"],
                       [[r["model"], r["config"], _f(r["file_size_gb"], 2), _f(r.get("peak_memory_gb"), 2),
                         _f(r["n_scored"]), _f(r["accuracy"], pct=True), _f(r["macro_f1"], 4)] for r in em["by_size"]]), ""]
        if em["pairs"]:
            L += [f"Pairs from different models whose file sizes are within {100 * em['tolerance']:.0f}%:", "",
                  md_table(["A", "B", "size ratio", "n items", "acc A", "acc B", "B minus A pp [95% CI]"],
                           [[f"{p['a']['model']}/{p['a']['config']} ({p['a']['file_size_gb']:.2f} GB)",
                             f"{p['b']['model']}/{p['b']['config']} ({p['b']['file_size_gb']:.2f} GB)",
                             _f(p["size_ratio"], 3), _f(p["n_items"]), _f(p["accuracy_a"], pct=True),
                             _f(p["accuracy_b"], pct=True), _ci(p["delta_accuracy_b_minus_a"])] for p in em["pairs"]]), ""]
        else:
            L += ["No cross-model pairs fall within the size tolerance.", ""]
    if em.get("sizes_absent"):
        L += ["Configs without file size (no meta.json): " + ", ".join(f"{r['model']}/{r['config']}" for r in em["sizes_absent"]), ""]

    ab = summary["absent"]
    L += ["## Absent data", ""]
    L += [f"- missing: {r['model']}/{r['config']}/{r['task']} ({r['reason']})" for r in ab["missing"]] or ["- no missing config/task files"]
    L += [f"- partial coverage: {r['model']}/{r['config']}/{r['task']} ({r['coverage']:.1%})" for r in ab["partial"]]
    L += [f"- model without a `ref` config (no paired metrics): {m}" for m in ab["models_without_ref"]]
    L += [f"- results for task without records: {t}" for t in ab["tasks_without_records"]]
    L += [f"- task with records but no results anywhere: {t}" for t in ab.get("tasks_without_results", [])]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------- private mode

FIXED_STRINGS = {"models", "configs", "tasks", "noise_floor", "pooled", "vs_ref", "absent", "meta", "equal_memory"}


def scrub_private(obj: Any, allowed: set[str]) -> Any:
    """Return obj unchanged if every key and string value is in `allowed`; otherwise raise ValueError."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) not in allowed:
                raise ValueError(f"unexpected key in private summary: {str(k)[:40]!r}")
            scrub_private(v, allowed)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            scrub_private(v, allowed)
    elif isinstance(obj, str) and obj not in allowed:
        raise ValueError(f"unexpected string in private summary: {obj[:40]!r}")
    return obj


def _structural_names(obj: Any, acc: set[str]) -> None:
    """All dict keys of an (aggregate) summary; every key is a fixed field name or a model/config/set name."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(str(k))
            _structural_names(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _structural_names(v, acc)


def private_aggregate(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep only aggregate numbers keyed by model/config/set name; drop everything free-form."""
    out = {"meta": {k: v for k, v in summary["meta"].items() if k != "tasks_with_records"},
           "sets": summary["meta"]["tasks_with_records"], "models": summary["models"],
           "equal_memory": summary["equal_memory"],
           "absent": {"missing": [{"model": r["model"], "config": r["config"], "set": r["task"]}
                                  for r in summary["absent"]["missing"]],
                      "models_without_ref": summary["absent"]["models_without_ref"]}}
    names = set()
    _structural_names(out["models"], names)
    for r in out["absent"]["missing"]:
        names |= {r["model"], r["config"], r["set"]}
    for m in out["absent"]["models_without_ref"]:
        names.add(m)
    names |= set(out["sets"]) | set(FIXED_STRINGS) | set(M.FLIP_LABELS) | set(QUARTILES) | {"noul", "choice", "score"}
    _structural_names(out["equal_memory"], names)
    _structural_names(out["absent"], names)
    _structural_names(out["meta"], names)
    names |= {"set", "sets", "model", "config"}
    for r in out["equal_memory"].get("by_size", []):
        names |= {r["model"], r["config"]}
    names |= {s for s in out["meta"].get("flip_buckets", [])}
    scrub_private(out, names)
    return out


def run_private(private_root: Path, out_dir: Path, n_boot: int = M.BOOTSTRAP_RESAMPLES,
                mem_tolerance: float = 0.15, models: Sequence[str] | None = None) -> dict[str, Any]:
    root = Path(private_root)
    summary = analyze(root / "records", root / "results", n_boot=n_boot, mem_tolerance=mem_tolerance, models=models)
    for entry in summary["models"].values():  # per-task 'tasks' are sets here; drop nothing but be explicit
        entry.setdefault("noise_floor", {})
    agg = private_aggregate(summary)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "private_summary.json").write_text(json.dumps(agg, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    shaped = {"meta": agg["meta"], "models": agg["models"], "equal_memory": agg["equal_memory"],
              "absent": {"missing": [{**r, "task": r["set"], "reason": "no results file"} for r in agg["absent"]["missing"]],
                         "partial": [], "models_without_ref": agg["absent"]["models_without_ref"],
                         "tasks_without_records": []}}
    (out_dir / "private_tables.md").write_text(
        render_tables(shaped, "Private sets: aggregate analysis (sets keyed by name; no record content)"), encoding="utf-8")
    return agg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=Path("data/records"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    ap.add_argument("--private", type=Path, default=None, help="private root with records/ and results/; aggregates only")
    ap.add_argument("--n-boot", type=int, default=M.BOOTSTRAP_RESAMPLES)
    ap.add_argument("--mem-tolerance", type=float, default=0.15, help="relative file-size window for equal-memory pairs")
    ap.add_argument("--models", nargs="+", default=None, help="analyse only these model directories (default: all)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.private is not None:
        run_private(args.private, args.out, args.n_boot, args.mem_tolerance, args.models)
        print(f"wrote {args.out}/private_summary.json and private_tables.md", file=sys.stderr)
        return 0
    summary = analyze(args.records, args.results, args.n_boot, args.mem_tolerance, args.models)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out / "tables.md").write_text(render_tables(summary), encoding="utf-8")
    print(f"wrote {args.out}/summary.json and tables.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
