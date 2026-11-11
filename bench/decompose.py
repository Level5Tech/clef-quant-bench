#!/usr/bin/env python
"""Noise-floor decomposition of a quantization comparison (PLAN.md, "Write-up decision").

Comparison unit: the first UNIT_N (200) records of every public task, the records that exist in every `-r2` repeat
and in `ref-fla`. Every statistic below is computed on exactly those records, for every pair of configs, so that the
non-quantization sources of disagreement (restart of the same server, batch size of a rerun, kernel selection,
engine) and the quantization effects are on the same footing.

Per pair (A = baseline, B = other):
  n questions / records, top-1 flip rate with a paired bootstrap 95% CI over records, mean total-variation distance
  with CI, max |dprob|, max |dlogit| (raw, and centred per question because softmax ignores a constant shift) where both
  sides have logits, and the number / share of records that are bit-identical.

Conventions (same as bench/metrics.py and bench/analyze.py):
  * distribution = softmax(logits) when logits are present, else the runner's probs; clipped at EPS and renormalised
    (metrics.dist). Top-1 = argmax of that vector, ties to the first option in the record's option order.
  * bit-identical record: every question's logits are exactly equal when BOTH sides have logits; otherwise (llama.cpp
    returns full-precision probabilities and no logits) every question's stored probs are exactly equal. The basis is
    recorded per pair. Logits from the native and vLLM paths are on bf16 granularity, so "identical" means identical at
    the precision the engines emit.
  * bootstrap: 2000 resamples of the 2,400 records (simple resampling, rng seed 20261015), ONE index matrix shared by
    every statistic and every difference, so differences are paired on the same resamples. A stratified-by-task variant
    (200 records resampled inside each task, same seed) is computed for the pre-registered difference tests as a
    sensitivity check. Intervals are percentile 95% unless stated.
  * Rare events: when a pair has 0 (or very few) flips the percentile bootstrap interval collapses, so an exact
    Clopper-Pearson interval for the flip rate (treating questions as independent) is reported next to it.

Usage:
  python -m bench.decompose --records data/records --results results --model clef-flash --out analysis
writes <out>/decomposition.json and <out>/decomposition.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats as sps

if __package__ in (None, ""):  # allow `python bench/decompose.py`
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from bench import metrics as M  # noqa: E402
from bench.schema import option_ids  # noqa: E402

UNIT_N = 200
SEED = M.SEED
N_BOOT = M.BOOTSTRAP_RESAMPLES

# group, A (baseline), B, label. Order is the order of the report tables.
PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("restart_vllm", "bf16-srv", "bf16-srv-r2", "vLLM restart, BF16"),
    ("restart_vllm", "fp8", "fp8-r2", "vLLM restart, FP8"),
    ("restart_vllm", "nvfp4", "nvfp4-r2", "vLLM restart, NVFP4"),
    ("restart_llamacpp", "gguf-bf16", "gguf-bf16-r2", "llama.cpp restart, BF16 (N5)"),
    ("native_rerun_bs8", "ref", "ref-r2",
     "native rerun at batch 8 vs batch 1 (a rerun AND a batch-size change, not a pure restart)"),
    ("kernel", "ref", "ref-fla", "Gated DeltaNet kernel: PyTorch fallback vs flash-linear-attention (N4)"),
    ("engine_bf16", "ref", "bf16-srv", "engine at BF16: transformers -> vLLM (N3)"),
    ("engine_bf16", "ref", "gguf-bf16", "engine at BF16: transformers -> llama.cpp (N3)"),
    ("quant_within_engine", "bf16-srv", "fp8", "FP8 vs BF16 inside vLLM (N1)"),
    ("quant_within_engine", "bf16-srv", "nvfp4", "NVFP4 vs BF16 inside vLLM (N2)"),
    ("quant_within_engine", "gguf-bf16", "q8_0", "Q8_0 vs BF16 inside llama.cpp (N3 8-bit)"),
    ("quant_within_engine", "gguf-bf16", "q4_k_m", "Q4_K_M vs BF16 inside llama.cpp (N2)"),
)
VS_REF = ("bf16-srv", "fp8", "nvfp4", "gguf-bf16", "q8_0", "q4_k_m", "ref-r2", "ref-fla",
          "bf16-srv-r2", "fp8-r2", "nvfp4-r2", "gguf-bf16-r2")
ALL_CONFIGS = ("ref", *VS_REF)

VLLM_RESTARTS = ("bf16-srv|bf16-srv-r2", "fp8|fp8-r2", "nvfp4|nvfp4-r2")
RESTARTS = (*VLLM_RESTARTS, "gguf-bf16|gguf-bf16-r2")
NON_QUANT_SOURCES = (*RESTARTS, "ref|ref-r2", "ref|ref-fla", "ref|bf16-srv", "ref|gguf-bf16")
ENGINE_CHANGES = ("ref|bf16-srv", "ref|gguf-bf16")
EIGHT_BIT = ("bf16-srv|fp8", "gguf-bf16|q8_0")
FOUR_BIT = ("bf16-srv|nvfp4", "gguf-bf16|q4_k_m")


def pk(a: str, b: str) -> str:
    return f"{a}|{b}"


# ----------------------------------------------------------------------------- loading

@dataclass
class Unit:
    """The comparison unit: questions in a fixed order, with their record index."""
    tasks: list[str]
    task_of: list[str]            # per question
    rec_ids: list[str]            # per record (unit order)
    rec_task: list[str]           # per record
    rec_of: np.ndarray            # per question -> record index
    qkeys: list[tuple[str, str]]  # per question (record id, question id)
    opts: list[tuple[str, ...]]
    n_records: int = 0
    n_questions: int = 0
    gold: list[str | None] = field(default_factory=list)   # per question; None when the record carries no label
    qtype: list[str] = field(default_factory=list)          # per question: noul | choice | score


@dataclass
class Cfg:
    name: str
    logits: list[np.ndarray | None]   # per question, record option order
    probs: list[np.ndarray]           # per question, stored probs reordered to record option order
    p: list[np.ndarray]               # per question, distribution used for metrics
    engines: dict[str, set[str]]      # task -> {"engine | engine_version | batch_size"}
    n_outside_unit: int = 0           # result lines for records that are not in the comparison unit (ignored)
    n_lines: int = 0                  # result lines read in total

    @property
    def has_logits(self) -> bool:
        return all(x is not None for x in self.logits)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_unit(records_dir: Path, unit_n: int | None = UNIT_N, tasks: Sequence[str] | None = None) -> Unit:
    """First `unit_n` lines of every data/records/<task>.jsonl, in file order (all records when unit_n is None).

    `tasks` restricts the record files used (default: every *.jsonl in the directory)."""
    names = sorted(tasks) if tasks is not None else sorted(p.stem for p in Path(records_dir).glob("*.jsonl"))
    if not names:
        raise FileNotFoundError(f"no records under {records_dir}")
    u = Unit(names, [], [], [], np.zeros(0, dtype=int), [], [])
    rec_of: list[int] = []
    for task in names:
        with open(Path(records_dir) / f"{task}.jsonl", encoding="utf-8") as fh:
            rows = []
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
                if unit_n is not None and len(rows) == unit_n:
                    break
        if unit_n is not None and len(rows) < unit_n:
            raise ValueError(f"{task}: only {len(rows)} records, need {unit_n}")
        for r in rows:
            ri = len(u.rec_ids)
            u.rec_ids.append(r["id"])
            u.rec_task.append(task)
            for qid, q in r["request"]["questions"].items():
                u.qkeys.append((r["id"], qid))
                u.opts.append(tuple(option_ids(q)))
                u.task_of.append(task)
                u.gold.append((r.get("gold") or {}).get(qid))
                u.qtype.append(q["type"])
                rec_of.append(ri)
    if len(set(u.rec_ids)) != len(u.rec_ids):
        raise ValueError("duplicate record ids inside the comparison unit")
    u.rec_of = np.array(rec_of, dtype=int)
    u.n_records = len(u.rec_ids)
    u.n_questions = len(u.qkeys)
    u.tasks = list(names)
    return u


def load_cfg(unit: Unit, results_dir: Path, model: str, name: str) -> Cfg:
    """Results of one config on the unit's questions; raises if any record is missing, duplicated or unusable."""
    root = Path(results_dir) / model / name
    by_task_ids: dict[str, set[str]] = {}
    for rid, task in zip(unit.rec_ids, unit.rec_task):
        by_task_ids.setdefault(task, set()).add(rid)
    lines: dict[str, dict] = {}
    engines: dict[str, set[str]] = {}
    n_outside = n_total = 0
    for task in unit.tasks:
        path = root / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{model}/{name}: missing {path.name}")
        seen: set[str] = set()
        for line in read_jsonl(path):
            rid = line.get("id")
            n_total += 1
            if rid not in by_task_ids[task]:
                n_outside += 1
                continue
            if rid in seen:
                raise ValueError(f"{model}/{name}/{task}: duplicate id in results")
            seen.add(rid)
            lines[rid] = line
            engines.setdefault(task, set()).add(
                f"{line.get('engine')} | {line.get('engine_version')} | batch_size={line.get('batch_size')}")
        if seen != by_task_ids[task]:
            raise ValueError(f"{model}/{name}/{task}: {len(by_task_ids[task] - seen)} unit records missing")
    logits: list[np.ndarray | None] = []
    probs: list[np.ndarray] = []
    pdist: list[np.ndarray] = []
    for (rid, qid), opts in zip(unit.qkeys, unit.opts):
        line = lines[rid]
        ans = None if line.get("error") else (line.get("answers") or {}).get(qid)
        if not isinstance(ans, dict):
            raise ValueError(f"{model}/{name}: unusable answer for a unit question (error or missing)")
        ids = [str(x) for x in ans.get("option_ids") or []]
        if sorted(ids) != sorted(opts):
            raise ValueError(f"{model}/{name}: option ids differ from the record's options")
        pos = {o: i for i, o in enumerate(ids)}
        order = [pos[o] for o in opts]
        raw_p = np.asarray(ans["probs"], dtype=np.float64)[order]
        raw_l = ans.get("logits")
        lg = None if raw_l is None else np.asarray(raw_l, dtype=np.float64)[order]
        if lg is not None:
            d = M.dist(logits=lg)
        else:
            d = M.dist(probs=raw_p)
        if not np.all(np.isfinite(d)):
            raise ValueError(f"{model}/{name}: non-finite distribution")
        logits.append(lg)
        probs.append(raw_p)
        pdist.append(d)
    present = [x is not None for x in logits]
    if any(present) and not all(present):
        raise ValueError(f"{model}/{name}: logits present for some unit questions and null for others")
    return Cfg(name, logits, probs, pdist, engines, n_outside, n_total)


# ------------------------------------------------------------------------- pair stats

@dataclass
class PairArrays:
    """Per-question arrays for one pair."""
    flip: np.ndarray        # bool
    tvd: np.ndarray
    dprob: np.ndarray       # max over options
    dlogit_raw: np.ndarray  # nan when a side has no logits
    dlogit_ctr: np.ndarray
    identical: np.ndarray   # bool per question
    basis: str


def pair_arrays(a: Cfg, b: Cfg) -> PairArrays:
    n = len(a.p)
    flip = np.zeros(n, dtype=bool)
    tvd = np.zeros(n)
    dprob = np.zeros(n)
    dl_raw = np.full(n, np.nan)
    dl_ctr = np.full(n, np.nan)
    ident = np.zeros(n, dtype=bool)
    both_logits = a.has_logits and b.has_logits
    for i in range(n):
        pa, pb = a.p[i], b.p[i]
        flip[i] = int(np.argmax(pa)) != int(np.argmax(pb))
        tvd[i] = M.tvd(pa, pb)
        dprob[i] = float(np.max(np.abs(pa - pb)))
        if both_logits:
            d = a.logits[i] - b.logits[i]
            dl_raw[i] = float(np.max(np.abs(d)))
            dl_ctr[i] = float(np.max(np.abs(d - d.mean())))
            ident[i] = bool(np.array_equal(a.logits[i], b.logits[i]))
        else:
            ident[i] = bool(np.array_equal(a.probs[i], b.probs[i]))
    return PairArrays(flip, tvd, dprob, dl_raw, dl_ctr, ident, "logits" if both_logits else "probs")


# ----------------------------------------------------------------------------- bootstrap

class Boot:
    """Shared resample indices over the unit's records. Statistics are ratio estimators over records."""

    def __init__(self, unit: Unit, n_boot: int = N_BOOT, seed: int = SEED):
        self.unit, self.n_boot, self.seed = unit, n_boot, seed
        n = unit.n_records
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, n, size=(n_boot, n))
        # stratified by task: resample inside each task's block of records, same seed
        rng2 = np.random.default_rng(seed)
        blocks = []
        rec_task = np.array(unit.rec_task)
        for task in unit.tasks:
            pos = np.flatnonzero(rec_task == task)
            blocks.append(pos[rng2.integers(0, len(pos), size=(n_boot, len(pos)))])
        self.idx_strat = np.concatenate(blocks, axis=1)
        self.q_per_rec = np.bincount(unit.rec_of, minlength=n).astype(np.float64)

    def rec_sum(self, per_question: np.ndarray) -> np.ndarray:
        return np.bincount(self.unit.rec_of, weights=per_question.astype(np.float64), minlength=self.unit.n_records)

    def ratio(self, per_question: np.ndarray, strat: bool = False) -> np.ndarray:
        """Bootstrap distribution of sum(x)/n_questions over resampled records."""
        idx = self.idx_strat if strat else self.idx
        num = self.rec_sum(per_question)
        out = np.empty(self.n_boot)
        step = 500
        for s in range(0, self.n_boot, step):
            ii = idx[s:s + step]
            out[s:s + step] = num[ii].sum(axis=1) / self.q_per_rec[ii].sum(axis=1)
        return out

    def ratio_all_identical(self, ident_q: np.ndarray, strat: bool = False) -> np.ndarray:
        """Share of records whose every question is identical, over resampled records."""
        rec_ident = (self.rec_sum(~ident_q) == 0).astype(np.float64)
        idx = self.idx_strat if strat else self.idx
        out = np.empty(self.n_boot)
        step = 500
        for s in range(0, self.n_boot, step):
            out[s:s + step] = rec_ident[idx[s:s + step]].mean(axis=1)
        return out


def ci(values: np.ndarray, alpha: float = 0.05) -> list[float]:
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    lo = 0.0 if k == 0 else float(sps.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


# ------------------------------------------------------------------------------ results

def summarize_pair(unit: Unit, boot: Boot, a: str, b: str, arr: PairArrays, label: str, group: str) -> dict[str, Any]:
    n_q = unit.n_questions
    flips = int(arr.flip.sum())
    rec_ident = (boot.rec_sum(~arr.identical) == 0)
    k_ident = int(rec_ident.sum())
    amax = int(np.argmax(arr.dprob))
    out: dict[str, Any] = {
        "a": a, "b": b, "label": label, "group": group,
        "n_questions": n_q, "n_records": unit.n_records,
        "flips": flips, "flip_rate": flips / n_q,
        "flip_ci95": ci(boot.ratio(arr.flip)),
        "flip_ci95_exact_cp": clopper_pearson(flips, n_q),
        "tvd_mean": float(arr.tvd.mean()), "tvd_ci95": ci(boot.ratio(arr.tvd)),
        "max_dprob": float(arr.dprob.max()),
        "max_dprob_at": {"task": unit.task_of[amax], "id": unit.qkeys[amax][0]},
        "identical_basis": arr.basis,
        "n_identical_records": k_ident, "share_identical": k_ident / unit.n_records,
        "share_identical_ci95": ci(boot.ratio_all_identical(arr.identical)),
        "flips_by_task": {t: int(arr.flip[np.array(unit.task_of) == t].sum()) for t in unit.tasks},
        "share_identical_by_task": {t: float(arr.identical[np.array(unit.task_of) == t].mean()) for t in unit.tasks},
    }
    if np.all(np.isfinite(arr.dlogit_raw)):
        out["max_dlogit_raw"] = float(arr.dlogit_raw.max())
        out["max_dlogit_centered"] = float(arr.dlogit_ctr.max())
    else:
        out["max_dlogit_raw"] = None
        out["max_dlogit_centered"] = None
    return out


def diff_test(boot: Boot, arr_a: PairArrays, arr_b: PairArrays, key_a: str, key_b: str, k: int = 1) -> dict[str, Any]:
    """flip(A) - flip(B) on the same records, paired bootstrap. k = number of comparisons in the family (Bonferroni)."""
    point = float(arr_a.flip.mean() - arr_b.flip.mean())
    d = boot.ratio(arr_a.flip) - boot.ratio(arr_b.flip)
    ds = boot.ratio(arr_a.flip, strat=True) - boot.ratio(arr_b.flip, strat=True)
    lo, hi = ci(d)
    slo, shi = ci(ds)
    out = {"minuend": key_a, "subtrahend": key_b,
           "flips_minuend": int(arr_a.flip.sum()), "flips_subtrahend": int(arr_b.flip.sum()),
           "diff": point, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
           "ci95_stratified": [slo, shi], "excludes_zero_stratified": bool(slo > 0 or shi < 0),
           "lower_gt_zero": bool(lo > 0), "upper_lt_zero": bool(hi < 0)}
    if k > 1:
        blo, bhi = ci(d, alpha=0.05 / k)
        out["ci95_bonferroni"] = [blo, bhi]
        out["bonferroni_k"] = k
        out["lower_gt_zero_bonferroni"] = bool(blo > 0)
    return out


def decompose(records_dir: Path, results_dir: Path, model: str = "clef-flash", unit_n: int = UNIT_N,
              n_boot: int = N_BOOT, seed: int = SEED, configs: Sequence[str] = ALL_CONFIGS) -> dict[str, Any]:
    unit = build_unit(Path(records_dir), unit_n)
    cfgs = {c: load_cfg(unit, Path(results_dir), model, c) for c in configs}
    boot = Boot(unit, n_boot, seed)

    arrays: dict[str, PairArrays] = {}
    labels: dict[str, tuple[str, str]] = {}
    groups: dict[str, list[str]] = {}

    def add(group: str, a: str, b: str, label: str) -> None:
        key = pk(a, b)
        if key not in arrays:
            arrays[key] = pair_arrays(cfgs[a], cfgs[b])
            labels[key] = (group, label)
        groups.setdefault(group, [])
        if key not in groups[group]:
            groups[group].append(key)

    for group, a, b, label in PAIRS:
        add(group, a, b, label)
    for c in VS_REF:
        add("vs_ref", "ref", c, f"{c} vs ref (for completeness)")

    pairs = {}
    for key, arr in arrays.items():
        a, b = key.split("|")
        pairs[key] = summarize_pair(unit, boot, a, b, arr, labels[key][1], labels[key][0])

    tests: dict[str, Any] = {}
    # N1
    n1 = diff_test(boot, arrays["bf16-srv|fp8"], arrays["bf16-srv|bf16-srv-r2"], "bf16-srv|fp8", "bf16-srv|bf16-srv-r2")
    n1["criterion"] = ("supported if FP8's flip rate against bf16-srv is NOT significantly larger than the bf16-srv "
                       "restart flip rate, i.e. the 95% CI of the difference does not lie above 0")
    n1["supported"] = not n1["lower_gt_zero"]
    # not pre-registered: FP8's own restart flip rate is larger than BF16's, so also compare against that floor
    n1["supplementary_not_preregistered"] = {
        "note": "FP8 vs BF16 minus FP8 against its own restart (fp8 vs fp8-r2)",
        **diff_test(boot, arrays["bf16-srv|fp8"], arrays["fp8|fp8-r2"], "bf16-srv|fp8", "fp8|fp8-r2")}
    tests["N1"] = n1
    # H1 restated at the decision level for the llama.cpp 8-bit build (not pre-registered)
    tests["H1_supplementary_not_preregistered"] = {
        "note": "Q8_0 vs BF16 inside llama.cpp minus the llama.cpp restart flip rate (N1's analogue for the GGUF 8-bit build)",
        **diff_test(boot, arrays["gguf-bf16|q8_0"], arrays["gguf-bf16|gguf-bf16-r2"], "gguf-bf16|q8_0", "gguf-bf16|gguf-bf16-r2")}
    # N2
    fam = [(q, s) for q in FOUR_BIT for s in NON_QUANT_SOURCES]
    n2_rows = [diff_test(boot, arrays[q], arrays[s], q, s, k=len(fam)) for q, s in fam]
    tests["N2"] = {
        "criterion": ("supported if every quantization flip rate (NVFP4 vs bf16-srv, Q4_K_M vs gguf-bf16) exceeds every "
                      "non-quantization source with a 95% CI of the difference that excludes 0 (lower bound > 0)"),
        "family_size": len(fam), "rows": n2_rows,
        "all_lower_gt_zero": all(r["lower_gt_zero"] for r in n2_rows),
        "all_lower_gt_zero_stratified": all(r["ci95_stratified"][0] > 0 for r in n2_rows),
        "all_lower_gt_zero_bonferroni": all(r["lower_gt_zero_bonferroni"] for r in n2_rows),
        "supported": all(r["lower_gt_zero"] for r in n2_rows),
        "weakest_row": min(n2_rows, key=lambda r: r["ci95"][0]),
    }
    # N3
    n3_rows = [diff_test(boot, arrays[e], arrays[q], e, q) for e in ENGINE_CHANGES for q in EIGHT_BIT]
    tests["N3"] = {
        "criterion": ("supported if each engine change at BF16 flips at least as many decisions as each 8-bit "
                      "quantization inside one engine: point estimate of (engine - 8-bit) >= 0; contradicted for a "
                      "combination when the 95% CI of the difference lies below 0"),
        "rows": n3_rows,
        "all_point_ge_zero": all(r["diff"] >= 0 for r in n3_rows),
        "any_significantly_smaller": any(r["upper_lt_zero"] for r in n3_rows),
        "supported": all(r["diff"] >= 0 for r in n3_rows),
    }
    # N4
    fla = pairs["ref|ref-fla"]
    vllm_flips = {k: pairs[k]["flips"] for k in VLLM_RESTARTS}
    lo_r, hi_r = min(vllm_flips.values()), max(vllm_flips.values())
    n4_rows = [diff_test(boot, arrays["ref|ref-fla"], arrays[k], "ref|ref-fla", k) for k in VLLM_RESTARTS]
    bf16_only = vllm_flips["bf16-srv|bf16-srv-r2"]
    tests["N4"] = {
        "criterion": ("supported if ref-fla flips a nonzero number of decisions against ref AND that count lies within "
                      "the range of the vLLM restart flip counts (the three measured vLLM restart pairs)"),
        "ref_fla_flips": fla["flips"], "ref_fla_flip_rate": fla["flip_rate"], "nonzero": fla["flips"] > 0,
        "vllm_restart_flips": vllm_flips, "vllm_restart_range": [lo_r, hi_r],
        "within_vllm_restart_range": bool(lo_r <= fla["flips"] <= hi_r),
        "bf16_only_vllm_restart_flips": bf16_only,
        "within_bf16_only_range": bool(fla["flips"] <= bf16_only),
        "differences_vs_each_vllm_restart": n4_rows,
        "supported": bool(fla["flips"] > 0 and lo_r <= fla["flips"] <= hi_r),
    }
    # N5
    g = pairs["gguf-bf16|gguf-bf16-r2"]
    tests["N5"] = {
        "criterion": "supported if every record of gguf-bf16 and gguf-bf16-r2 is bit-identical",
        "identical_basis": g["identical_basis"],
        "n_identical_records": g["n_identical_records"], "n_records": g["n_records"],
        "share_identical": g["share_identical"], "flips": g["flips"],
        "max_dprob": g["max_dprob"], "tvd_mean": g["tvd_mean"],
        "supported": bool(g["n_identical_records"] == g["n_records"]),
    }

    conditions = {}
    for c, cfg in cfgs.items():
        vals = sorted({v for s in cfg.engines.values() for v in s})
        conditions[c] = {"engine | engine_version | batch_size (distinct values on the unit)": vals,
                         "tasks_by_value": {v: sorted(t for t, s in cfg.engines.items() if v in s) for v in vals},
                         "result_lines_read": cfg.n_lines, "result_lines_outside_unit_ignored": cfg.n_outside_unit}
    return {
        "meta": {"model": model, "unit": f"first {unit_n} records of each of {len(unit.tasks)} public tasks",
                 "unit_n_per_task": unit_n, "tasks": unit.tasks, "n_records": unit.n_records,
                 "n_questions": unit.n_questions, "n_boot": n_boot, "seed": seed,
                 "bootstrap": "simple paired resampling of records, one shared index matrix; stratified-by-task variant "
                              "reported for the difference tests",
                 "configs": list(configs)},
        "conditions": conditions,
        "groups": groups, "pairs": pairs, "tests": tests,
    }


# ------------------------------------------------------------------------------ markdown

def _pct(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def _ci_pct(c: Sequence[float] | None, nd: int = 2) -> str:
    return "-" if c is None else f"[{100 * c[0]:.{nd}f}, {100 * c[1]:.{nd}f}]"


def _num(x: float | None, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


GROUP_TITLES = {
    "restart_vllm": "Restart of the same vLLM flags (new server instance)",
    "restart_llamacpp": "Restart of llama.cpp (new server instance)",
    "native_rerun_bs8": "Native rerun at batch 8 (rerun + batch-size change, NOT a pure restart)",
    "kernel": "Kernel selection (Gated DeltaNet: PyTorch fallback vs flash-linear-attention)",
    "engine_bf16": "Engine change at BF16",
    "quant_within_engine": "Quantization inside one engine",
    "vs_ref": "Every config against ref (for completeness)",
}


def render_md(res: dict[str, Any]) -> str:
    m = res["meta"]
    L = [f"# Noise-floor decomposition: {m['model']}", "",
         f"Comparison unit: {m['unit']} ({m['n_records']} records, {m['n_questions']} questions). "
         f"Bootstrap: {m['n_boot']} paired resamples over records, seed {m['seed']}, percentile 95% CI; one index "
         "matrix is shared by every row, so differences are paired. Flip = top-1 option differs between the two configs "
         "(rates are shares of questions; one question per record on these tasks, so questions = records). TVD = "
         "mean total-variation distance of the option distributions. Bit-identical = all logits equal (logits present on "
         "both sides) or all stored probabilities equal (llama.cpp, which returns full-precision probabilities and no "
         "logits). max |dlogit| raw includes any constant shift of the logit scale, which softmax ignores (it dominates "
         "NVFP4 pairs); the centred value removes it. Exact = Clopper-Pearson interval (questions treated as "
         "independent), shown because the percentile bootstrap collapses to a point at 0 flips.", ""]
    for group, keys in res["groups"].items():
        L += [f"## {GROUP_TITLES.get(group, group)}", ""]
        rows = []
        for k in keys:
            p = res["pairs"][k]
            rows.append([f"{p['a']} vs {p['b']}", p["label"] if group != "vs_ref" else "", str(p["n_questions"]),
                         f"{p['flips']}", f"{_pct(p['flip_rate'])} {_ci_pct(p['flip_ci95'])}",
                         _ci_pct(p["flip_ci95_exact_cp"]),
                         f"{_num(p['tvd_mean'], 5)} [{_num(p['tvd_ci95'][0], 5)}, {_num(p['tvd_ci95'][1], 5)}]",
                         _num(p["max_dprob"], 4), _num(p["max_dlogit_raw"], 3), _num(p["max_dlogit_centered"], 3),
                         f"{p['n_identical_records']} ({_pct(p['share_identical'], 1)}) [{p['identical_basis']}]"])
        hdr = ["pair", "what", "n", "flips", "flip rate [95% CI]", "flip rate exact 95%", "mean TVD [95% CI]",
               "max dprob", "max dlogit raw", "max dlogit centred", "bit-identical records (share) [basis]"]
        if group == "vs_ref":
            hdr.remove("what")
            rows = [r[:1] + r[2:] for r in rows]
        L += [md_table(hdr, rows), ""]
    t = res["tests"]
    L += ["## Pre-registered tests (PLAN.md, N1-N5)", "",
          "Each row is a paired-bootstrap interval of a DIFFERENCE of flip rates on the same records "
          "(percentage points). `strat` = same test with records resampled inside each task. Outcomes are reported as "
          "found.", ""]
    n1 = t["N1"]
    L += ["### N1 FP8 inside restart noise", "", f"Criterion: {n1['criterion']}.", "",
          md_table(["difference", "flips", "diff pp [95% CI]", "strat diff pp [95% CI]", "excludes 0"],
                   [[f"flip({n1['minuend']}) - flip({n1['subtrahend']})",
                     f"{n1['flips_minuend']} vs {n1['flips_subtrahend']}",
                     f"{100 * n1['diff']:+.3f} {_ci_pct(n1['ci95'], 3)}", _ci_pct(n1["ci95_stratified"], 3),
                     "yes" if n1["excludes_zero"] else "no"]]),
          "", f"N1 supported: {n1['supported']}.", "",
          (lambda x: f"Supplementary, not pre-registered ({x['note']}): {x['flips_minuend']} vs {x['flips_subtrahend']} "
                     f"flips, diff {100 * x['diff']:+.2f} pp {_ci_pct(x['ci95'])}, excludes 0: {x['excludes_zero']}.")(
              n1["supplementary_not_preregistered"]), ""]
    n2 = t["N2"]
    L += ["### N2 NVFP4 and Q4_K_M exceed every non-quantization source", "", f"Criterion: {n2['criterion']}.", "",
          md_table(["quantization", "minus source", "flips", "diff pp [95% CI]", "strat [95% CI]",
                    f"Bonferroni (k={n2['family_size']}) [CI]", "lower > 0"],
                   [[r["minuend"], r["subtrahend"], f"{r['flips_minuend']} vs {r['flips_subtrahend']}",
                     f"{100 * r['diff']:+.2f} {_ci_pct(r['ci95'])}", _ci_pct(r["ci95_stratified"]),
                     _ci_pct(r["ci95_bonferroni"]), "yes" if r["lower_gt_zero"] else "no"] for r in n2["rows"]]),
          "", "Bonferroni intervals use the alpha/k tails of the same 2000 resamples, so the extreme tail rests on about three "
              "resamples; read them as approximate.", "",
          f"Every CI excludes 0 (lower bound > 0): {n2['all_lower_gt_zero']} "
              f"(stratified: {n2['all_lower_gt_zero_stratified']}, Bonferroni: {n2['all_lower_gt_zero_bonferroni']}). "
              f"N2 supported: {n2['supported']}.", ""]
    n3 = t["N3"]
    L += ["### N3 engine change vs 8-bit quantization", "", f"Criterion: {n3['criterion']}.", "",
          md_table(["engine change", "minus 8-bit within engine", "flips", "diff pp [95% CI]", "strat [95% CI]",
                    "CI entirely below 0"],
                   [[r["minuend"], r["subtrahend"], f"{r['flips_minuend']} vs {r['flips_subtrahend']}",
                     f"{100 * r['diff']:+.2f} {_ci_pct(r['ci95'])}", _ci_pct(r["ci95_stratified"]),
                     "yes" if r["upper_lt_zero"] else "no"] for r in n3["rows"]]),
          "", f"All four point estimates >= 0: {n3['all_point_ge_zero']}. Any engine change significantly smaller than an "
              f"8-bit effect: {n3['any_significantly_smaller']}. N3 supported: {n3['supported']}.", ""]
    n4 = t["N4"]
    L += ["### N4 kernel selection", "", f"Criterion: {n4['criterion']}.", "",
          f"ref vs ref-fla: {n4['ref_fla_flips']} flips of {m['n_questions']} "
          f"({_pct(n4['ref_fla_flip_rate'])}); nonzero: {n4['nonzero']}. vLLM restart flip counts: "
          + ", ".join(f"{k} = {v}" for k, v in n4["vllm_restart_flips"].items())
          + f" (range {n4['vllm_restart_range'][0]} to {n4['vllm_restart_range'][1]}). Within that range: "
            f"{n4['within_vllm_restart_range']}. Against the BF16 vLLM restart alone ({n4['bf16_only_vllm_restart_flips']} "
            f"flips) it is {'within' if n4['within_bf16_only_range'] else 'above'}.", "",
          md_table(["difference", "flips", "diff pp [95% CI]", "excludes 0"],
                   [[f"flip({r['minuend']}) - flip({r['subtrahend']})", f"{r['flips_minuend']} vs {r['flips_subtrahend']}",
                     f"{100 * r['diff']:+.2f} {_ci_pct(r['ci95'])}", "yes" if r["excludes_zero"] else "no"]
                    for r in n4["differences_vs_each_vllm_restart"]]),
          "", f"N4 supported: {n4['supported']}.", ""]
    h1 = t["H1_supplementary_not_preregistered"]
    L += ["### Supplementary for H1 (not pre-registered)", "",
          f"{h1['note']}: {h1['flips_minuend']} vs {h1['flips_subtrahend']} flips, diff {100 * h1['diff']:+.2f} pp "
          f"{_ci_pct(h1['ci95'])}, excludes 0: {h1['excludes_zero']}.", ""]
    n5 = t["N5"]
    L += ["### N5 llama.cpp BF16 restart", "", f"Criterion: {n5['criterion']}.", "",
          f"gguf-bf16 vs gguf-bf16-r2: {n5['n_identical_records']} of {n5['n_records']} records bit-identical "
          f"({_pct(n5['share_identical'], 2)}; basis: {n5['identical_basis']}), {n5['flips']} flips, "
          f"max |dprob| {n5['max_dprob']:.6g}, mean TVD {n5['tvd_mean']:.6g}. N5 supported: {n5['supported']}.", ""]
    L += ["## Run conditions of each config on the unit (from the result lines)", ""]
    rows = []
    for c, d in res["conditions"].items():
        for v, tasks in d["tasks_by_value"].items():
            rows.append([c, v, str(len(tasks)), str(d["result_lines_read"]), str(d["result_lines_outside_unit_ignored"])])
    L += [md_table(["config", "engine | engine_version | batch_size", "tasks", "result lines read", "lines outside the unit (ignored)"], rows), "",
          "The `-r2` repeats and `ref-fla` hold exactly the first 200 records of each task (2,400 lines, none outside the unit); "
          "full-set configs hold the whole task and only the first 200 are used.", ""]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=Path("data/records"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--model", default="clef-flash")
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    ap.add_argument("--unit-n", type=int, default=UNIT_N)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    res = decompose(args.records, args.results, args.model, args.unit_n, args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "decomposition.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (args.out / "decomposition.md").write_text(render_md(res), encoding="utf-8")
    print(f"wrote {args.out}/decomposition.json and decomposition.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
