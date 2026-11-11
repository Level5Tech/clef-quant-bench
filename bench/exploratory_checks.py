"""Exploratory checks behind the paper's harm split, unit note, 8-bit separation and effect-to-floor ratios.

EXPLORATORY, NOT PRE-REGISTERED. Everything here was specified after the results existed, during review.

Public data only. Every quantized build is compared on top-1 decisions, and all statistics reuse bench.decompose
(build_unit, load_cfg, pair_arrays, Boot, diff_test), so the numbers match the decomposition exactly.

1. Harm split against `ref` (BF16, transformers): right->wrong ("worse"), wrong->right ("better"), wrong->different
   wrong ("neutral") and the net change in correct answers. On the full set and on the 2,400-record repeat unit.
2. Unit note: each build's flip rate against `ref` on the full set (pooled over questions), the same rate with every
   task weighted equally (how the repeat unit weights tasks), and the rate on the repeat unit.
3. Separation: every quantization effect inside one engine minus every non-quantization source, on the repeat unit,
   paired bootstrap as in N2 (no multiplicity correction).
4. Effect-to-floor ratios, derived from the counts: on the repeat unit, same-engine quantization flips over the largest
   non-quantization source; on the full set, flips against `ref` over the largest full-set source (engine changes
   and the batch-8 rerun, each against `ref`; restarts and the kernel switch ran on the repeat unit only).

Usage: python -m bench.exploratory_checks --records-dir data/records --results-dir results --out analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bench.decompose import (
    EIGHT_BIT, FOUR_BIT, N_BOOT, NON_QUANT_SOURCES, SEED, UNIT_N, Boot, build_unit, diff_test, load_cfg, md_table,
    pair_arrays,
)

MODELS = ("clef-flash", "clef")
QUANT = ("q8_0", "fp8", "q4_k_m", "nvfp4")
SAME_ENGINE = {"q8_0": "gguf-bf16|q8_0", "fp8": "bf16-srv|fp8", "q4_k_m": "gguf-bf16|q4_k_m", "nvfp4": "bf16-srv|nvfp4"}
FULL_SET_FLOORS = ("bf16-srv", "gguf-bf16", "ref-r2")  # each against ref, on the full set
HEADER = "Exploratory (not pre-registered). Public tasks only. Specified during review, after the results existed."


def _tasks(records_dir: Path) -> list[str]:
    return sorted(p.stem for p in records_dir.glob("*.jsonl"))


def _decisions(records_dir: Path, results_dir: Path, unit_n: int | None, cfgs: tuple[str, ...]):
    """-> per task: gold index array and {(model, cfg): top-1 index array}."""
    out = {}
    for t in _tasks(records_dir):
        u = build_unit(records_dir, unit_n, tasks=[t])
        gold = np.array([o.index(g) for g, o in zip(u.gold, u.opts)])
        preds = {(m, c): np.array([int(np.argmax(p)) for p in load_cfg(u, results_dir, m, c).p]) for m in MODELS for c in cfgs}
        out[t] = (gold, preds)
    return out


def harm_and_rates(records_dir: Path, results_dir: Path) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for label, n in (("full", None), ("unit", UNIT_N)):
        per = _decisions(records_dir, results_dir, n, ("ref", *QUANT, *(FULL_SET_FLOORS if n is None else ())))
        tasks = sorted(per)
        gold = np.concatenate([per[t][0] for t in tasks])
        cat = lambda m, c: np.concatenate([per[t][1][(m, c)] for t in tasks])  # noqa: E731
        block: dict[str, Any] = {"n_questions": int(len(gold))}
        for m in MODELS:
            rp = cat(m, "ref")
            rc = rp == gold
            for c in QUANT:
                qp = cat(m, c)
                qc = qp == gold
                flip = rp != qp
                block[f"{m}/{c}"] = {
                    "flips": int(flip.sum()), "flip_rate_pct": round(100 * float(flip.mean()), 2),
                    "worse_right_to_wrong": int((rc & ~qc).sum()), "better_wrong_to_right": int((~rc & qc).sum()),
                    "neutral_wrong_to_wrong": int((flip & ~rc & ~qc).sum()), "net": int(qc.sum() - rc.sum()),
                    "equal_task_weight_flip_rate_pct": round(
                        100 * float(np.mean([(per[t][1][(m, "ref")] != per[t][1][(m, c)]).mean() for t in tasks])), 2),
                }
            if n is None:
                block[f"{m}/floors"] = {c: int((rp != cat(m, c)).sum()) for c in FULL_SET_FLOORS}
        res[label] = block
    return res


def separation(records_dir: Path, results_dir: Path, n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    keys = (*NON_QUANT_SOURCES, *EIGHT_BIT, *FOUR_BIT)
    res: dict[str, Any] = {}
    unit = build_unit(records_dir, UNIT_N)
    for m in MODELS:
        cfgs = {c: load_cfg(unit, results_dir, m, c) for c in sorted({c for k in keys for c in k.split("|")})}
        boot = Boot(unit, n_boot, seed)
        arr = {k: pair_arrays(cfgs[k.split("|")[0]], cfgs[k.split("|")[1]]) for k in keys}
        rows = []
        for q in (*EIGHT_BIT, *FOUR_BIT):
            for s in NON_QUANT_SOURCES:
                r = diff_test(boot, arr[q], arr[s], q, s)
                rows.append({"quantization": q, "source": s, "flips_quantization": r["flips_minuend"],
                             "flips_source": r["flips_subtrahend"], "diff_pp": round(100 * r["diff"], 2),
                             "ci95_pp": [round(100 * x, 2) for x in r["ci95"]], "excludes_zero": r["lower_gt_zero"]})
        res[m] = {"rows": rows, "source_flips": {s: int(arr[s].flip.sum()) for s in NON_QUANT_SOURCES},
                  "quant_flips": {q: int(arr[q].flip.sum()) for q in (*EIGHT_BIT, *FOUR_BIT)}}
    return res


def ratios(rates: dict[str, Any], sep: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in MODELS:
        src = sep[m]["source_flips"]
        floor_key = max(src, key=src.get)
        out[f"unit/{m}"] = {"floor": src[floor_key], "floor_source": floor_key,
                            **{c: round(sep[m]["quant_flips"][SAME_ENGINE[c]] / src[floor_key], 2) for c in QUANT}}
        fl = rates["full"][f"{m}/floors"]
        fk = max(fl, key=fl.get)
        out[f"full/{m}"] = {"floor": fl[fk], "floor_source": f"ref|{fk}",
                            **{c: round(rates["full"][f"{m}/{c}"]["flips"] / fl[fk], 2) for c in QUANT}}
    return out


def render_md(res: dict[str, Any]) -> str:
    lines = ["# Exploratory checks", "", HEADER, "", "## Harm split against `ref`", ""]
    for label, title in (("full", "Full set"), ("unit", "2,400-record repeat unit")):
        lines += [f"{title}, {res['rates'][label]['n_questions']} questions.", ""]
        rows = [[k.split("/")[0], k.split("/")[1], str(v["flips"]), str(v["worse_right_to_wrong"]), str(v["better_wrong_to_right"]),
                 str(v["neutral_wrong_to_wrong"]), str(v["net"])]
                for k, v in res["rates"][label].items() if isinstance(v, dict) and "flips" in v]
        lines += [md_table(["model", "build", "flips", "worse", "better", "wrong to wrong", "net"], rows), ""]
    lines += ["## Unit note: flip rate against `ref`, %", ""]
    rows = []
    for m in MODELS:
        for c in QUANT:
            f, u = res["rates"]["full"][f"{m}/{c}"], res["rates"]["unit"][f"{m}/{c}"]
            rows.append([m, c, str(f["flip_rate_pct"]), str(f["equal_task_weight_flip_rate_pct"]), str(u["flip_rate_pct"]),
                         f"{f['equal_task_weight_flip_rate_pct'] / f['flip_rate_pct']:.2f}", f"{u['flip_rate_pct'] / f['flip_rate_pct']:.2f}"])
    lines += [md_table(["model", "build", "full set, pooled", "full set, tasks equal", "repeat unit", "equal / pooled",
                        "unit / pooled"], rows), ""]
    lines += ["## Separation on the repeat unit: quantization minus non-quantization source, pp [95% CI]", ""]
    for m in MODELS:
        rows = [[r["quantization"], r["source"], f"{r['flips_quantization']} vs {r['flips_source']}",
                 f"{r['diff_pp']:+.2f} [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]", "yes" if r["excludes_zero"] else "NO"]
                for r in res["separation"][m]["rows"]]
        lines += [f"**{m}**", "", md_table(["quantization", "source", "flips", "difference", "excludes 0"], rows), ""]
    lines += ["## Effect-to-floor ratios", ""]
    rows = [[k.split("/")[0], k.split("/")[1], f"{v['floor']} ({v['floor_source']})", *(f"{v[c]:.2f}" for c in QUANT)]
            for k, v in res["ratios"].items()]
    lines += [md_table(["unit", "model", "floor used", *QUANT], rows), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records-dir", type=Path, default=Path("data/records"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args(argv)
    rates = harm_and_rates(args.records_dir, args.results_dir)
    sep = separation(args.records_dir, args.results_dir)
    res = {"scope": HEADER, "rates": rates, "separation": sep, "ratios": ratios(rates, sep)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "exploratory_checks.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (args.out / "exploratory_checks.md").write_text(render_md(res), encoding="utf-8")
    print(f"wrote {args.out}/exploratory_checks.json and exploratory_checks.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
