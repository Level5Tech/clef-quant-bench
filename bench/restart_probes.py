"""Restart-repeat probes for the Clef (27B) vLLM formats: third instance per format.

EXPLORATORY, NOT PRE-REGISTERED. Nothing here was specified before the data existed.

For each vLLM format (fp8, nvfp4) the instances are [fmt, fmt-r2, fmt-r3]: servers started with identical flags,
scored on the same comparison unit as bench.decompose (first UNIT_N records of every public task). Present
instances only (at least two required). All statistics come from bench.decompose (build_unit, load_cfg,
pair_arrays, Boot, summarize_pair), so numbers are computed exactly as the decomposition computes them.

1. every pair of present instances (restart-to-restart decision flips and distribution drift);
2. every present instance against `ref` (the quantization effect a single run would report), with the min/max
   flip count across instances;
3. when all three instances exist: per record, whether r3's full logit vector equals base exactly, r2 exactly,
   both, or neither, plus the same split for base vs r2. "Equal" is decompose's bit-identical test: every
   question of the record has array_equal logits (probs when a config stores no logits).

Usage: python -m bench.restart_probes --records-dir data/records --results-dir results --model clef --out analysis_r3
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bench.decompose import (
    Boot, N_BOOT, SEED, UNIT_N, _ci_pct, _num, _pct, build_unit, load_cfg, md_table,
    pair_arrays, summarize_pair,
)

FORMATS = ("fp8", "nvfp4")
REF = "ref"
HEADER = ("Exploratory (not pre-registered). Third vLLM instance per format, 27B, "
          "same 2,400 records as the -r2 repeats.")


def instance_names(fmt: str) -> list[str]:
    return [fmt, f"{fmt}-r2", f"{fmt}-r3"]


def split4(ident_x: np.ndarray, ident_y: np.ndarray) -> dict[str, int]:
    """Counts of records equal to x only, y only, both, neither (inputs: per-record bool)."""
    return {"x_only": int((ident_x & ~ident_y).sum()), "y_only": int((~ident_x & ident_y).sum()),
            "both": int((ident_x & ident_y).sum()), "neither": int((~ident_x & ~ident_y).sum())}


def position_split(tasks: list[str], mask: np.ndarray) -> dict[str, int]:
    """Where the masked records sit inside their own task: first half, second half (odd task: middle goes second)."""
    seen: dict[str, int] = {}
    size = {t: tasks.count(t) for t in set(tasks)}
    first = second = 0
    for task, hit in zip(tasks, mask):
        rank = seen.get(task, 0)
        seen[task] = rank + 1
        if hit:
            if rank < size[task] // 2:
                first += 1
            else:
                second += 1
    return {"n": first + second, "first_half": first, "second_half": second}


def _with_shares(counts: dict[str, int], n: int) -> dict[str, Any]:
    return {k: {"n": v, "share": v / n} for k, v in counts.items()} | {"n_records": n}


def probe(records_dir: Path, results_dir: Path, model: str = "clef", unit_n: int = UNIT_N,
          n_boot: int = N_BOOT, seed: int = SEED, formats=FORMATS) -> dict[str, Any]:
    unit = build_unit(Path(records_dir), unit_n)
    boot = Boot(unit, n_boot, seed)
    ref = load_cfg(unit, Path(results_dir), model, REF)
    out: dict[str, Any] = {"model": model, "n_records": unit.n_records, "n_questions": unit.n_questions,
                           "exploratory_not_preregistered": True, "formats": {}}
    for fmt in formats:
        names = instance_names(fmt)
        present = [n for n in names if (Path(results_dir) / model / n).is_dir()]
        missing = [n for n in names if n not in present]
        if missing:
            print(f"{fmt}: missing instance dirs: {', '.join(missing)}", file=sys.stderr)
        if len(present) < 2:
            raise SystemExit(f"{fmt}: need at least 2 instances, found {present}")
        cfgs = {n: load_cfg(unit, Path(results_dir), model, n) for n in present}
        entry: dict[str, Any] = {"instances": present, "missing": missing, "pairs": {}, "vs_ref": {}}
        arrs = {}
        for a, b in itertools.combinations(present, 2):
            arr = pair_arrays(cfgs[a], cfgs[b])
            arrs[(a, b)] = arr
            entry["pairs"][f"{a}|{b}"] = summarize_pair(unit, boot, a, b, arr, f"{a} vs {b}", "restart_probes")
        for n in present:
            s = summarize_pair(unit, boot, REF, n, pair_arrays(ref, cfgs[n]), f"{n} vs ref", "restart_probes")
            entry["vs_ref"][n] = s
        counts = [s["flips"] for s in entry["vs_ref"].values()]
        entry["vs_ref_flips_min"], entry["vs_ref_flips_max"] = min(counts), max(counts)

        def rec_ident(a: str, b: str) -> np.ndarray:
            return boot.rec_sum(~pair_arrays(cfgs[a], cfgs[b]).identical) == 0

        base, r2, r3 = names
        n = unit.n_records
        # base vs r2 context: record identical to base-pair? 4-way is only defined for a third instance, so give
        # the 2-way split (identical / not) for the context pair.
        k = int(rec_ident(base, r2).sum()) if (base in cfgs and r2 in cfgs) else None
        disc = {"base_vs_r2_context": None if k is None else
                {"identical": {"n": k, "share": k / n}, "different": {"n": n - k, "share": (n - k) / n},
                 "n_records": n}}
        if len(present) == 3:
            e3b, e3r2 = rec_ident(r3, base), rec_ident(r3, r2)
            c = split4(e3b, e3r2)
            disc["r3_vs_base_and_r2"] = _with_shares(
                {"equals_base_only": c["x_only"], "equals_r2_only": c["y_only"],
                 "equals_both": c["both"], "equals_neither": c["neither"]}, n)
            # Records where the two later instances agree and the first differs, by position inside each task: if they
            # bunched up where the subsets end, the subset boundary would be the explanation.
            disc["first_instance_differs_by_position"] = position_split(unit.rec_task, e3r2 & ~e3b)
        entry["discreteness"] = disc
        out["formats"][fmt] = entry
    return out


def render_md(res: dict[str, Any]) -> str:
    L = [f"# Restart probes ({res['model']})", "", HEADER, "",
         f"Unit: {res['n_records']} records, {res['n_questions']} questions. Statistics are bench.decompose's "
         "(same bootstrap, same Clopper-Pearson, same bit-identical test).", ""]
    for fmt, e in res["formats"].items():
        L += [f"## {fmt}", "", f"Instances present: {', '.join(e['instances'])}"
              + (f" (missing: {', '.join(e['missing'])})" if e["missing"] else ""), "",
              "### Instance pairs", ""]
        rows = []
        for s in e["pairs"].values():
            rows.append([s["label"], str(s["n_records"]), str(s["flips"]), _pct(s["flip_rate"]),
                         _ci_pct(s["flip_ci95"]), _ci_pct(s["flip_ci95_exact_cp"]),
                         f"{s['n_identical_records']} ({_pct(s['share_identical'], 1)})",
                         f"{_num(s['tvd_mean'], 5)} {_ci_pct_num(s['tvd_ci95'])}",
                         _num(s["max_dprob"]), _num(s["max_dlogit_raw"])])
        L += [md_table(["pair", "records", "flips", "flip rate", "95% CI", "exact CP", "bit-identical",
                        "mean TVD [95% CI]", "max abs dprob", "max abs dlogit"], rows), "",
              "### Each instance against ref", ""]
        rows = [[n, str(s["flips"]), _pct(s["flip_rate"]), _ci_pct(s["flip_ci95"])] for n, s in e["vs_ref"].items()]
        L += [md_table(["instance", "flips", "flip rate", "95% CI"], rows), "",
              f"Flip count across instances: min {e['vs_ref_flips_min']}, max {e['vs_ref_flips_max']}.", "",
              "### Discreteness", ""]
        d = e["discreteness"]
        c = d["base_vs_r2_context"]
        if c:
            L += ["Context, base vs r2 (per record, all questions bit-identical):", "",
                  md_table(["split", "records", "share"],
                           [["identical", str(c["identical"]["n"]), _pct(c["identical"]["share"], 1)],
                            ["different", str(c["different"]["n"]), _pct(c["different"]["share"], 1)]]), ""]
        t = d.get("r3_vs_base_and_r2")
        if t:
            L += ["r3 against base and r2, per record:", "",
                  md_table(["r3 equals", "records", "share"],
                           [[k.replace("_", " "), str(t[k]["n"]), _pct(t[k]["share"], 1)]
                            for k in ("equals_base_only", "equals_r2_only", "equals_both", "equals_neither")]), ""]
            q = d["first_instance_differs_by_position"]
            L += [f"Records where r2 and r3 agree and base differs ({q['n']}), by position inside their task: "
                  f"{q['first_half']} in the first half, {q['second_half']} in the second.", ""]
        else:
            L += ["Third instance absent: 4-way split not computed.", ""]
    return "\n".join(L) + "\n"


def _ci_pct_num(c) -> str:
    return "-" if c is None else f"[{c[0]:.5f}, {c[1]:.5f}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records-dir", type=Path, default=Path("data/records"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--model", default="clef")
    ap.add_argument("--out", type=Path, default=Path("analysis_r3"))
    ap.add_argument("--unit-n", type=int, default=UNIT_N)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    res = probe(args.records_dir, args.results_dir, args.model, args.unit_n, args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "restart_probes.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (args.out / "restart_probes.md").write_text(render_md(res), encoding="utf-8")
    print(f"wrote {args.out}/restart_probes.json and restart_probes.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
