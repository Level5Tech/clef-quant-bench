#!/usr/bin/env python
"""Build analysis/findings.md: every pre-registered hypothesis (H1-H11) and report test (N1-N5) of PLAN.md, verbatim,
with the numbers that bear on it, a status, and a source for every number.

Nothing is typed in by hand except the status RULES (written out next to each status) and the prose around them.
Every number is read from one of the JSON files below and printed with a citation `[ALIAS key.path]` naming the file and
the key path it was read from; sentences that quote numbers are built from the data, not typed. The verbatim hypothesis
text is parsed out of PLAN.md.

Status vocabulary: supported | not supported | mixed | not yet evaluable | not evaluable. A hypothesis with several clauses is
`supported` only when every clause was evaluated and held; when only some clauses can be evaluated yet the status is
`not yet evaluable` and the evaluated clause is reported beside it. `not evaluable` means the pre-registered comparison
cannot be formed from the configurations this sweep built; unlike `not yet evaluable`, no further run of this sweep changes
that. Operational readings of vague words ("small fraction", "near") are stated in the rule and are this report's reading,
not the pre-registration's.

Aggregates only for private sets. Usage:
  python -m bench.findings --root . --out analysis/findings.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):  # allow `python bench/findings.py`
    sys.path.append(str(Path(__file__).resolve().parent.parent))

ALIASES = {
    "PUB": "analysis/flash_public/summary.json",
    "PUBALL": "analysis/all_public/summary.json",
    "PRIVA": "analysis/flash_private/private_summary.json",
    "DEC": "analysis/decomposition.json",
    "HYP": "analysis/hypotheses.json",
    "PRIV": "analysis/private_aggregates.json",
    "SPD": "analysis/speed.json",
    "MAN": "data/manifest.json",
}
STATUSES = ("supported", "not supported", "mixed", "not yet evaluable", "not evaluable")
FL = "models.clef-flash"
SETS = ("r1", "r2", "r4a", "r4b")
LAB = {"nvfp4": "NVFP4", "q4_k_m": "Q4_K_M", "fp8": "FP8", "q8_0": "Q8_0"}
MODEL_LAB = {"clef-flash": "Clef-flash", "clef": "Clef (27B)"}


class Facts:
    """Reads values out of the analysis JSON files and renders them with a citation."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[str, Any] = {}
        self.used: set[tuple[str, str]] = set()

    def load(self, alias: str) -> Any:
        if alias not in self._cache:
            self._cache[alias] = json.loads((self.root / ALIASES[alias]).read_text(encoding="utf-8"))
        return self._cache[alias]

    def v(self, alias: str, path: str) -> Any:
        """Walk `path`; keys may contain dots (e.g. the bucket '[0.5,0.7)'), so the longest-first join is tried per key."""
        obj = self.load(alias)
        toks = path.split(".")
        i = 0
        while i < len(toks):
            if isinstance(obj, list):
                obj = obj[int(toks[i])]
                i += 1
                continue
            for j in range(i + 1, len(toks) + 1):
                k = ".".join(toks[i:j])
                if k in obj:
                    obj, i = obj[k], j
                    break
            else:
                raise KeyError(f"{alias}: no key {toks[i]!r} in path {path!r}")
        self.used.add((alias, path))
        return obj

    @staticmethod
    def c(alias: str, path: str) -> str:
        return f"[{alias} {path}]"

    # value-only helpers (the caller puts one citation after a sentence) -------------------------
    def p(self, alias: str, path: str, nd: int = 2) -> str:
        return f"{100 * self.v(alias, path):.{nd}f}%"

    # value + citation renderers ------------------------------------------------------------------
    def pct(self, alias: str, path: str, nd: int = 2) -> str:
        return f"{self.p(alias, path, nd)} {self.c(alias, path)}"

    def pp_ci(self, alias: str, base: str, nd: int = 2, delta_key: str = "delta") -> str:
        d = self.v(alias, f"{base}.{delta_key}")
        lo, hi = self.v(alias, f"{base}.ci95")
        return f"{100 * d:+.{nd}f} pp [{100 * lo:+.{nd}f}, {100 * hi:+.{nd}f}] [{alias} {base}.{delta_key},ci95]"

    def raw_ci(self, alias: str, base: str, nd: int = 5) -> str:
        d = self.v(alias, f"{base}.delta")
        lo, hi = self.v(alias, f"{base}.ci95")
        return f"{d:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}] [{alias} {base}.delta,ci95]"

    def rate_ci(self, alias: str, rate_path: str, ci_path: str, nd: int = 2) -> str:
        r = self.v(alias, rate_path)
        lo, hi = self.v(alias, ci_path)
        return f"{100 * r:.{nd}f}% [{100 * lo:.{nd}f}, {100 * hi:.{nd}f}] [{alias} {rate_path},{ci_path.rsplit('.', 1)[-1]}]"

    def num(self, alias: str, path: str, nd: int = 4) -> str:
        return f"{self.v(alias, path):.{nd}f} {self.c(alias, path)}"

    def int_(self, alias: str, path: str) -> str:
        return f"{self.v(alias, path):,} {self.c(alias, path)}"


# ------------------------------------------------------------------------ PLAN.md parsing

def parse_plan(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Verbatim hypothesis texts H1..H11 and N1..N5 from PLAN.md."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## Pre-registered hypotheses"))
    para: list[str] = []
    for l in lines[start + 1:]:
        if l.startswith("#"):
            break
        if l.strip():
            para.append(l.strip())
    blob = " ".join(para)
    parts = re.split(r"(?:(?:Less )?[Oo]bvious: )?\bH(\d+) ", blob)
    hyp = {f"H{parts[k]}": f"H{parts[k]} " + parts[k + 1].strip() for k in range(1, len(parts) - 1, 2)}
    start = next(i for i, l in enumerate(lines) if l.startswith("Pre-registered for the report"))
    cur: str | None = None
    buf: dict[str, list[str]] = {}
    for l in lines[start + 1:]:
        m = re.match(r"^- (N\d) (.*)$", l)
        if m:
            cur = m.group(1)
            buf[cur] = [m.group(2).strip()]
        elif cur and l.startswith("  ") and l.strip():
            buf[cur].append(l.strip())
        else:
            break
    return hyp, {k: f"{k} " + " ".join(v) for k, v in buf.items()}


# --------------------------------------------------------------------------- builders

class Report:
    def __init__(self) -> None:
        self.status: dict[str, tuple[str, str]] = {}
        self.sections: dict[str, str] = {}

    def entry(self, key: str, verbatim: str, bullets: Sequence[str], status: str, rule: str, why: str) -> None:
        assert status in STATUSES, status
        self.status[key] = (status, why)
        L = [f"### {key}", "", f"> {verbatim}", ""]
        L += [f"- {b}" for b in bullets]
        L += ["", f"**Status: {status}.** {why}", "", f"Rule: {rule}", ""]
        self.sections[key] = "\n".join(L)


def build_report(root: Path) -> str:
    F = Facts(root)
    hyp, nulls = parse_plan((Path(root) / "PLAN.md").read_text(encoding="utf-8"))
    assert sorted(hyp, key=lambda k: int(k[1:])) == [f"H{i}" for i in range(1, 12)], sorted(hyp)
    assert sorted(nulls) == [f"N{i}" for i in range(1, 6)], sorted(nulls)
    R = Report()
    cfgs8, cfgs4 = ("fp8", "q8_0"), ("nvfp4", "q4_k_m")
    for i, t in ((3, 0.65), (4, 0.7), (6, 0.8)):   # cascade rows are addressed by index below
        assert F.v("PUB", f"meta.cascade_thresholds.{i}") == t
    for i, t in ((0, 0.5), (2, 0.6)):
        assert F.v("PRIVA", f"meta.cascade_thresholds.{i}") == t

    def pooled(c: str, k: str) -> str:
        return f"{FL}.configs.{c}.pooled.{k}"

    # ----------------------------------------------------------------------------- H1
    b = []
    for c in cfgs8:
        base = pooled(c, "vs_ref")
        b.append(f"{c} against ref, pooled over 12 tasks (31,275 questions): accuracy change {F.pp_ci('PUB', base + '.delta_accuracy')}; "
                 f"top-1 agreement {F.pct('PUB', base + '.agreement')} ({F.int_('PUB', base + '.n_flips')} flips).")
    nf = f"{FL}.noise_floor.ref.ref-r2.pooled"
    b.append(f"Measured full-set floor, ref vs ref-r2 (rerun at batch 8): agreement {F.pct('PUB', nf + '.agreement')}, "
             f"{F.int_('PUB', nf + '.n_flips')} flips.")
    b.append(f"Same records, first 200 per task (n = 2,400): FP8 vs bf16-srv flips "
             f"{F.int_('DEC', 'pairs.bf16-srv|fp8.flips')} ({F.rate_ci('DEC', 'pairs.bf16-srv|fp8.flip_rate', 'pairs.bf16-srv|fp8.flip_ci95')}); "
             f"vLLM BF16 restart {F.int_('DEC', 'pairs.bf16-srv|bf16-srv-r2.flips')} flips, FP8 restart "
             f"{F.int_('DEC', 'pairs.fp8|fp8-r2.flips')} flips.")
    h1s = "tests.H1_supplementary_not_preregistered"
    b.append(f"Q8_0 vs gguf-bf16 flips {F.int_('DEC', 'pairs.gguf-bf16|q8_0.flips')} "
             f"({F.rate_ci('DEC', 'pairs.gguf-bf16|q8_0.flip_rate', 'pairs.gguf-bf16|q8_0.flip_ci95')}) against the llama.cpp "
             f"restart's {F.int_('DEC', 'pairs.gguf-bf16|gguf-bf16-r2.flips')}; difference {F.pp_ci('DEC', h1s, delta_key='diff')} (not pre-registered).")
    for pair in ("bf16-srv|fp8", "gguf-bf16|q8_0"):
        base = f"calibration.within_engine.{pair}"
        b.append(f"{pair.split('|')[1]} against its own engine's BF16, full set: accuracy {F.pp_ci('HYP', base + '.accuracy')}, "
                 f"ECE {F.raw_ci('HYP', base + '.ece')}, Brier {F.raw_ci('HYP', base + '.brier')}.")
    fp8_acc, q8_acc = (F.v("PUB", pooled(c, "vs_ref") + ".delta_accuracy.ci95") for c in cfgs8)
    fp8_ok = fp8_acc[0] <= 0 <= fp8_acc[1] and not F.v("DEC", "tests.N1.lower_gt_zero")
    q8_ok = q8_acc[0] <= 0 <= q8_acc[1] and not F.v("DEC", h1s + ".lower_gt_zero")
    status = "supported" if (fp8_ok and q8_ok) else ("mixed" if (fp8_ok or q8_ok) else "not supported")
    R.entry("H1", hyp["H1"], b, status,
            "an 8-bit config is indistinguishable from BF16 only if its full-set accuracy-change CI contains 0 AND the CI of its flip rate "
            "minus the matching restart floor (200-record unit) contains 0; supported if both 8-bit configs pass, mixed if one does, not "
            "supported if neither does.",
            f"FP8: accuracy CI {'contains' if fp8_acc[0] <= 0 <= fp8_acc[1] else 'excludes'} 0 and flips exceed the restart floor "
            f"({F.v('DEC', 'pairs.bf16-srv|fp8.flips')} vs {F.v('DEC', 'pairs.bf16-srv|bf16-srv-r2.flips')}). Q8_0: accuracy CI "
            f"{'contains' if q8_acc[0] <= 0 <= q8_acc[1] else 'excludes'} 0, but it flips {F.p('DEC', 'pairs.gguf-bf16|q8_0.flip_rate')} of decisions "
            f"against a llama.cpp restart floor of {F.p('DEC', 'pairs.gguf-bf16|gguf-bf16-r2.flip_rate')}. Neither is indistinguishable within "
            "the measured floor." if status == "not supported" else f"FP8 passes: {fp8_ok}; Q8_0 passes: {q8_ok}.")

    # ----------------------------------------------------------------------------- H2
    b = []
    for c in cfgs4:
        base = pooled(c, "vs_ref")
        b.append(f"{c} against ref (Clef-flash, 31,275 questions): accuracy change {F.pp_ci('PUB', base + '.delta_accuracy')}; "
                 f"{F.int_('PUB', base + '.n_flips')} top-1 flips ({100 * (1 - F.v('PUB', base + '.agreement')):.2f}% of questions) "
                 f"{F.c('PUB', base + '.agreement')}.")
    pall = lambda m, c, k: f"models.{m}.configs.{c}.pooled.vs_ref.{k}"  # noqa: E731
    for c in cfgs4:
        big, small = pall("clef", c, ""), pall("clef-flash", c, "")
        b.append(f"{c} against ref (Clef (27B), {F.v('PUBALL', big + 'n_paired'):,} questions, analyzer over both models): accuracy change "
                 f"{F.pp_ci('PUBALL', big + 'delta_accuracy')}; {F.int_('PUBALL', big + 'n_flips')} top-1 flips "
                 f"({100 * (1 - F.v('PUBALL', big + 'agreement')):.2f}% of questions) {F.c('PUBALL', big + 'agreement')}; Clef-flash, same build: "
                 f"{F.int_('PUBALL', small + 'n_flips')} flips ({100 * (1 - F.v('PUBALL', small + 'agreement')):.2f}%) {F.c('PUBALL', small + 'agreement')}.")
    n_big = sum(F.v("PRIV", f"label_free_big.{s}.pooled.n_questions") for s in SETS)
    n_flash = sum(F.v("PRIV", f"label_free.{s}.pooled.n_questions") for s in SETS)
    for c in cfgs4:
        f_big = sum(F.v("PRIV", f"label_free_big.{s}.pooled.pairs.ref|{c}.flips") for s in SETS)
        f_flash = sum(F.v("PRIV", f"label_free.{s}.pooled.pairs.ref|{c}.flips") for s in SETS)
        b.append(f"Private sets, pooled over R1, R2, R4a and R4b, {c}, flips against each model's own ref: Clef (27B) {f_big:,} of {n_big:,} "
                 f"({100 * f_big / n_big:.2f}%; sum over [PRIV label_free_big.<set>.pooled.pairs.ref|{c}.flips]) vs Clef-flash {f_flash:,} of "
                 f"{n_flash:,} ({100 * f_flash / n_flash:.2f}%; sum over [PRIV label_free.<set>.pooled.pairs.ref|{c}.flips]).")
    first = {(m, c): F.v("PUBALL", pall(m, c, "delta_accuracy.ci95"))[1] < 0 for m in ("clef-flash", "clef") for c in cfgs4}
    second = {c: F.v("PUBALL", pall("clef", c, "n_flips")) < F.v("PUBALL", pall("clef-flash", c, "n_flips")) for c in cfgs4}
    status = "supported" if (all(first.values()) and all(second.values())) else (
        "not supported" if (not any(second.values()) or not any(first.values())) else "mixed")
    held = lambda ok, bad: (f"holds for {', '.join(ok)}" if ok else "holds for no build") + (f"; fails for {', '.join(bad)}" if bad else "")  # noqa: E731
    f_ok = [f"{MODEL_LAB[m]} {LAB[c]}" for (m, c), ok in first.items() if ok]
    f_bad = [f"{MODEL_LAB[m]} {LAB[c]} (CI upper bound {100 * F.v('PUBALL', pall(m, c, 'delta_accuracy.ci95'))[1]:+.2f} pp)"
             for (m, c), ok in first.items() if not ok]
    s_ok = [f"{LAB[c]} ({F.v('PUBALL', pall('clef', c, 'n_flips')):,} vs {F.v('PUBALL', pall('clef-flash', c, 'n_flips')):,})" for c, ok in second.items() if ok]
    s_bad = [f"{LAB[c]} ({F.v('PUBALL', pall('clef', c, 'n_flips')):,} vs {F.v('PUBALL', pall('clef-flash', c, 'n_flips')):,})" for c, ok in second.items() if not ok]
    R.entry("H2", hyp["H2"], b, status,
            "supported only if the first clause holds and the 27B 4-bit flip count on the same records is smaller than Clef-flash's."
            " Reading added 2026-10-23, after the 27B results were seen: the first clause is checked for each 4-bit build of both models "
            "(accuracy-change CI entirely below 0, each model against its own BF16 ref, full public set); the second compares, per 4-bit build, "
            "the two models' top-1 flips against their own ref on the same 31,275 public questions. Supported if every check holds, not "
            "supported if no 4-bit build has fewer 27B flips or no build loses accuracy, mixed otherwise. Private flips are reported beside "
            "the rule and are not part of it.",
            f"First clause (4-bit loses accuracy; CI entirely below 0 against the model's own ref): {held(f_ok, f_bad)}. Second clause "
            f"(fewer top-1 flips on Clef (27B) than on Clef-flash; counts are 27B vs Clef-flash): {held(s_ok, s_bad)}. "
            + {"supported": "Every check holds.",
               "not supported": "No 4-bit build has fewer 27B flips, or no build loses accuracy.",
               "mixed": "Some checks hold and some fail, so the hypothesis holds for some builds and not for others."}[status])

    # ----------------------------------------------------------------------------- H3
    b = []
    for c in ("nvfp4", "q4_k_m", "fp8", "q8_0"):
        base = f"confidence.per_config.{c}"
        b.append(f"{c}: {F.pct('HYP', base + '.share_of_flips_with_ref_conf_below_0_7', 1)} of its {F.int_('HYP', base + '.flips')} flips "
                 f"have ref confidence below 0.7, where only {F.pct('HYP', base + '.share_of_items_with_ref_conf_below_0_7', 1)} of "
                 f"questions sit; flip rate below 0.7 is {F.pct('HYP', base + '.flip_rate_ref_conf_below_0_7', 1)} vs "
                 f"{F.pct('HYP', base + '.flip_rate_ref_conf_at_least_0_7', 2)} at or above it; highest ref confidence of any flip "
                 f"{F.num('HYP', base + '.max_ref_confidence_among_flips', 3)}.")
    fb = pooled("nvfp4", "vs_ref") + ".flip_by_ref_confidence"
    b.append(f"By bucket (pooled, NVFP4): flip rate {F.pct('PUB', fb + '.0.flip_rate', 1)} below 0.5, {F.pct('PUB', fb + '.1.flip_rate', 1)} in "
             f"[0.5, 0.7), {F.pct('PUB', fb + '.3.flip_rate', 2)} in [0.9, 0.99), {F.pct('PUB', fb + '.4.flip_rate', 1)} at or above 0.99 "
             f"(n = {F.v('PUB', fb + '.4.n')}).")
    shares = {c: F.v("HYP", f"confidence.per_config.{c}.share_of_flips_with_ref_conf_below_0_7") for c in ("nvfp4", "q4_k_m", "fp8", "q8_0")}
    R.entry("H3", hyp["H3"], b, "supported" if min(shares.values()) > 0.8 else "not supported",
            "supported if, for every quantized config, more than 80% of its flips fall at ref confidence below 0.7 (a bucket holding about 16% of questions).",
            f"{100 * min(shares.values()):.0f}% to {100 * max(shares.values()):.0f}% of flips sit below 0.7. It is not a clean 'never above X' rule for 4-bit: "
            f"NVFP4 still flips {F.v('HYP', 'confidence.per_config.nvfp4.flips_with_ref_conf_at_least_0_9')} of "
            f"{F.v('HYP', 'confidence.per_config.nvfp4.items_with_ref_conf_at_least_0_9'):,} questions with ref confidence at or above 0.9, while the "
            f"8-bit llama.cpp build's highest flipping confidence is {F.v('HYP', 'confidence.per_config.q8_0.max_ref_confidence_among_flips'):.2f} and FP8's "
            f"{F.v('HYP', 'confidence.per_config.fp8.max_ref_confidence_among_flips'):.2f}.")

    # ----------------------------------------------------------------------------- H4
    pairs4 = ("bf16-srv|fp8", "gguf-bf16|q8_0", "bf16-srv|nvfp4", "gguf-bf16|q4_k_m")
    b = []
    for pair in pairs4:
        base = f"calibration.within_engine.{pair}"
        b.append(f"{pair} (quantized vs BF16 in the same engine, full set): accuracy {F.pp_ci('HYP', base + '.accuracy')}; "
                 f"ECE {F.raw_ci('HYP', base + '.ece')}; Brier {F.raw_ci('HYP', base + '.brier')}.")
    for c in ("bf16-srv", "gguf-bf16", "ref-r2"):
        base = f"calibration.per_config.{c}.delta_vs_ref"
        b.append(f"Non-quantization reference point, {c} against ref: accuracy {F.pp_ci('HYP', base + '.accuracy')}; ECE "
                 f"{F.raw_ci('HYP', base + '.ece')}; Brier {F.raw_ci('HYP', base + '.brier')}.")
    ev = {p: F.v("HYP", f"calibration.within_engine.{p}") for p in pairs4}
    floor_brier = max(F.v("HYP", f"calibration.per_config.{c}.delta_vs_ref.brier.delta") for c in ("bf16-srv", "gguf-bf16", "ref-r2"))
    before = []
    for p, d in ev.items():
        ece_sig = d["ece"]["excludes_zero"] and d["ece"]["delta"] > 0
        brier_excess = d["brier"]["excludes_zero"] and d["brier"]["delta"] > floor_brier
        if (ece_sig or brier_excess) and not d["accuracy"]["excludes_zero"]:
            before.append(p)
    R.entry("H4", hyp["H4"], b, "supported" if before else "not supported",
            "calibration degrades before accuracy if some quantized build (against its own engine's BF16) shows a significant ECE increase, or a "
            f"significant Brier increase larger than the largest non-quantization Brier change ({floor_brier:+.5f}), while its accuracy-change CI still "
            "contains 0.",
            ("Builds showing it: " + ", ".join(before) + ".") if before else
            "No build does. ECE does not move without accuracy moving (FP8's ECE rise comes with a significant accuracy loss; the other three ECE "
            f"intervals contain 0). Q8_0's Brier rise ({ev['gguf-bf16|q8_0']['brier']['delta']:+.5f}) is significant but no larger than the Brier change "
            f"from an engine change or a rerun alone ({floor_brier:+.5f}), so it is not attributed to quantization.")

    # ----------------------------------------------------------------------------- H5
    b = [f"CLINC150 (n = {F.v('HYP', 'oos.n'):,}, of which {F.v('HYP', 'oos.n_oos')} gold out-of-scope): reference out-of-scope recall "
         f"{F.pct('HYP', 'oos.per_config.ref.oos_recall')}, in-scope accuracy {F.pct('HYP', 'oos.per_config.ref.in_scope_acc')}."]
    oos_cfgs = ("fp8", "q8_0", "nvfp4", "q4_k_m")
    for c in oos_cfgs:
        base = f"oos.per_config.{c}.delta_vs_ref"
        b.append(f"{c}: out-of-scope recall {F.pp_ci('HYP', base + '.oos_recall')}; in-scope accuracy {F.pp_ci('HYP', base + '.in_scope_acc')}; "
                 f"difference of the two changes {F.pp_ci('HYP', base + '.oos_recall_minus_in_scope_acc')}; flip rate on out-of-scope items "
                 f"{F.pct('HYP', f'oos.per_config.{c}.flip_rate_oos_items', 2)} vs in-scope items {F.pct('HYP', f'oos.per_config.{c}.flip_rate_in_scope_items', 2)}.")
    sig_worse = [c for c in oos_cfgs if F.v("HYP", f"oos.per_config.{c}.delta_vs_ref.oos_recall_minus_in_scope_acc.ci95")[1] < 0]
    status = "supported" if len(sig_worse) == 4 else ("not supported" if not sig_worse else "mixed")
    ratio4 = {c: F.v("HYP", f"oos.per_config.{c}.flip_rate_oos_items") / max(F.v("HYP", f"oos.per_config.{c}.flip_rate_in_scope_items"), 1e-9)
              for c in cfgs4}
    R.entry("H5", hyp["H5"], b, status,
            "supported if, for every quantized config, (change in out-of-scope recall) minus (change in in-scope accuracy) has a 95% CI entirely below 0; "
            "not supported if none does; mixed otherwise.",
            (f"Only {', '.join(sig_worse)} shows out-of-scope detection degrading more than in-scope classification. Both 4-bit builds keep "
             "out-of-scope recall (changes are small and their CIs contain 0) even though out-of-scope items flip "
             f"{min(ratio4.values()):.0f}-{max(ratio4.values()):.0f} times as often as in-scope items; the flips go both ways.")
            if status == "mixed" else f"Builds with a significantly larger out-of-scope loss: {', '.join(sig_worse) or 'none'}.")

    # ----------------------------------------------------------------------------- H6
    b = []
    for c in ("fp8", "q8_0", "nvfp4", "q4_k_m", "bf16-srv", "ref-r2"):
        base = f"length.per_config.{c}"
        b.append(f"Public, {c}: flip rate against ref Q1 {F.p('HYP', base + '.by_quartile.Q1.flip_rate')} -> Q4 {F.p('HYP', base + '.by_quartile.Q4.flip_rate')} "
                 f"(quartiles of input tokens inside each task); Q4 minus Q1 {F.pp_ci('HYP', base + '.q4_minus_q1')}.")
    cells: dict[str, tuple[int, int, int]] = {}
    for c in cfgs4:
        up = flat = down = 0
        for s in SETS:
            for q in F.v("PRIV", f"set_sizes.{s}.questions"):
                d = F.v("PRIV", f"label_free.{s}.{q}.pairs.ref|{c}.flip_q4_minus_q1")
                if d["delta"] is None:
                    continue
                lo, hi = d["ci95"]
                up, down, flat = up + (lo > 0), down + (hi < 0), flat + (lo <= 0 <= hi)
        cells[c] = (up, flat, down)
        b.append(f"Private sets, {c}: of {up + flat + down} (set, question) cells, Q4 flips significantly more than Q1 in {up}, significantly less in "
                 f"{down}, no significant difference in {flat} [PRIV label_free.<set>.<question>.pairs.ref|{c}.flip_q4_minus_q1].")
    n_priv = sum(F.v("PRIV", f"label_free.{s}.pooled.n_questions") for s in SETS)
    higher = {}
    for c in ("fp8", "q8_0", "nvfp4", "q4_k_m"):
        f_priv = sum(F.v("PRIV", f"label_free.{s}.pooled.pairs.ref|{c}.flips") for s in SETS)
        f_pub, n_pub = F.v("PUB", pooled(c, "vs_ref") + ".n_flips"), F.v("PUB", pooled(c, "vs_ref") + ".n_paired")
        higher[c] = f_priv / n_priv > f_pub / n_pub
        b.append(f"Flip rate against ref, {c}: private sets pooled {100 * f_priv / n_priv:.2f}% ({f_priv:,} of {n_priv:,}; sum over "
                 f"[PRIV label_free.<set>.pooled.pairs.ref|{c}.flips]) vs public pooled {100 * f_pub / n_pub:.2f}% ({f_pub:,} of {n_pub:,}) "
                 f"{F.c('PUB', pooled(c, 'vs_ref') + '.n_flips')}.")
    low_priv = sum(F.v("PRIV", f"label_free.{s}.pooled.pairs.ref|nvfp4.flip_by_ref_confidence.{k}.n") for s in SETS for k in ("[0,0.5)", "[0.5,0.7)"))
    b.append(f"Confidence mix: {100 * low_priv / n_priv:.1f}% of private questions have ref confidence below 0.7 against "
             f"{F.pct('HYP', 'confidence.per_config.nvfp4.share_of_items_with_ref_conf_below_0_7', 1)} of public questions, and H3 puts the flips there "
             "[PRIV label_free.<set>.pooled.pairs.ref|nvfp4.flip_by_ref_confidence].")
    six = ("fp8", "q8_0", "nvfp4", "q4_k_m", "bf16-srv", "ref-r2")
    grow = [c for c in six if F.v("HYP", f"length.per_config.{c}.q4_minus_q1.ci95")[0] > 0]
    no_grow = [c for c in six if c not in grow]
    public_grow = all(c in grow for c in cfgs4)
    private_grow = all(cells[c][0] > cells[c][1] + cells[c][2] for c in cfgs4)
    status = "supported" if (public_grow and private_grow and all(higher.values())) else (
        "not supported" if (not public_grow and not any(higher.values())) else "mixed")
    R.entry("H6", hyp["H6"], b, status,
            "clause 1 (drift grows with input length): Q4 minus Q1 flip-rate CI above 0 for both 4-bit builds on the public sets AND in most private "
            "(set, question) cells; clause 2 (real-world data drifts more): private pooled flip rate above the public pooled flip rate for all four "
            "quantized builds. supported if both hold, not supported if neither, mixed otherwise.",
            f"Clause 1 (CI of Q4 minus Q1 above 0) holds on the public sets for {', '.join(grow)} and not for {', '.join(no_grow)}; in the private sets only "
            f"{cells['nvfp4'][0]} of {sum(cells['nvfp4'])} (NVFP4) and {cells['q4_k_m'][0]} of {sum(cells['q4_k_m'])} (Q4_K_M) cells show it. Clause 2 "
            f"holds on pooled rates for {sum(higher.values())} of 4 builds; the private sets also hold far more low-confidence questions "
            f"({100 * low_priv / n_priv:.0f}% vs {100 * F.v('HYP', 'confidence.per_config.nvfp4.share_of_items_with_ref_conf_below_0_7'):.0f}%), where flips concentrate (H3), "
            "while length alone does not explain the private gap."
            if status == "mixed" else f"Clause 1 public: {public_grow}, private: {private_grow}; clause 2: {higher}.")

    # ----------------------------------------------------------------------------- H7
    b = []
    for c in ("bf16-srv", "gguf-bf16", "fp8", "q8_0"):
        base = pooled(c, "vs_ref")
        b.append(f"Full set, {c} against ref: {F.int_('PUB', base + '.n_flips')} flips ({100 * (1 - F.v('PUB', base + '.agreement')):.2f}%).")
    for row in range(4):
        r = f"tests.N3.rows.{row}"
        b.append(f"200-record unit, flip({F.v('DEC', r + '.minuend')}) minus flip({F.v('DEC', r + '.subtrahend')}): {F.v('DEC', r + '.flips_minuend')} vs "
                 f"{F.v('DEC', r + '.flips_subtrahend')} flips, {F.pp_ci('DEC', r, delta_key='diff')}.")
    n3 = F.v("DEC", "tests.N3.supported")
    rate = lambda c: 100 * (1 - F.v("PUB", pooled(c, "vs_ref") + ".agreement"))  # noqa: E731
    R.entry("H7", hyp["H7"], b, "supported" if n3 else "not supported",
            "supported if each engine change at BF16 (transformers -> vLLM, transformers -> llama.cpp) flips at least as many decisions as each 8-bit "
            "quantization inside one engine (this is N3).",
            f"Against ref, an engine change at BF16 flips {rate('bf16-srv'):.2f}% (vLLM) and {rate('gguf-bf16'):.2f}% (llama.cpp) of decisions; FP8 flips "
            f"{rate('fp8'):.2f}% and Q8_0 {rate('q8_0'):.2f}%. On the 200-record unit every engine-minus-8-bit difference is negative with a CI below 0."
            if not n3 else "Engine changes flip at least as many decisions as 8-bit quantization.")

    # ----------------------------------------------------------------------------- H8
    tol = F.v("PUBALL", "equal_memory.tolerance")
    gb = lambda m, c: F.v("SPD", f"configs.{m}.{c}.meta.file_size_gb")  # noqa: E731
    ratio = {(c4, c8): max(gb("clef", c4), gb("clef-flash", c8)) / min(gb("clef", c4), gb("clef-flash", c8)) for c4 in cfgs4 for c8 in cfgs8}
    in_window = [k for k in ratio if max(gb("clef", k[0]), gb("clef-flash", k[1])) <= (1 + tol) * min(gb("clef", k[0]), gb("clef-flash", k[1]))]
    ep = F.v("PUBALL", "equal_memory.pairs")
    cross_in_ep = [i for i, p in enumerate(ep) for x, y in ((p["a"], p["b"]), (p["b"], p["a"]))
                   if x["model"] == "clef" and x["config"] in cfgs4 and y["model"] == "clef-flash" and y["config"] in cfgs8]
    if in_window or cross_in_ep:
        raise ValueError(f"H8: a 27B 4-bit / Clef-flash 8-bit pair lies within the {100 * tol:.0f}% size window (from file sizes: "
                         f"{in_window}; in PUBALL equal_memory.pairs: {cross_in_ep}); the status rule for that case is not written")
    closest = min(ratio, key=lambda k: ratio[k])
    b = ["File sizes: " + "; ".join(f"Clef (27B) {LAB[c]} {F.num('SPD', f'configs.clef.{c}.meta.file_size_gb', 2)} GB" for c in cfgs4) + "; "
         + "; ".join(f"Clef-flash {LAB[c]} {F.num('SPD', f'configs.clef-flash.{c}.meta.file_size_gb', 2)} GB" for c in cfgs8)
         + ". Size ratios, larger over smaller as the analyzer defines them: "
         + "; ".join(f"{LAB[c4]} / {LAB[c8]} {ratio[(c4, c8)]:.2f}" for c4 in cfgs4 for c8 in cfgs8)
         + f". Window: {100 * tol:.0f}% {F.c('PUBALL', 'equal_memory.tolerance')} (two files are equal-memory when the larger is at most "
           f"{1 + tol:.2f} times the smaller).",
         f"Check: none of the {len(ratio)} pairs of a 27B 4-bit build with a Clef-flash 8-bit build is within the window (closest: "
         f"{LAB[closest[0]]} / {LAB[closest[1]]} at {ratio[closest]:.2f}), and PUBALL `equal_memory.pairs` ({len(ep)} pairs) contains no such pair "
         f"{F.c('PUBALL', 'equal_memory.pairs')}."]
    dif = "cross_size_4bit_vs_8bit.difficulty"
    for c4 in cfgs4:
        for c8 in cfgs8:
            base = f"cross_size_4bit_vs_8bit.pairs.clef/{c4}|clef-flash/{c8}"
            b.append(f"Supplementary, not pre-registered, outside the size window: Clef (27B) {LAB[c4]} vs Clef-flash {LAB[c8]}, size ratio "
                     f"{ratio[(c4, c8)]:.2f}, accuracy of the 27B minus Clef-flash on the same questions: pooled hard tasks "
                     f"{F.pp_ci('HYP', base + '.pooled_hard.delta_accuracy_big_minus_small')}, pooled easy tasks "
                     f"{F.pp_ci('HYP', base + '.pooled_easy.delta_accuracy_big_minus_small')}, hard minus easy {F.pp_ci('HYP', base + '.hard_minus_easy')}.")
    b.append(f"The hard/easy split behind those bullets ({F.v('HYP', dif + '.basis')}): hard = {', '.join(F.v('HYP', dif + '.hard'))}; easy = "
             f"{', '.join(F.v('HYP', dif + '.easy'))} {F.c('HYP', dif)}. The split was chosen after the 27B results were seen.")
    sens = "cross_size_4bit_vs_8bit.difficulty_sensitivity"
    alts = list(F.v("HYP", sens))
    delta_path = lambda a, c4, c8, g: f"{sens}.{a}.pairs.clef/{c4}|clef-flash/{c8}.pooled_{g}.delta_accuracy_big_minus_small.delta"  # noqa: E731
    cell = lambda a, c4, c8, g: F.v("HYP", delta_path(a, c4, c8, g))  # noqa: E731
    shown = lambda a, c4, c8, g: f"{100 * cell(a, c4, c8, g):+.2f} pp {F.c('HYP', delta_path(a, c4, c8, g))} ({a})"  # noqa: E731
    descr = []
    for a in alts:
        same = f"; the same hard and easy sets as {F.v('HYP', f'{sens}.{a}.same_as')}" if "same_as" in F.v("HYP", f"{sens}.{a}") else ""
        descr.append(f"{a} ({F.v('HYP', f'{sens}.{a}.basis')}: hard = {', '.join(F.v('HYP', f'{sens}.{a}.hard'))}; easy = "
                     f"{', '.join(F.v('HYP', f'{sens}.{a}.easy'))}{same})")
    ranges = []
    for c4 in cfgs4:
        for c8 in cfgs8:
            span = {}
            for g in ("hard", "easy"):
                lo, hi = min(alts, key=lambda a: cell(a, c4, c8, g)), max(alts, key=lambda a: cell(a, c4, c8, g))
                span[g] = f"{shown(lo, c4, c8, g)} to {shown(hi, c4, c8, g)}"
            ranges.append(f"{LAB[c4]} vs {LAB[c8]}: pooled hard {span['hard']}, pooled easy {span['easy']}")
    fails = [f"{LAB[c4]} vs {LAB[c8]} under {a}" for c4 in cfgs4 for c8 in cfgs8 for a in alts
             if not (cell(a, c4, c8, "hard") > 0 and cell(a, c4, c8, "easy") <= cell(a, c4, c8, "hard"))]
    n_cells = len(alts) * len(cfgs4) * len(cfgs8)
    rule = "pooled hard delta above 0 and pooled easy delta at or below the pooled hard delta, on the point estimates"
    holds = (f"The pattern ({rule}) holds in all {n_cells} pair and split combinations." if not fails else
             f"The pattern ({rule}) fails in {len(fails)} of {n_cells} pair and split combinations: {'; '.join(fails)}.")
    b.append(f"Sensitivity of the split: the primary median split above stays the reported one, and {len(alts)} alternate splits were added as a "
             f"robustness check, also after the 27B results were seen: {'; '.join(descr)} {F.c('HYP', sens)}. Range of the pooled deltas across "
             f"the alternate splits, Clef (27B) minus Clef-flash on the same questions, with the split that gives each end: {'; '.join(ranges)}. "
             f"{holds} These are robustness checks of a post-hoc split and are not pre-registered.")
    inside = [(i, p) for i, p in enumerate(ep) if p["a"]["model"] == "clef" and p["a"]["config"] in cfgs4
              and p["b"]["model"] == "clef-flash" and p["b"]["config"] in ("ref", "bf16-srv", "gguf-bf16")]
    fla = [p for p in ep if p["b"]["config"] == "ref-fla"]
    b.append(f"Pairs that are inside the window: PUBALL `equal_memory.pairs` lists {len(inside)} pairs of a 27B 4-bit build with a Clef-flash BF16 build "
             f"({len(fla)} further pairs involve ref-fla, a {max(p['n_items'] for p in fla):,}-question subset, and are skipped). These pair 27B 4-bit with Clef-flash BF16, which H8 "
             "does not ask about; they are listed for completeness and are not a test of H8.")
    for i, p in inside:
        pp = f"equal_memory.pairs.{i}"
        b.append(f"Inside the window: Clef (27B) {LAB[p['a']['config']]} ({p['a']['file_size_gb']:.2f} GB) vs Clef-flash {p['b']['config']} "
                 f"({p['b']['file_size_gb']:.2f} GB), size ratio {F.num('PUBALL', pp + '.size_ratio', 3)}: Clef-flash minus 27B accuracy "
                 f"{F.pp_ci('PUBALL', pp + '.delta_accuracy_b_minus_a')} on {p['n_items']:,} questions.")
    status = "not evaluable"      # computed above: no pair in the window, else the ValueError fired
    R.entry("H8", hyp["H8"], b, status,
            "needs 27B 4-bit accuracy on the same records and a 4-bit Clef / 8-bit Clef-flash pair within the analyzer's 15% size window."
            " No 4-bit Clef build lies within that window of an 8-bit Clef-flash build, so the pre-registered comparison cannot be formed with the "
            "builds this sweep made; the supplementary comparison is outside the window and is not pre-registered.",
            f"No 27B 4-bit / Clef-flash 8-bit pair is within the {100 * tol:.0f}% window. The closest is Clef (27B) {LAB[closest[0]]} "
            f"({gb('clef', closest[0]):.2f} GB) against Clef-flash {LAB[closest[1]]} ({gb('clef-flash', closest[1]):.2f} GB), a size ratio of "
            f"{ratio[closest]:.2f} against a limit of {1 + tol:.2f}. The comparison H8 pre-registers cannot be formed from the builds this sweep "
            "made, and no further run of this sweep changes that.")

    # ----------------------------------------------------------------------------- H9
    b = []
    for c in ("nvfp4", "q4_k_m", "fp8", "q8_0"):
        base = f"score_shift.per_config.{c}"
        b.append(f"sst5 (5-point scale, public), {c}: center pull {F.raw_ci('HYP', base + '.center_pull')} (negative = toward the middle); mean "
                 f"signed shift {F.raw_ci('HYP', base + '.mean_signed_shift')}.")
    priv_cells = []
    for c in cfgs4:
        for s, q in (("r1", "score6"), ("r4a", "score5")):
            p = f"label_free.{s}.{q}.pairs.ref|{c}.score_shift"
            lo, hi = F.v("PRIV", p + ".center_pull_ci95")
            priv_cells.append(hi < 0)
            b.append(f"Private {s} {q}, {c}: center pull {F.v('PRIV', p + '.center_pull'):+.4f} [{lo:+.4f}, {hi:+.4f}] "
                     f"{F.c('PRIV', p + '.center_pull,center_pull_ci95')}; mean signed shift {F.v('PRIV', p + '.mean_signed_shift'):+.4f}.")
    pub_ok = all(F.v("HYP", f"score_shift.per_config.{c}.center_pull.ci95")[1] < 0 for c in cfgs4)
    mags = [abs(F.v("PRIV", f"label_free.{s}.{q}.pairs.ref|{c}.score_shift.center_pull")) for c in cfgs4 for s, q in (("r1", "score6"), ("r4a", "score5"))] + \
           [abs(F.v("HYP", f"score_shift.per_config.{c}.center_pull.delta")) for c in cfgs4]
    status = "supported" if (pub_ok and all(priv_cells)) else ("not supported" if not pub_ok and not any(priv_cells) else "mixed")
    R.entry("H9", hyp["H9"], b, status,
            "supported if the center-pull 95% CI (|expected score - middle|, quantized minus ref) lies below 0 for both 4-bit builds on sst5 and on "
            "every private score question (R1 6-point score, R4a score5 bucket).",
            f"All {len(priv_cells) + 2 * int(pub_ok)} cells (two 4-bit builds x sst5, R1 score, R4a score5) show a pull toward the middle with CIs below 0; "
            f"the effect is small, {min(mags):.3f} to {max(mags):.3f} of a scale point." if status == "supported" else
            f"public: {pub_ok}; private cells below 0: {sum(priv_cells)} of {len(priv_cells)}.")

    # ----------------------------------------------------------------------------- H10
    b = []
    for c in cfgs4:
        for i, t in ((3, 0.65), (4, 0.7), (6, 0.8)):
            base = f"{FL}.configs.{c}.cascade.pooled.{i}"
            b.append(f"Public, {c}, escalate when its top probability < {t}: escalation {F.pct('PUB', base + '.escalation_rate', 1)}, agreement with ref "
                     f"{F.pct('PUB', base + '.agreement_with_ref', 2)} (standalone {F.pct('PUB', pooled(c, 'vs_ref') + '.agreement', 2)}), accuracy "
                     f"{F.pct('PUB', base + '.accuracy', 2)} (ref {F.pct('PUB', pooled('ref', 'accuracy'), 2)}).")
    for c in cfgs4:
        for i, t in ((0, 0.5), (2, 0.6)):
            base = f"{FL}.configs.{c}.cascade.pooled.{i}"
            b.append(f"Private R4a+R4b (the gold-bearing items the analyzer pairs), {c}, t = {t}: escalation {F.pct('PRIVA', base + '.escalation_rate', 1)}, "
                     f"agreement with ref {F.pct('PRIVA', base + '.agreement_with_ref', 2)}.")
    ok = lambda alias, c, i: (F.v(alias, f"{FL}.configs.{c}.cascade.pooled.{i}.agreement_with_ref") >= 0.99  # noqa: E731
                              and F.v(alias, f"{FL}.configs.{c}.cascade.pooled.{i}.escalation_rate") <= 0.20)
    pub_ok = all(ok("PUB", c, 4) for c in cfgs4)
    priv_ok = all(ok("PRIVA", c, 0) for c in cfgs4)
    esc = [F.v("PUB", f"{FL}.configs.{c}.cascade.pooled.4.escalation_rate") for c in cfgs4]
    agr = [F.v("PUB", f"{FL}.configs.{c}.cascade.pooled.4.agreement_with_ref") for c in cfgs4]
    esc_p = [F.v("PRIVA", f"{FL}.configs.{c}.cascade.pooled.0.escalation_rate") for c in cfgs4]
    esc_p6 = [F.v("PRIVA", f"{FL}.configs.{c}.cascade.pooled.2.escalation_rate") for c in cfgs4]
    R.entry("H10", hyp["H10"], b, "supported" if pub_ok and priv_ok else ("mixed" if pub_ok or priv_ok else "not supported"),
            "this report's operational reading: near-BF16 agreement = at least 99% top-1 agreement with ref (the level of the two 8-bit builds), small "
            "fraction = at most 20% escalated, checked at t = 0.7 (public) and t = 0.5 (private R4a/R4b, where it is the lowest threshold tabulated). "
            "supported if both public and private reach that; mixed if only one does.",
            f"Public: t = 0.7 escalates {100 * min(esc):.0f}-{100 * max(esc):.0f}% of questions and lifts agreement to {100 * min(agr):.1f}-{100 * max(agr):.1f}% "
            f"(the 8-bit builds agree {F.p('PUB', pooled('fp8', 'vs_ref') + '.agreement', 1)} and {F.p('PUB', pooled('q8_0', 'vs_ref') + '.agreement', 1)} standalone). Private R4a/R4b: the same 99% needs t = 0.5 and escalates {100 * min(esc_p):.0f}-{100 * max(esc_p):.0f}% (t = 0.6: "
            f"{100 * min(esc_p6):.0f}-{100 * max(esc_p6):.0f}%); there are many more low-confidence questions." if not (pub_ok and priv_ok) else "Both hold.")

    # ----------------------------------------------------------------------------- H11
    b = [f"Q4_K_M vs NVFP4 directly, full public set: they disagree on {F.int_('HYP', 'q4_pair.flips_between')} questions "
         f"({F.rate_ci('HYP', 'q4_pair.flip_rate_between', 'q4_pair.flip_rate_between_ci95')}); each differs from ref on "
         f"{F.pct('HYP', 'q4_pair.flip_rate_vs_ref.q4_k_m', 2)} (Q4_K_M) and {F.pct('HYP', 'q4_pair.flip_rate_vs_ref.nvfp4', 2)} (NVFP4) of questions.",
         f"Accuracy, NVFP4 minus Q4_K_M: {F.pp_ci('HYP', 'q4_pair.delta_accuracy_nvfp4_minus_q4_k_m')}; Brier "
         f"{F.raw_ci('HYP', 'q4_pair.delta_brier_nvfp4_minus_q4_k_m')}; ECE {F.raw_ci('HYP', 'q4_pair.delta_ece_nvfp4_minus_q4_k_m')}.",
         f"Q4_K_M is the more accurate build on {F.v('HYP', 'q4_pair.tasks_where_q4_k_m_more_accurate')} of 12 tasks, NVFP4 on "
         f"{F.v('HYP', 'q4_pair.tasks_where_nvfp4_more_accurate')} {F.c('HYP', 'q4_pair.per_task')}.",
         f"Same nominal bit width is not the same size: NVFP4 {F.num('SPD', 'configs.clef-flash.nvfp4.meta.file_size_gb', 2)} GB vs Q4_K_M "
         f"{F.num('SPD', 'configs.clef-flash.q4_k_m.meta.file_size_gb', 2)} GB."]
    ok11 = F.v("HYP", "q4_pair.delta_accuracy_nvfp4_minus_q4_k_m.excludes_zero")
    R.entry("H11", hyp["H11"], b, "supported" if ok11 else "not supported",
            "supported if the paired 95% CI of the accuracy difference between the two 4-bit builds excludes 0.",
            "They differ measurably: Q4_K_M is the more accurate and the better scored (Brier) of the two on this benchmark, and it changes fewer "
            "decisions against ref than NVFP4 does.")

    # ======================================================================================= N1-N5
    n1 = "tests.N1"
    sup = n1 + ".supplementary_not_preregistered"
    b = [f"flip(bf16-srv, fp8) = {F.v('DEC', n1 + '.flips_minuend')} of 2,400 questions "
         f"({F.rate_ci('DEC', 'pairs.bf16-srv|fp8.flip_rate', 'pairs.bf16-srv|fp8.flip_ci95')}); flip(bf16-srv, bf16-srv-r2) = "
         f"{F.v('DEC', n1 + '.flips_subtrahend')} ({F.rate_ci('DEC', 'pairs.bf16-srv|bf16-srv-r2.flip_rate', 'pairs.bf16-srv|bf16-srv-r2.flip_ci95')}; exact "
         f"95% 0.00-{F.p('DEC', 'pairs.bf16-srv|bf16-srv-r2.flip_ci95_exact_cp.1')}).",
         f"Difference {F.pp_ci('DEC', n1, delta_key='diff')}; stratified-by-task bootstrap [{100 * F.v('DEC', n1 + '.ci95_stratified.0'):+.2f}, "
         f"{100 * F.v('DEC', n1 + '.ci95_stratified.1'):+.2f}] {F.c('DEC', n1 + '.ci95_stratified')}.",
         f"Supplementary, not pre-registered: against FP8's own restart ({F.v('DEC', sup + '.flips_subtrahend')} flips) the difference is "
         f"{F.pp_ci('DEC', sup, delta_key='diff')}."]
    R.entry("N1", nulls["N1"], b, "supported" if F.v("DEC", n1 + ".supported") else "not supported", F.v("DEC", n1 + ".criterion"),
            f"FP8 flips {F.v('DEC', n1 + '.flips_minuend')} decisions where the BF16 restart flips {F.v('DEC', n1 + '.flips_subtrahend')}; the interval of the "
            f"difference excludes 0, and FP8's own restart noise ({F.v('DEC', sup + '.flips_subtrahend')} flips) is "
            f"{F.v('DEC', sup + '.flips_minuend') / max(F.v('DEC', sup + '.flips_subtrahend'), 1):.0f}x smaller than its effect.")

    n2 = "tests.N2"
    nonq = max(F.v("DEC", f"pairs.{k}.flip_rate") for k in ("bf16-srv|bf16-srv-r2", "fp8|fp8-r2", "nvfp4|nvfp4-r2", "gguf-bf16|gguf-bf16-r2",
                                                             "ref|ref-r2", "ref|ref-fla", "ref|bf16-srv", "ref|gguf-bf16"))
    b = [f"{F.v('DEC', n2 + '.family_size')} comparisons (NVFP4 vs bf16-srv and Q4_K_M vs gguf-bf16, each minus the 8 non-quantization flip rates: four "
         f"restart pairs, ref-r2, ref-fla, both engine changes). Flip counts on 2,400 questions: NVFP4 {F.v('DEC', 'pairs.bf16-srv|nvfp4.flips')}, "
         f"Q4_K_M {F.v('DEC', 'pairs.gguf-bf16|q4_k_m.flips')} {F.c('DEC', 'pairs.bf16-srv|nvfp4.flips')}; the largest non-quantization source is the "
         f"NVFP4 restart with {F.v('DEC', 'pairs.nvfp4|nvfp4-r2.flips')} flips {F.c('DEC', 'pairs.nvfp4|nvfp4-r2.flips')}.",
         f"Every CI excludes 0 (lower bound > 0): {F.v('DEC', n2 + '.all_lower_gt_zero')} {F.c('DEC', n2 + '.all_lower_gt_zero')}; with stratified resampling: "
         f"{F.v('DEC', n2 + '.all_lower_gt_zero_stratified')}; with Bonferroni (k = 16): {F.v('DEC', n2 + '.all_lower_gt_zero_bonferroni')} "
         f"{F.c('DEC', n2 + '.all_lower_gt_zero_bonferroni')}.",
         f"Weakest row: {F.v('DEC', n2 + '.weakest_row.minuend')} minus {F.v('DEC', n2 + '.weakest_row.subtrahend')}: "
         f"{F.pp_ci('DEC', n2 + '.weakest_row', delta_key='diff')} (Bonferroni lower bound {100 * F.v('DEC', n2 + '.weakest_row.ci95_bonferroni.0'):+.2f} pp)."]
    R.entry("N2", nulls["N2"], b, "supported" if F.v("DEC", n2 + ".supported") else "not supported", F.v("DEC", n2 + ".criterion"),
            f"Both 4-bit builds change {F.p('DEC', 'pairs.bf16-srv|nvfp4.flip_rate', 1)} of decisions on this unit against at most {100 * nonq:.1f}% for any "
            "non-quantization source.")

    b = []
    for row in range(4):
        r = f"tests.N3.rows.{row}"
        b.append(f"flip({F.v('DEC', r + '.minuend')}) - flip({F.v('DEC', r + '.subtrahend')}): {F.v('DEC', r + '.flips_minuend')} vs "
                 f"{F.v('DEC', r + '.flips_subtrahend')} flips, {F.pp_ci('DEC', r, delta_key='diff')}; CI entirely below 0: {F.v('DEC', r + '.upper_lt_zero')}.")
    R.entry("N3", nulls["N3"], b, "supported" if F.v("DEC", "tests.N3.supported") else "not supported", F.v("DEC", "tests.N3.criterion"),
            f"All four engine-minus-8-bit differences are negative and significant: on this unit an engine change at BF16 moves "
            f"{F.p('DEC', 'pairs.ref|gguf-bf16.flip_rate')}-{F.p('DEC', 'pairs.ref|bf16-srv.flip_rate')} of decisions, FP8 {F.p('DEC', 'pairs.bf16-srv|fp8.flip_rate')} "
            f"and Q8_0 {F.p('DEC', 'pairs.gguf-bf16|q8_0.flip_rate')}. (Same as H7.)"
            if not F.v("DEC", "tests.N3.supported") else "Engine changes flip at least as many decisions as 8-bit quantization.")

    n4 = "tests.N4"
    b = [f"ref vs ref-fla: {F.int_('DEC', n4 + '.ref_fla_flips')} flips of 2,400 ({F.pct('DEC', n4 + '.ref_fla_flip_rate', 2)}); nonzero: "
         f"{F.v('DEC', n4 + '.nonzero')}.",
         f"vLLM restart flip counts: BF16 {F.v('DEC', n4 + '.vllm_restart_flips.bf16-srv|bf16-srv-r2')}, FP8 {F.v('DEC', n4 + '.vllm_restart_flips.fp8|fp8-r2')}, "
         f"NVFP4 {F.v('DEC', n4 + '.vllm_restart_flips.nvfp4|nvfp4-r2')}; range {F.v('DEC', n4 + '.vllm_restart_range.0')} to "
         f"{F.v('DEC', n4 + '.vllm_restart_range.1')} {F.c('DEC', n4 + '.vllm_restart_range')}. Within that range: {F.v('DEC', n4 + '.within_vllm_restart_range')}. "
         f"Against the BF16 restart alone ({F.v('DEC', n4 + '.bf16_only_vllm_restart_flips')} flips) it is above: {not F.v('DEC', n4 + '.within_bf16_only_range')}.",
         "Differences of flip rates, ref-fla minus each vLLM restart: "
         + "; ".join(f"{F.v('DEC', n4 + f'.differences_vs_each_vllm_restart.{i}.subtrahend')}: "
                     f"{F.pp_ci('DEC', n4 + f'.differences_vs_each_vllm_restart.{i}', delta_key='diff')}" for i in range(3)) + "."]
    R.entry("N4", nulls["N4"], b, "supported" if F.v("DEC", n4 + ".supported") else "not supported", F.v("DEC", n4 + ".criterion"),
            f"The kernel switch alone flips {F.v('DEC', n4 + '.ref_fla_flips')} of 2,400 decisions: more than the FP8 restart "
            f"({F.v('DEC', n4 + '.vllm_restart_flips.fp8|fp8-r2')}), fewer than the NVFP4 restart ({F.v('DEC', n4 + '.vllm_restart_flips.nvfp4|nvfp4-r2')}), and "
            f"significantly above the BF16 vLLM restart ({F.v('DEC', n4 + '.bf16_only_vllm_restart_flips')}). Whether it is 'within the range' depends on "
            "which restart pairs make up 'the vLLM restart drift'.")

    n5 = "tests.N5"
    b = [f"gguf-bf16 vs gguf-bf16-r2: {F.int_('DEC', n5 + '.n_identical_records')} of {F.int_('DEC', n5 + '.n_records')} records bit-identical "
         f"({F.pct('DEC', n5 + '.share_identical', 1)}); basis: {F.v('DEC', n5 + '.identical_basis')} (llama.cpp returns full-precision probabilities and no "
         f"logits); flips {F.v('DEC', n5 + '.flips')}, max |dprob| {F.v('DEC', n5 + '.max_dprob'):g}, mean TVD {F.v('DEC', n5 + '.tvd_mean'):g} "
         f"{F.c('DEC', n5 + '.max_dprob')}.",
         f"For contrast, the vLLM restarts are bit-identical on {F.pct('DEC', 'pairs.bf16-srv|bf16-srv-r2.share_identical', 1)} (BF16), "
         f"{F.pct('DEC', 'pairs.fp8|fp8-r2.share_identical', 1)} (FP8) and {F.pct('DEC', 'pairs.nvfp4|nvfp4-r2.share_identical', 1)} (NVFP4) of records."]
    R.entry("N5", nulls["N5"], b, "supported" if F.v("DEC", n5 + ".supported") else "not supported", F.v("DEC", n5 + ".criterion"),
            "All 2,400 records are identical, so the llama.cpp restart noise floor is exactly zero on this unit.")

    # ======================================================================================= assemble
    out: list[str] = ["# Findings: Clef quantization benchmark", ""]
    out += ["Every number below is read from a JSON file in this directory by `bench/findings.py` and carries a citation `[ALIAS key.path]`; the verbatim "
            "hypothesis text is parsed from `PLAN.md`. Hypotheses are fixed: outcomes are reported as found.", "",
            "Sources: " + "; ".join(f"`{a}` = `{p}`" for a, p in ALIASES.items() if a != "PRIVA")
            + f"; `PRIVA` = `{ALIASES['PRIVA']}` (analyzer output for the private sets). Private sets appear as aggregates only.", "",
            f"Scope: {clef_scope(F)} Comparison unit for the noise-floor tests: the first 200 records of each public task (2,400 questions); full-set numbers use all 31,275 "
            "public questions. 'pp' = percentage points; CIs are 95%.", ""]
    out += ["## Status at a glance", "", "| id | status |", "|---|---|"]
    for k in [*(f"H{i}" for i in range(1, 12)), *(f"N{i}" for i in range(1, 6))]:
        out.append(f"| {k} | {R.status[k][0]} |")
    out += ["", "## Hypotheses H1-H11", ""] + [R.sections[f"H{i}"] for i in range(1, 12)]
    out += ["## Report tests N1-N5", ""] + [R.sections[f"N{i}"] for i in range(1, 6)]
    out += two_seven_b(F) + private_section(F) + caveats(F)
    out += ["## Reproduce", "", "```",
            "python -m bench.analyze --records data/records --results results --out analysis/flash_public --models clef-flash",
            "python -m bench.analyze --records data/records --results results --out analysis/all_public",
            "python -m bench.analyze --private private --out analysis/flash_private --models clef-flash",
            "python -m bench.decompose --out analysis", "python -m bench.speed", "python -m bench.hypotheses", "python -m bench.private_agg",
            "python -m bench.findings --root . --out analysis/findings.md", "```", ""]
    return "\n".join(out) + "\n"


def clef_coverage(F: Facts) -> dict[str, Any]:
    """Which Clef (27B) configs have full public-set results and which only a subset (from the analyzer over both models)."""
    cfgs = F.v("PUBALL", "models.clef.configs")
    n_ref = cfgs["ref"]["pooled"]["n_scored"]
    n_all = len(F.v("PUBALL", "meta.tasks_with_records"))
    return {"full": [c for c, v in cfgs.items() if v["pooled"]["n_scored"] == n_ref],
            "subset": {c: v["pooled"]["n_scored"] for c, v in cfgs.items() if v["pooled"]["n_scored"] != n_ref},
            "tasks": (f"all {n_all} public tasks" if len(F.v("HYP", "big_vs_small.tasks_finished")) == n_all
                      else f"{len(F.v('HYP', 'big_vs_small.tasks_finished'))} of {n_all} public tasks"),
            "private": sorted(k.split("|")[1] for k in F.v("PRIV", "label_free_big.r1.pooled.pairs") if k.startswith("ref|"))}


def clef_scope(F: Facts) -> str:
    cov = clef_coverage(F)
    n_flash, n_big = len(F.v("SPD", "configs.clef-flash")), len(F.v("SPD", "configs.clef"))
    return (f"Clef-flash (9B) has results for {n_flash} configs {F.c('SPD', 'configs.clef-flash')}; Clef (27B) for {n_big} {F.c('SPD', 'configs.clef')}. "
            f"Clef (27B) covers {cov['tasks']} with {', '.join(cov['full'])}"
            + "".join(f", and only {n:,} public questions with {c}" for c, n in cov["subset"].items())
            + f"; its private-set pairs against ref are {', '.join(cov['private'])} {F.c('PRIV', 'label_free_big')}.")


def two_seven_b(F: Facts) -> list[str]:
    done = F.v("HYP", "big_vs_small.tasks_finished")
    missing = F.v("HYP", "big_vs_small.tasks_missing")
    cov = clef_coverage(F)
    meta = "configs.clef.ref.meta"
    lowest = min(((c, F.v("PUBALL", f"models.clef.configs.{c}.pooled.vs_ref.agreement")) for c in cov["full"] if c != "ref"), key=lambda x: x[1])
    pooled_agree = F.v("HYP", "big_vs_small.pooled.top1_agreement_27b_vs_9b")
    L = ["## 27B", "",
         f"Clef (27B) public-set results {F.c('PUBALL', 'models.clef.configs')}: complete on {cov['tasks']} for "
         + ", ".join(cov["full"]) + "".join(f"; {c} on {n:,} questions only" for c, n in cov["subset"].items())
         + f". `ref` is the BF16 reference ({F.v('SPD', meta + '.engine')}, {F.v('SPD', meta + '.engine_version')}, batch {F.v('SPD', meta + '.batch_size')}) "
           f"{F.c('SPD', meta)}; every other 27B config is compared with the 27B's own `ref`. The table below sets the 27B `ref` against the 9B `ref` "
           f"in absolute terms. Tasks finished: {', '.join(done)}; "
         + (f"missing: {', '.join(missing)}" if missing else "none missing") + f" {F.c('HYP', 'big_vs_small.tasks_missing')}. "
         "Pairing is by record id; intervals are paired bootstraps over questions.", "",
         "| task | n | 27B accuracy | 9B accuracy | 27B minus 9B pp [95% CI] | top-1 agreement 27B vs 9B | macro-F1 27B / 9B |", "|---|---|---|---|---|---|---|"]
    ahead, level, behind = [], [], []
    agree = []
    for t in done:
        p = f"big_vs_small.per_task.{t}"
        d = F.v("HYP", p + ".delta_accuracy_27b_minus_9b")
        agree.append(F.v("HYP", p + ".top1_agreement_27b_vs_9b"))
        (ahead if d["ci95"][0] > 0 else behind if d["ci95"][1] < 0 else level).append(t)
        L.append(f"| {t} | {F.v('HYP', p + '.n'):,} | {100 * F.v('HYP', p + '.acc_27b'):.2f}% | {100 * F.v('HYP', p + '.acc_9b'):.2f}% | "
                 f"{100 * d['delta']:+.2f} [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}] | {100 * agree[-1]:.1f}% | "
                 f"{F.v('HYP', p + '.macro_f1_27b'):.3f} / {F.v('HYP', p + '.macro_f1_9b'):.3f} |")
    same = ("both figures cover the same tasks" if not missing else
            f"the 9B figure here excludes {', '.join(missing)}, so it is not the 9B all-task accuracy")
    L += ["", f"Source for every row: [HYP big_vs_small.per_task.<task>]. Pooled over the {len(done)} finished public tasks "
          f"({F.v('HYP', 'big_vs_small.pooled.n'):,} questions): 27B {F.pct('HYP', 'big_vs_small.pooled.acc_27b')}, 9B "
          f"{F.pct('HYP', 'big_vs_small.pooled.acc_9b')}, difference {F.pp_ci('HYP', 'big_vs_small.pooled.delta_accuracy_27b_minus_9b')} (pooled over "
          f"questions, not stratified by task; {same}).", "",
          f"Reading (from the intervals above): the 27B is ahead on {', '.join(ahead)}; level on {', '.join(level)}; behind on {', '.join(behind)}. "
          f"Its top-1 agreement with the 9B reference ranges {100 * min(agree):.0f}% to {100 * max(agree):.0f}% by task; pooled it is "
          f"{F.pct('HYP', 'big_vs_small.pooled.top1_agreement_27b_vs_9b', 1)}, {'below' if pooled_agree < lowest[1] else 'not below'} the lowest pooled "
          f"agreement of a 27B build with its own ref ({100 * lowest[1]:.1f}%, {lowest[0]}) "
          f"{F.c('PUBALL', f'models.clef.configs.{lowest[0]}.pooled.vs_ref.agreement')}.", ""]
    ratio = "ratio_27b_over_9b.pooled_with_sidecar_fallback"
    L += [f"Speed (quality-run throughput, not the latency benchmark): 27B / 9B per-record time {F.num('SPD', ratio + '.pooled_ratio', 2)} pooled over "
          f"{F.v('SPD', ratio + '.n_sets')} sets (per-set range {F.v('SPD', ratio + '.min_ratio'):.2f} to {F.v('SPD', ratio + '.max_ratio'):.2f}); the four 9B "
          f"sets without an END line use the per-set sidecar [SPD ratio_27b_over_9b.fallback_sets].", ""]
    for s, q in (("r4a", "choice5"), ("r4a", "score5"), ("r4b", "choice3")):
        p = f"label_agreement.{s}.{q}.models"
        L.append(f"- Private {s} / {q}: agreement with the hosted label, 27B ref {F.p('PRIV', p + '.clef.configs.ref.accuracy', 1)} (macro-F1 "
                 f"{F.v('PRIV', p + '.clef.configs.ref.macro_f1'):.3f}) vs Clef-flash ref {F.p('PRIV', p + '.clef-flash.configs.ref.accuracy', 1)} (macro-F1 "
                 f"{F.v('PRIV', p + '.clef-flash.configs.ref.macro_f1'):.3f}) {F.c('PRIV', p)}.")
    L += ["", "### 27B quantization", ""]
    qcfgs = [c for c in ("bf16-srv", "gguf-bf16", "fp8", "q8_0", "nvfp4", "q4_k_m") if c in cov["full"]] + \
            sorted(c for c in cov["full"] if c not in ("ref", "bf16-srv", "gguf-bf16", "fp8", "q8_0", "nvfp4", "q4_k_m"))
    vs = lambda m, c, k: f"models.{m}.configs.{c}.pooled.vs_ref.{k}"  # noqa: E731
    n_pub = F.v("PUBALL", vs("clef", qcfgs[0], "n_paired"))
    nosize = [m["config"] for m in F.v("SPD", "file_sizes_missing") if m["model"] == "clef"]
    L += [f"Each 27B config against the 27B `ref`, full public set ({n_pub:,} questions), paired bootstrap over questions; the last column is the "
          "Clef-flash flip count for the same config against the Clef-flash `ref`. "
          + ("" if not nosize else f"File sizes are missing for {', '.join(nosize)}. ")
          + f"`ref-fla` is left out because the 27B has it on a subset only.", "",
          "| config | file size GB | accuracy change vs 27B ref pp [95% CI] | flips (% of questions) | top-1 agreement | Clef-flash flips (% of questions) |",
          "|---|---|---|---|---|---|"]
    for c in qcfgs:
        d = F.v("PUBALL", vs("clef", c, "delta_accuracy"))
        gb_c = F.v("SPD", f"configs.clef.{c}.meta.file_size_gb")
        fb, ff = F.v("PUBALL", vs("clef", c, "n_flips")), F.v("PUBALL", vs("clef-flash", c, "n_flips"))
        ab, af = F.v("PUBALL", vs("clef", c, "agreement")), F.v("PUBALL", vs("clef-flash", c, "agreement"))
        L.append(f"| {c} | {'not recorded' if gb_c is None else f'{gb_c:.2f}'} | {100 * d['delta']:+.2f} [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}] | "
                 f"{fb:,} ({100 * (1 - ab):.2f}%) | {100 * ab:.2f}% | {ff:,} ({100 * (1 - af):.2f}%) |")
    nf = "models.{m}.noise_floor.ref.ref-r2.pooled.n_flips"
    L += ["", "Source for every row: [SPD configs.clef.<cfg>.meta.file_size_gb]; [PUBALL models.clef.configs.<cfg>.pooled.vs_ref.delta_accuracy.delta,ci95], "
          "[PUBALL models.clef.configs.<cfg>.pooled.vs_ref.n_flips] and [PUBALL models.clef.configs.<cfg>.pooled.vs_ref.agreement]; Clef-flash flips "
          "[PUBALL models.clef-flash.configs.<cfg>.pooled.vs_ref.n_flips]. Rerun floor on the same questions, ref against ref-r2: Clef (27B) "
          f"{F.int_('PUBALL', nf.format(m='clef'))} flips, Clef-flash {F.int_('PUBALL', nf.format(m='clef-flash'))} flips.", ""]
    n_b = sum(F.v("PRIV", f"label_free_big.{s}.pooled.n_questions") for s in SETS)
    n_f = sum(F.v("PRIV", f"label_free.{s}.pooled.n_questions") for s in SETS)
    L += [f"Private sets R1, R2, R4a and R4b pooled ({n_b:,} questions for the 27B, {n_f:,} for Clef-flash), flips against each model's own `ref`:", "",
          "| config | Clef (27B) flips (% of questions) | Clef-flash flips (% of questions) |", "|---|---|---|"]
    for c in [*qcfgs, "ref-r2"]:
        fb = sum(F.v("PRIV", f"label_free_big.{s}.pooled.pairs.ref|{c}.flips") for s in SETS)
        ff = sum(F.v("PRIV", f"label_free.{s}.pooled.pairs.ref|{c}.flips") for s in SETS)
        L.append(f"| {c}{' (rerun floor)' if c == 'ref-r2' else ''} | {fb:,} ({100 * fb / n_b:.2f}%) | {ff:,} ({100 * ff / n_f:.2f}%) |")
    L += ["", "Source for every row: sums over the four sets of [PRIV label_free_big.<set>.pooled.pairs.ref|<cfg>.flips] (27B) and "
          "[PRIV label_free.<set>.pooled.pairs.ref|<cfg>.flips] (Clef-flash); denominators from [PRIV label_free_big.<set>.pooled.n_questions] and "
          "[PRIV label_free.<set>.pooled.n_questions].", ""]
    for s, q in (("r4a", "choice5"), ("r4a", "score5"), ("r4b", "choice3")):
        p = f"label_agreement.{s}.{q}.models.clef.configs"
        L.append(f"- Private {s} / {q}: change in agreement with the hosted label, 27B config against the 27B ref (paired): "
                 + "; ".join(f"{c}: {F.pp_ci('PRIV', f'{p}.{c}.delta_vs_ref.delta_accuracy')}" for c in ("fp8", "q8_0", "nvfp4", "q4_k_m")) + ".")
    L += [""]
    return L


def private_section(F: Facts) -> list[str]:
    L = ["## Private sets (aggregates only)", "",
         "Full tables: `analysis/private_aggregates.md` / `.json`; analyzer view: `analysis/flash_private/`. R1/R2 have no trustworthy gold, so only label-free "
         "numbers exist for them. The hosted model's recorded label is run a of the hosted repeat (verified: the recorded label equals run a on every repeat "
         "pair).", ""]
    for q, pq in (("choice5", 0.077), ("score5", 0.281)):
        p = f"hosted_repeat.r4a.{q}"
        L.append(f"- R4a {q}: the hosted model disagrees with its own repeat on {F.rate_ci('PRIV', p + '.disagreement', p + '.disagreement_ci95')} of "
                 f"{F.v('PRIV', p + '.n_pairs'):,} pairs ({F.v('PRIV', p + '.disagreement_n')} pairs); PLAN.md quotes {100 * pq:.1f}%; reproduced: "
                 f"{F.v('PRIV', p + '.reproduces_plan_quote')} {F.c('PRIV', p + '.reproduces_plan_quote')}. Macro-F1 of the repeat against the recorded label "
                 f"{F.num('PRIV', p + '.run_b_vs_run_a.macro_f1', 3)}.")
    for q in ("choice5", "score5"):
        sub = f"label_agreement.r4a.{q}.hosted_repeat_subset"
        mc = f"{sub}.models.clef-flash.configs"
        L.append(f"- R4a {q}, on the {F.v('PRIV', sub + '.n'):,} repeat records, agreement with the hosted label: majority-class baseline "
                 f"{F.p('PRIV', sub + '.majority_baseline.accuracy', 1)}, hosted repeat {F.p('PRIV', sub + '.hosted_run_b_vs_recorded.accuracy', 1)} (macro-F1 "
                 f"{F.v('PRIV', sub + '.hosted_run_b_vs_recorded.macro_f1'):.3f}), Clef-flash ref {F.p('PRIV', mc + '.ref.accuracy', 1)} (macro-F1 "
                 f"{F.v('PRIV', mc + '.ref.macro_f1'):.3f}), NVFP4 {F.p('PRIV', mc + '.nvfp4.accuracy', 1)}, Q4_K_M {F.p('PRIV', mc + '.q4_k_m.accuracy', 1)}, Q8_0 "
                 f"{F.p('PRIV', mc + '.q8_0.accuracy', 1)} {F.c('PRIV', sub)}. Clef-flash ref minus hosted repeat: "
                 f"{F.pp_ci('PRIV', mc + '.ref.delta_vs_hosted_run_b.delta_accuracy')} agreement, "
                 f"{F.pp_ci('PRIV', mc + '.ref.delta_vs_hosted_run_b.delta_macro_f1')} macro-F1.")
    sub = "label_agreement.r4a.choice5.hosted_repeat_subset"
    big = f"{sub}.models.clef.configs.ref"
    flash = f"{sub}.models.clef-flash.configs.ref"
    hosted = f"{sub}.hosted_run_b_vs_recorded"
    L.append(f"- Reading, R4a choice5: the majority-class baseline ({F.p('PRIV', sub + '.majority_baseline.accuracy', 1)}) beats even the hosted repeat "
             f"({F.p('PRIV', hosted + '.accuracy', 1)}) on raw agreement, so agreement alone says nothing; per-class recall separates the predictors. "
             f"Recall of the second, third and fourth class: hosted repeat {F.p('PRIV', hosted + '.recall.c2.recall', 0)}, {F.p('PRIV', hosted + '.recall.c3.recall', 0)}, "
             f"{F.p('PRIV', hosted + '.recall.c4.recall', 0)}; Clef-flash ref {F.p('PRIV', flash + '.recall.c2.recall', 0)}, {F.p('PRIV', flash + '.recall.c3.recall', 0)}, "
             f"{F.p('PRIV', flash + '.recall.c4.recall', 0)}; 27B ref {F.p('PRIV', big + '.recall.c2.recall', 0)}, {F.p('PRIV', big + '.recall.c3.recall', 0)}, "
             f"{F.p('PRIV', big + '.recall.c4.recall', 0)} (supports {F.v('PRIV', hosted + '.recall.c2.support')}, {F.v('PRIV', hosted + '.recall.c3.support')}, "
             f"{F.v('PRIV', hosted + '.recall.c4.support')}) {F.c('PRIV', sub)}. Clef-flash is not a replacement for the hosted call on R4a choice5 or score5, "
             "in BF16 or 8-bit; quantization moves these aggregates by far less than that gap.")
    L.append("- Quantization effect on R4a choice5 agreement with the hosted label (change against ref, paired): "
             + "; ".join(f"{c}: {F.pp_ci('PRIV', f'label_agreement.r4a.choice5.models.clef-flash.configs.{c}.delta_vs_ref.delta_accuracy')}"
                         for c in ("fp8", "nvfp4", "q8_0", "q4_k_m")) + ".")
    L.append("- R1 and R2, label-free, flip rate against ref pooled over all questions: "
             + "; ".join(f"{s.upper()} {c}: {F.rate_ci('PRIV', f'label_free.{s}.pooled.pairs.ref|{c}.flip_rate', f'label_free.{s}.pooled.pairs.ref|{c}.flip_ci95', 1)}"
                         for s in ("r1", "r2") for c in ("q8_0", "nvfp4", "q4_k_m")) + "; rerun floor (ref-r2): "
             + "; ".join(f"{s.upper()} {F.rate_ci('PRIV', f'label_free.{s}.pooled.pairs.ref|ref-r2.flip_rate', f'label_free.{s}.pooled.pairs.ref|ref-r2.flip_ci95', 2)}"
                         for s in ("r1", "r2")) + ".")
    L += [""]
    return L


def caveats(F: Facts) -> list[str]:
    tag = F.v("DEC", "conditions.ref.tasks_by_value")
    untagged = sorted(t for v, ts in tag.items() if "gdn=" not in v for t in ts)
    tagged = sorted(t for v, ts in tag.items() if "gdn=" in v for t in ts)
    ident = lambda pair, ts: [F.v("DEC", f"pairs.{pair}.share_identical_by_task.{t}") for t in ts]  # noqa: E731
    L = ["## Data anomalies and caveats found while computing these numbers", ""]
    L.append("1. **PLAN.md record counts.** PLAN.md says held-out splits of 'about 500 records each'; the public task files hold "
             + ", ".join(f"{t} {F.v('MAN', f'tasks.{t}.written'):,}" for t in ("banking77", "clinc150", "mmlu", "when2call"))
             + " and so on, 31,275 questions in total [MAN tasks.<task>.written]. All numbers here use what is on disk.")
    L.append(f"2. **Untagged reference runs (provenance gap, not verified from the outputs).** `ref` result lines for {', '.join(untagged)} carry "
             "engine_version 'transformers 5.16.1' without the `gdn=` kernel tag that the later runs record "
             f"{F.c('DEC', 'conditions.ref.tasks_by_value')}. PLAN.md says the container had no flash-linear-attention then, so they ran the PyTorch "
             f"fallback, and the config-level meta.json says gdn_kernel = {F.v('SPD', 'configs.clef-flash.ref.meta.gdn_kernel')} "
             f"{F.c('SPD', 'configs.clef-flash.ref.meta.gdn_kernel')}, but that file was written afterwards from the config definition, not observed at run "
             "time. The outputs cannot prove it either way: the share of bit-identical records between ref and ref-fla is "
             f"{100 * min(ident('ref|ref-fla', untagged)):.1f}-{100 * max(ident('ref|ref-fla', untagged)):.1f}% on the untagged tasks and "
             f"{100 * min(ident('ref|ref-fla', tagged)):.1f}-{100 * max(ident('ref|ref-fla', tagged)):.1f}% on the tagged ones "
             "[DEC pairs.ref|ref-fla.share_identical_by_task] (no untagged task is anywhere near 100%, as it would be if ref had run the same kernel as ref-fla "
             "and that kernel were reproducible, which has not been tested). The N4 comparison and every ref-based number assume the fallback.")
    miss = F.v("SPD", "missing_end_lines")
    L.append("3. **Timing gaps.** The sweep log has no END line for " + ", ".join(f"{m['model']}/{m['config']}/{m['set']}" for m in miss)
             + f" {F.c('SPD', 'missing_end_lines')}; per-config totals exclude them (the like-for-like table uses common sets), and the 27B/9B ratio takes the 9B "
               "time for those sets from the per-set sidecar (`wall_seconds`, load time not included).")
    L.append("4. **Analyzer columns that are not throughput.** `p50 ms`, `p95 ms` and `dec/s` in `analysis/flash_public/tables.md` come from per-request latency; "
             "for vLLM those requests ran 16 in flight, so dec/s there (bf16-srv: "
             f"{F.num('PUB', 'models.clef-flash.configs.bf16-srv.pooled.latency.decisions_per_sec', 2)}) is not a throughput. Use `analysis/speed.md`.")
    L.append("5. **Analyzer limits fixed or worked around.** (a) `--models clef-flash` keeps the 27B out of `flash_public` (it was unfinished when that directory was built; test added); "
             "`analysis/all_public/` is the same analyzer over both models and is the source of every 27B quantization number. (b) For "
             "private sets the analyzer pairs only items with gold, so R1/R2 get no paired numbers, and it blends the two R4a questions into one macro-F1: "
             "`bench/private_agg.py` computes per-question and label-free numbers instead. (c) The analyzer's `noise_floor_max_*` annotation takes the maximum over "
             "per-task blocks of every repeat of the model (mixing engines); the decomposition uses matched pairs instead. (d) `ref-fla` appears in the analyzer's "
             "equal-memory table with accuracy on its 2,400-question subset next to full-set rows; do not compare them.")
    L.append("6. **'Bit-identical' has two bases.** Native and vLLM logits are on bf16 granularity and are compared as logits; llama.cpp emits full-precision "
             "probabilities and no logits, so its records are compared on probabilities. Raw max |dlogit| contains any constant offset of the logit scale (large "
             "for NVFP4 vs BF16), so the centred value is reported next to it.")
    L.append("7. **Rare classes.** R4a choice5 has classes with 3 and 45 records and R4b's 3-way choice a class with 1; all-class macro-F1 intervals are approximate "
             "there, and a macro-F1 over classes with support of at least 30 is reported beside it.")
    L.append("8. **Zero-flip pairs.** The percentile bootstrap interval of a flip rate collapses to [0, 0] when a pair has 0 flips; exact Clopper-Pearson intervals are "
             "shown next to it (0 of 2,400 gives an upper bound of 0.15%).")
    cov = clef_coverage(F)
    nosize = [m["config"] for m in F.v("SPD", "file_sizes_missing") if m["model"] == "clef"]
    L.append("9. **27B scope.** Clef (27B) has results for " + ", ".join(cov["full"]) + f" on {cov['tasks']} "
             f"{F.c('PUBALL', 'models.clef.configs')}"
             + "".join(f"; {c} only on {n:,} public questions (a subset: never compare it with a full-set row)" for c, n in cov["subset"].items())
             + ". File sizes are " + ("recorded for every 27B config" if not nosize else "missing for " + ", ".join(nosize))
             + f" {F.c('SPD', 'file_sizes_missing')}. Every 27B quantization number is measured against the 27B's own ref; the cross-size "
               "comparison H8 asks for cannot be formed from this sweep's builds (see H8).")
    L.append("")
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("analysis/findings.md"))
    args = ap.parse_args(argv)
    text = build_report(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
