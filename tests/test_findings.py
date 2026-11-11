"""Tests of bench.findings: verbatim PLAN.md parsing, citation integrity, and status consistency with the sources."""

import json
import re
import shutil
from pathlib import Path

import pytest

from bench import findings as FI

REAL = Path(__file__).resolve().parent.parent
needs_real = pytest.mark.skipif(
    not all((REAL / p).exists() for p in ("PLAN.md", *FI.ALIASES.values())), reason="real analysis outputs not present")

PLAN_SNIPPET = """\
# Plan

## Pre-registered hypotheses (written before any results; published as-is with the outcome)

Obvious: H1 first thing is true. H2 second thing, more here than
there. H3 third thing holds.
Less obvious: H4 fourth thing. H5 fifth (with a parenthesis). H6 sixth.
H7 seventh. H8 eighth. H9 ninth. H10 tenth, with a
second line. H11 eleventh.

## Next section

Pre-registered for the report (in addition to H1-H11; outcomes published as-is):
- N1 first null `x` is not larger than `y`
  (continued on a second line).
- N2 second null.
- N3 third null (H7).
- N4 fourth null,
  continued.
- N5 fifth null.

Extra work the report needs, after the sweep.
"""


def test_parse_plan_extracts_each_hypothesis_verbatim_without_section_labels():
    hyp, nulls = FI.parse_plan(PLAN_SNIPPET)
    assert sorted(hyp, key=lambda k: int(k[1:])) == [f"H{i}" for i in range(1, 12)]
    assert hyp["H1"] == "H1 first thing is true."
    assert hyp["H2"] == "H2 second thing, more here than there."
    assert hyp["H3"] == "H3 third thing holds."            # 'Less obvious:' belongs to the next paragraph label, not to H3
    assert hyp["H4"] == "H4 fourth thing."
    assert hyp["H10"] == "H10 tenth, with a second line."
    assert hyp["H11"] == "H11 eleventh."
    assert sorted(nulls) == ["N1", "N2", "N3", "N4", "N5"]
    assert nulls["N1"] == "N1 first null `x` is not larger than `y` (continued on a second line)."
    assert nulls["N4"] == "N4 fourth null, continued."
    assert "Extra work" not in nulls["N5"]


@pytest.mark.skipif(not (REAL / "PLAN.md").exists(), reason="PLAN.md not present")
def test_parse_plan_on_the_real_plan_keeps_the_pre_registered_wording():
    hyp, nulls = FI.parse_plan((REAL / "PLAN.md").read_text())
    assert hyp["H1"] == "H1 8-bit is indistinguishable from BF16 within the noise floor."
    assert hyp["H3"] == "H3 flips concentrate where ref confidence is low."
    assert hyp["H4"].startswith("H4 calibration (ECE/Brier) degrades before accuracy does.")
    assert hyp["H11"] == "H11 GGUF Q4_K_M and NVFP4 differ at the same nominal bit width."
    assert "H8 at equal memory, 4-bit Clef beats 8-bit Clef-flash on hard tasks but not easy ones." == hyp["H8"]
    assert nulls["N1"].endswith("(8-bit inside restart noise; H1 restated against a measured floor).")
    assert nulls["N5"].endswith("(`gguf-bf16` vs `gguf-bf16-r2`).")
    plan_text = " ".join((REAL / "PLAN.md").read_text().split())
    for k, v in {**hyp, **nulls}.items():          # every parsed text is a verbatim substring of the plan (modulo line breaks)
        assert v in plan_text, k


def test_facts_walk_handles_dotted_keys_lists_and_missing_keys(tmp_path):
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "decomposition.json").write_text(json.dumps(
        {"a": {"[0.5,0.7)": {"n": 3}, "lst": [{"x": 1}, {"x": 2}], "p|q": 0.25}}))
    F = FI.Facts(tmp_path)
    assert F.v("DEC", "a.[0.5,0.7).n") == 3
    assert F.v("DEC", "a.lst.1.x") == 2
    assert F.v("DEC", "a.p|q") == 0.25
    assert ("DEC", "a.lst.1.x") in F.used
    with pytest.raises(KeyError):
        F.v("DEC", "a.nope")
    assert F.pct("DEC", "a.p|q", 1) == "25.0% [DEC a.p|q]"
    (tmp_path / "analysis" / "hypotheses.json").write_text(json.dumps({"x": {"delta": 0.012, "ci95": [0.01, 0.02]}}))
    assert F.pp_ci("HYP", "x") == "+1.20 pp [+1.00, +2.00] [HYP x.delta,ci95]"


def test_report_entry_rejects_unknown_status():
    R = FI.Report()
    with pytest.raises(AssertionError):
        R.entry("H1", "H1 x", ["b"], "kind of supported", "rule", "why")
    R.entry("H1", "H1 x", ["b"], "mixed", "rule", "why")
    assert "**Status: mixed.**" in R.sections["H1"] and "> H1 x" in R.sections["H1"]


def test_not_evaluable_is_a_status_and_is_defined_in_the_module_docstring_apart_from_not_yet_evaluable():
    R = FI.Report()
    R.entry("H8", "H8 x", ["b"], "not evaluable", "rule", "why")
    assert "**Status: not evaluable.**" in R.sections["H8"]
    assert FI.ALIASES["PUBALL"] == "analysis/all_public/summary.json"
    doc = " ".join(FI.__doc__.split())
    assert "`not evaluable` means the pre-registered comparison cannot be formed from the configurations this sweep built" in doc
    assert "unlike `not yet evaluable`, no further run of this sweep changes that" in doc


@pytest.fixture(scope="module")
def real_report():
    return FI.build_report(REAL)


@needs_real
def test_every_hypothesis_and_test_has_a_status_and_the_verbatim_text(real_report):
    hyp, nulls = FI.parse_plan((REAL / "PLAN.md").read_text())
    for k, text in {**hyp, **nulls}.items():
        assert f"### {k}\n\n> {text}\n" in real_report, k
        m = re.search(rf"### {k}\n.*?\*\*Status: ([a-z ]+)\.\*\*", real_report, re.S)
        assert m and m.group(1) in FI.STATUSES, k


@needs_real
def test_status_table_matches_the_sections_and_the_pre_registered_tests_follow_the_decomposition(real_report):
    dec = json.loads((REAL / FI.ALIASES["DEC"]).read_text())["tests"]
    for n in ("N1", "N2", "N3", "N4", "N5"):
        want = "supported" if dec[n]["supported"] else "not supported"
        assert re.search(rf"\| {n} \| {want} \|", real_report), n
    for k in [f"H{i}" for i in range(1, 12)]:
        row = re.search(rf"\| {k} \| ([a-z ]+) \|", real_report).group(1)
        sec = re.search(rf"### {k}\n.*?\*\*Status: ([a-z ]+)\.\*\*", real_report, re.S).group(1)
        assert row == sec, k
    # H8 pairs a 4-bit Clef with an 8-bit Clef-flash of the same size; no pair of this sweep's builds is that close
    assert re.search(r"\| H8 \| not evaluable \|", real_report)
    assert "not evaluable" in FI.STATUSES and "not yet evaluable" in FI.STATUSES
    # H2 is whatever the written rule computes from the analyzer output over both models
    pa = json.loads((REAL / FI.ALIASES["PUBALL"]).read_text())["models"]
    vs = lambda m, c: pa[m]["configs"][c]["pooled"]["vs_ref"]  # noqa: E731
    cfgs = ("nvfp4", "q4_k_m")
    first = [vs(m, c)["delta_accuracy"]["ci95"][1] < 0 for m in ("clef-flash", "clef") for c in cfgs]
    second = [vs("clef", c)["n_flips"] < vs("clef-flash", c)["n_flips"] for c in cfgs]
    want = "supported" if all(first) and all(second) else ("not supported" if not any(first) or not any(second) else "mixed")
    assert re.search(rf"\| H2 \| {want} \|", real_report)


@needs_real
def test_every_citation_resolves_to_a_key_in_its_source_file(real_report):
    F = FI.Facts(REAL)
    cites = re.findall(r"\[(PUBALL|PUB|PRIVA|DEC|HYP|PRIV|SPD|MAN) ([^\]]+)\]", real_report)
    assert any(a == "PUBALL" for a, _ in cites)
    assert len(cites) > 200
    checked = 0
    for alias, path in cites:
        if "<" in path:                      # placeholders like <set>
            continue
        first, *rest = path.split(",")
        F.v(alias, first)
        base = first.rsplit(".", 1)[0]
        for q in rest:
            F.v(alias, f"{base}.{q}")
        checked += 1
    assert checked > 180


@needs_real
def test_report_has_no_unrendered_values_and_no_private_identifiers(real_report):
    for bad in (r"\bnan\b", r"\bNone\b", r"\binf\b", r"[{}]"):
        assert re.search(bad, real_report) is None, bad
    priv = REAL / "private" / "records"
    if priv.exists():
        ids = set()
        for name in ("r4a", "r4b"):
            with open(priv / f"{name}.jsonl") as fh:
                for _, line in zip(range(300), fh):
                    ids.add(json.loads(line)["id"])
        assert not [i for i in ids if i in real_report]


@needs_real
def test_findings_cli_is_deterministic(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    assert FI.main(["--root", str(REAL), "--out", str(a)]) == 0
    assert FI.main(["--root", str(REAL), "--out", str(b)]) == 0
    assert a.read_text() == b.read_text()


H2_RULE_ORIGINAL = "supported only if the first clause holds and the 27B 4-bit flip count on the same records is smaller than Clef-flash's."
H8_RULE_ORIGINAL = ("needs 27B 4-bit accuracy on the same records and a 4-bit Clef / 8-bit Clef-flash pair within the analyzer's "
                    "15% size window.")


def section(report, key):
    return re.search(rf"### {key}\n.*?(?=\n### |\n## )", report, re.S).group(0)


@needs_real
def test_h2_and_h8_keep_the_original_rule_sentence_and_append_the_dated_reading(real_report):
    h2, h8 = section(real_report, "H2"), section(real_report, "H8")
    assert f"Rule: {H2_RULE_ORIGINAL} Reading added 2026-10-23, after the 27B results were seen: the first clause is checked" in h2
    assert "Private flips are reported beside the rule and are not part of it." in h2
    assert f"Rule: {H8_RULE_ORIGINAL} No 4-bit Clef build lies within that window of an 8-bit Clef-flash build" in h8
    assert "the supplementary comparison is outside the window and is not pre-registered." in h8


@needs_real
def test_h2_reports_both_models_for_both_4bit_builds_and_the_private_flips(real_report):
    pa = json.loads((REAL / FI.ALIASES["PUBALL"]).read_text())["models"]
    pr = json.loads((REAL / FI.ALIASES["PRIV"]).read_text())
    h2 = section(real_report, "H2")
    for c in ("nvfp4", "q4_k_m"):
        big = pa["clef"]["configs"][c]["pooled"]["vs_ref"]["n_flips"]
        small = pa["clef-flash"]["configs"][c]["pooled"]["vs_ref"]["n_flips"]
        assert f"{big:,} [PUBALL models.clef.configs.{c}.pooled.vs_ref.n_flips]" in h2
        assert f"{small:,} [PUBALL models.clef-flash.configs.{c}.pooled.vs_ref.n_flips]" in h2
        fb = sum(pr["label_free_big"][s]["pooled"]["pairs"][f"ref|{c}"]["flips"] for s in FI.SETS)
        ff = sum(pr["label_free"][s]["pooled"]["pairs"][f"ref|{c}"]["flips"] for s in FI.SETS)
        assert f"Clef (27B) {fb:,} of " in h2 and f"vs Clef-flash {ff:,} of " in h2
        assert f"[PRIV label_free_big.<set>.pooled.pairs.ref|{c}.flips]" in h2
    assert "**Status: mixed.**" in h2 and "fails for Clef (27B) Q4_K_M" in h2 and "fails for NVFP4 (1,502 vs 1,426)" in h2


@needs_real
def test_h8_bullets_state_sizes_window_supplementary_pairs_and_the_inside_window_pairs(real_report):
    hy = json.loads((REAL / FI.ALIASES["HYP"]).read_text())["cross_size_4bit_vs_8bit"]
    pa = json.loads((REAL / FI.ALIASES["PUBALL"]).read_text())["equal_memory"]
    h8 = section(real_report, "H8")
    assert "Window: 15% [PUBALL equal_memory.tolerance]" in h8
    assert h8.count("Supplementary, not pre-registered, outside the size window:") == 4
    for key, row in hy["pairs"].items():
        d = row["pooled_hard"]["delta_accuracy_big_minus_small"]["delta"]
        assert f"{100 * d:+.2f} pp" in h8 and f"pairs.{key}.hard_minus_easy.delta,ci95" in h8
    assert "The split was chosen after the 27B results were seen." in h8
    assert ", ".join(hy["difficulty"]["hard"]) in h8 and ", ".join(hy["difficulty"]["easy"]) in h8
    inside = [i for i, p in enumerate(pa["pairs"]) if p["a"]["config"] in ("nvfp4", "q4_k_m") and p["b"]["config"] in ("ref", "bf16-srv", "gguf-bf16")]
    assert len(inside) == 6 and h8.count("Inside the window:") == 6
    assert all(f"equal_memory.pairs.{i}.delta_accuracy_b_minus_a.delta,ci95" in h8 for i in inside)
    assert "ref-fla" in h8 and "which H8 does not ask about" in h8
    assert not any(f"equal_memory.pairs.{i}." in h8 for i, p in enumerate(pa["pairs"]) if p["b"]["config"] == "ref-fla")


@needs_real
def test_h8_sensitivity_bullet_ranges_and_pattern_check_are_computed_from_the_json(real_report):
    hy = json.loads((REAL / FI.ALIASES["HYP"]).read_text())["cross_size_4bit_vs_8bit"]
    sens = hy["difficulty_sensitivity"]
    h8 = section(real_report, "H8")
    bullets = [x for x in h8.split("\n") if x.startswith("- Sensitivity of the split")]
    assert len(bullets) == 1
    bullet = bullets[0]
    assert "\u2014" not in bullet and "\u2013" not in bullet
    assert "robustness checks of a post-hoc split and are not pre-registered" in bullet
    assert "[HYP cross_size_4bit_vs_8bit.difficulty_sensitivity]" in bullet
    for name, e in sens.items():
        assert f"{name} ({e['basis']}: hard = {', '.join(e['hard'])}; easy = {', '.join(e['easy'])}" in bullet
        if "same_as" in e:
            assert f"the same hard and easy sets as {e['same_as']}" in bullet
        else:
            assert "the same hard and easy sets as" not in bullet.split(f"{name} (")[1].split("); ")[0]
    cells = []
    for key in hy["pairs"]:
        for g in ("hard", "easy"):
            vals = {n: e["pairs"][key][f"pooled_{g}"]["delta_accuracy_big_minus_small"]["delta"] for n, e in sens.items()}
            lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
            for end in (lo, hi):
                cite = f"[HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.{end}.pairs.{key}.pooled_{g}.delta_accuracy_big_minus_small.delta]"
                assert f"{100 * vals[end]:+.2f} pp {cite} ({end})" in bullet
        for n, e in sens.items():
            r = e["pairs"][key]
            h, ez = (r[f"pooled_{g}"]["delta_accuracy_big_minus_small"]["delta"] for g in ("hard", "easy"))
            cells.append(h > 0 and ez <= h)
    assert (f"holds in all {len(cells)} pair and split combinations" in bullet) == all(cells)
    assert (f"fails in {cells.count(False)} of {len(cells)} pair and split combinations" in bullet) == (not all(cells))


def test_h8_sensitivity_pattern_check_reports_a_failure_when_the_data_break_the_pattern(tmp_path):
    if not all((REAL / p).exists() for p in ("PLAN.md", *FI.ALIASES.values())):
        pytest.skip("real analysis outputs not present")
    _copy_inputs(tmp_path)
    path = tmp_path / FI.ALIASES["HYP"]
    data = json.loads(path.read_text())
    sens = data["cross_size_4bit_vs_8bit"]["difficulty_sensitivity"]
    first = next(iter(sens))
    key = "clef/q4_k_m|clef-flash/q8_0"
    sens[first]["pairs"][key]["pooled_easy"]["delta_accuracy_big_minus_small"]["delta"] = 0.5      # easy above hard: the pattern breaks
    path.write_text(json.dumps(data))
    h8 = section(FI.build_report(tmp_path), "H8")
    assert f"fails in 1 of {3 * 4} pair and split combinations: Q4_K_M vs Q8_0 under {first}." in h8
    assert "holds in all" not in h8


@needs_real
def test_27b_section_is_data_driven_has_the_quantization_subsection_and_no_stale_sentences(real_report):
    sec = re.search(r"## 27B\n.*?(?=\n## )", real_report, re.S).group(0)
    assert "## 27B so far" not in real_report and "### 27B quantization" in sec
    for stale in ("`ref` only", "no quantization claim is made for the 27B", "excludes clinc150", "no 4-bit 27B config has run",
                  "only the BF16 reference", "No clinc150 yet", "no quantized config", "Clef (27B) has only its BF16 reference"):
        assert stale not in real_report, stale
    assert "both figures cover the same tasks" in sec
    pa = json.loads((REAL / FI.ALIASES["PUBALL"]).read_text())["models"]
    for c in ("bf16-srv", "gguf-bf16", "fp8", "q8_0", "nvfp4", "q4_k_m"):
        row = re.search(rf"\| {c} \| ([^|]*) \| ([^|]*) \| ([\d,]+) \(", sec)
        assert row, c
        assert row.group(3) == f"{pa['clef']['configs'][c]['pooled']['vs_ref']['n_flips']:,}"
    assert "| ref-fla |" not in sec
    for q in ("r4a / choice5", "r4a / score5", "r4b / choice3"):
        assert re.search(rf"Private {q}: change in agreement with the hosted label, 27B config against the 27B ref \(paired\): fp8: .*q4_k_m: ", sec)


@needs_real
def test_scope_and_27b_caveat_are_true_of_the_data(real_report):
    pa = json.loads((REAL / FI.ALIASES["PUBALL"]).read_text())
    assert "has only its BF16 reference" not in real_report and "11 of 12" not in real_report
    scope = re.search(r"\nScope: (.*)\n", real_report).group(1)
    assert f"covers all {len(pa['meta']['tasks_with_records'])} public tasks" in scope
    assert "only 2,400 public questions with ref-fla" in scope
    cav = re.search(r"\n9\. \*\*27B scope\.\*\* (.*)\n", real_report).group(1)
    assert "ref-fla only on 2,400 public questions" in cav and "File sizes are recorded for every 27B config" in cav
    assert "`analysis/all_public/` is the same analyzer over both models and is the source of every 27B quantization number" in real_report
    assert "python -m bench.analyze --records data/records --results results --out analysis/all_public\n" in real_report


def _copy_inputs(dst: Path) -> None:
    for rel in ["PLAN.md", *FI.ALIASES.values()]:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REAL / rel, dst / rel)


@needs_real
def test_h8_raises_when_the_analyzer_lists_a_4bit_27b_and_8bit_clef_flash_pair(tmp_path):
    _copy_inputs(tmp_path)
    path = tmp_path / FI.ALIASES["PUBALL"]
    data = json.loads(path.read_text())
    data["equal_memory"]["pairs"].append(
        {"a": {"model": "clef", "config": "nvfp4", "file_size_gb": 20.0}, "b": {"model": "clef-flash", "config": "fp8", "file_size_gb": 19.0},
         "size_ratio": 20.0 / 19.0, "n_items": 31275, "tasks": [], "accuracy_a": 0.8, "accuracy_b": 0.8,
         "delta_accuracy_b_minus_a": {"delta": 0.0, "ci95": [0.0, 0.0]}})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="within the 15% size window"):
        FI.build_report(tmp_path)


@needs_real
def test_h8_raises_when_the_recorded_file_sizes_put_a_pair_inside_the_window(tmp_path):
    _copy_inputs(tmp_path)
    path = tmp_path / FI.ALIASES["SPD"]
    data = json.loads(path.read_text())
    data["configs"]["clef"]["q4_k_m"]["meta"]["file_size_gb"] = 12.5      # 1.03 x the Clef-flash FP8 file (12.154 GB)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="status rule for that case is not written"):
        FI.build_report(tmp_path)
