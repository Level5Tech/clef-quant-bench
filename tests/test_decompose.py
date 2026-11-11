"""Tests of bench.decompose on small synthetic result trees with hand-checkable flip counts."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from bench import analyze as A
from bench import decompose as D
from bench import metrics as M

UNIT = 20            # records per task in the unit (the real unit is 200)
FULL = 30            # records per task in the full records files (extras beyond the unit must be ignored)
BASE = np.array([2.0, 1.0, 0.0])      # choice: argmax is option "a"
FLIPPED = np.array([0.0, 2.0, 1.0])   # choice: argmax is option "b"
NOUL_BASE = np.array([1.0, 0.0])      # noul options ["true", "false"]
NOUL_FLIPPED = np.array([0.0, 1.0])


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def make_records(root: Path):
    rec_dir = root / "records"
    rec_dir.mkdir(parents=True)
    for task, qtype in (("toy_a", "choice"), ("toy_b", "noul")):
        rows = []
        for i in range(FULL):
            q = ({"type": "choice", "instructions": "pick", "criteria": {"c": "3", "a": "1", "b": "2"}}
                 if qtype == "choice" else {"type": "noul", "instructions": "yes or no"})
            rows.append({"id": f"{task}-{i}", "task": task,
                         "request": {"model": "clef-flash", "state": {"t": i}, "questions": {"q": q}},
                         "gold": {"q": "a" if qtype == "choice" else "true"}})
        (rec_dir / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rec_dir


# unit-record indices 0..19 are toy_a, 20..39 are toy_b.
FLIP_SETS = {
    "ref": set(), "ref-r2": {30}, "ref-fla": {31, 32},
    "bf16-srv": {0}, "bf16-srv-r2": {0}, "fp8": {0, 1, 2, 3, 4}, "fp8-r2": {0, 1, 2, 3, 4, 6},
    "nvfp4": set(range(12)), "nvfp4-r2": set(range(12)) | {25},
    "gguf-bf16": {2}, "gguf-bf16-r2": {2}, "q8_0": {2, 7, 8}, "q4_k_m": {2} | set(range(7, 19)),
}
JITTER = {"bf16-srv-r2": {5}}       # tiny logit change, no flip
LLAMA = {"gguf-bf16", "gguf-bf16-r2", "q8_0", "q4_k_m"}
SUBSET = {"bf16-srv-r2", "fp8-r2", "nvfp4-r2", "gguf-bf16-r2", "ref-fla"}


def unit_index(task: str, i: int) -> int:
    return i if task == "toy_a" else UNIT + i


def write_cfg(root: Path, cfg: str, model: str = "clef-flash", flip_sets=None):
    flip_sets = FLIP_SETS if flip_sets is None else flip_sets
    for task in ("toy_a", "toy_b"):
        n = UNIT if cfg in SUBSET else FULL
        rows = []
        for i in range(n):
            ui = unit_index(task, i)
            flipped = ui in flip_sets[cfg]
            if task == "toy_a":
                opts = ["a", "b", "c"]
                lg = FLIPPED.copy() if flipped else BASE.copy()
            else:
                opts = ["true", "false"]
                lg = NOUL_FLIPPED.copy() if flipped else NOUL_BASE.copy()
            if ui in JITTER.get(cfg, set()):
                lg = lg + np.array([0.01] + [0.0] * (len(lg) - 1))
            ans = {"option_ids": opts, "logits": None if cfg in LLAMA else lg.tolist(), "probs": softmax(lg).tolist()}
            rows.append({"id": f"{task}-{i}", "task": task, "model": model, "config": cfg,
                         "answers": {"q": ans}, "latency_ms": 1.0, "input_tokens": 5, "engine": "toy",
                         "engine_version": "1", "error": None})
        d = root / "results" / model / cfg
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("decomp")
    rec = make_records(root)
    for cfg in FLIP_SETS:
        write_cfg(root, cfg)
    return rec, root / "results"


@pytest.fixture(scope="module")
def res(tree):
    rec, results = tree
    return D.decompose(rec, results, "clef-flash", unit_n=UNIT, n_boot=400)


# ----------------------------------------------------------------------------- unit

def test_unit_is_first_n_lines_of_every_task_in_file_order(tree):
    rec, _ = tree
    u = D.build_unit(rec, UNIT)
    assert u.n_records == 2 * UNIT and u.n_questions == 2 * UNIT
    assert u.rec_ids[:3] == ["toy_a-0", "toy_a-1", "toy_a-2"] and u.rec_ids[UNIT] == "toy_b-0"
    assert "toy_a-20" not in u.rec_ids  # records beyond the unit are not in it
    assert u.tasks == ["toy_a", "toy_b"]


def test_unit_needs_enough_records(tree):
    rec, _ = tree
    with pytest.raises(ValueError, match="need"):
        D.build_unit(rec, FULL + 1)


def test_records_outside_the_unit_are_counted_and_ignored(tree, res):
    rec, results = tree
    u = D.build_unit(rec, UNIT)
    full = D.load_cfg(u, results, "clef-flash", "ref")           # holds FULL records per task, unit uses UNIT of them
    assert full.n_lines == 2 * FULL and full.n_outside_unit == 2 * (FULL - UNIT)
    sub = D.load_cfg(u, results, "clef-flash", "ref-fla")        # a subset config: exactly the unit
    assert sub.n_lines == 2 * UNIT and sub.n_outside_unit == 0
    cond = res["conditions"]
    assert cond["ref-fla"]["result_lines_outside_unit_ignored"] == 0
    assert cond["ref"]["result_lines_outside_unit_ignored"] == 2 * (FULL - UNIT)


def test_load_cfg_ignores_extra_records_and_reorders_options(tree, tmp_path):
    rec, results = tree
    u = D.build_unit(rec, UNIT)
    cfg = D.load_cfg(u, results, "clef-flash", "ref")
    assert len(cfg.p) == 2 * UNIT
    np.testing.assert_allclose(cfg.p[0], softmax(BASE))
    # shuffle option order in a copy: the loader must map back to record (sorted) order
    d = tmp_path / "results" / "clef-flash" / "ref"
    d.mkdir(parents=True)
    for task in ("toy_a", "toy_b"):
        rows = [json.loads(l) for l in (results / "clef-flash" / "ref" / f"{task}.jsonl").read_text().splitlines()]
        for r in rows:
            a = r["answers"]["q"]
            a["option_ids"] = a["option_ids"][::-1]
            a["logits"] = a["logits"][::-1]
            a["probs"] = a["probs"][::-1]
        (d / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg2 = D.load_cfg(u, tmp_path / "results", "clef-flash", "ref")
    for x, y in zip(cfg.p, cfg2.p):
        np.testing.assert_allclose(x, y)


def _copy_cfg(results: Path, dst_root: Path, mutate):
    d = dst_root / "results" / "clef-flash" / "ref"
    d.mkdir(parents=True, exist_ok=True)
    for task in ("toy_a", "toy_b"):
        rows = [json.loads(l) for l in (results / "clef-flash" / "ref" / f"{task}.jsonl").read_text().splitlines()]
        rows = mutate(task, rows)
        (d / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return dst_root / "results"


def test_load_cfg_fails_loudly_on_missing_duplicate_error_and_option_mismatch(tree, tmp_path):
    rec, results = tree
    u = D.build_unit(rec, UNIT)

    def drop(task, rows):
        return [r for r in rows if r["id"] != "toy_a-3"]
    with pytest.raises(ValueError, match="missing"):
        D.load_cfg(u, _copy_cfg(results, tmp_path / "m", drop), "clef-flash", "ref")

    def dup(task, rows):
        return rows + [rows[0]]
    with pytest.raises(ValueError, match="duplicate"):
        D.load_cfg(u, _copy_cfg(results, tmp_path / "d", dup), "clef-flash", "ref")

    def err(task, rows):
        rows[0]["error"] = "boom"
        return rows
    with pytest.raises(ValueError, match="unusable"):
        D.load_cfg(u, _copy_cfg(results, tmp_path / "e", err), "clef-flash", "ref")

    def badopt(task, rows):
        rows[0]["answers"]["q"]["option_ids"] = ["x", "y", "z"][: len(rows[0]["answers"]["q"]["option_ids"])]
        return rows
    with pytest.raises(ValueError, match="option ids"):
        D.load_cfg(u, _copy_cfg(results, tmp_path / "o", badopt), "clef-flash", "ref")


# ----------------------------------------------------------------------- pair numbers

def sym(a: str, b: str) -> int:
    return len(FLIP_SETS[a] ^ FLIP_SETS[b])


def test_flip_counts_equal_symmetric_difference_of_flip_sets(res):
    for key, p in res["pairs"].items():
        a, b = key.split("|")
        assert p["flips"] == sym(a, b), key
        assert p["n_questions"] == 2 * UNIT and p["n_records"] == 2 * UNIT
        assert p["flip_rate"] == pytest.approx(p["flips"] / (2 * UNIT))
        lo, hi = p["flip_ci95"]
        assert lo <= p["flip_rate"] <= hi or p["flips"] in (0,)


def test_every_requested_pair_is_present(res):
    keys = set(res["pairs"])
    for _, a, b, _ in D.PAIRS:
        assert D.pk(a, b) in keys
    for c in D.VS_REF:
        assert D.pk("ref", c) in keys
    assert len(keys) == len(D.PAIRS) + len(D.VS_REF) - 4  # four vs-ref pairs are also named pairs


def test_tvd_max_dprob_and_dlogit_hand_computed(res):
    p = res["pairs"]["ref|bf16-srv"]  # one choice record flipped from BASE to FLIPPED (record 0)
    tv_choice = 0.5 * np.abs(softmax(BASE) - softmax(FLIPPED)).sum()
    assert p["tvd_mean"] == pytest.approx(tv_choice / (2 * UNIT))
    assert p["max_dprob"] == pytest.approx(np.abs(softmax(BASE) - softmax(FLIPPED)).max())
    d = BASE - FLIPPED
    assert p["max_dlogit_raw"] == pytest.approx(np.abs(d).max())
    assert p["max_dlogit_centered"] == pytest.approx(np.abs(d - d.mean()).max())
    assert p["max_dprob_at"] == {"task": "toy_a", "id": "toy_a-0"}
    assert p["identical_basis"] == "logits"
    assert p["n_identical_records"] == 2 * UNIT - 1


def test_noul_flip_tvd_is_two_option_distance(res):
    p = res["pairs"]["ref|ref-r2"]  # record index 30 -> toy_b-10 flipped
    assert p["flips"] == 1
    assert p["max_dprob_at"] == {"task": "toy_b", "id": "toy_b-10"}
    assert p["tvd_mean"] == pytest.approx(abs(softmax(NOUL_BASE)[0] - softmax(NOUL_FLIPPED)[0]) / (2 * UNIT))


def test_centered_dlogit_ignores_constant_shift(tmp_path):
    rec = make_records(tmp_path)
    for cfg in ("ref", "bf16-srv"):
        write_cfg(tmp_path, cfg, flip_sets={"ref": set(), "bf16-srv": set()})
    # shift every logit of bf16-srv by +3: softmax unchanged, raw dlogit = 3, centred = 0
    d = tmp_path / "results" / "clef-flash" / "bf16-srv"
    for task in ("toy_a", "toy_b"):
        rows = [json.loads(l) for l in (d / f"{task}.jsonl").read_text().splitlines()]
        for r in rows:
            r["answers"]["q"]["logits"] = [x + 3.0 for x in r["answers"]["q"]["logits"]]
        (d / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    u = D.build_unit(rec, UNIT)
    arr = D.pair_arrays(D.load_cfg(u, tmp_path / "results", "clef-flash", "ref"),
                        D.load_cfg(u, tmp_path / "results", "clef-flash", "bf16-srv"))
    assert arr.dlogit_raw.max() == pytest.approx(3.0)
    assert arr.dlogit_ctr.max() == pytest.approx(0.0, abs=1e-12)
    assert arr.flip.sum() == 0 and arr.dprob.max() < 1e-12
    assert not arr.identical.any()  # raw logits differ even though the decisions do not


def test_bit_identical_uses_probs_when_logits_are_null(res):
    g = res["pairs"]["gguf-bf16|gguf-bf16-r2"]
    assert g["identical_basis"] == "probs"
    assert g["n_identical_records"] == 2 * UNIT and g["share_identical"] == 1.0
    assert g["max_dlogit_raw"] is None and g["max_dlogit_centered"] is None
    assert g["max_dprob"] == 0.0 and g["tvd_mean"] == 0.0
    # a pair of a logits-config against a probs-config falls back to probs and is never identical here
    assert res["pairs"]["ref|gguf-bf16"]["identical_basis"] == "probs"


def test_jitter_is_not_a_flip_but_breaks_bit_identity(res):
    p = res["pairs"]["bf16-srv|bf16-srv-r2"]
    assert p["flips"] == 0
    assert p["n_identical_records"] == 2 * UNIT - 1
    assert p["tvd_mean"] > 0
    assert p["flip_ci95"] == [0.0, 0.0]
    lo, hi = p["flip_ci95_exact_cp"]
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / (2 * UNIT)))


def test_clopper_pearson_known_values():
    lo, hi = D.clopper_pearson(5, 100)
    assert lo == pytest.approx(0.01643, abs=1e-4) and hi == pytest.approx(0.11283, abs=1e-4)
    assert D.clopper_pearson(0, 2400)[1] == pytest.approx(1 - 0.025 ** (1 / 2400))
    assert D.clopper_pearson(10, 10) == [pytest.approx(0.6915, abs=1e-3), 1.0]


def test_flips_by_task_sum_to_total(res):
    for p in res["pairs"].values():
        assert sum(p["flips_by_task"].values()) == p["flips"]
    assert res["pairs"]["ref|ref-fla"]["flips_by_task"] == {"toy_a": 0, "toy_b": 2}


# ------------------------------------------------------------------------- bootstrap

def test_bootstrap_is_deterministic_and_seeded(tree):
    rec, _ = tree
    u = D.build_unit(rec, UNIT)
    b1, b2, b3 = D.Boot(u, 100, 7), D.Boot(u, 100, 7), D.Boot(u, 100, 8)
    np.testing.assert_array_equal(b1.idx, b2.idx)
    assert not np.array_equal(b1.idx, b3.idx)
    assert b1.idx.shape == (100, 2 * UNIT)


def test_stratified_resamples_stay_inside_their_task(tree):
    rec, _ = tree
    u = D.Boot(D.build_unit(rec, UNIT), 50, 3)
    s = u.idx_strat
    assert s[:, :UNIT].max() < UNIT and s[:, UNIT:].min() >= UNIT


def test_ratio_estimator_edge_cases(tree):
    rec, _ = tree
    u = D.build_unit(rec, UNIT)
    b = D.Boot(u, 60, 1)
    assert np.all(b.ratio(np.ones(u.n_questions)) == 1.0)
    assert np.all(b.ratio(np.zeros(u.n_questions)) == 0.0)
    x = np.zeros(u.n_questions)
    x[:4] = 1
    assert b.ratio(x).mean() == pytest.approx(4 / u.n_questions, abs=0.05)


def test_difference_is_paired_on_the_same_resamples(tree):
    rec, results = tree
    u = D.build_unit(rec, UNIT)
    b = D.Boot(u, 300, 11)
    cfgs = {c: D.load_cfg(u, results, "clef-flash", c) for c in ("ref", "bf16-srv", "fp8")}
    a = D.pair_arrays(cfgs["bf16-srv"], cfgs["fp8"])
    same = D.diff_test(b, a, a, "x", "x")
    assert same["diff"] == 0.0 and same["ci95"] == [0.0, 0.0] and not same["excludes_zero"]
    zero = D.pair_arrays(cfgs["ref"], cfgs["ref"])
    d = D.diff_test(b, a, zero, "x", "none")
    assert d["diff"] == pytest.approx(4 / (2 * UNIT))
    # with only 4 flips among 40 records the bootstrap difference equals the bootstrap flip rate of `a`
    np.testing.assert_allclose(sorted(b.ratio(a.flip) - b.ratio(zero.flip)), sorted(b.ratio(a.flip)))
    assert d["lower_gt_zero"] == (d["ci95"][0] > 0)
    assert d["flips_minuend"] == 4 and d["flips_subtrahend"] == 0


def test_bonferroni_interval_is_wider(tree):
    rec, results = tree
    u = D.build_unit(rec, UNIT)
    b = D.Boot(u, 1000, 11)
    cfgs = {c: D.load_cfg(u, results, "clef-flash", c) for c in ("ref", "nvfp4", "bf16-srv")}
    q = D.pair_arrays(cfgs["bf16-srv"], cfgs["nvfp4"])
    s = D.pair_arrays(cfgs["ref"], cfgs["bf16-srv"])
    d = D.diff_test(b, q, s, "q", "s", k=16)
    assert d["ci95_bonferroni"][0] <= d["ci95"][0] and d["ci95_bonferroni"][1] >= d["ci95"][1]
    assert d["bonferroni_k"] == 16


# ----------------------------------------------------------------------- pre-registered

def test_n_tests_follow_their_stated_criteria_on_the_synthetic_tree(res):
    t = res["tests"]
    n1 = t["N1"]
    assert (n1["flips_minuend"], n1["flips_subtrahend"]) == (4, 0)
    assert n1["supported"] == (not n1["lower_gt_zero"])
    assert n1["supplementary_not_preregistered"]["flips_subtrahend"] == sym("fp8", "fp8-r2")
    n2 = t["N2"]
    assert n2["family_size"] == 16 and len(n2["rows"]) == 16
    assert {r["minuend"] for r in n2["rows"]} == set(D.FOUR_BIT)
    assert {r["subtrahend"] for r in n2["rows"]} == set(D.NON_QUANT_SOURCES)
    assert n2["supported"] == all(r["lower_gt_zero"] for r in n2["rows"])
    n3 = t["N3"]
    assert len(n3["rows"]) == 4
    # engine change (ref vs bf16-srv: 1 flip) minus FP8 (4 flips) is negative here
    row = [r for r in n3["rows"] if (r["minuend"], r["subtrahend"]) == ("ref|bf16-srv", "bf16-srv|fp8")][0]
    assert row["diff"] == pytest.approx((1 - 4) / (2 * UNIT))
    assert n3["all_point_ge_zero"] is False and n3["supported"] is False
    n4 = t["N4"]
    assert n4["ref_fla_flips"] == 2 and n4["nonzero"] is True
    assert n4["vllm_restart_flips"] == {"bf16-srv|bf16-srv-r2": 0, "fp8|fp8-r2": 1, "nvfp4|nvfp4-r2": 1}
    assert n4["vllm_restart_range"] == [0, 1]
    assert n4["within_vllm_restart_range"] is False and n4["supported"] is False  # 2 flips > range max 1
    n5 = t["N5"]
    assert n5["supported"] is True and n5["n_identical_records"] == n5["n_records"]


def test_n4_is_supported_when_kernel_flips_fall_inside_the_vllm_range(tmp_path):
    rec = make_records(tmp_path)
    flips = dict(FLIP_SETS)
    flips["ref-fla"] = {31}  # one flip: inside the vLLM restart range [0, 1]
    for cfg in flips:
        write_cfg(tmp_path, cfg, flip_sets=flips)
    r = D.decompose(rec, tmp_path / "results", "clef-flash", unit_n=UNIT, n_boot=100)
    assert r["tests"]["N4"]["ref_fla_flips"] == 1
    assert r["tests"]["N4"]["within_vllm_restart_range"] is True and r["tests"]["N4"]["supported"] is True


def test_n4_is_not_supported_when_the_kernel_changes_nothing(tmp_path):
    rec = make_records(tmp_path)
    flips = dict(FLIP_SETS)
    flips["ref-fla"] = set()
    for cfg in flips:
        write_cfg(tmp_path, cfg, flip_sets=flips)
    r = D.decompose(rec, tmp_path / "results", "clef-flash", unit_n=UNIT, n_boot=100)
    assert r["tests"]["N4"]["nonzero"] is False and r["tests"]["N4"]["supported"] is False


def test_n5_fails_when_one_record_differs(tmp_path):
    rec = make_records(tmp_path)
    for cfg in FLIP_SETS:
        write_cfg(tmp_path, cfg)
    d = tmp_path / "results" / "clef-flash" / "gguf-bf16-r2"
    rows = [json.loads(l) for l in (d / "toy_a.jsonl").read_text().splitlines()]
    rows[4]["answers"]["q"]["probs"][0] += 1e-9
    (d / "toy_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = D.decompose(rec, tmp_path / "results", "clef-flash", unit_n=UNIT, n_boot=100)
    n5 = r["tests"]["N5"]
    assert n5["n_identical_records"] == 2 * UNIT - 1 and n5["supported"] is False and n5["flips"] == 0


def test_decompose_is_deterministic_and_markdown_renders(tree):
    rec, results = tree
    a = D.decompose(rec, results, "clef-flash", unit_n=UNIT, n_boot=100)
    b = D.decompose(rec, results, "clef-flash", unit_n=UNIT, n_boot=100)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    md = D.render_md(a)
    for needle in ("N1 FP8 inside restart noise", "N2 NVFP4", "N3 engine change", "N4 kernel", "N5 llama.cpp",
                   "rerun AND a batch-size change"):
        assert needle in md


def test_cli_writes_json_and_markdown(tree, tmp_path):
    rec, results = tree
    out = tmp_path / "out"
    assert D.main(["--records", str(rec), "--results", str(results), "--out", str(out),
                   "--unit-n", str(UNIT), "--n-boot", "50"]) == 0
    assert json.loads((out / "decomposition.json").read_text())["meta"]["n_records"] == 2 * UNIT
    assert (out / "decomposition.md").read_text().startswith("# Noise-floor decomposition")


# --------------------------------------------------- cross-check against the analyzer

REAL = Path(__file__).resolve().parent.parent
needs_real = pytest.mark.skipif(not (REAL / "results" / "clef-flash" / "ref-fla").is_dir(),
                                reason="real results not present")


@needs_real
def test_real_flip_counts_agree_with_the_analyzers_pairing():
    """Independent path: analyze.build_items / make_pair on the real subset files give the same per-task flips."""
    rec_dir, res_dir = REAL / "data" / "records", REAL / "results" / "clef-flash"
    u = D.build_unit(rec_dir, D.UNIT_N)
    out = D.decompose(rec_dir, REAL / "results", "clef-flash", n_boot=50, configs=D.ALL_CONFIGS)
    records = A.load_records(rec_dir)
    first = {t: set(u.rec_ids[i] for i in range(u.n_records) if u.rec_task[i] == t) for t in u.tasks}
    for base, other in [("fp8", "fp8-r2"), ("ref", "ref-fla"), ("bf16-srv", "bf16-srv-r2"), ("gguf-bf16", "gguf-bf16-r2")]:
        per_task = {}
        for t in u.tasks:
            ia, _ = A.build_items(t, records[t], A.read_jsonl(res_dir / base / f"{t}.jsonl"))
            ib, _ = A.build_items(t, records[t], A.read_jsonl(res_dir / other / f"{t}.jsonl"))
            keys = [k for k in ia.keys() & ib.keys() if k[0] in first[t]]
            assert len(keys) == D.UNIT_N
            per_task[t] = sum(ia[k].pred != ib[k].pred for k in keys)
        assert per_task == out["pairs"][D.pk(base, other)]["flips_by_task"], (base, other)
