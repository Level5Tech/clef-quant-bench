#!/usr/bin/env python
"""Build the Clef quantization benchmark datasets from public Hugging Face sources.

Outputs (default --out data):
  data/records/<task>.jsonl   scored records (full eval split, capped at CAP per task)
  data/calibration.jsonl      ~768 records from TRAIN splits only, mixed across tasks
  data/record_ids.jsonl       source split + row index/id for every record (ids-only publishing)
  data/manifest.json          counts, label distributions, pinned revisions, file hashes

Reproducibility: every source is loaded at a pinned dataset revision (commit hash) and every
random choice derives from SEED, so two runs produce byte-identical files.

Usage (from the project root):
  .venv/bin/python datasets/build.py                  # full build
  .venv/bin/python datasets/build.py --limit 200      # dev subset (first 200 of each task's order)
  .venv/bin/python datasets/build.py --tasks boolq,sst5 --out /tmp/x

Record order: each task's records are ordered so that EVERY PREFIX is approximately label-stratified
(see stratified_order). The cap and --limit both take prefixes, so the dev subset is a stratified
subset of the full set and has the same ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # append: never shadow the Hugging Face `datasets` package

from bench.schema import validate_record  # noqa: E402

SEED = 20261015
CAP = 5000  # max scored records per task; stratified when the eval split is larger
CALIBRATION_TOTAL = 768
QID = "answer"  # question id used by every record
MODEL = "clef"  # request.model; runners override it per model


# --------------------------------------------------------------------------- sources

@dataclass(frozen=True)
class Source:
    hf_id: str
    revision: str  # pinned dataset commit hash
    config: str | None
    license: str  # as declared on the dataset card
    release: str  # "records" | "ids_only": how derived records are published


SOURCES: dict[str, Source] = {
    "banking77": Source("legacy-datasets/banking77", "f54121560de48f2852f90be299010d1d6dc612ec", None, "cc-by-4.0", "records"),
    "clinc150": Source("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec", "plus", "cc-by-3.0", "records"),
    "anli": Source("facebook/anli", "8e4813d81f46d313dac7892e1c28076917cfcdf9", None, "cc-by-nc-4.0", "ids_only"),
    "contractnli": Source("kiddothe2b/contract-nli", "059927b91122a6827e7dbb4f296f6da8f5dcee1c", None, "cc-by-nc-sa-4.0", "ids_only"),
    "mmlu": Source("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all", "mit", "records"),
    "arc_challenge": Source("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge", "cc-by-sa-4.0", "records"),
    "boolq": Source("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5", None, "cc-by-sa-3.0", "records"),
    "ragtruth": Source("wandb/RAGTruth-processed", "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b", None, "mit", "ids_only"),
    "when2call": Source("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53", None, "cc-by-4.0", "records"),
    "sst5": Source("SetFit/sst5", "e51bdcd8cd3a30da231967c1a249ba59361279a3", None, "none declared", "ids_only"),
}


# ------------------------------------------------------------------ split abstraction

@dataclass
class SplitData:
    labels: list[str]  # gold option id per source row (used for stratification and gold)
    get: Callable[[int], dict[str, Any]]  # raw source row by index
    source_id: Callable[[int], str | None]  # stable row id when the source has one


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source: str
    eval_split: str
    train_split: str
    qtype: str
    load: Callable[[str], SplitData]  # split name -> SplitData
    make: Callable[[dict[str, Any]], tuple[Any, dict[str, Any]]]  # raw row -> (state, question)


def _hf(source_key: str, split: str, config_override: str | None = None):
    from datasets import load_dataset

    src = SOURCES[source_key]
    cfg = config_override if config_override is not None else src.config
    args = [src.hf_id] + ([cfg] if cfg else [])
    return load_dataset(*args, revision=src.revision, split=split)


def _readable(name: str) -> str:
    return name.replace("_", " ").strip()


def _from_hf(ds, label_of: Callable[[dict[str, Any]], str], id_col: str | None = None, label_col: str | None = None,
             label_map: Callable[[Any], str] | None = None) -> SplitData:
    if label_col is not None:
        labels = [label_map(v) for v in ds[label_col]]
    else:
        labels = [label_of(ds[i]) for i in range(len(ds))]
    ids = ds[id_col] if id_col else None
    return SplitData(labels, lambda i: ds[int(i)], (lambda i: str(ids[int(i)])) if ids is not None else (lambda i: None))


def _question(qtype: str, instructions: str, criteria: Any = None) -> dict[str, Any]:
    q: dict[str, Any] = {"type": qtype, "instructions": instructions}
    if criteria is not None:
        q["criteria"] = criteria
    return q


# ----------------------------------------------------------------------- task builders

def _banking77() -> TaskSpec:
    cache: dict[str, Any] = {}

    def norm(name: str) -> str:
        return name.lower().replace("?", "").strip()

    def load(split: str) -> SplitData:
        ds = _hf("banking77", split)
        names = [norm(n) for n in ds.features["label"].names]
        cache["criteria"] = {n: _readable(n) for n in names}
        return _from_hf(ds, None, label_col="label", label_map=lambda v: names[int(v)])

    def make(row):
        return {"message": row["text"]}, _question(
            "choice", "Select the intent that best describes the message in the state.", dict(cache["criteria"]))

    return TaskSpec("banking77", "banking77", "test", "train", "choice", load, make)


def _clinc150() -> TaskSpec:
    cache: dict[str, Any] = {}
    oos_key = "out_of_scope"

    def load(split: str) -> SplitData:
        ds = _hf("clinc150", split)
        raw = ds.features["intent"].names
        names = [oos_key if n == "oos" else n for n in raw]
        crit = {n: _readable(n) for n in names}
        crit[oos_key] = "None of the other intents applies; the request is outside the supported intents."
        cache["criteria"] = crit
        return _from_hf(ds, None, label_col="intent", label_map=lambda v: names[int(v)])

    def make(row):
        return {"utterance": row["text"]}, _question(
            "choice",
            "Select the intent that best describes the utterance in the state. "
            f"If none of the other listed intents applies, select {oos_key}.",
            dict(cache["criteria"]))

    return TaskSpec("clinc150", "clinc150", "test", "train", "choice", load, make)


NLI_INSTRUCTIONS = "Decide how the hypothesis relates to the premise in the state."
NLI_CRITERIA = {
    "entailment": "The hypothesis must be true if the premise is true.",
    "neutral": "The hypothesis may or may not be true given the premise.",
    "contradiction": "The hypothesis must be false if the premise is true.",
}


def _anli(rnd: int) -> TaskSpec:
    def load(split: str) -> SplitData:
        ds = _hf("anli", split)
        names = ds.features["label"].names
        return _from_hf(ds, None, id_col="uid", label_col="label", label_map=lambda v: names[int(v)])

    def make(row):
        return {"premise": row["premise"], "hypothesis": row["hypothesis"]}, _question(
            "choice", NLI_INSTRUCTIONS, dict(NLI_CRITERIA))

    return TaskSpec(f"anli_r{rnd}", "anli", f"test_r{rnd}", f"train_r{rnd}", "choice", load, make)


def _contractnli() -> TaskSpec:
    def load(split: str) -> SplitData:
        from huggingface_hub import hf_hub_download

        src = SOURCES["contractnli"]
        path = hf_hub_download(src.hf_id, "contract_nli.zip", repo_type="dataset", revision=src.revision)
        with zipfile.ZipFile(path) as zf:  # read only the data file; the archive also holds macOS junk
            lines = zf.read("contract_nli_v1.jsonl").decode("utf-8").splitlines()
        all_rows = [json.loads(line) for line in lines if line.strip()]
        picked = [(n, r) for n, r in enumerate(all_rows) if r["subset"] == split]
        return SplitData([r["label"] for _, r in picked], lambda i: picked[int(i)][1],
                         lambda i: f"line{picked[int(i)][0]}")

    def make(row):
        return {"premise": row["premise"], "hypothesis": row["hypothesis"]}, _question(
            "choice", NLI_INSTRUCTIONS, dict(NLI_CRITERIA))

    return TaskSpec("contractnli", "contractnli", "test", "train", "choice", load, make)


MCQ_INSTRUCTIONS = "Select the correct answer to the question in the state."
LETTERS = "ABCDEFGHIJ"


def _mmlu() -> TaskSpec:
    def load(split: str) -> SplitData:
        ds = _hf("mmlu", split)
        return _from_hf(ds, None, label_col="answer", label_map=lambda v: LETTERS[int(v)])

    def make(row):
        state = {"question": row["question"]}
        if row.get("subject"):
            state["subject"] = _readable(row["subject"])
        crit = {LETTERS[i]: str(t) for i, t in enumerate(row["choices"])}
        return state, _question("choice", MCQ_INSTRUCTIONS, crit)

    return TaskSpec("mmlu", "mmlu", "test", "auxiliary_train", "choice", load, make)


def _arc() -> TaskSpec:
    def load(split: str) -> SplitData:
        ds = _hf("arc_challenge", split)
        rows = [ds[i] for i in range(len(ds))]  # small split; needed to map answerKey -> position

        def label(r):
            return LETTERS[r["choices"]["label"].index(r["answerKey"])]

        return SplitData([label(r) for r in rows], lambda i: rows[int(i)], lambda i: rows[int(i)]["id"])

    def make(row):
        crit = {LETTERS[i]: str(t) for i, t in enumerate(row["choices"]["text"])}
        return {"question": row["question"]}, _question("choice", MCQ_INSTRUCTIONS, crit)

    return TaskSpec("arc_challenge", "arc_challenge", "test", "train", "choice", load, make)


def _boolq() -> TaskSpec:
    def load(split: str) -> SplitData:
        ds = _hf("boolq", split)
        return _from_hf(ds, None, label_col="answer", label_map=lambda v: "true" if v else "false")

    def make(row):
        return {"passage": row["passage"], "question": row["question"]}, _question(
            "noul", "The answer to the question in the state is yes, based on the passage in the state.")

    # BoolQ test labels are hidden, so validation is the held-out split.
    return TaskSpec("boolq", "boolq", "validation", "train", "noul", load, make)


def _ragtruth() -> TaskSpec:
    def load(split: str) -> SplitData:
        ds = _hf("ragtruth", split)
        proc = ds["hallucination_labels_processed"]
        labels = ["true" if (p["evident_conflict"] or p["baseless_info"]) else "false" for p in proc]
        return SplitData(labels, lambda i: ds[int(i)], lambda i: str(ds[int(i)]["id"]))

    def make(row):
        state = {"task_type": row["task_type"], "request": row["query"],
                 "source_passages": row["context"], "response": row["output"]}
        return state, _question(
            "noul",
            "The response in the state contains at least one claim that is not supported by, "
            "or that contradicts, the source passages in the state.")

    return TaskSpec("ragtruth", "ragtruth", "test", "train", "noul", load, make)


W2C_CRITERIA = {
    "tool_call": "Call one of the available tools, using the details the user already provided.",
    "request_for_info": "Ask the user a follow-up question to obtain information needed before a tool can be called.",
    "cannot_answer": "Say that the request cannot be fulfilled with the available tools.",
    "direct": "Answer the user directly without calling a tool.",
}


def _w2c_tools(raw) -> list[Any]:
    out = []
    for t in raw or []:
        try:
            out.append(json.loads(t))
        except (TypeError, ValueError):
            out.append(t)
    return out


def _w2c_train_label(chosen: str) -> str:
    """Heuristic category for a preference-train 'chosen' response (calibration only, never scored)."""
    text = (chosen or "").strip()
    if text.startswith("<TOOLCALL>"):
        return "tool_call"
    return "request_for_info" if "?" in text else "cannot_answer"


def _when2call() -> TaskSpec:
    from datasets import load_dataset

    src = SOURCES["when2call"]

    def load(split: str) -> SplitData:
        if split == "mcq":
            ds = load_dataset(src.hf_id, "test", revision=src.revision, split="mcq")
            return SplitData(list(ds["correct_answer"]), lambda i: ds[int(i)], lambda i: str(ds[int(i)]["uuid"]))
        ds = load_dataset(src.hf_id, "train_pref", revision=src.revision, split="train")
        keep = [i for i in range(len(ds)) if len(ds[i]["messages"]) == 1 and ds[i]["messages"][0]["role"] == "user"]
        rows = [ds[i] for i in keep]
        labels = [_w2c_train_label(r["chosen_response"]["content"]) for r in rows]
        return SplitData(labels, lambda i: {**rows[int(i)], "_src_index": keep[int(i)]}, lambda i: None)

    def make(row):
        if "question" in row:
            user = row["question"]
        else:
            user = row["messages"][0]["content"]
        state = {"user_message": user, "available_tools": _w2c_tools(row.get("tools"))}
        return state, _question("choice", "Select the best next action for the assistant.", dict(W2C_CRITERIA))

    return TaskSpec("when2call", "when2call", "mcq", "train_pref", "choice", load, make)


SST5_LEVELS = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]


def _sst5() -> TaskSpec:
    def load(split: str) -> SplitData:
        ds = _hf("sst5", split)
        texts = {t: l for l, t in zip(ds["label"], ds["label_text"])}
        expected = {"very negative": 0, "negative": 1, "neutral": 2, "positive": 3, "very positive": 4}
        assert texts == expected, f"unexpected SST-5 label mapping: {texts}"
        return _from_hf(ds, None, label_col="label", label_map=lambda v: str(int(v)))

    def make(row):
        return {"text": row["text"]}, _question(
            "score", "Rate the sentiment of the text in the state on the ordered scale provided.", list(SST5_LEVELS))

    return TaskSpec("sst5", "sst5", "test", "train", "score", load, make)


def all_tasks() -> dict[str, TaskSpec]:
    specs = [_banking77(), _clinc150(), _anli(1), _anli(2), _anli(3), _contractnli(), _mmlu(), _arc(),
             _boolq(), _ragtruth(), _when2call(), _sst5()]
    return {s.name: s for s in specs}


# ------------------------------------------------------------------------- ordering

def stratified_order(labels: list[str], rng: random.Random, limit: int | None = None) -> list[int]:
    """Return row indices ordered so that every prefix is approximately label-stratified.

    Rows are shuffled within each label (seeded), then labels are interleaved with the
    largest-deficit rule: at step t pick the label maximising share*(t+1) - taken. A prefix
    of length k therefore holds each label's share of the split to within one record.
    """
    by: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by[lab].append(i)
    classes = sorted(by)
    for c in classes:
        rng.shuffle(by[c])
    n = len(labels)
    share = {c: len(by[c]) / n for c in classes}
    taken = {c: 0 for c in classes}
    out: list[int] = []
    for t in range(n if limit is None else min(limit, n)):
        best, best_d = None, float("-inf")
        for c in classes:
            if taken[c] < len(by[c]):
                d = share[c] * (t + 1) - taken[c]
                if d > best_d + 1e-12:
                    best, best_d = c, d
        out.append(by[best][taken[best]])
        taken[best] += 1
    return out


def _rng(task: str, split: str) -> random.Random:
    return random.Random(f"{SEED}:{task}:{split}")  # str seeds are hash-randomisation independent


def text_key(state: Any) -> str:
    """Normalised text of a record's state (all string leaves except 'subject'); used for overlap checks."""
    parts: list[str] = []

    def walk(x: Any, key: str | None = None) -> None:
        if isinstance(x, str):
            if key != "subject":
                parts.append(x)
        elif isinstance(x, dict):
            for k in sorted(x):
                walk(x[k], k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)

    walk(state)
    return re.sub(r"\s+", " ", " ".join(parts)).strip().lower()


# --------------------------------------------------------------------------- build

def make_record(spec: TaskSpec, rid: str, row: dict[str, Any], gold: str) -> dict[str, Any]:
    state, question = spec.make(row)
    rec = {"id": rid, "task": spec.name,
           "request": {"model": MODEL, "state": state, "questions": {QID: question}},
           "gold": {QID: gold}}
    errs = validate_record(rec)
    if errs:
        raise ValueError(f"{rid}: invalid record: {errs}")
    return rec


def source_row_index(spec: TaskSpec, split_data: SplitData, i: int) -> int:
    row = split_data.get(i)
    return int(row["_src_index"]) if isinstance(row, dict) and "_src_index" in row else int(i)


def build_scored(spec: TaskSpec, limit: int | None) -> tuple[list[dict], list[dict], dict[str, Any]]:
    sd = spec.load(spec.eval_split)
    order = stratified_order(sd.labels, _rng(spec.name, spec.eval_split))
    n_split = len(order)
    selected = order[: min(n_split, CAP)]
    keep = selected if limit is None else selected[:limit]
    src = SOURCES[spec.source]
    records, ids = [], []
    for n, i in enumerate(keep):
        rid = f"{spec.name}-{n}"
        records.append(make_record(spec, rid, sd.get(i), sd.labels[i]))
        ids.append({"id": rid, "task": spec.name, "kind": "scored", "dataset": src.hf_id, "config": src.config,
                    "revision": src.revision, "source_split": spec.eval_split,
                    "source_index": source_row_index(spec, sd, i), "source_id": sd.source_id(i),
                    "release": src.release})
    # state keys of the FULL selected set (not just the --limit slice), for the calibration overlap check
    all_keys = {text_key(spec.make(sd.get(i))[0]) for i in selected} if limit is not None else {
        text_key(r["request"]["state"]) for r in records}
    info = {"eval_split": spec.eval_split, "split_size": n_split, "cap": CAP, "capped": n_split > CAP,
            "written": len(records), "labels": dict(sorted(Counter(r["gold"][QID] for r in records).items())),
            "text_keys": all_keys}
    return records, ids, info


def build_calibration(spec: TaskSpec, quota: int, forbidden: set[str]) -> tuple[list[dict], list[dict], int]:
    sd = spec.load(spec.train_split)
    candidates = stratified_order(sd.labels, _rng(spec.name, spec.train_split), limit=quota * 6 + 64)
    src = SOURCES[spec.source]
    records, ids, seen, dropped = [], [], set(), 0
    for i in candidates:
        if len(records) >= quota:
            break
        rid = f"{spec.name}-cal-{len(records)}"
        rec = make_record(spec, rid, sd.get(i), sd.labels[i])
        key = text_key(rec["request"]["state"])
        if key in forbidden or key in seen:
            dropped += 1
            continue
        seen.add(key)
        records.append(rec)
        ids.append({"id": rid, "task": spec.name, "kind": "calibration", "dataset": src.hf_id, "config": src.config,
                    "revision": src.revision, "source_split": spec.train_split,
                    "source_index": source_row_index(spec, sd, i), "source_id": sd.source_id(i),
                    "release": src.release})
    if len(records) < quota:
        raise RuntimeError(f"{spec.name}: only {len(records)} calibration records after overlap filtering")
    return records, ids, dropped


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n" for r in rows)
    path.write_text(blob, encoding="utf-8")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


COUNTS_BEGIN, COUNTS_END = "<!-- BEGIN GENERATED COUNTS -->", "<!-- END GENERATED COUNTS -->"


def render_counts_md(manifest: dict[str, Any]) -> str:
    lines = [COUNTS_BEGIN, "",
             f"Seed {manifest['seed']}; cap {manifest['cap']} scored records per task; "
             f"calibration target {manifest['calibration_total']}.", "",
             "| task | eval split | split size | scored records | capped | calibration records | release |",
             "|---|---|---:|---:|---|---:|---|"]
    for t, d in manifest["tasks"].items():
        lines.append(f"| {t} | {d['eval_split']} | {d['split_size']} | {d['written']} | "
                     f"{'yes (stratified)' if d['capped'] else 'no (full split)'} | {d['calibration']} | {d['release']} |")
    lines.append(f"| **total** | | | **{sum(d['written'] for d in manifest['tasks'].values())}** | | "
                 f"**{manifest['calibration_written']}** | |")
    if manifest["skipped"]:
        lines += ["", "Skipped tasks:"] + [f"- {t}: {why}" for t, why in manifest["skipped"].items()]
    else:
        lines += ["", "Skipped tasks: none."]
    lines += ["", f"Calibration records dropped for overlapping a scored record's text: {manifest['calibration_dropped_overlap']}.",
              "", "Label distribution of the scored records (counts):", ""]
    for t, d in manifest["tasks"].items():
        labs = d["labels"]
        shown = ", ".join(f"{k}: {v}" for k, v in list(labs.items())[:8])
        more = f", ... ({len(labs)} labels)" if len(labs) > 8 else ""
        lines.append(f"- {t}: {shown}{more}")
    lines += ["", COUNTS_END]
    return "\n".join(lines)


def update_datasets_md(md_path: Path, counts_md: str) -> None:
    text = md_path.read_text(encoding="utf-8")
    if COUNTS_BEGIN in text and COUNTS_END in text:
        head, rest = text.split(COUNTS_BEGIN, 1)
        _, tail = rest.split(COUNTS_END, 1)
        text = head + counts_md + tail
    else:
        text = text.rstrip() + "\n\n" + counts_md + "\n"
    md_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="output dir (default data, or data_dev with --limit)")
    ap.add_argument("--limit", type=int, default=None, help="keep only the first N scored records per task")
    ap.add_argument("--tasks", default=None, help="comma-separated subset of tasks (default all)")
    ap.add_argument("--calibration-total", type=int, default=CALIBRATION_TOTAL)
    ap.add_argument("--allow-skip", action="store_true",
                    help="record a task as skipped when its source cannot be loaded instead of aborting")
    args = ap.parse_args(argv)

    out = args.out or (ROOT / ("data_dev" if args.limit else "data"))
    specs = all_tasks()
    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(",")]
        unknown = [t for t in wanted if t not in specs]
        if unknown:
            ap.error(f"unknown tasks {unknown}; choose from {sorted(specs)}")
        specs = {t: specs[t] for t in wanted}

    scored: dict[str, list[dict]] = {}
    ids_rows: list[dict] = []
    infos: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for name, spec in specs.items():
        try:
            recs, ids, info = build_scored(spec, args.limit)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
            if not args.allow_skip:
                raise
            skipped[name] = f"{type(exc).__name__}: {exc}"
            print(f"SKIPPED {name}: {skipped[name]}", file=sys.stderr)
            continue
        scored[name], infos[name] = recs, info
        ids_rows += ids
        print(f"{name}: {len(recs)} scored records (split {info['split_size']})", file=sys.stderr)

    forbidden: set[str] = set()
    for info in infos.values():
        forbidden |= info.pop("text_keys")

    names = list(scored)
    base, extra = divmod(args.calibration_total, max(len(names), 1))
    calib: list[dict] = []
    dropped_total = 0
    for k, name in enumerate(names):
        quota = base + (1 if k < extra else 0)
        recs, ids, dropped = build_calibration(specs[name], quota, forbidden)
        calib += recs
        ids_rows += ids
        dropped_total += dropped
        infos[name]["calibration"] = len(recs)
    random.Random(f"{SEED}:calibration-mix").shuffle(calib)

    # Hard guarantees, checked on the final data rather than assumed.
    scored_ids = {r["id"] for rs in scored.values() for r in rs}
    assert not scored_ids & {r["id"] for r in calib}, "calibration id collides with a scored id"
    calib_keys = {text_key(r["request"]["state"]) for r in calib}
    assert not calib_keys & forbidden, "a calibration record overlaps a scored record's text"
    assert len(calib_keys) == len(calib), "duplicate texts inside the calibration set"
    for rs in list(scored.values()) + [calib]:
        for r in rs:
            errs = validate_record(r)
            assert not errs, (r["id"], errs)

    hashes: dict[str, str] = {}
    for name, rs in scored.items():
        hashes[f"records/{name}.jsonl"] = _write_jsonl(out / "records" / f"{name}.jsonl", rs)
    hashes["calibration.jsonl"] = _write_jsonl(out / "calibration.jsonl", calib)
    hashes["record_ids.jsonl"] = _write_jsonl(out / "record_ids.jsonl", ids_rows)

    manifest = {
        "seed": SEED, "cap": CAP, "limit": args.limit, "calibration_total": args.calibration_total,
        "calibration_written": len(calib), "calibration_dropped_overlap": dropped_total, "skipped": skipped,
        "sources": {k: {"hf_id": s.hf_id, "revision": s.revision, "config": s.config, "license": s.license,
                        "release": s.release} for k, s in SOURCES.items()},
        "tasks": {t: {**infos[t], "release": SOURCES[specs[t].source].release} for t in names},
        "sha256": hashes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {sum(len(r) for r in scored.values())} scored + {len(calib)} calibration records to {out}",
          file=sys.stderr)

    md = ROOT / "datasets" / "DATASETS.md"
    if out == ROOT / "data" and args.limit is None and not args.tasks and md.exists():
        update_datasets_md(md, render_counts_md(manifest))
        print(f"updated counts in {md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
