"""Build Clef-format records from public labeled datasets.

calib.jsonl  — from train splits, used for NVFP4/FP8-static calibration
eval.jsonl   — from test/validation splits, scored against gold labels and BF16
Each line: {"task", "gold": {question_id: option_id}, "record": {state, questions}}
"""
import json
import random

from datasets import load_dataset

random.seed(0)
LETTERS = "ABCDEFGHIJ"


def banking77(split):
    ds = load_dataset("mteb/banking77", split=split)
    names = sorted(set(ds["label_text"]))
    criteria = {n: n.replace("_", " ") for n in names}
    for r in ds:
        yield {"state": r["text"], "questions": {"intent": {
            "type": "choice", "instructions": "Which banking intent does the customer message express?",
            "criteria": criteria}}}, {"intent": r["label_text"]}


def arc(split):
    for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split):
        crit = dict(zip(r["choices"]["label"], r["choices"]["text"]))
        if r["answerKey"] not in crit:
            continue
        yield {"state": {"question": r["question"]}, "questions": {"answer": {
            "type": "choice", "instructions": "Which option correctly answers the question?",
            "criteria": crit}}}, {"answer": r["answerKey"]}


def mmlu(split):
    for r in load_dataset("cais/mmlu", "all", split=split):
        yield {"state": {"subject": r["subject"], "question": r["question"]}, "questions": {"answer": {
            "type": "choice", "instructions": "Which option correctly answers the question?",
            "criteria": dict(zip(LETTERS, r["choices"]))}}}, {"answer": LETTERS[r["answer"]]}


def hellaswag(split):
    for r in load_dataset("Rowan/hellaswag", split=split):
        if r["label"] == "":
            continue
        yield {"state": {"context": r["ctx"]}, "questions": {"ending": {
            "type": "choice", "instructions": "Which ending most plausibly continues the context?",
            "criteria": dict(zip(LETTERS, r["endings"]))}}}, {"ending": LETTERS[int(r["label"])]}


def anli(split):
    names = ["entailment", "neutral", "contradiction"]
    for r in load_dataset("facebook/anli", split=split):
        yield {"state": {"premise": r["premise"], "hypothesis": r["hypothesis"]}, "questions": {"relation": {
            "type": "choice", "instructions": "How does the premise relate to the hypothesis?",
            "criteria": {"entailment": "The premise entails the hypothesis.",
                         "neutral": "The premise neither entails nor contradicts the hypothesis.",
                         "contradiction": "The premise contradicts the hypothesis."}}}}, {"relation": names[r["label"]]}


def boolq(split):
    for r in load_dataset("google/boolq", split=split):
        yield {"state": {"passage": r["passage"]}, "questions": {"answer": {
            "type": "noul", "instructions": r["question"].rstrip("?") + "?"}}}, {"answer": "true" if r["answer"] else "false"}


def multi(split):
    """Multi-question records: ANLI relation + a score + a noul over the same state."""
    names = ["entailment", "neutral", "contradiction"]
    for r in load_dataset("facebook/anli", split=split):
        yield {"state": {"premise": r["premise"], "hypothesis": r["hypothesis"]}, "questions": {
            "relation": {"type": "choice", "instructions": "How does the premise relate to the hypothesis?",
                         "criteria": {"entailment": "Entails.", "neutral": "Neither.", "contradiction": "Contradicts."}},
            "supported": {"type": "noul", "instructions": "Does the premise support the hypothesis?"},
            "overlap": {"type": "score", "instructions": "How much topical overlap do premise and hypothesis have?",
                        "criteria": ["None", "Some", "A lot"]},
        }}, {"relation": names[r["label"]], "supported": "true" if r["label"] == 0 else "false"}


TASKS = {
    "banking77": (banking77, "train", "test"),
    "arc_challenge": (arc, "train", "test"),
    "mmlu": (mmlu, "auxiliary_train", "test"),
    "hellaswag": (hellaswag, "train", "validation"),
    "anli": (anli, "train_r3", "test_r3"),
    "boolq": (boolq, "train", "validation"),
    "multi_anli": (multi, "train_r1", "test_r1"),
}
CALIB_PER_TASK, EVAL_PER_TASK = 96, 200

for out, idx, n in (("calib.jsonl", 1, CALIB_PER_TASK), ("eval.jsonl", 2, EVAL_PER_TASK)):
    with open(out, "w") as f:
        for task, spec in TASKS.items():
            rows = list(spec[0](spec[idx]))
            random.shuffle(rows)
            for record, gold in rows[:n]:
                f.write(json.dumps({"task": task, "gold": gold, "record": record}) + "\n")
            print(out, task, min(n, len(rows)))
