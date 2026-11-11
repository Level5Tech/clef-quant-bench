"""Compare two runner-output JSONL files (determinism, engine-vs-native, quant-vs-ref).

    python -m runners.compare A.jsonl B.jsonl [--json]

Reports over all questions present and error-free in both files:
  max_abs_logit_diff  (None if either side lacks logits), max_abs_prob_diff, mean_abs_prob_diff,
  top1_agreement, and the largest-gap items.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from runners.common import read_jsonl


def compare(rows_a: Sequence[dict[str, Any]], rows_b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_id_b = {r["id"]: r for r in rows_b}
    n_q = n_agree = errors = missing = 0
    max_logit: float | None = 0.0
    logits_ok = True
    max_prob = 0.0
    sum_prob = 0.0
    n_prob = 0
    worst: tuple[float, str] = (0.0, "")
    for ra in rows_a:
        rb = by_id_b.get(ra["id"])
        if rb is None:
            missing += 1
            continue
        if ra.get("error") or rb.get("error"):
            errors += 1
            continue
        for qid, aa in ra["answers"].items():
            ab = rb["answers"].get(qid)
            if ab is None or aa["option_ids"] != ab["option_ids"]:
                missing += 1
                continue
            n_q += 1
            pa, pb = aa["probs"], ab["probs"]
            n_agree += int(max(range(len(pa)), key=pa.__getitem__) == max(range(len(pb)), key=pb.__getitem__))
            diffs = [abs(x - y) for x, y in zip(pa, pb)]
            d = max(diffs)
            if d > worst[0]:
                worst = (d, f"{ra['id']}/{qid}")
            max_prob = max(max_prob, d)
            sum_prob += sum(diffs)
            n_prob += len(diffs)
            if aa["logits"] is None or ab["logits"] is None:
                logits_ok = False
            elif logits_ok:
                max_logit = max(max_logit, max(abs(x - y) for x, y in zip(aa["logits"], ab["logits"])))
    return {
        "questions": n_q,
        "errors": errors,
        "missing": missing,
        "max_abs_logit_diff": max_logit if (logits_ok and n_q) else None,
        "max_abs_prob_diff": max_prob,
        "mean_abs_prob_diff": (sum_prob / n_prob) if n_prob else None,
        "top1_agreement": (n_agree / n_q) if n_q else None,
        "worst_prob_item": worst[1] or None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    out = compare(list(read_jsonl(args.a)), list(read_jsonl(args.b)))
    print(json.dumps(out, indent=None if args.json else 2))


if __name__ == "__main__":
    main()
