"""Compare two runner-output directories task by task (matching <stem>.jsonl names) and overall.

    python -m runners.compare_dirs results/clef-flash/ref results/clef-flash/fp8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runners.common import read_jsonl
from runners.compare import compare


def compare_dirs(dir_a: str | Path, dir_b: str | Path) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    all_a: list[dict[str, Any]] = []
    all_b: list[dict[str, Any]] = []
    for fa in sorted(Path(dir_a).glob("*.jsonl")):
        fb = Path(dir_b) / fa.name
        if not fb.exists():
            continue
        a, b = list(read_jsonl(fa)), list(read_jsonl(fb))
        per_task[fa.stem] = compare(a, b)
        all_a += a
        all_b += b
    return {"per_task": per_task, "overall": compare(all_a, all_b)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--brief", action="store_true", help="print only the overall numbers")
    args = ap.parse_args()
    out = compare_dirs(args.dir_a, args.dir_b)
    if args.brief:
        print(json.dumps(out["overall"]))
        return
    for task, r in out["per_task"].items():
        print(f"{task:16s} q={r['questions']:4d} top1={r['top1_agreement']} max|dprob|={r['max_abs_prob_diff']:.4f} "
              f"max|dlogit|={r['max_abs_logit_diff']}")
    print("OVERALL", json.dumps(out["overall"]))


if __name__ == "__main__":
    main()
