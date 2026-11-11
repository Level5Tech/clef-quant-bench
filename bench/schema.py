"""Shared record / runner-output schema checks (see PLAN.md, "Shared interface").

Option ids follow Cloudflare's reference encoder (joint_schema_model.question_options):
  noul   -> ["true", "false"]
  choice -> sorted(str(key) for key in criteria)
  score  -> ["0", ..., "k-1"]
"""

from __future__ import annotations

import json
import re
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")
RECORD_ID_RE = re.compile(r"^(?P<task>[a-z0-9_]+)-(?:cal-)?\d+$")


def option_ids(question: dict[str, Any]) -> list[str]:
    """Allowed option ids for a question, in the reference encoder's order."""
    qtype = str(question["type"])
    if qtype == "noul":
        return ["true", "false"]
    if qtype == "choice":
        return sorted(str(key) for key in question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def validate_record(rec: Any) -> list[str]:
    """Return a list of problems (empty list means the record is valid)."""
    errs: list[str] = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    for key in ("id", "task", "request", "gold"):
        if key not in rec:
            errs.append(f"missing key {key!r}")
    if errs:
        return errs
    rid, task = rec["id"], rec["task"]
    if not isinstance(rid, str) or not isinstance(task, str):
        return ["id and task must be strings"]
    m = RECORD_ID_RE.match(rid)
    if not m:
        errs.append(f"id {rid!r} does not match '<task>-<n>' or '<task>-cal-<n>'")
    elif m.group("task") != task:
        errs.append(f"id {rid!r} does not start with task {task!r}")
    req = rec["request"]
    if not isinstance(req, dict):
        return errs + ["request is not an object"]
    if not isinstance(req.get("model"), str) or not req["model"]:
        errs.append("request.model must be a non-empty string")
    if "state" not in req:
        errs.append("request.state is missing")
    questions = req.get("questions")
    if not isinstance(questions, dict) or not questions:
        return errs + ["request.questions must be a non-empty object"]
    gold = rec["gold"]
    if not isinstance(gold, dict):
        return errs + ["gold is not an object"]
    if set(gold) != set(questions):
        errs.append(f"gold keys {sorted(gold)} != question keys {sorted(questions)}")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in QUESTION_TYPES:
            errs.append(f"{qid}: type must be one of {QUESTION_TYPES}")
            continue
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            errs.append(f"{qid}: instructions must be a non-empty string")
        crit = q.get("criteria")
        if q["type"] == "choice":
            if not isinstance(crit, dict) or len(crit) < 2:
                errs.append(f"{qid}: choice criteria must be an object with >= 2 options")
                continue
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in crit.items()):
                errs.append(f"{qid}: choice criteria must map str -> str")
        elif q["type"] == "score":
            if not isinstance(crit, list) or len(crit) < 2 or not all(isinstance(v, str) for v in crit):
                errs.append(f"{qid}: score criteria must be a list of >= 2 strings")
                continue
        else:
            if crit is not None and (not isinstance(crit, dict) or not set(crit) <= {"true", "false"}):
                errs.append(f"{qid}: noul criteria may only override 'true'/'false'")
        if qid in gold and gold[qid] not in option_ids(q):
            errs.append(f"{qid}: gold {gold[qid]!r} is not an allowed option id")
    try:
        json.dumps(rec, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        errs.append(f"record is not JSON serialisable: {exc}")
    return errs
