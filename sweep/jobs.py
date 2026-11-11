"""Jobs file: one JSON object per line, one line per (model, config, set) plus latency and build jobs.

  {"id": "clef-flash/ref/anli_r1", "kind": "quality", "model": "clef-flash", "config": "ref", "set": "anli_r1", "private": false}
  {"id": "clef-flash/ref/latency", "kind": "latency", "model": "clef-flash", "config": "ref"}
  {"id": "build/clef/fp8",         "kind": "build",   "model": "clef", "fmt": "fp8"}

The driver groups consecutive quality+latency lines of one (model, config) into a session (one server start).

    python3 -m sweep.jobs --records-dir data/records --out sweep/jobs.jsonl            # full sweep
    python3 -m sweep.jobs --records-dir data_dev/records --models clef-flash --configs ref --private-sets r4b \
        --out dryrun.jsonl                                                              # dry run
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sweep.configs import BUILD_ARGS, CONFIG_BY_NAME, CONFIGS, MODELS, PRIVATE_SETS


def public_sets(records_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(records_dir).glob("*.jsonl") if not p.stem.endswith("_hosted_repeat"))


def _session_jobs(model: str, config: str, public: Sequence[str], private: Sequence[str],
                  latency: bool) -> list[dict[str, Any]]:
    js: list[dict[str, Any]] = []
    for s in public:
        js.append({"id": f"{model}/{config}/{s}", "kind": "quality", "model": model, "config": config,
                   "set": s, "private": False})
    for s in private:
        js.append({"id": f"{model}/{config}/{s}", "kind": "quality", "model": model, "config": config,
                   "set": s, "private": True})
    if latency:
        js.append({"id": f"{model}/{config}/latency", "kind": "latency", "model": model, "config": config})
    return js


def make_jobs(public: Sequence[str], private: Sequence[str], models: Sequence[str] = MODELS,
              configs: Sequence[str] | None = None, builds: bool = True, latency: bool = True) -> list[dict[str, Any]]:
    """Flash configs first, then the 27B quant builds, then the 27B configs (the order in the brief)."""
    names = list(configs) if configs else [c.name for c in CONFIGS]
    order = [c.name for c in CONFIGS if c.name in names]
    js: list[dict[str, Any]] = []
    for model in models:
        if model == "clef" and builds:
            for fmt in BUILD_ARGS:
                if {"fp8": "fp8", "nvfp4": "nvfp4"}[fmt] in order:
                    js.append({"id": f"build/{model}/{fmt}", "kind": "build", "model": model, "fmt": fmt})
        for name in order:
            if CONFIG_BY_NAME[name].on_subset:   # cross-restart repeats and subset configs: public subset only, no latency
                js.extend(_session_jobs(model, name, public, [], False))
            else:
                js.extend(_session_jobs(model, name, public, private, latency))
    return js


def ensure_subset(records_dir: str | Path, out_dir: str | Path, n: int) -> None:
    """<out_dir>/<set>.jsonl = first n records of <records_dir>/<set>.jsonl (deterministic, so ids pair with the base run)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(records_dir).glob("*.jsonl")):
        if p.stem.endswith("_hosted_repeat"):
            continue
        lines = []
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    lines.append(line if line.endswith("\n") else line + "\n")
                if len(lines) == n:
                    break
        target = out / p.name
        if not target.is_file() or target.read_text(encoding="utf-8") != "".join(lines):
            target.write_text("".join(lines), encoding="utf-8")


def group(js: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for j in js:
        if j["kind"] == "build":
            items.append({"type": "build", "model": j["model"], "fmt": j["fmt"], "id": j["id"]})
            continue
        last = items[-1] if items else None
        if last and last["type"] == "session" and (last["model"], last["config"]) == (j["model"], j["config"]):
            last["jobs"].append(j)
        else:
            items.append({"type": "session", "model": j["model"], "config": j["config"], "jobs": [j]})
    return items


def stages(js: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Interpret a jobs file for the concurrent driver:
    quality[model] = [(config, public sets, private sets)] in order; builds = [build jobs]; latency = [(model, config, public, private)]."""
    quality: dict[str, list[tuple[str, list[str], list[str]]]] = {}
    sets: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    order: list[tuple[str, str]] = []
    builds, latency = [], []
    for j in js:
        if j["kind"] == "build":
            builds.append(j)
        elif j["kind"] == "quality":
            key = (j["model"], j["config"])
            if key not in sets:
                sets[key] = ([], [])
                order.append(key)
            sets[key][1 if j["private"] else 0].append(j["set"])
        elif j["kind"] == "latency":
            latency.append((j["model"], j["config"]))
    for model, config in order:
        quality.setdefault(model, []).append((config, *sets[(model, config)]))
    lat = [(m, c, *sets.get((m, c), ([], []))) for m, c in latency]
    return {"quality": quality, "builds": builds, "latency": lat}


def write_jobs(path: str | Path, js: Sequence[dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(j) + "\n" for j in js))


def read_jobs(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--private-sets", nargs="*", default=list(PRIVATE_SETS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--no-builds", action="store_true")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    js = make_jobs(public_sets(a.records_dir), a.private_sets, a.models, a.configs, not a.no_builds, not a.no_latency)
    write_jobs(a.out, js)
    print(f"{len(js)} jobs -> {a.out}")


if __name__ == "__main__":
    main()
