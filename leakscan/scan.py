"""Publication leak scanner.

Fails (exit 1) when any publish-bound text file under a directory contains:
  ngram      an 8-token sequence (lowercase word tokens) that also occurs in the private corpus
  email      an email address
  phone      a phone number
  ipv4       an IPv4 address (tailnet addresses included; 127.0.0.1 and 0.0.0.0 are exempt)
  home_path  a home-directory path (/Users/<name>, /home/<name>)
  denylist   a term listed in the private denylist file

Public benchmark data (default: data/records/, data/calibration.jsonl and data_dev/ under the scanned root, override
with --public-data) is checked with ngram and denylist only; every other file gets all rules.

8-gram windows with fewer than 4 distinct tokens or fewer than 2 alphabetic tokens (number runs,
repeated tokens) are ignored on both sides: they collide by coincidence, not by leakage.

Matched text is never printed: every hit is reported as file, line, rule and a 12-hex digest of the
matched text.

    uv run python -m leakscan.scan <publish_tree> [--corpus private/records] [--denylist private/denylist.txt]
                                   [--exclude NAME_OR_REL_PATH ...] [--allow reviewed_digests.txt] [--json]

Exit codes: 0 clean, 1 hits, 2 usage or corpus error.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

NGRAM = 8
MAX_TEXT_BYTES = 50 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "private" / "records"
DEFAULT_DENYLIST = PROJECT_ROOT / "private" / "denylist.txt"

# Public benchmark data (redistributed third-party datasets): their emails, phones and IPs are dataset
# content, so only the ngram and denylist rules apply. Paths are relative to the scanned root.
DEFAULT_PUBLIC_DATA = ("data/records", "data/calibration.jsonl", "data_dev")
PUBLIC_DATA_RULES = frozenset({"ngram", "denylist"})
ALL_RULES = frozenset({"ngram", "denylist", "email", "phone", "ipv4", "home_path"})

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "private",
}

TOKEN_RE = re.compile(r"\w+")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
PHONE_RES = (
    re.compile(r"(?<![\w.])(?:\+?1[\s.-])?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\w)"),
    re.compile(r"(?<![\w.])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}(?!\w)"),
)
IPV4_RE = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\.?\d)"
)
IPV4_EXEMPT = {"127.0.0.1", "0.0.0.0"}
HOME_PATH_RE = re.compile(r"(?<![\w.:/-])/(?:Users|home)/[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    rule: str
    digest: str
    detail: str = ""

    def format(self) -> str:
        return f"{self.path}:{self.line}\t{self.rule}\t{self.digest}" + (f"\t{self.detail}" if self.detail else "")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _informative(words: Sequence[str]) -> bool:
    """Skip low-information windows (runs of numbers, repeated tokens): they match by coincidence."""
    return len(set(words)) >= 4 and sum(1 for w in words if w.isalpha()) >= 2


def _gram_hash(words: Sequence[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest(), "big")


def _flatten(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _flatten(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten(v)
    elif value is not None:
        yield str(value)


def build_ngram_set(corpus_dir: Path) -> set[int]:
    """Hashed 8-gram set over request.state of every record in corpus_dir/*.jsonl."""
    grams: set[int] = set()
    for path in sorted(Path(corpus_dir).glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for text in _flatten(rec.get("request", {}).get("state")):
                    words = tokens(text)
                    for i in range(len(words) - NGRAM + 1):
                        window = words[i:i + NGRAM]
                        if _informative(window):
                            grams.add(_gram_hash(window))
    return grams


def load_denylist(path: Path) -> list[str]:
    terms = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        t = raw.strip()
        if t and not t.startswith("#"):
            terms.append(t.lower())
    return terms


def _denylist_regexes(terms: Sequence[str]) -> list[tuple[str, re.Pattern]]:
    out = []
    for t in terms:
        pat = re.escape(t)
        pat = (r"(?<!\w)" if t[:1].isalnum() or t[:1] == "_" else "") + pat + (r"(?!\w)" if t[-1:].isalnum() or t[-1:] == "_" else "")
        out.append((t, re.compile(pat, re.IGNORECASE)))
    return out


def scan_text(text: str, rel: str, grams: set[int], deny: Sequence[tuple[str, re.Pattern]],
              rules: frozenset = ALL_RULES) -> list[Hit]:
    hits: list[Hit] = []
    line_starts = [0]
    for m in re.finditer(r"\n", text):
        line_starts.append(m.end())

    def line_of(offset: int) -> int:
        return bisect.bisect_right(line_starts, offset)

    # pattern rules
    for m in EMAIL_RE.finditer(text) if "email" in rules else ():
        hits.append(Hit(rel, line_of(m.start()), "email", digest(m.group())))
    spans = sorted((m.start(), m.end(), m.group()) for rx in PHONE_RES for m in rx.finditer(text)) if "phone" in rules else []
    last_end = -1
    for start, end, grp in spans:
        if start < last_end:  # overlapping match from the other phone pattern
            continue
        last_end = end
        hits.append(Hit(rel, line_of(start), "phone", digest(grp)))
    for m in IPV4_RE.finditer(text) if "ipv4" in rules else ():
        if m.group() not in IPV4_EXEMPT:
            hits.append(Hit(rel, line_of(m.start()), "ipv4", digest(m.group())))
    for m in HOME_PATH_RE.finditer(text) if "home_path" in rules else ():
        hits.append(Hit(rel, line_of(m.start()), "home_path", digest(m.group())))
    for _term, rx in deny if "denylist" in rules else ():
        for m in rx.finditer(text):
            hits.append(Hit(rel, line_of(m.start()), "denylist", digest(m.group().lower())))

    # 8-gram rule: one hit per run of consecutive overlapping windows
    if grams and "ngram" in rules:
        toks = [(m.group().lower(), m.start()) for m in TOKEN_RE.finditer(text)]
        words = [t for t, _ in toks]
        run_start = None
        run_len = 0
        for i in range(len(words) - NGRAM + 1):
            window = words[i:i + NGRAM]
            if _informative(window) and _gram_hash(window) in grams:
                if run_start is None:
                    run_start, run_len = i, 0
                run_len += 1
            elif run_start is not None:
                hits.append(_ngram_hit(rel, toks, words, run_start, run_len, line_of))
                run_start = None
        if run_start is not None:
            hits.append(_ngram_hit(rel, toks, words, run_start, run_len, line_of))
    return hits


def _ngram_hit(rel, toks, words, start, run_len, line_of) -> Hit:
    span = " ".join(words[start:start + NGRAM + run_len - 1])
    return Hit(rel, line_of(toks[start][1]), "ngram", digest(span), f"tokens={NGRAM + run_len - 1}")


def _is_excluded(rel_parts: Sequence[str], excludes: Sequence[str]) -> bool:
    rel = "/".join(rel_parts)
    for e in excludes:
        e = e.strip("/")
        if e in rel_parts or rel == e or rel.startswith(e + "/"):
            return True
    return False


@dataclass
class ScanStats:
    files_scanned: int = 0
    skipped_binary: int = 0
    skipped_large: int = 0
    public_data_files: int = 0


def _is_public(rel_parts: Sequence[str], public_paths: Sequence[str]) -> bool:
    rel = "/".join(rel_parts)
    return any(rel == p.strip("/") or rel.startswith(p.strip("/") + "/") for p in public_paths)


def scan_tree(root: Path, grams: set[int], denylist: Sequence[str],
              extra_excludes: Sequence[str] = (), stats: ScanStats | None = None,
              public_paths: Sequence[str] = DEFAULT_PUBLIC_DATA) -> list[Hit]:
    root = Path(root)
    stats = stats if stats is not None else ScanStats()
    excludes = list(DEFAULT_EXCLUDE_DIRS) + list(extra_excludes)
    deny = _denylist_regexes(denylist)
    hits: list[Hit] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).parts
        dirnames[:] = sorted(d for d in dirnames if not _is_excluded(rel_dir + (d,), excludes))
        for name in sorted(filenames):
            rel_parts = rel_dir + (name,)
            if _is_excluded(rel_parts, excludes):
                continue
            path = Path(dirpath) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_TEXT_BYTES:
                stats.skipped_large += 1
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data:
                stats.skipped_binary += 1
                continue
            text = data.decode("utf-8", "replace")
            stats.files_scanned += 1
            public = _is_public(rel_parts, public_paths)
            stats.public_data_files += public
            hits.extend(scan_text(text, "/".join(rel_parts), grams, deny, PUBLIC_DATA_RULES if public else ALL_RULES))
    return hits


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Leak scanner for publish-bound trees.")
    ap.add_argument("root", type=Path, help="publish tree to scan")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="directory of private *.jsonl records")
    ap.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    ap.add_argument("--exclude", action="append", default=[], help="extra dir name or relative path to skip")
    ap.add_argument("--public-data", action="append", default=None, metavar="REL_PATH",
                    help="relative file or dir holding public benchmark data: ngram and denylist rules only "
                         f"(repeatable; default: {' '.join(DEFAULT_PUBLIC_DATA)})")
    ap.add_argument("--allow", type=Path, help="file of reviewed hit digests (one per line) to suppress")
    ap.add_argument("--json", action="store_true", help="emit hits as JSON lines")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2
    grams = build_ngram_set(args.corpus) if args.corpus.is_dir() else set()
    if not grams:
        print(f"error: no 8-grams built from corpus {args.corpus}; refusing to scan without a private corpus",
              file=sys.stderr)
        return 2
    if args.denylist.is_file():
        deny = load_denylist(args.denylist)
    else:
        deny = []
        print(f"warning: denylist {args.denylist} not found; denylist rule inactive", file=sys.stderr)

    stats = ScanStats()
    public = args.public_data if args.public_data is not None else list(DEFAULT_PUBLIC_DATA)
    hits = scan_tree(args.root, grams, deny, args.exclude, stats, public)
    suppressed = 0
    if args.allow:
        allowed = {ln.strip() for ln in args.allow.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
        kept = [h for h in hits if h.digest not in allowed]
        suppressed = len(hits) - len(kept)
        hits = kept
    for h in hits:
        if args.json:
            print(json.dumps({"path": h.path, "line": h.line, "rule": h.rule, "digest": h.digest, "detail": h.detail}))
        else:
            print(h.format())
    by_rule: dict[str, int] = {}
    for h in hits:
        by_rule[h.rule] = by_rule.get(h.rule, 0) + 1
    print(f"scanned {stats.files_scanned} text files ({stats.skipped_binary} binary, {stats.skipped_large} oversize "
          f"skipped; {stats.public_data_files} public-data files checked with ngram+denylist only); corpus 8-grams: {len(grams)}; denylist terms: {len(deny)}; "
          f"hits: {len(hits)} {json.dumps(by_rule, sort_keys=True)}; allow-suppressed: {suppressed}", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
