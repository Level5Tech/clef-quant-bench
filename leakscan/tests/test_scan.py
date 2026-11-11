"""Tests for the publication leak scanner. Synthetic data only.

Strings that would themselves trip the scanner (emails, IPs, home paths, phone numbers) are assembled
at runtime so this source file stays clean when the scanner is pointed at the project root.
"""
import json
from pathlib import Path

import pytest

from leakscan import scan

AT = "@"
DOT = "."
DIGITS = "555"  # phone/IP fixtures are assembled at runtime so this file stays clean under the scanner
SECRET_SENTENCE = "zephyr quokka marmalade orbit twelve lantern velvet harbor candle meadow"  # 10 tokens


def _corpus(tmp_path: Path, states) -> Path:
    d = tmp_path / "records"
    d.mkdir()
    with open(d / "r9.jsonl", "w") as fh:
        for i, s in enumerate(states):
            fh.write(json.dumps({"id": f"r9-{i}", "task": "r9",
                                 "request": {"model": "clef", "state": s, "questions": {}},
                                 "gold": {}}) + "\n")
    return d


def _tree(tmp_path: Path, files: dict) -> Path:
    root = tmp_path / "publish"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _run(tmp_path, files, states=(SECRET_SENTENCE,), denylist=None, extra=()):
    corpus = _corpus(tmp_path, list(states))
    root = _tree(tmp_path, files)
    ngrams = scan.build_ngram_set(corpus)
    return scan.scan_tree(root, ngrams, denylist or [], extra_excludes=list(extra))


def rules(hits):
    return sorted({h.rule for h in hits})


def test_planted_8gram_is_caught(tmp_path):
    hits = _run(tmp_path, {"post.md": "Intro text. Here: Zephyr quokka MARMALADE orbit twelve lantern velvet harbor. Done."})
    assert rules(hits) == ["ngram"]
    assert hits[0].path.endswith("post.md") and hits[0].line == 1


def test_7_token_overlap_is_not_caught(tmp_path):
    hits = _run(tmp_path, {"post.md": "zephyr quokka marmalade orbit twelve lantern velvet then something else"})
    assert hits == []


def test_ngram_across_punctuation_and_lines(tmp_path):
    hits = _run(tmp_path, {"a.txt": "intro\nzephyr, quokka;\nmarmalade -- orbit twelve\nlantern velvet harbor\n"})
    assert rules(hits) == ["ngram"]
    assert hits[0].line == 2


def test_long_overlap_is_one_hit_per_run(tmp_path):
    hits = _run(tmp_path, {"a.txt": SECRET_SENTENCE})
    assert len(hits) == 1


def test_structured_state_is_flattened(tmp_path):
    state = {"premise": "alpha bravo charlie delta echo foxtrot golf hotel india", "hypothesis": "short"}
    hits = _run(tmp_path, {"a.txt": "x alpha bravo charlie delta echo foxtrot golf hotel x"}, states=[state])
    assert rules(hits) == ["ngram"]


def test_email_is_caught(tmp_path):
    hits = _run(tmp_path, {"a.md": f"contact someone{AT}example{DOT}com for details"})
    assert rules(hits) == ["email"]


def test_phone_is_caught(tmp_path):
    hits = _run(tmp_path, {"a.md": f"call ({DIGITS}) 123-" + "4567 today", "b.md": f"or {DIGITS}-987-" + "6543",
                           "c.md": f"+1 {DIGITS} 222 " + "3344"})
    assert rules(hits) == ["phone"]
    assert len(hits) == 3


def test_ipv4_and_tailnet_caught_loopback_ok(tmp_path):
    a = ".".join(["192", "168", "1", "20"])
    b = ".".join(["100", "64", "12", "9"])
    loop = ".".join(["127", "0", "0", "1"])
    anyaddr = ".".join(["0", "0", "0", "0"])
    hits = _run(tmp_path, {"a.md": f"host {a} and {b}", "b.md": f"listen on {loop} and {anyaddr}"})
    assert rules(hits) == ["ipv4"]
    assert len(hits) == 2
    assert all(h.path.endswith("a.md") for h in hits)


def test_version_strings_are_not_phone_or_ip(tmp_path):
    hits = _run(tmp_path, {"a.md": "python 3.12.13 and torch 2.9.1 built 2026-10-15 batch 4096 x 1024"})
    assert hits == []


def test_home_paths_caught(tmp_path):
    hits = _run(tmp_path, {"a.sh": "cd /" + "Users/someone/work\nls /" + "home/someone/x"})
    assert rules(hits) == ["home_path"]
    assert len(hits) == 2


def test_denylist_term_caught_case_insensitive(tmp_path):
    hits = _run(tmp_path, {"a.md": "Our client Acmecorp asked"}, denylist=["acmecorp"])
    assert rules(hits) == ["denylist"]


def test_clean_text_passes(tmp_path):
    hits = _run(tmp_path, {"a.md": "Quantization trades bits for speed. Calibration matters more than people expect."})
    assert hits == []


def test_binary_files_skipped_and_excluded_dirs_skipped(tmp_path):
    corpus = _corpus(tmp_path, [SECRET_SENTENCE])
    root = _tree(tmp_path, {"private/x.md": SECRET_SENTENCE, ".venv/lib/y.py": SECRET_SENTENCE, "ok.md": "fine"})
    (root / "img.bin").write_bytes(b"\x00\x01" + SECRET_SENTENCE.encode())
    hits = scan.scan_tree(root, scan.build_ngram_set(corpus), [])
    assert hits == []


def test_extra_exclude(tmp_path):
    hits = _run(tmp_path, {"skipme/x.md": SECRET_SENTENCE, "keep.md": "clean"}, extra=["skipme"])
    assert hits == []


def test_hits_report_hash_not_text(tmp_path, capsys):
    corpus = _corpus(tmp_path, [SECRET_SENTENCE])
    root = _tree(tmp_path, {"a.md": f"{SECRET_SENTENCE} and someone{AT}example{DOT}com"})
    code = scan.main([str(root), "--corpus", str(corpus), "--denylist", str(tmp_path / "none.txt")])
    out = capsys.readouterr()
    text = out.out + out.err
    assert code == 1
    assert "zephyr" not in text.lower() and "example" not in text.lower()
    assert "ngram" in text and "email" in text and "a.md" in text


def test_main_exit_zero_when_clean(tmp_path, capsys):
    corpus = _corpus(tmp_path, [SECRET_SENTENCE])
    root = _tree(tmp_path, {"a.md": "nothing to see here"})
    assert scan.main([str(root), "--corpus", str(corpus), "--denylist", str(tmp_path / "none.txt")]) == 0


def test_empty_corpus_is_an_error(tmp_path, capsys):
    empty = tmp_path / "records"
    empty.mkdir()
    root = _tree(tmp_path, {"a.md": "x"})
    assert scan.main([str(root), "--corpus", str(empty)]) == 2


def test_denylist_file_comments_and_blank_lines(tmp_path):
    f = tmp_path / "deny.txt"
    f.write_text("# comment\n\n  Foo Bar  \n")
    assert scan.load_denylist(f) == ["foo bar"]


def test_low_information_grams_are_ignored(tmp_path):
    states = ["list 1 2 3 4 5 6 7 8 9", "0 0 0 0 0 0 0 0 0 0 0", "ha ha ha ha ha ha ha ha ha ha"]
    hits = _run(tmp_path, {"a.md": "see list 1 2 3 4 5 6 7 8 9 and 0 0 0 0 0 0 0 0 0 0 0 and ha ha ha ha ha ha ha ha ha ha"},
                states=states)
    assert hits == []


def test_allowlisted_digest_is_suppressed(tmp_path, capsys):
    corpus = _corpus(tmp_path, [SECRET_SENTENCE])
    root = _tree(tmp_path, {"a.md": f"mail someone{AT}example{DOT}com"})
    args = [str(root), "--corpus", str(corpus), "--denylist", str(tmp_path / "none.txt"), "--json"]
    assert scan.main(args) == 1
    digest = json.loads(capsys.readouterr().out.splitlines()[0])["digest"]
    allow = tmp_path / "allow.txt"
    allow.write_text(f"# reviewed\n{digest}\n")
    assert scan.main(args + ["--allow", str(allow)]) == 0


def test_public_data_paths_check_only_ngram_and_denylist(tmp_path):
    email = f"someone{AT}example{DOT}com"
    phone = f"({DIGITS}) 123-" + "4567"
    ip = ".".join(["192", "168", "1", "20"])
    pii = f"{email} {phone} {ip} /" + "Users/someone/x"
    files = {
        "data/records/public.jsonl": f"{pii} Acmecorp {SECRET_SENTENCE}",
        "data/calibration.jsonl": pii,
        "data_dev/records/x.jsonl": pii,
        "data/other.jsonl": pii,
        "docs/post.md": pii,
    }
    hits = _run(tmp_path, files, denylist=["acmecorp"])
    by_file = {}
    for h in hits:
        by_file.setdefault(h.path, set()).add(h.rule)
    assert by_file["data/records/public.jsonl"] == {"ngram", "denylist"}
    assert "data/calibration.jsonl" not in by_file
    assert "data_dev/records/x.jsonl" not in by_file
    assert by_file["data/other.jsonl"] == {"email", "phone", "ipv4", "home_path"}
    assert by_file["docs/post.md"] == {"email", "phone", "ipv4", "home_path"}


def test_public_data_flag_overrides_default(tmp_path):
    corpus = _corpus(tmp_path, [SECRET_SENTENCE])
    root = _tree(tmp_path, {"pub/x.jsonl": f"someone{AT}example{DOT}com", "data/records/y.jsonl": f"a{AT}example{DOT}com"})
    args = [str(root), "--corpus", str(corpus), "--denylist", str(tmp_path / "none.txt"), "--json"]
    assert scan.main(args + ["--public-data", "pub"]) == 1  # data/records no longer special, pub/ is
    assert scan.main(args + ["--public-data", "pub", "--public-data", "data/records"]) == 0
