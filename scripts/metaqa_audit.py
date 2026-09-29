#!/usr/bin/env python3
"""Audit MetaQA candidates against their fetched Wikipedia docs.

Run: uv run python -m scripts.metaqa_audit

A question is kept only if
  (a) every movie in `evidence_movies` has a verified doc (per fetch_report.json
      and the file on disk) — including the topic movie for movie-anchored chains, and
  (b) every triple in `evidence_triples` passes its per-relation check against
      that triple's own movie doc.

Outputs: <base>/metaqa_eval.jsonl (hotpot_eval.jsonl shape) and
<base>/audit_dropped.jsonl (first failing reason per dropped question).
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from scripts.metaqa_fetch_wiki import safe_name
from scripts.metaqa_sample import load_kb, resolve_root

YEAR_WINDOW = 1500
MIN_PER_HOP = 30
NAME_RELATIONS = ("directed_by", "written_by", "starred_actors")
GENRE_SYNONYMS = {
    "music": ("music", "musical"),
    "musical": ("musical",),
    "sci-fi": ("science fiction", "sci-fi"),
    "sport": ("sport", "sports"),
    "biography": ("biography", "biographical", "biopic"),
    "animation": ("animation", "animated"),
    "romance": ("romance", "romantic"),
    "comedy": ("comedy", "comedic"),
    "family": ("family",),
    "short": ("short film",),
}
LANGUAGE_SYNONYMS = {"filipino": ("tagalog",)}  # the only language synonym


def norm(text: str) -> str:
    """lowercase, strip accents and punctuation, drop a leading article.

    Runs of capitalised single-letter initials collapse, so "A. J. Bowen",
    "A.J. Bowen" and "AJ Bowen" all normalise to "aj bowen". Capitalisation is
    read before lowercasing so a lowercase article is not absorbed into a run
    ("a J. Edgar Hoover biopic" keeps its own "a").
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    words, run = [], ""
    for token in re.findall(r"[A-Za-z]+|\d+", text):
        if len(token) == 1 and token.isupper():
            run += token
            continue
        if run:
            words.append(run)
            run = ""
        words.append(token)
    if run:
        words.append(run)
    words = [word.lower() for word in words]
    if words and words[0] in ("the", "a", "an"):
        words = words[1:]
    return " ".join(words)


def mentions(haystack_norm: str, name: str) -> bool:
    needle = norm(name)
    return bool(needle) and re.search(rf"\b{re.escape(needle)}\b", haystack_norm) is not None


def load_doc(path: Path) -> dict:
    """Parse a generated doc: raw text, first 1500 chars, and the infobox fields."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().lower() == "infobox:"), None)
    fields: dict[str, str] = {}
    if start is not None:
        for line in lines[start + 1:]:
            if not line.strip():
                break
            if ":" in line:
                key, value = line.split(":", 1)
                fields.setdefault(key.strip().lower(), value.strip())
    head = text[:YEAR_WINDOW]
    return {"text": text, "head": head, "norm": norm(text), "head_norm": norm(head),
            "language": norm(fields.get("language", ""))}


def genre_match(head_norm: str, genre: str):
    """Exact genre first, then the small synonym map. Returns the match kind."""
    if mentions(head_norm, genre):
        return "exact"
    for synonym in GENRE_SYNONYMS.get(genre.lower(), ()):
        if mentions(head_norm, synonym):
            return f"synonym:{synonym}"
    return None


def language_match(language_norm: str, language: str):
    """Exact language first, then the single Filipino -> Tagalog synonym."""
    if mentions(language_norm, language):
        return "exact"
    for synonym in LANGUAGE_SYNONYMS.get(language.lower(), ()):
        if mentions(language_norm, synonym):
            return f"synonym:{synonym}"
    return None


def check_triple(triple, doc) -> tuple[bool, str, str]:
    """Per-relation check against the triple's own movie doc: (ok, reason, match_kind)."""
    _, rel, obj = triple
    if rel in NAME_RELATIONS:
        if mentions(doc["norm"], obj):
            return True, "", ""
        return False, f"{rel}: name not found in doc", ""
    if rel == "in_language":
        kind = language_match(doc["language"], obj)
        if kind:
            return True, "", kind
        return False, f"in_language: '{obj}' not in infobox Language line", ""
    if rel == "release_year":
        if obj in doc["head"]:
            return True, "", ""
        return False, f"release_year: {obj} not in first {YEAR_WINDOW} chars", ""
    if rel == "has_genre":
        kind = genre_match(doc["head_norm"], obj)
        if kind:
            return True, "", kind
        return False, f"has_genre: '{obj}' not in first {YEAR_WINDOW} chars", ""
    return True, "", "unchecked_relation"


def audit(rows, verified, docs_dir, kb_triples):
    kept, dropped = [], []
    passed, failed = Counter(), Counter()
    failures = Counter()
    kinds = Counter()
    no_doc_triples = not_in_kb = 0
    doc_cache: dict[str, dict | None] = {}

    def doc_for(title):
        if title not in doc_cache:
            path = docs_dir / f"{safe_name(title)}.txt"
            doc_cache[title] = load_doc(path) if path.exists() else None
        return doc_cache[title]

    for row in rows:
        missing = next((m for m in row["evidence_movies"] if m not in verified), None)
        first = {"reason": "movie missing", "movie": missing} if missing else None

        for triple in row["evidence_triples"]:
            movie, rel, obj = triple
            if tuple(triple[1:]) not in kb_triples.get(movie, ()):
                not_in_kb += 1
                if first is None:
                    first = {"reason": "triple not in kb.txt", "triple": triple, "relation": rel}
                continue
            doc = doc_for(movie)
            if doc is None:
                no_doc_triples += 1
                continue
            ok, reason, kind = check_triple(triple, doc)
            if ok:
                passed[rel] += 1
                if kind:
                    kinds[kind] += 1
            else:
                failed[rel] += 1
                failures[(rel, reason)] += 1
                if first is None:
                    first = {"reason": "triple failed", "triple": triple, "relation": rel, "detail": reason}

        if first is None:
            kept.append(row)
        else:
            dropped.append({"id": row["id"], "qtype": row["qtype"], "hop": row["hop"], **first})

    return kept, dropped, passed, failed, failures, kinds, no_doc_triples, not_in_kb


def report(rows, kept, dropped, passed, failed, failures, kinds, no_doc_triples, not_in_kb):
    print(f"\nSurvival: {len(kept):,}/{len(rows):,} questions kept "
          f"({len(kept) / max(len(rows), 1):.1%}), {len(dropped):,} dropped")
    if no_doc_triples:
        print(f"  triples skipped (movie doc missing): {no_doc_triples:,}")
    if not_in_kb:
        print(f"  WARNING: {not_in_kb:,} triples are not in kb.txt")

    print("\nPer hop")
    for hop in sorted({r["hop"] for r in rows}):
        total = sum(1 for r in rows if r["hop"] == hop)
        keep = sum(1 for r in kept if r["hop"] == hop)
        print(f"  {hop}-hop: {keep:>4,}/{total:<5,} kept ({keep / total:.1%})")

    print("\nPer qtype")
    for qtype in sorted({r["qtype"] for r in rows}):
        total = sum(1 for r in rows if r["qtype"] == qtype)
        keep = sum(1 for r in kept if r["qtype"] == qtype)
        print(f"  {qtype:<40} {keep:>3,}/{total:<4,} ({keep / total:>5.1%})")

    print("\nPer relation (triples checked, all candidates with a doc)")
    for rel in sorted(set(passed) | set(failed)):
        ok, bad = passed[rel], failed[rel]
        print(f"  {rel:<18} passed {ok:>6,}  failed {bad:>5,}  ({ok / (ok + bad):.1%} pass)")

    print("\nTop 10 failing (relation, reason)")
    for (rel, reason), count in failures.most_common(10):
        print(f"  {count:>5,}  {rel}: {reason}")

    if kinds:
        genre_kinds = {k: v for k, v in kinds.items() if k != "unchecked_relation"}
        print("\nMatch kinds (genre/language): " + ", ".join(f"{k}={v:,}" for k, v in sorted(genre_kinds.items())))
        if kinds["unchecked_relation"]:
            print(f"  WARNING: {kinds['unchecked_relation']:,} triples used an unchecked relation")

    print("\nKept questions with a single evidence movie, per hop")
    for hop in sorted({r["hop"] for r in rows}):
        single = sum(1 for r in kept if r["hop"] == hop and len(r["evidence_movies"]) == 1)
        print(f"  {hop}-hop: {single:,}")

    low = [(hop, sum(1 for r in kept if r["hop"] == hop)) for hop in sorted({r["hop"] for r in rows})]
    low = [(hop, n) for hop, n in low if n < MIN_PER_HOP]
    if low:
        print("\n" + "!" * 72)
        for hop, n in low:
            print(f"!! WARNING: {hop}-hop has only {n} kept questions (< {MIN_PER_HOP})")
        print("!! Fetch more docs (scripts.metaqa_fetch_wiki) before benchmarking.")
        print("!" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", default="metaqa_candidates.jsonl")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--report", default="fetch_report.json")
    args = ap.parse_args()

    root = resolve_root()
    base = root.parent if root.name == "raw" else root
    rows = [json.loads(line) for line in
            (base / args.candidates).read_text(encoding="utf-8").splitlines() if line.strip()]
    fetch_report = json.loads((base / args.report).read_text(encoding="utf-8"))
    docs_dir = base / args.docs
    _, _, _, kb_triples = load_kb(root)

    resolved = {entry["title"] for entry in fetch_report["resolved"]}
    verified = {title for title in resolved if (docs_dir / f"{safe_name(title)}.txt").exists()}
    print(f"base: {base} | candidates: {len(rows):,} | fetch_report resolved: {len(resolved):,} | "
          f"verified docs on disk: {len(verified):,}")
    if len(verified) < len(resolved):
        print(f"  WARNING: {len(resolved) - len(verified):,} resolved titles have no doc file")

    kept, dropped, *stats = audit(rows, verified, docs_dir, kb_triples)
    eval_path, dropped_path = base / "metaqa_eval.jsonl", base / "audit_dropped.jsonl"
    with open(eval_path, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps({"question": row["question"], "answer": row["answer"],
                                     "supporting_facts_titles": row["evidence_movies"],
                                     "answers": row["answers"], "hop": row["hop"], "qtype": row["qtype"],
                                     "id": row["id"], "evidence_triples": row["evidence_triples"]},
                                    ensure_ascii=False) + "\n")
    with open(dropped_path, "w", encoding="utf-8") as handle:
        for row in dropped:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report(rows, kept, dropped, *stats)
    print(f"\nWrote {len(kept):,} kept questions -> {eval_path}")
    print(f"Wrote {len(dropped):,} dropped questions -> {dropped_path}")


if __name__ == "__main__":
    main()
