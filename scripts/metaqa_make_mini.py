#!/usr/bin/env python3
"""Build the MetaQA mini track: a hop-balanced slice of the frozen corpus.

Run: uv run python -m scripts.metaqa_make_mini

Writes (never touching the full frozen artifacts):
  MetaQA/corpus_manifest_mini.txt  — --docs docs: the evidence docs of the sampled
                                     questions plus random distractors, so retrieval
                                     is not trivial
  MetaQA/metaqa_eval_mini.jsonl    — --per-hop questions per hop (seed 42), a
                                     subsequence of metaqa_eval_v1.jsonl

Why stratified: a slice of the corpus can only cover questions whose evidence movies
are all inside it. Picking the 50 first docs covers 6/212 questions, all hop 1 — a
50-doc slice cannot cover hop 3 at all unless the questions are chosen deliberately.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scripts.metaqa_sample import resolve_root

FULL_EVAL = "metaqa_eval_v1.jsonl"
FULL_MANIFEST = "corpus_manifest.txt"
MINI_EVAL = "metaqa_eval_mini.jsonl"
MINI_MANIFEST = "corpus_manifest_mini.txt"


def doc_title(path: str) -> str:
    """Movie title of a generated doc (its first line is "Title: <movie>")."""
    first = Path(path).read_text(encoding="utf-8").splitlines()[0]
    return first.split(":", 1)[1].strip() if first.lower().startswith("title:") else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-hop", type=int, default=4, help="questions sampled per hop level")
    ap.add_argument("--docs", type=int, default=50, help="total docs in the mini corpus")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = resolve_root()
    manifest = [line.strip() for line in (root / FULL_MANIFEST).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    eval_rows = [json.loads(line) for line in (root / FULL_EVAL).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    title_doc = {doc_title(path): path for path in manifest}

    rng = random.Random(args.seed)
    picked = []
    for hop in (1, 2, 3):
        rows = [row for row in eval_rows if row["hop"] == hop]
        picked += rng.sample(rows, min(args.per_hop, len(rows)))
    picked.sort(key=lambda row: eval_rows.index(row))  # keep frozen eval order

    evidence = {title for row in picked for title in row["supporting_facts_titles"]}
    unknown = sorted(evidence - set(title_doc))
    if unknown:
        raise SystemExit(f"{len(unknown)} evidence movies have no doc in the manifest, e.g. {unknown[:3]}")
    evidence_docs = sorted(title_doc[title] for title in evidence)

    budget = args.docs - len(evidence_docs)
    if budget < 0:
        raise SystemExit(f"--per-hop {args.per_hop} needs {len(evidence_docs)} evidence docs, "
                         f"more than --docs {args.docs}. Lower --per-hop or raise --docs.")
    pool = [path for path in manifest if path not in set(evidence_docs)]
    docs = sorted(evidence_docs + rng.sample(pool, min(budget, len(pool))))

    (root / MINI_MANIFEST).write_text("\n".join(docs) + "\n", encoding="utf-8")
    with open(root / MINI_EVAL, "w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    hops = Counter(row["hop"] for row in picked)
    print(f"{MINI_EVAL}: {len(picked)} questions {dict(sorted(hops.items()))}")
    print(f"{MINI_MANIFEST}: {len(docs)} docs = {len(evidence_docs)} evidence + {len(docs) - len(evidence_docs)} distractor")
    print(f"  -> {root / MINI_EVAL}\n  -> {root / MINI_MANIFEST}")


if __name__ == "__main__":
    main()
