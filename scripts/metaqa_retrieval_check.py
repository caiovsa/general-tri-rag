#!/usr/bin/env python3
"""Retrieval-only diagnostic for the MetaQA track (no LLM calls).

Run: DATASET=metaqa uv run python -m scripts.metaqa_retrieval_check [--phase 1]

For every question in the frozen eval set this retrieves the top 50 chunks ONCE
(Phase 1 vector search, similarity threshold disabled) and scores three metric
families at k = 8, 20, 50 by slicing that one ranked list, so the numbers line up
with what the phase actually feeds the model at --top-k:

- evidence_recall@k / full_hit@k — evidence movies whose [Movie: <title>] tag is
  in the top-k chunks (shared.benchmark.evidence_recall).
- answer_in_context_recall@k / full_answer@k — gold `answers` present in the
  concatenated top-k text (shared.benchmark.gold_recall, same normalizer).
- triple_recall@k — evidence triples carried by a top-k chunk of that movie: the
  chunk's tag matches the triple's movie AND its content passes the audit's
  per-relation check (scripts.metaqa_audit.check_triple — name containment with
  initials normalization, infobox `Language:` line for in_language, first 1500
  chars for release_year/has_genre with the genre synonyms).

It also records where the first evidence chunk ranked, which answers "how far
below the cut did the right movie land?".

No generation: the only API calls are query embeddings.

Outputs: results/metaqa/phase<phase>/retrieval_check.jsonl (one row per question).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from phase1_vector_rag.retriever import retrieve_chunks
from scripts.metaqa_audit import check_triple, doc_from_text
from shared.benchmark import evidence_recall, gold_recall
from shared.config import dataset_config

KS = (8, 20, 50)
TOP_K_MAX = max(KS)
BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10 ** 9, "6+"))
TAG_RE = re.compile(r"\[Movie:\s*(.+?)\]")


def bucket_of(size: int) -> str:
    for low, high, label in BUCKETS:
        if low <= size <= high:
            return label
    return BUCKETS[-1][2]


def movies_in(text: str) -> list[str]:
    """Movie titles tagged in a chunk ([Movie: <title>] is prepended by ingest)."""
    return TAG_RE.findall(text or "")


def triple_support_ranks(triples: list, texts: list[str]) -> dict[int, int | None]:
    """Rank of the first chunk supporting each evidence triple (1-based), else None.

    A chunk supports a triple when it carries that movie's tag and its content
    passes the audit's per-relation check (check_triple), so genres use the
    synonym map and names use the initials normalization.
    """
    tags = [set(movies_in(text)) for text in texts]
    docs: dict[int, dict] = {}
    ranks: dict[int, int | None] = {}
    for index, triple in enumerate(triples):
        movie = triple[0]
        ranks[index] = None
        for rank, text in enumerate(texts, 1):
            if movie not in tags[rank - 1]:
                continue
            doc = docs.setdefault(rank - 1, doc_from_text(text))
            if check_triple(triple, doc)[0]:
                ranks[index] = rank
                break
    return ranks


def evaluate(question: dict, nodes) -> dict:
    """Evidence / answer-in-context / triple metrics for one question."""
    titles = question.get("supporting_facts_titles") or []
    answers = question.get("answers") or []
    triples = question.get("evidence_triples") or []
    texts = [node.text for node in nodes]

    ranks: dict[str, int] = {}
    for rank, text in enumerate(texts, 1):
        for movie in movies_in(text):
            ranks.setdefault(movie, rank)
    evidence_ranks = {title: ranks.get(title) for title in titles}
    found_ranks = [rank for rank in evidence_ranks.values() if rank]
    triple_ranks = triple_support_ranks(triples, texts)

    row = {
        "n_chunks": len(texts),
        "rank_first_evidence": min(found_ranks) if found_ranks else None,
        "evidence_ranks": evidence_ranks,
        "distinct_movies@8": len({movie for text in texts[:8] for movie in movies_in(text)}),
        "missing_triples": [f"{m}|{rel}|{obj}" for i, (m, rel, obj) in enumerate(triples)
                            if triple_ranks[i] is None],
    }
    for k in KS:
        recall = evidence_recall(texts[:k], titles)
        row[f"evidence_recall@{k}"] = recall
        row[f"full_hit@{k}"] = None if recall is None else int(recall == 1)

        answer_recall = gold_recall("\n".join(texts[:k]), answers)
        row[f"answer_in_context_recall@{k}"] = answer_recall
        row[f"full_answer@{k}"] = None if answer_recall is None else int(answer_recall == 1)

        found = sum(1 for rank in triple_ranks.values() if rank and rank <= k)
        row[f"triple_recall@{k}"] = found / len(triples) if triples else None
    return row


def groups(rows: list[dict]) -> list[tuple[str, list[int]]]:
    """Overall, per hop, pooled hop 2+3, per anchor, per hop x anchor, per evidence bucket."""
    result = [("ALL", list(range(len(rows))))]
    hops = sorted({row["hop"] for row in rows if row["hop"] is not None})
    for hop in hops:
        result.append((f"hop {hop}", [i for i, r in enumerate(rows) if r["hop"] == hop]))
    if any(hop >= 2 for hop in hops):
        result.append(("hop 2+3", [i for i, r in enumerate(rows) if (r["hop"] or 0) >= 2]))
    anchors = sorted({row["anchor"] for row in rows})
    for anchor in anchors:
        result.append((f"anchor {anchor}", [i for i, r in enumerate(rows) if r["anchor"] == anchor]))
    for hop in hops:
        for anchor in anchors:
            if hop == 2 and anchor == "movie":
                continue  # n=11 — too small to read, dropped on purpose
            idx = [i for i, r in enumerate(rows) if r["hop"] == hop and r["anchor"] == anchor]
            if idx:
                result.append((f"hop {hop} x {anchor}", idx))
    for low, high, label in BUCKETS:
        idx = [i for i, r in enumerate(rows) if low <= r["evidence_n"] <= high]
        if idx:
            result.append((f"evidence {label}", idx))
    return [(name, idx) for name, idx in result if idx]


def mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def print_table(title: str, rows: list[dict], specs: list[tuple[str, str]]) -> None:
    """specs = [(column header, row key)] over the shared group rows."""
    head = f"{'group':<20} {'n':>4} " + " ".join(f"{header:>9}" for header, _ in specs)
    print(f"\n{title}")
    print(head)
    print("-" * len(head))
    for name, idx in groups(rows):
        subset = [rows[i] for i in idx]
        cells = " ".join(f"{(mean([row[key] for row in subset]) or 0):>9.3f}" for _, key in specs)
        print(f"{name:<20} {len(subset):>4} {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", type=int, choices=(1, 2, 3), default=1,
                    help="phase to check (only phase 1 is implemented)")
    args = ap.parse_args()
    if args.phase != 1:
        raise SystemExit(f"Retrieval check for phase {args.phase} is not implemented yet — "
                         "only phase 1 is. Phases 2 and 3 need their own retriever wiring.")

    dataset = dataset_config(phase=args.phase)
    if dataset.name != "metaqa":
        raise SystemExit("This check is MetaQA-only. Run it as: "
                         "DATASET=metaqa uv run python -m scripts.metaqa_retrieval_check")

    questions = [json.loads(line) for line in
                 dataset.eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"dataset={dataset.name} | phase={args.phase} | collection={dataset.qdrant_collection} | "
          f"questions={len(questions)} | retrieving top {TOP_K_MAX} (threshold ignored, no LLM)")

    rows = []
    for question in tqdm(questions, desc="retrieval check"):
        nodes = retrieve_chunks(question["question"], top_k=TOP_K_MAX, node_threshold=-1.0)
        qtype = question.get("qtype", "")
        rows.append({
            "id": question.get("id"),
            "question": question["question"],
            "hop": question.get("hop"),
            "qtype": qtype,
            "anchor": "movie" if qtype.startswith("movie_") else "person",
            "evidence_n": len(question.get("supporting_facts_titles") or []),
            "evidence_movies": question.get("supporting_facts_titles") or [],
            **evaluate(question, nodes),
        })

    evidence_specs = ([(f"rec@{k}", f"evidence_recall@{k}") for k in KS]
                      + [(f"full@{k}", f"full_hit@{k}") for k in KS]
                      + [("movies@8", "distinct_movies@8")])
    answer_specs = ([(f"ans@{k}", f"answer_in_context_recall@{k}") for k in KS]
                    + [(f"fullA@{k}", f"full_answer@{k}") for k in KS])
    triple_specs = [(f"triple@{k}", f"triple_recall@{k}") for k in KS]
    print_table("Evidence recall by group (movies whose [Movie: ...] tag is retrieved)", rows, evidence_specs)
    print_table("Answer-in-context recall by group (gold answers present in the text)", rows, answer_specs)
    print_table("Triple recall by group (evidence triples carried by a retrieved chunk)", rows, triple_specs)

    missing = [row for row in rows if row["n_chunks"] == 0]
    if missing:
        print(f"\nWARNING: {len(missing)} question(s) returned no chunks at all")
    never_found = [row for row in rows if row["rank_first_evidence"] is None]
    print(f"\nQuestions with no evidence chunk in the top {TOP_K_MAX}: "
          f"{len(never_found)}/{len(rows)}")
    deep = sorted((row for row in rows if row["rank_first_evidence"]),
                  key=lambda r: -r["rank_first_evidence"])[:5]
    if deep:
        print("Deepest first-evidence ranks:")
        for row in deep:
            print(f"  rank {row['rank_first_evidence']:>2} | hop{row['hop']} {row['qtype']:<38} "
                  f"{row['question'][:52]}")

    out_path = dataset.results_file.parent / "retrieval_check.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nPer-question rows saved to: {out_path}")


if __name__ == "__main__":
    main()
