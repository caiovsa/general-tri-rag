#!/usr/bin/env python3
"""Phase 1b retrieval-only hybrid check (dense + BM25, no generation).

Uses the frozen P1 dense top-20 context (no regeneration) and builds an in-memory
BM25 index over the same 4,522 MetaQA chunks from Qdrant. Reports dense, BM25-only,
and RRF hybrid retrieval metrics. No LLM calls.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient

from shared.benchmark import evidence_recall, normalize_answer
from shared.config import dataset_config, settings

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "MetaQA/metaqa_eval_v1.jsonl"
DENSE_DETAILED = ROOT / "results/metaqa/phase1/results_detailed.json"
TOKEN_RE = re.compile(r"[a-z0-9]+")
K1, B = 1.5, 0.75
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


class BM25:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.tokens = [tokenize(d["text"]) for d in docs]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.df: Counter = Counter()
        for tokens in self.tokens:
            self.df.update(set(tokens))
        n = len(self.tokens)
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5))
                    for term, count in self.df.items()}

    def score(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = []
        for tokens, length in zip(self.tokens, self.lengths):
            tf = Counter(tokens)
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                freq = tf[term]
                score += self.idf.get(term, 0.0) * (freq * (K1 + 1)) / (
                    freq + K1 * (1 - B + B * length / self.avgdl))
            scores.append(score)
        return scores


def rank_dense(detail: dict) -> list[str]:
    return [str(text) for text in (detail.get("contexts") or [])[:20]]


def rank_bm25(bm25: BM25, question: str, limit: int) -> list[str]:
    scores = bm25.score(question)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
    return [bm25.docs[i]["text"] for i in order]


def rrf(dense: list[str], sparse: list[str], limit: int = 20) -> list[str]:
    scores: dict[str, float] = {}
    for rank, text in enumerate(dense, start=1):
        scores[text] = scores.get(text, 0.0) + 1.0 / (RRF_K + rank)
    for rank, text in enumerate(sparse, start=1):
        scores[text] = scores.get(text, 0.0) + 1.0 / (RRF_K + rank)
    return [text for text, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def contains(text: str, needle: str) -> bool:
    return normalize_answer(needle) in normalize_answer(text)


def metrics(data: list[dict]) -> dict:
    gold_scores, found, evidence = [], [], []
    for item in data:
        context = "\n".join(item.get("contexts") or [])
        answers = [str(answer) for answer in (item.get("answers") or [])]
        if answers:
            present = [contains(context, answer) for answer in answers]
            gold_scores.append(sum(present) / len(present))
            found.append(int(all(present)))
        else:
            gold_scores.append(None)
            found.append(None)
        evidence.append(evidence_recall(item.get("contexts", []), item.get("supporting_facts", []) or []))
    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None
    return {
        "gold_in_context_recall": mean(gold_scores),
        "all_gold_in_context": mean(found),
        "all_found_count": sum(1 for v in found if v == 1),
        "evidence_recall": mean(evidence),
        "n": len(data),
    }


def main() -> None:
    dataset = dataset_config(phase=1)
    qdrant = QdrantClient(url=settings.QDRANT_URL)
    docs, offset = [], None
    while True:
        points, offset = qdrant.scroll(
            dataset.qdrant_collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False)
        for point in points:
            payload = point.payload or {}
            raw = payload.get("_node_content", "")
            text = ""
            if raw:
                try:
                    text = json.loads(raw).get("text", "")
                except json.JSONDecodeError:
                    pass
            if text:
                docs.append({"text": text, "movie": payload.get("movie_title")})
        if offset is None:
            break
    print(f"BM25 chunks: {len(docs)}")
    bm25 = BM25(docs)
    dense_by_id = {row["id"]: row for row in json.load(DENSE_DETAILED.open())}
    eval_rows = [json.loads(line) for line in EVAL.read_text().splitlines() if line.strip()]

    variants = {"dense": [], "bm25": [], "hybrid": []}
    for row in eval_rows:
        qid = row["id"]
        dense = rank_dense(dense_by_id[qid]) if qid in dense_by_id else []
        sparse = rank_bm25(bm25, row["question"], 50)
        contexts = {
            "dense": dense,
            "bm25": sparse[:20],
            "hybrid": rrf(dense, sparse, 20),
        }
        for name, ctxs in contexts.items():
            variants[name].append({
                "question": row["question"],
                "contexts": ctxs,
                "answers": row.get("answers") or [],
                "supporting_facts": row.get("supporting_facts_titles") or [],
                "hop": row.get("hop"),
                "qtype": row.get("qtype"),
            })

    result = {}
    for name, data in variants.items():
        print(f"\n--- {name} ---")
        result[name] = metrics(data)
        print(result[name])

    out = ROOT / "results/diagnostics/phase1b_hybrid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def fmt(v): return "—" if v is None else f"{v:.4f}"
    lines = [
        "# Phase 1b — dense vs BM25 vs hybrid (retrieval-only)",
        "",
        "No generation. Dense ranking is the frozen P1 top-20 context; BM25 is built",
        "over the same 4,522 MetaQA chunks; hybrid is reciprocal-rank fusion (RRF, k=60).",
        "",
        "| Retriever | n | gold-in-context recall | all gold in context | evidence recall | all-gold count |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["dense", "bm25", "hybrid"]:
        r = result[name]
        lines.append(f"| {name} | {r['n']} | {fmt(r['gold_in_context_recall'])} | {fmt(r['all_gold_in_context'])} | {fmt(r['evidence_recall'])} | {r['all_found_count']} |")
    lines += [
        "",
        "Dense is the same retrieval as Phase 1, so its metrics should match the P1",
        "retrieval numbers (gold .5538 / all_found .4764 / evidence .6550).",
        "",
        "This is a retrieval-only screen. Run answers only if the hybrid closes a",
        "meaningful part of the dense→graph gap.",
    ]
    (ROOT / "docs/RESULTS_PHASE1B.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)} and docs/RESULTS_PHASE1B.md")


if __name__ == "__main__":
    main()
