"""Phase 1b: dense + BM25 hybrid retrieval, same prompt/model as Phase 1."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache

from qdrant_client import QdrantClient

from phase1_vector_rag.rag import extract_chunk_text
from phase1_vector_rag.retriever import retrieve_chunks
from shared.config import dataset_config, settings
from shared.llm import generate_completion
from shared.prompts import build_metaqa_prompt, format_context

TOKEN_RE = re.compile(r"[a-z0-9]+")
K1, B = 1.5, 0.75
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


class _BM25:
    def __init__(self, docs: list[str]):
        self.docs = docs
        self.tokens = [_tokenize(text) for text in docs]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter()
        for tokens in self.tokens:
            df.update(set(tokens))
        n = len(self.tokens)
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5))
                    for term, count in df.items()}

    def rank(self, query: str, limit: int) -> list[str]:
        q = _tokenize(query)
        scores = []
        for tokens, length in zip(self.tokens, self.lengths):
            tf = Counter(tokens)
            score = 0.0
            for term in q:
                freq = tf.get(term, 0)
                if freq:
                    score += self.idf.get(term, 0.0) * (freq * (K1 + 1)) / (
                        freq + K1 * (1 - B + B * length / self.avgdl))
            scores.append(score)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        return [self.docs[i] for i in order]


@lru_cache(maxsize=1)
def _chunks() -> list[str]:
    dataset = dataset_config(phase=1)
    qdrant = QdrantClient(url=settings.QDRANT_URL)
    docs, offset = [], None
    while True:
        points, offset = qdrant.scroll(dataset.qdrant_collection, limit=1000,
                                       offset=offset, with_payload=True,
                                       with_vectors=False)
        for point in points:
            raw = (point.payload or {}).get("_node_content", "")
            if not raw:
                continue
            try:
                text = json.loads(raw).get("text", "")
            except json.JSONDecodeError:
                continue
            if text:
                docs.append(text)
        if offset is None:
            break
    return docs


@lru_cache(maxsize=1)
def _bm25() -> _BM25:
    return _BM25(_chunks())


def _rrf(dense: list[str], sparse: list[str], limit: int = 20) -> list[str]:
    scores: dict[str, float] = {}
    for rank, text in enumerate(dense, start=1):
        scores[text] = scores.get(text, 0.0) + 1.0 / (RRF_K + rank)
    for rank, text in enumerate(sparse, start=1):
        scores[text] = scores.get(text, 0.0) + 1.0 / (RRF_K + rank)
    return [text for text, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def run_hybrid_rag_pipeline(user_query: str) -> dict:
    """Dense top-20 + BM25 top-50 -> RRF -> top-20, then the shared MetaQA prompt."""
    print(f"Executing Phase 1b hybrid RAG for: '{user_query}'\n")
    nodes = retrieve_chunks(user_query, top_k=20, apply_threshold=False)
    dense = [extract_chunk_text(node) for node in nodes]
    sparse = _bm25().rank(user_query, 50)
    contexts = _rrf(dense, sparse, 20)
    print(f"-> Dense {len(dense)}, BM25 {len(sparse)}, hybrid context {len(contexts)}/20.\n")
    prompt = build_metaqa_prompt(user_query, format_context(contexts))
    print(f"Generating answer using model: {settings.GENERATION_MODEL}...\n")
    answer = generate_completion(prompt)
    return {
        "answer": answer,
        "contexts": contexts,
        "retrieval": {"chunks_in_prompt": len(contexts), "dense": len(dense),
                      "bm25": len(sparse), "variant": "hybrid_rrf"},
    }


if __name__ == "__main__":
    print(run_hybrid_rag_pipeline("what movies did Cher act in")["answer"])
