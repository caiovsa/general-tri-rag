#!/usr/bin/env python3
"""Easy cross-ablation: 3b graph + Phase 2 generic retriever (retrieval-only).

Adds a compatibility MENTIONS view from chunks to the movie/person nodes already
present in the 3b graph, runs Phase 2's vector + full-text + 1-hop neighbor
retrieval on the 3b graph, then removes the compatibility edges so the 3b graph
stays in its original state.

The Phase 2 generic retriever uses the persisted Phase 2 entity cache, so this
retrieval-only experiment makes no LLM calls. Query embeddings are still needed
for the vector index.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from neo4j import GraphDatabase

from shared.benchmark import evidence_recall, normalize_answer
from shared.config import dataset_config, settings
from phase2_graph_rag.retriever import (
    NEIGHBOR_QUERY, _RawStore, _row_chunk_text, cap_metaqa_chunks,
    embed_model, extract_entities_from_query,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "MetaQA/metaqa_eval_v1.jsonl"
TYPED_DETAILED = ROOT / "results/metaqa/phase3b/results_detailed.json"
DATABASE = "metaqa-ontology-b-graph"
TOP_K = 20


def contains(text: str, needle: str) -> bool:
    return normalize_answer(needle) in normalize_answer(text)


def score_contexts(rows: list[dict]) -> dict:
    gold, found, evidence = [], [], []
    for row in rows:
        context = "\n".join(row.get("contexts") or [])
        answers = [str(a) for a in (row.get("answers") or [])]
        present = [contains(context, answer) for answer in answers] if answers else []
        gold.append(sum(present) / len(present) if present else None)
        found.append(int(all(present)) if present else None)
        evidence.append(evidence_recall(row.get("contexts", []), row.get("supporting_facts_titles", []) or []))
    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None
    return {
        "n": len(rows),
        "gold_in_context_recall": mean(gold),
        "all_gold_in_context": mean(found),
        "all_gold_count": sum(1 for v in found if v == 1),
        "evidence_recall": mean(evidence),
    }


def add_mentions(driver) -> dict:
    with driver.session(database=DATABASE) as session:
        session.run(
            "MATCH (c:Chunk)-[:ABOUT]->(m:Movie) MERGE (c)-[:MENTIONS]->(m)"
        ).consume()
        session.run(
            "MATCH (c:Chunk)-[:ABOUT]->(m:Movie)-[:STARRED_ACTORS|DIRECTED_BY|WRITTEN_BY]->(p:Person) "
            "MERGE (c)-[:MENTIONS]->(p)"
        ).consume()
        return {"mentions": session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r)").single()[0]}


def remove_mentions(driver) -> None:
    with driver.session(database=DATABASE) as session:
        session.run("MATCH ()-[r:MENTIONS]->() DELETE r").consume()


def retrieve(dataset, question: str) -> list[str]:
    graph_store = _RawStore(dataset)
    try:
        embedding = embed_model.get_query_embedding(question)
        seed_limit = TOP_K // 2
        vector_results = graph_store.structured_query(
            f"""
            CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $embedding)
            YIELD node AS seed_chunk, score
            {NEIGHBOR_QUERY}
            """,
            {"embedding": embedding, "limit": seed_limit},
        ) or []
        entities = extract_entities_from_query(question)
        keyword_results = []
        for entity in entities:
            try:
                results = graph_store.structured_query(
                    f"""
                    CALL db.index.fulltext.queryNodes('chunk_text_fulltext', $keyword)
                    YIELD node AS seed_chunk, score
                    {NEIGHBOR_QUERY}
                    LIMIT $limit
                    """,
                    {"keyword": entity, "limit": 3},
                )
            except Exception:
                results = graph_store.structured_query(
                    f"""
                    MATCH (seed_chunk)
                    WHERE seed_chunk.text IS NOT NULL
                      AND toLower(seed_chunk.text) CONTAINS toLower($keyword)
                    WITH seed_chunk, 1.0 AS score
                    {NEIGHBOR_QUERY}
                    LIMIT $limit
                    """,
                    {"keyword": entity, "limit": 3},
                )
            keyword_results.extend(results or [])
        seeds = [_row_chunk_text(row) for row in vector_results]
        anchors = [_row_chunk_text(row) for row in keyword_results]
        neighbours = []
        for row in list(vector_results) + list(keyword_results):
            neighbours.extend(row.get("neighbor_texts") or [])
        entries, _ = cap_metaqa_chunks(seeds, anchors, neighbours, TOP_K, include_neighbours=True)
        return [entry["text"] for entry in entries]
    finally:
        graph_store.close()


def main() -> None:
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    base = dataset_config(phase=2)
    dataset = SimpleNamespace(neo4j_uri=base.neo4j_uri, neo4j_database=DATABASE,
                              tag_movies=True, top_k=TOP_K)
    eval_rows = [json.loads(line) for line in EVAL.read_text().splitlines() if line.strip()]
    typed = {row["id"]: row for row in json.load(TYPED_DETAILED.open())}
    adapter = {}
    try:
        adapter = add_mentions(driver)
        print("adapter:", adapter)
        rows = []
        for i, row in enumerate(eval_rows, start=1):
            contexts = retrieve(dataset, row["question"])
            rows.append({
                "id": row["id"],
                "question": row["question"],
                "contexts": contexts,
                "answers": row.get("answers") or [],
                "supporting_facts_titles": row.get("supporting_facts_titles") or [],
                "hop": row.get("hop"),
            })
            if i % 20 == 0 or i == len(eval_rows):
                print(f"retrieved {i}/{len(eval_rows)}")
        generic = score_contexts(rows)
        typed_rows = [{"contexts": typed[row["id"]].get("contexts", []),
                       "answers": row.get("answers") or [],
                       "supporting_facts_titles": row.get("supporting_facts_titles") or []}
                      for row in eval_rows]
        typed_score = score_contexts(typed_rows)
        # Remove compatibility edges; the 3b graph returns to its original relation count.
        remove_mentions(driver)
        adapter["removed"] = True
        result = {
            "cell": "3b graph + Phase 2 generic retriever",
            "not_identifiable": "Phase 2 graph + typed retriever",
            "adapter": adapter,
            "typed_3b": typed_score,
            "generic_3b": generic,
            "delta_generic_minus_typed": {
                key: (generic[key] - typed_score[key])
                for key in ["gold_in_context_recall", "all_gold_in_context", "evidence_recall"]
                if generic[key] is not None and typed_score[key] is not None
            },
        }
        out = ROOT / "results/diagnostics/phase3b_generic_cross_ablation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

        def fmt(v): return "—" if v is None else f"{v:.4f}"
        lines = [
            "# Easy cross-ablation — 3b graph, generic retriever (retrieval-only)",
            "",
            "No generation. The only graph modification is a temporary `MENTIONS` compatibility",
            "view from chunks to the movie/person nodes already in the 3b graph; it is removed",
            "after the run.",
            "",
            "| Retriever on 3b graph | n | gold-in-context recall | all gold in context | evidence recall |",
            "|---|---:|---:|---:|---:|",
            f"| typed 3a/3b retriever | {typed_score['n']} | {fmt(typed_score['gold_in_context_recall'])} | {fmt(typed_score['all_gold_in_context'])} | {fmt(typed_score['evidence_recall'])} |",
            f"| Phase 2 generic retriever | {generic['n']} | {fmt(generic['gold_in_context_recall'])} | {fmt(generic['all_gold_in_context'])} | {fmt(generic['evidence_recall'])} |",
            "",
            "The Phase 2 graph + typed retriever cell is **not identifiable**: the Phase 2 graph",
            "has generic free-text relationships and no typed Movie/Person edges.",
            "",
            "The generic retriever is a compatibility check, not a clean 2x2 identification.",
        ]
        (ROOT / "docs/RESULTS_CROSS_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("typed:", typed_score)
        print("generic:", generic)
        print(f"wrote {out.relative_to(ROOT)} and docs/RESULTS_CROSS_ABLATION.md")
    finally:
        try:
            remove_mentions(driver)
        except Exception:
            pass
        driver.close()


if __name__ == "__main__":
    main()
