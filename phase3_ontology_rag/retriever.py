"""
Phase 3a — Ontology RAG retrieval (typed KB graph, no extraction noise).

For a question:
  1. Match the topic entity against the typed graph (longest KB name contained
     in the question; no LLM, no qtype leakage).
  2. Walk only typed person edges (STARRED_ACTORS / DIRECTED_BY / WRITTEN_BY)
     up to max_hops, collecting the movies on the chain:
        movie start  -> movies at distance 0 and 2
        person start -> movies at distance 1 and 3
  3. Rank those movies' chunks with the chunk vector index, then fill from the
     candidate movies by graph distance, capped to top_k.

The result is an upper bound on graph retrieval quality: the graph is the KB
itself, so failures point at linking/coverage, not at the LLM extractor.
"""

from __future__ import annotations

import re
import json

import llama_index.core
from neo4j import GraphDatabase

from phase3_ontology_rag.schema import PERSON_EDGE_TYPES
from shared.config import dataset_config, get_llamaindex_settings, phase3_database, settings

_, embed_model = get_llamaindex_settings()
llama_index.core.Settings.embed_model = embed_model

_NAME_CACHE: dict[tuple[str, str], dict] = {}
MAX_HOPS = 3
FRONTIER_CAP = 2000
MIN_NAME_LEN = 3


def _driver(dataset):
    return GraphDatabase.driver(dataset.neo4j_uri,
                                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))


def _load_names(driver, dataset, database: str) -> dict:
    key = (dataset.neo4j_uri, database)
    if key in _NAME_CACHE:
        return _NAME_CACHE[key]
    with driver.session(database=database) as session:
        movies = [row["title"] for row in session.run(
            "MATCH (m:Movie) RETURN m.title AS title ORDER BY title")]
        persons = [row["name"] for row in session.run(
            "MATCH (p:Person) RETURN p.name AS name ORDER BY name")]
    _NAME_CACHE[key] = {"movies": movies, "persons": persons}
    return _NAME_CACHE[key]


def _contains(question_lower: str, name: str) -> bool:
    needle = name.lower()
    if len(needle) < MIN_NAME_LEN or needle not in question_lower:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", question_lower) is not None


def match_entity(question: str, names: dict) -> tuple[str, str] | None:
    """Longest typed KB name contained in the question (movie wins a tie)."""
    question_lower = question.lower()
    best = None
    for kind, collection in (("Movie", names["movies"]), ("Person", names["persons"])):
        for name in collection:
            if _contains(question_lower, name):
                candidate = (len(name), 1 if kind == "Movie" else 0, name, kind)
                if best is None or candidate > best:
                    best = candidate
    return (best[2], best[3]) if best else None


def _neighbours(session, kind: str, keys: list[str], limit: int) -> list[str]:
    if not keys:
        return []
    if kind == "Movie":
        query = (
            "UNWIND $keys AS key "
            "MATCH (m:Movie {title: key})-[r:STARRED_ACTORS|DIRECTED_BY|WRITTEN_BY]->(p:Person) "
            "RETURN DISTINCT p.name AS name ORDER BY name LIMIT $limit"
        )
    else:
        query = (
            "UNWIND $keys AS key "
            "MATCH (p:Person {name: key})<-[r:STARRED_ACTORS|DIRECTED_BY|WRITTEN_BY]-(m:Movie) "
            "RETURN DISTINCT m.title AS name ORDER BY name LIMIT $limit"
        )
    return [row["name"] for row in session.run(query, keys=keys, limit=limit)]


def expand_movies(session, entity: str, kind: str) -> tuple[dict[str, int], bool]:
    """BFS over typed person edges; returns {movie_title: distance} and truncation flag."""
    movies: dict[str, int] = {}
    if kind == "Movie":
        movies[entity] = 0
    frontier = [entity]
    current_kind = kind
    truncated = False
    for hop in range(1, MAX_HOPS + 1):
        neighbours = _neighbours(session, current_kind, frontier, FRONTIER_CAP)
        truncated = truncated or len(neighbours) >= FRONTIER_CAP
        next_kind = "Person" if current_kind == "Movie" else "Movie"
        current_kind = next_kind
        frontier = neighbours
        if not frontier:
            break
        if current_kind == "Movie":
            for title in frontier:
                movies.setdefault(title, hop)
    # Evidence movies only sit at even distances for a movie topic and odd for a person.
    parity = 0 if kind == "Movie" else 1
    return {title: dist for title, dist in movies.items() if dist % 2 == parity}, truncated


def _vector_hits(session, embedding, limit: int) -> list[dict]:
    return list(session.run(
        """
        CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $embedding)
        YIELD node, score
        OPTIONAL MATCH (node)-[:ABOUT]->(m:Movie)
        RETURN node.text AS text, score, m.title AS movie
        """,
        embedding=embedding, limit=limit,
    ))


def _chunks_for_titles(session, candidates: dict[str, int]) -> list[dict]:
    if not candidates:
        return []
    rows = [{"title": title, "distance": distance} for title, distance in candidates.items()]
    return list(session.run(
        "UNWIND $rows AS row MATCH (c:Chunk)-[:ABOUT]->(m:Movie {title: row.title}) "
        "RETURN c.text AS text, m.title AS movie, row.distance AS distance "
        "ORDER BY row.distance, m.title, c.id",
        rows=rows,
    ))


def retrieve_ontology_chunks(user_query: str, top_k: int | None = None):
    dataset = dataset_config(phase=3)
    database = phase3_database(dataset)
    if top_k is None:
        top_k = dataset.top_k
    empty = {"chunks_in_prompt": 0, "seeds": 0, "anchors": 0, "neighbours": 0,
             "candidates": 0, "movies_used": 0, "entity": None, "entity_type": None,
             "fallback": False}
    if not user_query or not user_query.strip():
        return [], empty

    driver = _driver(dataset)
    try:
        with driver.session(database=database) as session:
            session.run("RETURN 1 AS ok").consume()
            names = _load_names(driver, dataset, database)
            match = match_entity(user_query, names)

            if match is None:
                # No typed entity: fall back to plain vector search (recorded).
                embedding = embed_model.get_query_embedding(user_query)
                hits = _vector_hits(session, embedding, max(top_k * 20, 200))
                entries, seen = [], set()
                for row in hits:
                    if row["text"] and row["text"] not in seen:
                        seen.add(row["text"])
                        entries.append({"text": row["text"], "source": "vector_fallback",
                                        "movie": row["movie"], "distance": None})
                    if len(entries) >= top_k:
                        break
                counts = {**empty, "chunks_in_prompt": len(entries), "seeds": len(entries),
                          "fallback": True}
                return entries, counts

            entity, kind = match
            with driver.session(database=database) as session:
                candidates, truncated = expand_movies(session, entity, kind)
                embedding = embed_model.get_query_embedding(user_query)
                hits = _vector_hits(session, embedding, max(top_k * 20, 200))
                entries, seen = [], set()
                for row in hits:
                    if row["movie"] not in candidates or not row["text"] or row["text"] in seen:
                        continue
                    seen.add(row["text"])
                    entries.append({"text": row["text"], "source": "graph_vector",
                                    "movie": row["movie"], "distance": candidates[row["movie"]],
                                    "score": row["score"]})
                    if len(entries) >= top_k:
                        break

                if len(entries) < top_k:
                    for row in _chunks_for_titles(session, candidates):
                        if not row["text"] or row["text"] in seen:
                            continue
                        seen.add(row["text"])
                        entries.append({"text": row["text"], "source": "graph_only",
                                        "movie": row["movie"], "distance": candidates[row["movie"]]})
                        if len(entries) >= top_k:
                            break

        counts = {
            "chunks_in_prompt": len(entries),
            "seeds": len(entries),
            "anchors": 0,
            "neighbours": 0,
            "candidates": len(candidates),
            "movies_used": len({entry["movie"] for entry in entries}),
            "entity": entity,
            "entity_type": kind,
            "fallback": False,
            "truncated": truncated,
            "hops": MAX_HOPS,
        }
        print(f"-> Ontology retrieval: entity={entity!r} ({kind}), candidates={len(candidates)}, "
              f"chunks={len(entries)}/{top_k}, truncated={truncated}")
        return entries, counts
    finally:
        driver.close()


def retrieve_ontology_contexts(user_query: str, top_k: int | None = None) -> list[str]:
    entries, _ = retrieve_ontology_chunks(user_query, top_k)
    return [entry["text"] for entry in entries]


if __name__ == "__main__":
    query = "what movies did Cher act in"
    entries, counts = retrieve_ontology_chunks(query)
    print(json.dumps(counts, indent=2))
    for entry in entries[:5]:
        print(f"\n[{entry['source']}] movie={entry['movie']!r}\n{entry['text'][:400]}")
