"""
Phase 3b — schema-constrained LLM extraction ingest.

Same MetaQA corpus, chunks and embeddings as Phase 3a, but the typed graph is
built by gpt-6-luna from the documents under the ontology schema instead of
being copied from kb.txt. The result is stored in a separate
`*-ontology-b-graph` database; Phase 3a is never touched.

Run:
  DATASET=metaqa_mini30 python -m phase3b_ontology_rag.ingest --reset
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import defaultdict
from pathlib import Path

from phase3_ontology_rag.ingest import (
    connect,
    create_schema,
    database_counts,
    guard_reset,
    load_chunk_documents,
    reset_database,
    run_batches,
)
from phase3_ontology_rag.ingest import embed_model
from llama_index.core.node_parser import TokenTextSplitter
from phase3_ontology_rag.schema import ONTOLOGY_FILE, ontology_hash
from phase3b_ontology_rag.extraction import LIST_KEYS, extract_facts
from shared.config import (
    check_neo4j_target,
    phase3_database,
    phase3b_dataset,
    settings,
    write_ingest_meta,
)
from shared.extraction_llm import LitellmExtractionLLM
from shared.ingest import MovieTagger, prepare_documents

llm = LitellmExtractionLLM()


def manifest_paths(dataset, limit_docs: int | None) -> list[str]:
    paths = [line.strip() for line in Path(dataset.doc_list).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if limit_docs:
        paths = paths[:limit_docs]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest file(s) missing, e.g. {missing[:3]}")
    return paths


def write_facts(session, facts_by_movie: dict) -> dict:
    """Write the extracted typed facts in batches; returns fact counts."""
    rows_by_relation = {key: [] for key in LIST_KEYS}
    year_rows = []
    for movie, facts in facts_by_movie.items():
        for key in LIST_KEYS:
            rows_by_relation[key].extend({"movie": movie, "obj": value}
                                         for value in sorted(facts[key]))
        if facts.get("year"):
            year_rows.append({"movie": movie, "year": facts["year"]})

    run_batches(
        session,
        "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
        "MERGE (p:Person {name: row.obj}) SET p:Actor MERGE (m)-[:STARRED_ACTORS]->(p)",
        rows_by_relation["actors"],
    )
    run_batches(
        session,
        "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
        "MERGE (p:Person {name: row.obj}) SET p:Director MERGE (m)-[:DIRECTED_BY]->(p)",
        rows_by_relation["directors"],
    )
    run_batches(
        session,
        "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
        "MERGE (p:Person {name: row.obj}) SET p:Writer MERGE (m)-[:WRITTEN_BY]->(p)",
        rows_by_relation["writers"],
    )
    run_batches(
        session,
        "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
        "MERGE (g:Genre {name: row.obj}) MERGE (m)-[:HAS_GENRE]->(g)",
        rows_by_relation["genres"],
    )
    run_batches(
        session,
        "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
        "MERGE (l:Language {name: row.obj}) MERGE (m)-[:IN_LANGUAGE]->(l)",
        rows_by_relation["languages"],
    )
    run_batches(
        session,
        "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
        "MERGE (y:Year {value: row.year}) MERGE (m)-[:RELEASED_IN]->(y) "
        "SET m.release_year = row.year",
        year_rows,
    )
    return {key: len(rows) for key, rows in rows_by_relation.items()} | {"year": len(year_rows)}


async def ingest(dataset, reset: bool = False, limit_docs: int | None = None,
                 workers: int = 8) -> dict:
    database = phase3_database(dataset)
    check_neo4j_target(dataset.neo4j_uri, database)
    driver = connect(dataset)
    started = time.monotonic()
    facts_by_movie = defaultdict(lambda: {key: set() for key in LIST_KEYS} | {"year": None})
    parse_errors: list[str] = []
    docs = chunks = 0

    try:
        if reset:
            guard_reset(dataset, database)
            reset_database(driver, database)
        create_schema(driver, database)

        paths = manifest_paths(dataset, limit_docs)
        print(f"Phase 3b: {len(paths)} docs -> {database} (gpt-6-luna extraction)")
        text_splitter = TokenTextSplitter(chunk_size=dataset.chunk_size,
                                          chunk_overlap=dataset.chunk_overlap)
        tagger = MovieTagger()
        semaphore = asyncio.Semaphore(max(1, workers))

        async def one(movie: str, text: str):
            async with semaphore:
                return await extract_facts(llm, movie, text)

        for start in range(0, len(paths), 20):
            batch_paths = paths[start:start + 20]
            documents = prepare_documents(load_chunk_documents(batch_paths), dataset)
            nodes = tagger(text_splitter.get_nodes_from_documents(documents))
            rows = []
            for node in nodes:
                movie = node.metadata.get("movie_title")
                if not movie:
                    continue
                rows.append({"id": node.node_id, "text": node.text, "movie": movie})
            if not rows:
                continue

            embeddings = embed_model.get_text_embedding_batch([row["text"] for row in rows],
                                                              show_progress=False)
            for row, embedding in zip(rows, embeddings):
                row["embedding"] = embedding

            with driver.session(database=database) as session:
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (m:Movie {title: row.movie})
                    ON CREATE SET m.orphan = true
                    MERGE (c:Chunk {id: row.id})
                    SET c.text = row.text, c.embedding = row.embedding, c.movie_title = row.movie
                    MERGE (c)-[:ABOUT]->(m)
                    """,
                    rows=rows,
                ).consume()

            results = await asyncio.gather(*(one(row["movie"], row["text"]) for row in rows))
            for row, (facts, error) in zip(rows, results):
                if error:
                    parse_errors.append(f"{row['movie']}: {error}")
                    continue
                bucket = facts_by_movie[row["movie"]]
                for key in LIST_KEYS:
                    bucket[key].update(facts[key])
                if facts.get("year"):
                    bucket["year"] = facts["year"]

            docs += len(documents)
            chunks += len(rows)
            print(f"  docs {docs}/{len(paths)} | chunks {chunks} | facts for {len(facts_by_movie)} movies")

        with driver.session(database=database) as session:
            fact_counts = write_facts(session, facts_by_movie)

        counts = database_counts(driver, database)
        elapsed = time.monotonic() - started
        print("\n" + "=" * 62)
        print("  Phase 3b ingest report")
        print("=" * 62)
        print(f"  database          : {dataset.neo4j_uri} / {database}")
        print(f"  docs / chunks     : {docs:,} / {chunks:,}")
        print(f"  extraction calls  : {llm.calls:,}")
        print(f"  prompt tokens     : {llm.prompt_tokens:,}")
        print(f"  completion tokens : {llm.completion_tokens:,}")
        print(f"  total tokens      : {llm.prompt_tokens + llm.completion_tokens:,}")
        print(f"  errors            : {llm.errors:,}")
        print(f"  policy skips      : {llm.policy_skips:,}")
        print(f"  parse errors      : {len(parse_errors):,}")
        print(f"  facts             : " + ", ".join(f"{k}={v:,}" for k, v in fact_counts.items()))
        print(f"  movies/persons    : {counts['movies']:,} / {counts['persons']:,}")
        print(f"  chunks embedded   : {counts['embedded_chunks']:,}")
        print(f"  relationships     : {counts['relationships']:,}")
        print(f"  wall clock        : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        write_ingest_meta(
            dataset, 3,
            variant="3b",
            graph_database=database,
            ontology_sha256=ontology_hash(),
            ontology_file=str(ONTOLOGY_FILE),
            extraction_model=settings.EXTRACTION_MODEL,
            extraction_temperature=settings.EXTRACTION_TEMPERATURE if settings.EXTRACTION_TEMPERATURE is not None else "default",
            docs=docs, chunks=chunks, facts=fact_counts, parse_errors=len(parse_errors),
            errors=llm.errors, policy_skips=llm.policy_skips, reset=reset,
        )
        return {"docs": docs, "chunks": chunks, "facts": fact_counts,
                "parse_errors": len(parse_errors)}
    finally:
        driver.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3b schema-constrained ingest")
    parser.add_argument("--reset", action="store_true", help="wipe this dataset's Phase 3b database")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = phase3b_dataset(3)
    asyncio.run(ingest(dataset, reset=args.reset, limit_docs=args.limit_docs, workers=args.workers))
    # litellm's async HTTP client can leave non-daemon threads behind; the graph
    # and ingest_meta.json are already written, so force a clean CLI exit.
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
