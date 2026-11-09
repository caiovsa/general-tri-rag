"""
Phase 3a — Ontology RAG ingestion (oracle KB graph, no LLM extraction).

Builds a typed graph directly from MetaQA/kb.txt plus the same chunks Phase 1/2
use, so retrieval quality can be measured without extraction noise:

  (:Movie)-[:STARRED_ACTORS|DIRECTED_BY|WRITTEN_BY]->(:Person)
  (:Movie)-[:HAS_GENRE]->(:Genre) / [:IN_LANGUAGE]->(:Language)
  (:Movie)-[:RELEASED_IN]->(:Year) / [:HAS_TAG]->(:Tag)
  (:Chunk)-[:ABOUT]->(:Movie)  +  chunk_embeddings vector index

Run:
  DATASET=metaqa_mini30 python -m phase3_ontology_rag.ingest --reset
  DATASET=metaqa python -m phase3_ontology_rag.ingest --reset
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import llama_index.core
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from neo4j import GraphDatabase

from phase3_ontology_rag.schema import (
    ONTOLOGY_FILE,
    RELATIONS,
    ROLE_LABELS,
    ontology_hash,
)
from scripts.metaqa_sample import load_kb, resolve_root
from shared.config import (
    check_neo4j_target,
    dataset_config,
    phase3_database,
    get_llamaindex_settings,
    settings,
    write_ingest_meta,
)
from shared.ingest import MovieTagger, prepare_documents

_, embed_model = get_llamaindex_settings()
llama_index.core.Settings.embed_model = embed_model

BATCH = 5000


def connect(dataset):
    return GraphDatabase.driver(dataset.neo4j_uri, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))


def guard_reset(dataset, database: str) -> None:
    """--reset may only wipe this dataset's own Phase 3 ontology database."""
    expected = phase3_database(dataset)
    if database != expected:
        raise SystemExit(f"Refusing --reset: {database!r} is not the configured Phase 3 database "
                         f"({expected!r}).")
    if database == dataset.neo4j_database:
        raise SystemExit(f"Refusing --reset: {database!r} is the Phase 2 database.")
    if database == settings.NEO4J_DATABASE:
        raise SystemExit(f"Refusing --reset: {database!r} is the HotpotQA database.")


def reset_database(driver, database: str) -> None:
    with driver.session(database=database) as session:
        before = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        print(f"Resetting {database}: {before:,} node(s) to delete")
        session.run("MATCH (n) DETACH DELETE n").consume()
        after = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        print(f"  database now holds {after:,} node(s)")


def create_schema(driver, database: str) -> None:
    constraints = {
        "movie_title": "FOR (m:Movie) REQUIRE m.title IS UNIQUE",
        "person_name": "FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "genre_name": "FOR (g:Genre) REQUIRE g.name IS UNIQUE",
        "language_name": "FOR (l:Language) REQUIRE l.name IS UNIQUE",
        "tag_name": "FOR (t:Tag) REQUIRE t.name IS UNIQUE",
        "year_value": "FOR (y:Year) REQUIRE y.value IS UNIQUE",
        "chunk_id": "FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    }
    with driver.session(database=database) as session:
        for name, body in constraints.items():
            session.run(f"CREATE CONSTRAINT {name} IF NOT EXISTS {body}").consume()
        session.run(
            "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS FOR (c:Chunk) ON c.embedding "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {settings.EMBEDDING_DIMENSION}, "
            "`vector.similarity_function`: 'cosine'}}"
        ).consume()
        session.run("CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS "
                    "FOR (c:Chunk) ON EACH [c.text]").consume()
        for _ in range(30):
            row = session.run(
                "SHOW VECTOR INDEXES YIELD name, state WHERE name = 'chunk_embeddings' RETURN state"
            ).single()
            if row and row["state"] == "READY":
                print("  vector index: READY")
                return
            time.sleep(1)
    print("  vector index: not READY yet (ingest continues)")


def run_batches(session, query: str, rows: list[dict], batch: int = BATCH) -> None:
    for start in range(0, len(rows), batch):
        session.run(query, rows=rows[start:start + batch]).consume()


def ingest_kb(driver, database: str, kb) -> dict:
    movies, forward, backward, triples = kb
    years = {movie: sorted(forward["release_year"].get(movie, ()))[0]
             for movie in movies if forward["release_year"].get(movie)}
    with driver.session(database=database) as session:
        run_batches(
            session,
            "UNWIND $rows AS row MERGE (m:Movie {title: row.title}) SET m.release_year = row.year",
            [{"title": title, "year": years.get(title)} for title in sorted(movies)],
        )
        counts = {}
        for relation, (subject_label, edge, object_label) in RELATIONS.items():
            rows = [{"movie": movie, "obj": obj}
                    for movie, facts in triples.items()
                    for rel, obj in facts if rel == relation]
            if not rows:
                continue
            if relation in ROLE_LABELS:
                role = ROLE_LABELS[relation]
                query = (
                    f"UNWIND $rows AS row "
                    f"MATCH (m:Movie {{title: row.movie}}) "
                    f"MERGE (o:Person {{name: row.obj}}) SET o:{role} "
                    f"MERGE (m)-[:{edge}]->(o)"
                )
            elif object_label == "Year":
                query = (
                    "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
                    "MERGE (y:Year {value: row.obj}) MERGE (m)-[:RELEASED_IN]->(y)"
                )
            elif object_label == "Language":
                query = (
                    "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
                    f"MERGE (o:Language {{name: row.obj}}) MERGE (m)-[:{edge}]->(o)"
                )
            elif object_label == "Genre":
                query = (
                    "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
                    f"MERGE (o:Genre {{name: row.obj}}) MERGE (m)-[:{edge}]->(o)"
                )
            elif object_label == "Tag":
                query = (
                    "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
                    f"MERGE (o:Tag {{name: row.obj}}) MERGE (m)-[:{edge}]->(o)"
                )
            else:  # Rating / Votes
                query = (
                    "UNWIND $rows AS row MATCH (m:Movie {title: row.movie}) "
                    f"MERGE (o:{object_label} {{name: row.obj}}) MERGE (m)-[:{edge}]->(o)"
                )
            run_batches(session, query, rows)
            counts[relation] = len(rows)
        session.run(
            "UNWIND $rows AS row MATCH (m:Movie {title: row.title}) "
            "SET m.release_year = row.year",
            rows=[{"title": title, "year": years.get(title)} for title in sorted(movies)],
        ).consume()
    return counts


def load_chunk_documents(paths: list[str]):
    documents = SimpleDirectoryReader(input_files=paths).load_data()
    return documents


def ingest_chunks(driver, database: str, dataset, limit_docs: int | None = None) -> dict:
    manifest = [line.strip() for line in Path(dataset.doc_list).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if limit_docs:
        manifest = manifest[:limit_docs]
    missing = [path for path in manifest if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest file(s) missing, e.g. {missing[:3]}")

    splitter = TokenTextSplitter(chunk_size=dataset.chunk_size, chunk_overlap=dataset.chunk_overlap)
    tagger = MovieTagger()
    chunks_written = 0
    docs = 0
    missing_movies = 0
    with driver.session(database=database) as session:
        for start in range(0, len(manifest), 20):
            batch_paths = manifest[start:start + 20]
            documents = prepare_documents(load_chunk_documents(batch_paths), dataset)
            nodes = splitter.get_nodes_from_documents(documents)
            nodes = tagger(nodes)
            docs += len(documents)

            rows = []
            for node in nodes:
                movie = node.metadata.get("movie_title")
                if not movie:
                    first = node.text.splitlines()[0] if node.text else ""
                    movie = first.split(":", 1)[1].strip() if first.lower().startswith("[movie:") else ""
                if not movie:
                    missing_movies += 1
                    continue
                rows.append({"id": node.node_id, "text": node.text, "movie": movie})
            if not rows:
                continue

            embeddings = embed_model.get_text_embedding_batch([row["text"] for row in rows],
                                                              show_progress=False)
            for row, embedding in zip(rows, embeddings):
                row["embedding"] = embedding

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
            chunks_written += len(rows)
            print(f"  chunks {chunks_written:,} ({docs}/{len(manifest)} docs)")
    return {"docs": docs, "chunks": chunks_written, "missing_movie": missing_movies}


def database_counts(driver, database: str) -> dict:
    with driver.session(database=database) as session:
        def one(query):
            row = session.run(query).single()
            return row[0] if row else 0
        return {
            "movies": one("MATCH (m:Movie) RETURN count(m)"),
            "persons": one("MATCH (p:Person) RETURN count(p)"),
            "chunks": one("MATCH (c:Chunk) RETURN count(c)"),
            "embedded_chunks": one("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c)"),
            "relationships": one("MATCH ()-[r]->() RETURN count(r)"),
            "orphan_movies": one("MATCH (m:Movie) WHERE m.orphan RETURN count(m)"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3a ontology ingest (oracle KB graph)")
    parser.add_argument("--reset", action="store_true", help="wipe this dataset's Phase 3 database first")
    parser.add_argument("--limit-docs", type=int, default=None, help="only index the first N docs")
    parser.add_argument("--skip-kb", action="store_true", help="do not rebuild the typed KB graph")
    parser.add_argument("--skip-chunks", action="store_true", help="do not index documents/chunks")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = dataset_config(phase=3)
    if not dataset.tag_movies:
        raise SystemExit("Phase 3a (oracle KB graph) is MetaQA-only. "
                         "Use DATASET=metaqa|metaqa_mini|metaqa_mini30.")
    database = phase3_database(dataset)
    check_neo4j_target(dataset.neo4j_uri, database)
    driver = connect(dataset)
    try:
        if args.reset:
            guard_reset(dataset, database)
            reset_database(driver, database)
        create_schema(driver, database)

        if not args.skip_kb:
            print(f"Loading KB from {resolve_root() / 'kb.txt'} ...")
            kb = load_kb(resolve_root())
            counts = ingest_kb(driver, database, kb)
            print("  triples by relation: " + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))

        if not args.skip_chunks:
            stats = ingest_chunks(driver, database, dataset, args.limit_docs)
            print(f"  docs={stats['docs']:,} chunks={stats['chunks']:,} "
                  f"missing_movie={stats['missing_movie']:,}")

        counts = database_counts(driver, database)
        print("\n" + "=" * 60)
        print("  Phase 3a ingest report")
        print("=" * 60)
        print(f"  dataset  : {dataset.name} | ontology: {ontology_hash()[:16]}...")
        print(f"  database : {dataset.neo4j_uri} / {database}")
        for key, value in counts.items():
            print(f"  {key:<10}: {value:,}")
        print(f"  ontology file: {ONTOLOGY_FILE}")
        write_ingest_meta(
            dataset, 3,
            graph_database=database,
            ontology_sha256=ontology_hash(),
            ontology_file=str(ONTOLOGY_FILE),
            reset=args.reset,
            **counts,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
