import nest_asyncio
nest_asyncio.apply()

import argparse
import asyncio
import os
import time
from pathlib import Path

import llama_index.core
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TransformComponent

from shared.config import check_neo4j_target, dataset_config, metaqa_database, settings, get_llamaindex_settings
from shared.ingest import MovieTagger, prepare_documents

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model


class RunStats:
    """Counters for the end-of-run report (extraction settings are untouched)."""

    def __init__(self):
        self.calls = 0
        self.errors = 0
        self.retries = 0


class CountingExtractionLLM(OpenAI):
    """EXTRACTION_LLM with identical settings (model/temperature/retries), plus counters.

    `_chat`/`_achat` are the single funnel every entry point funnels through
    (`apredict` -> `achat` -> `_achat`, `complete` -> `_chat`), so each logical
    extraction call is counted exactly once, and failures are visible even when
    the extractor swallows them.
    """

    calls: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def _record_tokens(self, response) -> None:
        self.prompt_tokens += _usage_value(response, "prompt_tokens")
        self.completion_tokens += _usage_value(response, "completion_tokens")

    def _chat(self, messages, **kwargs):
        self.calls += 1
        try:
            response = super()._chat(messages, **kwargs)
        except Exception:
            self.errors += 1
            raise
        self._record_tokens(response)
        return response

    async def _achat(self, messages, **kwargs):
        self.calls += 1
        try:
            response = await super()._achat(messages, **kwargs)
        except Exception:
            self.errors += 1
            raise
        self._record_tokens(response)
        return response


# Real retrieval chunks only: the store also keeps empty document-id placeholders as Chunk nodes.
TEXT_CHUNKS = "MATCH (c:Chunk) WHERE c.text <> '' RETURN count(c) AS n"
# The [Movie: <title>] tag makes the LLM extractor emit an entity literally named "Movie".
TAG_ARTIFACT_NAME = "movie"


def _usage_value(response, key: str) -> int:
    """Token count from a ChatResponse's raw openai usage (object or dict)."""
    usage = getattr(getattr(response, "raw", None), "usage", None)
    if usage is None:
        return 0
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    return int(value or 0)


class MetadataScrubber(TransformComponent):
    """Clear node metadata before the graph extractors run.

    The LLM extractor reads `node.get_content(metadata_mode=LLM)`, which renders
    every metadata key that is not in `excluded_llm_metadata_keys` — so the file
    metadata (file_path, file_name, file_type, file_size, creation_date,
    last_modified_date, movie_title) was extracted as triples and became entities
    ("Text/plain", "2026-09-28") and edges (File_path, Creation_date, ...).

    Clearing the metadata at the source removes both, for every extractor. The
    `[Movie: <title>]` tag lives in the node *text* so it is unaffected, and the
    store's Chunk->Entity MENTIONS links are rebuilt from `node.id_` by
    PropertyGraphIndex (base.py: triplet_source_id), not from metadata.
    """

    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.metadata.clear()
        return nodes


EXTRACTION_LLM = CountingExtractionLLM(model=settings.EXTRACTION_MODEL, temperature=0.0)


def instrument_retries(llm, stats: RunStats) -> None:
    """Count SDK-level retries by observing the OpenAI client's retry hook.

    Only wraps the existing hook — retry configuration itself is unchanged.
    """
    for get_client in (llm._get_client, llm._get_aclient):
        try:
            client = get_client()
        except Exception as error:  # noqa: BLE001 - counter is best effort
            print(f"  (retry counter unavailable: {type(error).__name__})")
            continue
        original = getattr(client, "_calculate_retry_timeout", None)
        if original is None or getattr(client, "_metaqa_counted", False):
            continue

        def counting(remaining_retries, options, response_headers=None, _original=original):
            stats.retries += 1
            return _original(remaining_retries, options, response_headers)

        client._calculate_retry_timeout = counting
        client._metaqa_counted = True


def guard_reset(dataset) -> None:
    """--reset may only ever touch this dataset's own MetaQA database."""
    if dataset.name not in ("metaqa", "metaqa_mini"):
        raise SystemExit("--reset is MetaQA-only: refusing to wipe a HotpotQA graph.")
    expected = metaqa_database(dataset.name)
    if dataset.neo4j_database != expected:
        raise SystemExit(f"Refusing --reset: target database {dataset.neo4j_database!r} is not the "
                         f"configured {dataset.name} database ({expected!r}).")
    if dataset.neo4j_database == settings.NEO4J_DATABASE:
        raise SystemExit(f"Refusing --reset: {dataset.neo4j_database!r} is the HotpotQA database.")
    other = metaqa_database("metaqa" if dataset.name == "metaqa_mini" else "metaqa_mini")
    if dataset.neo4j_database == other:
        raise SystemExit(f"Refusing --reset: {dataset.neo4j_database!r} is the other MetaQA "
                         "track's database.")


def reset_metaqa_graph(graph_store, dataset) -> None:
    """Wipe the MetaQA database's nodes/relationships (batched, then verified)."""
    before = graph_store.structured_query("MATCH (n) RETURN count(n) AS n")[0]["n"]
    print(f"Resetting {dataset.neo4j_uri} / {dataset.neo4j_database}: {before:,} node(s) to delete")
    try:
        graph_store.structured_query(
            "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS"
        )
    except Exception as error:  # noqa: BLE001 - older servers: fall back to one transaction
        print(f"  batched delete unavailable ({type(error).__name__}) — using a single transaction")
        graph_store.structured_query("MATCH (n) DETACH DELETE n")
    after = graph_store.structured_query("MATCH (n) RETURN count(n) AS n")[0]["n"]
    print(f"  database now holds {after:,} node(s)")


def cleanup_tag_artifact(graph_store, enabled: bool = True) -> None:
    """Delete the entity the [Movie: <title>] tag creates (name 'Movie'), nothing else."""
    if not enabled:
        print("  cleanup           : skipped (--no-cleanup)")
        return

    found = graph_store.structured_query(
        """
        MATCH (e:__Entity__) WHERE toLower(e.name) = $name
        OPTIONAL MATCH (c:Chunk)-[m:MENTIONS]->(e)
        RETURN count(DISTINCT e) AS nodes, count(m) AS mentions
        """,
        param_map={"name": TAG_ARTIFACT_NAME},
    )[0]
    nodes, mentions = found["nodes"], found["mentions"]
    if not nodes:
        print("  cleanup           : no tag-artifact entity found")
        return

    relations = graph_store.structured_query(
        "MATCH (e:__Entity__) WHERE toLower(e.name) = $name OPTIONAL MATCH (e)-[r]-() RETURN count(r) AS n",
        param_map={"name": TAG_ARTIFACT_NAME},
    )[0]["n"]
    graph_store.structured_query(
        "MATCH (e:__Entity__) WHERE toLower(e.name) = $name DETACH DELETE e",
        param_map={"name": TAG_ARTIFACT_NAME},
    )
    extra = f" (+{relations - mentions} other edge(s) to it)" if relations > mentions else ""
    print(f"  cleanup           : removed {nodes} entity '{TAG_ARTIFACT_NAME}' "
          f"+ {mentions} MENTIONS edge(s){extra}")


def get_extraction_chunker(dataset=None) -> TokenTextSplitter:
    # HotpotQA paragraphs are ~100-200 tokens (256/20 keeps multi-hop separation);
    # MetaQA docs are longer pages and use the dataset's single chunk size (256/32).
    dataset = dataset or dataset_config(phase=2)
    return TokenTextSplitter(
        chunk_size=dataset.chunk_size,
        chunk_overlap=dataset.chunk_overlap,
    )


def build_graph_store(dataset=None):
    dataset = dataset or dataset_config(phase=2)
    print(f"Connecting to Neo4j at {dataset.neo4j_uri} (database={dataset.neo4j_database})...")
    # HotpotQA and MetaQA must never share a graph: see dataset_config().
    graph_store = Neo4jPropertyGraphStore(
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        url=dataset.neo4j_uri,
        database=dataset.neo4j_database,
    )
    return graph_store


def create_vector_index(graph_store):
    """Create vector + full-text indexes on Chunk nodes for similarity + keyword search."""
    print("Creating vector index on Chunk nodes...")
    graph_store.structured_query(f"""
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON c.embedding
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {settings.EMBEDDING_DIMENSION},
            `vector.similarity_function`: 'cosine'
        }}}}
    """)
    time.sleep(2)
    result = graph_store.structured_query(
        "SHOW VECTOR INDEXES YIELD name, state WHERE name = 'chunk_embeddings' RETURN state"
    )
    if result:
        state = result[0].get('state', 'unknown')
        print(f"  Vector index state: {state}")
    else:
        print("  Warning: Could not verify vector index state")

    # Full-text index for fast keyword anchor search (replaces CONTAINS scan)
    print("Creating full-text index on Chunk.text (if not exists)...")
    try:
        graph_store.structured_query("""
            CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
        """)
        time.sleep(1)
        ft = graph_store.structured_query(
            "SHOW FULLTEXT INDEXES YIELD name, state WHERE name = 'chunk_text_fulltext' RETURN state"
        )
        if ft:
            print(f"  Full-text index state: {ft[0].get('state', 'unknown')}")
    except Exception as e:
        # Neo4j <5 or permission issue — keyword search will fall back to CONTAINS
        print(f"  Warning: full-text index not created ({e}) — retriever will use CONTAINS fallback")


def build_extractors():
    return [
        ImplicitPathExtractor(),
        SimpleLLMPathExtractor(
            llm=EXTRACTION_LLM,
            num_workers=8, # Parallel LLM calls for extraction
            max_paths_per_chunk=12, # Increased slightly because chunks are smaller and focused
        ),
    ]


def iter_document_batches(data_dir: str = None, batch_size: int = 50, dataset=None, limit_docs: int = None):
    """Yield document batches: glob a directory (hotpot) or read the manifest (metaqa)."""
    dataset = dataset or dataset_config(phase=2)

    if dataset.doc_list:
        all_files = [line.strip() for line in Path(dataset.doc_list).read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        missing = [path for path in all_files if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest file(s) missing, e.g. {missing[:3]}")
    else:
        data_dir = data_dir or str(dataset.docs_dir)
        all_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".txt")
        )
    available = len(all_files)
    if limit_docs:
        all_files = all_files[:limit_docs]
    total = len(all_files)
    limited = f" (first {total} of {available})" if limit_docs else ""
    print(f"Found {total} TXT(s){limited} — processing in batches of {batch_size}")

    for start in range(0, total, batch_size):
        batch_files = all_files[start : start + batch_size]
        batch_num = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch_files)} files) ---")
        documents = SimpleDirectoryReader(input_files=batch_files).load_data()
        yield prepare_documents(documents, dataset) if dataset.tag_movies else documents


async def ingest_phase_2_graph(data_dir: str = None, batch_size: int = 50, limit_docs: int = None,
                               reset: bool = False, cleanup: bool = True):
    """
    Pure Graph RAG ingestion: Neo4j only (no Qdrant).

    Stores chunks with embeddings in Neo4j, extracts graph relationships,
    and creates a vector index on Chunk nodes for similarity search.
    """
    started = time.monotonic()
    stats = RunStats()
    dataset = dataset_config(phase=2)
    if dataset.name == "metaqa":
        check_neo4j_target(dataset.neo4j_uri, dataset.neo4j_database)
    if reset:
        guard_reset(dataset)

    graph_store = build_graph_store(dataset)
    if reset:
        reset_metaqa_graph(graph_store, dataset)

    instrument_retries(EXTRACTION_LLM, stats)
    extractors = build_extractors()
    chunker = get_extraction_chunker(dataset)
    # [Movie: <title>] tags every chunk before extraction, identically to Phase 1
    transformations = [chunker, MovieTagger(), MetadataScrubber()] if dataset.tag_movies else [chunker]

    # Create the vector index on Chunk nodes
    create_vector_index(graph_store)

    chunks_before = graph_store.structured_query(TEXT_CHUNKS)[0]["n"]

    from llama_index.core import PropertyGraphIndex
    
    index = None
    docs_ingested = 0

    for batch_docs in iter_document_batches(data_dir, batch_size, dataset, limit_docs):
        if index is None:
            print("Building PropertyGraphIndex from first batch...")
            index = PropertyGraphIndex.from_documents(
                batch_docs,
                property_graph_store=graph_store,
                kg_extractors=extractors,
                transformations=transformations,
                embed_model=embed_model,
                embed_kg_nodes=True,
                show_progress=True,
            )
        else:
            print("Inserting batch into existing index...")
            insert_tasks = [
                asyncio.create_task(index.ainsert(doc))
                for doc in batch_docs
            ]
            await asyncio.gather(*insert_tasks)

        docs_ingested += len(batch_docs)
        print(f"Batch complete. {len(batch_docs)} document(s) ingested.")

    # Verify ingestion (cleanup first, so the numbers below are post-cleanup)
    cleanup_tag_artifact(graph_store, enabled=cleanup)
    print("\nVerifying ingestion...")
    result = graph_store.structured_query(
        "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS chunk_count"
    )
    chunk_count = result[0]['chunk_count'] if result else 0
    
    result = graph_store.structured_query(
        "MATCH ()-[r]->() RETURN count(r) AS rel_count"
    )
    rel_count = result[0]['rel_count'] if result else 0
    
    print(f"  Chunks with embeddings: {chunk_count}")
    print(f"  Total relationships: {rel_count}")

    elapsed = time.monotonic() - started
    chunks_total = graph_store.structured_query(TEXT_CHUNKS)[0]["n"]
    all_chunk_nodes = graph_store.structured_query("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"]
    print("\n" + "=" * 62)
    print("  Phase 2 ingest report")
    print("=" * 62)
    print(f"  dataset           : {dataset.name}")
    print(f"  neo4j             : {dataset.neo4j_uri} / {dataset.neo4j_database}")
    print(f"  docs ingested     : {docs_ingested:,}")
    print(f"  chunks added      : {chunks_total - chunks_before:,}  (db text chunks {chunks_total:,})")
    if all_chunk_nodes != chunks_total:
        print(f"  empty placeholders: {all_chunk_nodes - chunks_total:,}  "
              "(document ids stored as Chunk nodes: no text, no embedding, no extraction)")
    print(f"  extraction calls  : {EXTRACTION_LLM.calls:,}")
    print(f"  prompt tokens     : {EXTRACTION_LLM.prompt_tokens:,}")
    print(f"  completion tokens : {EXTRACTION_LLM.completion_tokens:,}")
    print(f"  total tokens      : {EXTRACTION_LLM.prompt_tokens + EXTRACTION_LLM.completion_tokens:,}")
    print(f"  errors            : {EXTRACTION_LLM.errors:,}")
    print(f"  retries           : {stats.retries:,}")
    print(f"  relationships     : {rel_count:,}  (db total)")
    print(f"  wall clock        : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if EXTRACTION_LLM.errors:
        print(f"  WARNING: {EXTRACTION_LLM.errors:,} extraction call(s) failed — those chunks have no triples")
    print("\nPhase 2 ingestion complete! Data is in Neo4j.")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 (Graph RAG) ingest for the active DATASET")
    parser.add_argument("--limit-docs", type=int, default=None,
                        help="ingest only the first N documents of the corpus")
    parser.add_argument("--reset", action="store_true",
                        help="wipe the MetaQA database before ingesting (refuses any other database)")
    parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True,
                        help="delete the tag-artifact entity 'Movie' after ingest (default: on)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(ingest_phase_2_graph(batch_size=2000, limit_docs=args.limit_docs, reset=args.reset,
                                     cleanup=args.cleanup))
    # python -m phase2_graph_rag.ingest                      (DATASET=hotpot)
    # DATASET=metaqa python -m phase2_graph_rag.ingest --reset --limit-docs 50
    # DATASET=metaqa_mini python -m phase2_graph_rag.ingest --reset