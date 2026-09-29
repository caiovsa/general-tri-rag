import nest_asyncio
nest_asyncio.apply()

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

from shared.config import check_neo4j_target, dataset_config, settings, get_llamaindex_settings
from shared.ingest import MovieTagger, prepare_documents

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model

EXTRACTION_LLM = OpenAI(model=settings.EXTRACTION_MODEL, temperature=0.0)


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


def iter_document_batches(data_dir: str = None, batch_size: int = 50, dataset=None):
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
    total = len(all_files)
    print(f"Found {total} TXT(s) — processing in batches of {batch_size}")

    for start in range(0, total, batch_size):
        batch_files = all_files[start : start + batch_size]
        batch_num = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch_files)} files) ---")
        documents = SimpleDirectoryReader(input_files=batch_files).load_data()
        yield prepare_documents(documents, dataset) if dataset.tag_movies else documents


async def ingest_phase_2_graph(data_dir: str = None, batch_size: int = 50):
    """
    Pure Graph RAG ingestion: Neo4j only (no Qdrant).

    Stores chunks with embeddings in Neo4j, extracts graph relationships,
    and creates a vector index on Chunk nodes for similarity search.
    """
    dataset = dataset_config(phase=2)
    if dataset.name == "metaqa":
        check_neo4j_target(dataset.neo4j_uri, dataset.neo4j_database)

    graph_store = build_graph_store(dataset)
    extractors = build_extractors()
    chunker = get_extraction_chunker(dataset)
    # [Movie: <title>] tags every chunk before extraction, identically to Phase 1
    transformations = [chunker, MovieTagger()] if dataset.tag_movies else [chunker]

    # Create the vector index on Chunk nodes
    create_vector_index(graph_store)

    from llama_index.core import PropertyGraphIndex
    
    index = None

    for batch_docs in iter_document_batches(data_dir, batch_size, dataset):
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

        print(f"Batch complete. {len(batch_docs)} document(s) ingested.")

    # Verify ingestion
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
    print("\nPhase 2 ingestion complete! Data is in Neo4j.")
    return index


if __name__ == "__main__":
    asyncio.run(ingest_phase_2_graph(batch_size=2000))
    # python -m phase2_graph_rag.ingest                      (DATASET=hotpot)
    # DATASET=metaqa python -m phase2_graph_rag.ingest