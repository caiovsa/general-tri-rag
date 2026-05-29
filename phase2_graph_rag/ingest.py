import nest_asyncio
nest_asyncio.apply()

import asyncio
import os
import json

import llama_index.core
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI

from shared.config import settings, get_llamaindex_settings

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model

EXTRACTION_LLM = OpenAI(model="gpt-5-mini")

def get_extraction_chunker() -> TokenTextSplitter:
    return TokenTextSplitter(chunk_size=1024, chunk_overlap=128)


def build_graph_store():
    print("Connecting to Neo4j...")
    graph_store = Neo4jPropertyGraphStore(
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        url=settings.NEO4J_URI,
        database=settings.NEO4J_DATABASE,
    )
    return graph_store


def create_vector_index(graph_store):
    """Create a vector index on Chunk nodes for similarity search."""
    print("Creating vector index on Chunk nodes...")
    graph_store.structured_query("""
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON c.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
    """)
    import time
    time.sleep(2)
    result = graph_store.structured_query(
        "SHOW VECTOR INDEXES YIELD name, state WHERE name = 'chunk_embeddings' RETURN state"
    )
    if result:
        state = result[0].get('state', 'unknown')
        print(f"  Vector index state: {state}")
    else:
        print("  Warning: Could not verify index state")


def build_extractors():
    return [
        ImplicitPathExtractor(),
        SimpleLLMPathExtractor(
            llm=EXTRACTION_LLM,
            num_workers=8,
            max_paths_per_chunk=8,
        ),
    ]


def iter_document_batches(data_dir: str, batch_size: int = 20):
    all_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    )
    total = len(all_files)
    print(f"Found {total} PDF(s) — processing in batches of {batch_size}")

    for start in range(0, total, batch_size):
        batch_files = all_files[start : start + batch_size]
        batch_num = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch_files)} files) ---")
        yield SimpleDirectoryReader(input_files=batch_files).load_data()


async def ingest_phase_2_graph(data_dir: str, batch_size: int = 20):
    """
    Pure Graph RAG ingestion: Neo4j only (no Qdrant).
    
    Stores chunks with embeddings in Neo4j, extracts graph relationships,
    and creates a vector index on Chunk nodes for similarity search.
    """
    graph_store = build_graph_store()
    extractors = build_extractors()
    chunker = get_extraction_chunker()

    # Create the vector index on Chunk nodes
    create_vector_index(graph_store)

    from llama_index.core import PropertyGraphIndex
    
    index = None

    for batch_docs in iter_document_batches(data_dir, batch_size):
        if index is None:
            print("Building PropertyGraphIndex from first batch...")
            index = PropertyGraphIndex.from_documents(
                batch_docs,
                property_graph_store=graph_store,
                kg_extractors=extractors,
                transformations=[chunker],
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
    asyncio.run(ingest_phase_2_graph("data", batch_size=2))
    # python -m phase2_graph_rag.ingest
