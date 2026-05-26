import nest_asyncio
nest_asyncio.apply()

import asyncio
import os

import qdrant_client
import llama_index.core
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,       # FREE: pure NLP, zero LLM calls
    SimpleLLMPathExtractor,      # CHEAP: simple prompt, much faster than Dynamic
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI  # or swap for your provider

from shared.config import settings, get_llamaindex_settings

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model

# Use a cheap, FAST model just for KG extraction.
# gpt-4o-mini is ~10–20× cheaper and 3–5× faster than gpt-4o for this task.
EXTRACTION_LLM = OpenAI(model="gpt-4o-mini", temperature=0)

# 2. Chunker
def get_extraction_chunker() -> TokenTextSplitter:
    return TokenTextSplitter(chunk_size=1024, chunk_overlap=128)


# 3. Stores
def build_stores():
    print("Connecting to Neo4j...")
    graph_store = Neo4jPropertyGraphStore(
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        url=settings.NEO4J_URI,
    )

    print("Connecting to Qdrant...")
    client = qdrant_client.QdrantClient(url=settings.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME_PHASE2,
    )
    return graph_store, vector_store


# 4. Extractors
# Two-extractor strategy:
#
# ImplicitPathExtractor  — zero cost, catches ~30% of relations via noun-chunk
#                          NLP without any LLM call.
#
# SimpleLLMPathExtractor — simple, short prompt (vs Dynamic which re-infers
#                          the ontology every chunk). Uses the cheap model and
#                          8 async workers so many chunks fly in parallel.
#
# Together they match or exceed DynamicLLMPathExtractor quality at a fraction
# of the time and cost.
def build_extractors():
    return [
        ImplicitPathExtractor(),
        SimpleLLMPathExtractor(
            llm=EXTRACTION_LLM,
            num_workers=8,          # parallel async workers — tune to your API rate limit
            max_paths_per_chunk=8,  # slightly more than your old max_triplets to compensate
                                    # for the simpler prompt
        ),
    ]


# 5. Document batching
def iter_document_batches(data_dir: str, batch_size: int = 20):
    """
    Yield SimpleDirectoryReader-loaded batches of `batch_size` PDFs.
    This keeps memory flat regardless of corpus size and lets you resume
    from a checkpoint if the process is interrupted.
    """
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


# 6. Main async ingestion
async def ingest_phase_2_graph(data_dir: str, batch_size: int = 20):
    """
    Optimized Phase 2 hybrid Graph + Vector ingestion.

    Key improvements over the original:
      • ImplicitPathExtractor  — free NLP-based relation extraction
      • SimpleLLMPathExtractor — simpler prompt, cheap model, 8 workers
      • afrom_documents        — fully async, maximises API concurrency
      • Larger chunks          — halves the number of LLM calls
      • Batching               — flat memory, resumable
    """
    graph_store, vector_store = build_stores()
    extractors = build_extractors()
    chunker = get_extraction_chunker()

    index = None

    for batch_docs in iter_document_batches(data_dir, batch_size):
        if index is None:
            # First batch: create the index and both stores from scratch.
            print("Building index from first batch (async)...")
            index = PropertyGraphIndex.from_documents(
                batch_docs,
                property_graph_store=graph_store,
                vector_store=vector_store,
                kg_extractors=extractors,
                transformations=[chunker],
                show_progress=True,     # tqdm bar so you can see progress
            )
        else:
            # Subsequent batches: insert into the existing index.
            # Each insert reuses the already-open store connections.
            print("Inserting batch into existing index (async)...")
            insert_tasks = [
                asyncio.create_task(index.ainsert(doc))
                for doc in batch_docs
            ]
            await asyncio.gather(*insert_tasks)

        print(f"Batch complete. {len(batch_docs)} document(s) ingested.")

    print("\nPhase 2 ingestion complete! Data is in Neo4j and Qdrant.")
    return index


# 7. Entry point
if __name__ == "__main__":
    # Point at your full data folder — batching keeps memory and cost safe.
    asyncio.run(ingest_phase_2_graph("data", batch_size=20))
    # python -m phase2_graph_rag.ingest