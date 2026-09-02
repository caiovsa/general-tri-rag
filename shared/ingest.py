"""
Shared ingest helper — dedup for Phase 1 Vector RAG.

Phase1 had two ~90% identical ingest scripts (ingest_hot.py vs ingest_pdf.py)
differing only in file extension, chunk size/overlap, and target dir.
This helper centralizes that logic; the phase-specific modules now delegate
to it while preserving their CLI entrypoints.

Usage (from phase1_vector_rag.ingest_hot / ingest_pdf):
    from shared.ingest import ingest_directory
    ingest_directory("data_hotpot", pattern="**/*.txt",
                     chunk_size=settings.CHUNK_SIZE_HOTPOT, ...,
                     collection_name=settings.QDRANT_COLLECTION_NAME)
"""

from pathlib import Path

import qdrant_client
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore


def load_documents(file_path: Path):
    """Load a single file via LlamaIndex SimpleDirectoryReader."""
    reader = SimpleDirectoryReader(input_files=[str(file_path)])
    return reader.load_data()


def ingest_directory(
    directory_path: str,
    pattern: str = "**/*.txt",
    chunk_size: int = 200,
    chunk_overlap: int = 20,
    collection_name: str = "rag_phase_1_baseline",
    qdrant_url: str = "http://localhost:6333",
    embed_model=None,
    verbose: bool = True,
):
    """
    Generic Qdrant ingest: first file bootstraps the collection, remaining
    files upsert via IngestionPipeline.

    Args:
        directory_path: folder to scan (e.g., "data_hotpot")
        pattern: glob pattern (e.g., "**/*.txt" or "**/*.pdf")
        chunk_size / chunk_overlap: TokenTextSplitter params
        collection_name / qdrant_url: Qdrant target
        embed_model: LlamaIndex embedding model (required for pipeline)
        verbose: print progress
    """
    if embed_model is None:
        raise ValueError("embed_model is required — pass get_llamaindex_settings()[1]")

    doc_files = sorted(Path(directory_path).glob(pattern))
    if not doc_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory_path}")

    if verbose:
        print(f"Found {len(doc_files)} file(s) in: {directory_path} (pattern={pattern})\n")
        print(f"Chunk size={chunk_size}, overlap={chunk_overlap}, collection={collection_name}\n")

    client = qdrant_client.QdrantClient(url=qdrant_url)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    node_parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    total_chunks = 0

    # First doc bootstraps collection
    first_doc = doc_files[0]
    if verbose:
        print(f"[1/{len(doc_files)}] Bootstrapping collection with: {first_doc.name}")
    try:
        documents = load_documents(first_doc)
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser],
            show_progress=True,
        )
        if verbose:
            print("  -> Collection created and first doc indexed.\n")
        total_chunks += len(documents)
    except Exception as e:
        print(f"  -> ERROR on first doc {first_doc.name}: {e}")
        raise

    # Remaining docs via pipeline
    pipeline = IngestionPipeline(
        transformations=[node_parser, embed_model],
        vector_store=vector_store,
    )

    for i, doc_path in enumerate(doc_files[1:], start=2):
        if verbose:
            print(f"[{i}/{len(doc_files)}] Processing: {doc_path.name}")
        try:
            documents = load_documents(doc_path)
            if verbose and documents and hasattr(documents[0], "text"):
                # hint for PDFs with multiple pages
                if len(documents) > 1:
                    print(f"  -> Loaded {len(documents)} pages")
            nodes = pipeline.run(documents=documents, show_progress=True, num_workers=1)
            total_chunks += len(nodes)
            if verbose:
                print(f"  -> Indexed {len(nodes)} chunks (total so far: {total_chunks})")
        except Exception as e:
            print(f"  -> ERROR on {doc_path.name}: {e} — skipping")
            continue

    if verbose:
        print(f"\nDone! Indexed {total_chunks} total chunks into: {collection_name}")
    return total_chunks
