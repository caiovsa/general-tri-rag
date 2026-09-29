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
from llama_index.core.schema import TransformComponent
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Metadata keys that must never reach the embedding text (identical in every phase).
EMBED_METADATA_EXCLUDE = ("file_path", "file_name", "movie_title")
MOVIE_TAG = "[Movie: {title}]"


def movie_title(text: str) -> str:
    """Title of a generated MetaQA doc: its first line is "Title: <movie>"."""
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if first.lower().startswith("title:"):
        return first.split(":", 1)[1].strip()
    return ""


class MovieTagger(TransformComponent):
    """Prepend "[Movie: <title>]" to every chunk, before embedding or extraction.

    MetaQA evidence is matched against chunk text (and Phase 2 extracts from it),
    so the tag has to live in the text rather than in metadata. Shared by all phases.
    Titles come from document metadata (prepare_documents) with a fallback to the
    doc's own "Title:" line, cached per file so every chunk of a file is tagged.
    """

    _titles: dict = {}

    def __call__(self, nodes, **kwargs):
        for node in nodes:
            path = node.metadata.get("file_path", "")
            title = node.metadata.get("movie_title") or self._titles.get(path) or movie_title(node.text)
            if title:
                self._titles[path] = title
                node.metadata["movie_title"] = title
                if not node.text.startswith(MOVIE_TAG.format(title=title)):
                    node.text = f"{MOVIE_TAG.format(title=title)}\n{node.text}"
            node.excluded_embed_metadata_keys = sorted(set(node.excluded_embed_metadata_keys)
                                                       | set(EMBED_METADATA_EXCLUDE))
            node.excluded_llm_metadata_keys = sorted(set(node.excluded_llm_metadata_keys)
                                                     | set(EMBED_METADATA_EXCLUDE))
        return nodes


def prepare_documents(documents, dataset):
    """Apply the dataset policy shared by every phase (tag + metadata exclusions)."""
    for document in documents:
        # Union with the reader's defaults so file_size/dates stay out of embeddings
        document.excluded_embed_metadata_keys = sorted(set(document.excluded_embed_metadata_keys)
                                                       | set(EMBED_METADATA_EXCLUDE))
        document.excluded_llm_metadata_keys = sorted(set(document.excluded_llm_metadata_keys)
                                                     | set(EMBED_METADATA_EXCLUDE))
        if dataset.tag_movies:
            title = movie_title(document.text)
            if title:
                document.metadata["movie_title"] = title
    return documents


def load_manifest_documents(doc_list: Path, verbose: bool = True):
    """Load exactly the files listed in a corpus manifest (one path per line)."""
    paths = [line.strip() for line in Path(doc_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths:
        raise ValueError(f"Empty manifest: {doc_list}")
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest file(s) missing, e.g. {missing[:3]}")
    if verbose:
        print(f"Manifest {doc_list}: {len(paths)} file(s)")
    documents = []
    for path in paths:
        documents.extend(load_documents(Path(path)))
    return documents


def ingest_documents(
    documents,
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
    qdrant_url: str,
    embed_model=None,
    dataset=None,
    batch_size: int = 50,
    verbose: bool = True,
):
    """Qdrant ingest for a resolved dataset (manifest-driven, movie-tagged).

    Batched so the embedding model is called with full batches; the first batch
    creates the collection. Only used when a `doc_list` dataset is active — the
    HotpotQA path keeps using ingest_directory().
    """
    if embed_model is None:
        raise ValueError("embed_model is required")

    client = qdrant_client.QdrantClient(url=qdrant_url)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    transformations = [TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)]
    if dataset is not None and dataset.tag_movies:
        transformations.append(MovieTagger())
    pipeline = IngestionPipeline(transformations=[*transformations, embed_model], vector_store=vector_store)

    total_chunks = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        if dataset is not None:
            batch = prepare_documents(batch, dataset)
        nodes = pipeline.run(documents=batch, show_progress=verbose, num_workers=1)
        total_chunks += len(nodes)
        if verbose:
            done = min(start + batch_size, len(documents))
            print(f"  -> {done}/{len(documents)} docs, {total_chunks} chunks so far")

    if verbose:
        print(f"\nDone! Indexed {total_chunks} total chunks into: {collection_name}")
    return total_chunks


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
