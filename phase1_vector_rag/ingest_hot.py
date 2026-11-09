import argparse
from pathlib import Path
from shared.config import dataset_config, settings, get_llamaindex_settings, write_ingest_meta
from shared.ingest import (
    ingest_directory,
    ingest_documents,
    load_documents,
    load_manifest_documents,
)
import llama_index.core

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model

# Backwards-compat alias (used by older callers / notebooks)
def load_document(file_path: Path):
    return load_documents(file_path)


def ingest_directory_with_framework(directory_path: str = None, reset: bool = False):
    """Ingest the active dataset into Qdrant.

    DATASET=hotpot (default): globs data_hotpot/ exactly as before.
    DATASET=metaqa|metaqa_mini|metaqa_mini30: loads ONLY the files listed in the
    dataset manifest and tags every chunk with "[Movie: <title>]" before embedding.
    DATASET=local: globs LOCAL_DOCS_DIR with LOCAL_PATTERN (txt/pdf/md).
    """
    dataset = dataset_config(phase=1)

    if dataset.doc_list:
        print(f"Ingesting {dataset.name} into Qdrant collection: {dataset.qdrant_collection}")
        print(f"Chunk size={dataset.chunk_size}, overlap={dataset.chunk_overlap}, docs from {dataset.doc_list}\n")
        documents = load_manifest_documents(dataset.doc_list)
        chunks = ingest_documents(
            documents=documents,
            chunk_size=dataset.chunk_size,
            chunk_overlap=dataset.chunk_overlap,
            collection_name=dataset.qdrant_collection,
            qdrant_url=settings.QDRANT_URL,
            embed_model=embed_model,
            dataset=dataset,
            reset=reset,
        )
    else:
        chunks = ingest_directory(
            directory_path=directory_path or str(dataset.docs_dir),
            pattern=dataset.file_pattern,
            chunk_size=dataset.chunk_size,
            chunk_overlap=dataset.chunk_overlap,
            collection_name=dataset.qdrant_collection,
            qdrant_url=settings.QDRANT_URL,
            embed_model=embed_model,
            reset=reset,
        )
    write_ingest_meta(
        dataset, 1,
        collection=dataset.qdrant_collection,
        chunks=chunks,
        embedding_model=settings.EMBEDDING_MODEL,
        reset=reset,
    )
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 (Vector RAG) ingest for the active DATASET")
    parser.add_argument("--reset", action="store_true",
                        help="delete the active dataset's Qdrant collection before ingesting")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_directory_with_framework(reset=args.reset)
    # python -m phase1_vector_rag.ingest_hot              (DATASET=hotpot)
    # DATASET=metaqa python -m phase1_vector_rag.ingest_hot --reset
