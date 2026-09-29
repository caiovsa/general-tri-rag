from pathlib import Path
from shared.config import dataset_config, settings, get_llamaindex_settings
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


def ingest_directory_with_framework(directory_path: str = None):
    """Ingest the active dataset into Qdrant.

    DATASET=hotpot (default): globs data_hotpot/ exactly as before.
    DATASET=metaqa: loads ONLY the files listed in MetaQA/corpus_manifest.txt and
    tags every chunk with "[Movie: <title>]" before embedding.
    """
    dataset = dataset_config(phase=1)

    if dataset.doc_list:
        print(f"Ingesting {dataset.name} into Qdrant collection: {dataset.qdrant_collection}")
        print(f"Chunk size={dataset.chunk_size}, overlap={dataset.chunk_overlap}, docs from {dataset.doc_list}\n")
        documents = load_manifest_documents(dataset.doc_list)
        return ingest_documents(
            documents=documents,
            chunk_size=dataset.chunk_size,
            chunk_overlap=dataset.chunk_overlap,
            collection_name=dataset.qdrant_collection,
            qdrant_url=settings.QDRANT_URL,
            embed_model=embed_model,
            dataset=dataset,
        )

    return ingest_directory(
        directory_path=directory_path or str(dataset.docs_dir),
        pattern="**/*.txt",
        chunk_size=dataset.chunk_size,
        chunk_overlap=dataset.chunk_overlap,
        collection_name=dataset.qdrant_collection,
        qdrant_url=settings.QDRANT_URL,
        embed_model=embed_model,
    )


if __name__ == "__main__":
    ingest_directory_with_framework()
    # python -m phase1_vector_rag.ingest_hot          (DATASET=hotpot)
    # DATASET=metaqa python -m phase1_vector_rag.ingest_hot
