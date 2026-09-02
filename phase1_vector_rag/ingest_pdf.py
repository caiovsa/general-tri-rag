from pathlib import Path
from shared.config import settings, get_llamaindex_settings
from shared.ingest import ingest_directory, load_documents
import llama_index.core

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model


def load_pdf(pdf_path: Path):
    return load_documents(pdf_path)


def ingest_directory_with_framework(directory_path: str):
    """Ingest finance PDFs into Qdrant (chunk_size from shared/config.py)."""
    return ingest_directory(
        directory_path=directory_path,
        pattern="**/*.pdf",
        chunk_size=settings.CHUNK_SIZE_PDF,
        chunk_overlap=settings.CHUNK_OVERLAP_PDF,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        qdrant_url=settings.QDRANT_URL,
        embed_model=embed_model,
    )


if __name__ == "__main__":
    ingest_directory_with_framework("data_finance")
    # python -m phase1_vector_rag.ingest