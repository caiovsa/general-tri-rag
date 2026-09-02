from pathlib import Path
from shared.config import settings, get_llamaindex_settings
from shared.ingest import ingest_directory, load_documents
import llama_index.core

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model

# Backwards-compat alias (used by older callers / notebooks)
def load_document(file_path: Path):
    return load_documents(file_path)


def ingest_directory_with_framework(directory_path: str):
    """Ingest HotpotQA TXT files into Qdrant (chunk_size from shared/config.py)."""
    return ingest_directory(
        directory_path=directory_path,
        pattern="**/*.txt",
        chunk_size=settings.CHUNK_SIZE_HOTPOT,
        chunk_overlap=settings.CHUNK_OVERLAP_HOTPOT,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        qdrant_url=settings.QDRANT_URL,
        embed_model=embed_model,
    )


if __name__ == "__main__":
    # 4. Point to the new Hotpot data folder!
    ingest_directory_with_framework("data_hotpot")
    # python -m phase1_vector_rag.ingest_hot