import argparse
from pathlib import Path
from shared.config import settings, get_llamaindex_settings
from shared.ingest import ingest_directory, load_documents
import llama_index.core

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model


def load_pdf(pdf_path: Path):
    return load_documents(pdf_path)


def ingest_directory_with_framework(directory_path: str, collection_name: str = None, reset: bool = False):
    """Ingest finance PDFs into an ISOLATED Qdrant collection.

    Historical helper: it must never upsert into the HotpotQA baseline
    collection, so the default target is `<QDRANT_COLLECTION_NAME>_finance`.
    """
    return ingest_directory(
        directory_path=directory_path,
        pattern="**/*.pdf",
        chunk_size=settings.CHUNK_SIZE_PDF,
        chunk_overlap=settings.CHUNK_OVERLAP_PDF,
        collection_name=collection_name or f"{settings.QDRANT_COLLECTION_NAME}_finance",
        qdrant_url=settings.QDRANT_URL,
        embed_model=embed_model,
        reset=reset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 finance PDF ingest (isolated collection)")
    parser.add_argument("--dir", default="data_finance", help="PDF directory (default: data_finance)")
    parser.add_argument("--collection", default=None,
                        help="target collection (default: <QDRANT_COLLECTION_NAME>_finance)")
    parser.add_argument("--reset", action="store_true", help="delete the target collection first")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_directory_with_framework(args.dir, args.collection, reset=args.reset)
    # python -m phase1_vector_rag.ingest_pdf --dir data_finance