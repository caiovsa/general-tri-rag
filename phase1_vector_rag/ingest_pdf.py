from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from pathlib import Path
from shared.config import settings, get_llamaindex_settings
import llama_index.core

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model


def load_pdf(pdf_path: Path):
    reader = SimpleDirectoryReader(input_files=[str(pdf_path)])
    return reader.load_data()


def ingest_directory_with_framework(directory_path: str):
    pdf_files = list(Path(directory_path).glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in: {directory_path}\n")

    client = qdrant_client.QdrantClient(url=settings.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    node_parser = TokenTextSplitter(chunk_size=512, chunk_overlap=64)

    total_chunks = 0

    # --- First PDF: creates the collection ---
    first_pdf = pdf_files[0]
    print(f"[1/{len(pdf_files)}] Bootstrapping collection with: {first_pdf.name}")
    try:
        documents = load_pdf(first_pdf)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser],
            show_progress=True,
        )
        print(f"  -> Collection created and first PDF indexed.\n")
        total_chunks += len(documents)
    except Exception as e:
        print(f"  -> ERROR on first PDF: {e}")
        raise  # Can't continue without the collection

    # --- Remaining PDFs: pipeline upserts into existing collection ---
    pipeline = IngestionPipeline(
        transformations=[node_parser, embed_model],
        vector_store=vector_store,
    )

    for i, pdf_path in enumerate(pdf_files[1:], start=2):
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        try:
            documents = load_pdf(pdf_path)
            print(f"  -> Loaded {len(documents)} pages")

            nodes = pipeline.run(
                documents=documents,
                show_progress=True,
                num_workers=1,
            )

            total_chunks += len(nodes)
            print(f"  -> Indexed {len(nodes)} chunks (total so far: {total_chunks})")

        except Exception as e:
            print(f"  -> ERROR on {pdf_path.name}: {e} — skipping")
            continue

    print(f"\nDone! Indexed {total_chunks} total chunks into: {settings.QDRANT_COLLECTION_NAME}")
    return total_chunks


if __name__ == "__main__":
    ingest_directory_with_framework("data_finance")
    # python -m phase1_vector_rag.ingest