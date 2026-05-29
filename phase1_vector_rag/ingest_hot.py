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

# 1. Renamed function from load_pdf to load_document
def load_document(file_path: Path):
    reader = SimpleDirectoryReader(input_files=[str(file_path)])    # SimpleDirectoryReader natively supports .txt files!
    return reader.load_data()


def ingest_directory_with_framework(directory_path: str):
    doc_files = sorted(list(Path(directory_path).glob("**/*.txt")))
    print(f"Found {len(doc_files)} TXT files in: {directory_path}\n")

    client = qdrant_client.QdrantClient(url=settings.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 3. CRITICAL CHANGE: Adjusted chunk size for HotpotQA!
    node_parser = TokenTextSplitter(chunk_size=200, chunk_overlap=20)

    total_chunks = 0

    # First Doc: creates the collection
    first_doc = doc_files[0]
    print(f"[1/{len(doc_files)}] Bootstrapping collection with: {first_doc.name}")
    try:
        documents = load_document(first_doc)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser],
            show_progress=True,
        )
        print(f"  -> Collection created and first doc indexed.\n")
        total_chunks += len(documents)
    except Exception as e:
        print(f"  -> ERROR on first doc: {e}")
        raise  # Can't continue without the collection

    # --- Remaining Docs: pipeline upserts into existing collection ---
    pipeline = IngestionPipeline(
        transformations=[node_parser, embed_model],
        vector_store=vector_store,
    )

    for i, doc_path in enumerate(doc_files[1:], start=2):
        print(f"[{i}/{len(doc_files)}] Processing: {doc_path.name}")
        try:
            documents = load_document(doc_path)

            nodes = pipeline.run(
                documents=documents,
                show_progress=True,
                num_workers=1,
            )

            total_chunks += len(nodes)
            print(f"  -> Indexed {len(nodes)} chunks (total so far: {total_chunks})")

        except Exception as e:
            print(f"  -> ERROR on {doc_path.name}: {e} — skipping")
            continue

    print(f"\nDone! Indexed {total_chunks} total chunks into: {settings.QDRANT_COLLECTION_NAME}")
    return total_chunks


if __name__ == "__main__":
    # 4. Point to the new Hotpot data folder!
    ingest_directory_with_framework("data_hotpot")
    # python -m phase1_vector_rag.ingest_hot