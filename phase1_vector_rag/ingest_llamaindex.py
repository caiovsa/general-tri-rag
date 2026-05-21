from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from shared.config import settings, get_llamaindex_settings
import llama_index.core

# 1. Apply global LLM and Embedding models to LlamaIndex
llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model

def ingest_directory_with_framework(directory_path: str):
    # 2. Load all documents in the directory
    print(f"Loading documents from: {directory_path}")
    reader = SimpleDirectoryReader(input_dir=directory_path)
    documents = reader.load_data()
    
    # 3. Define the exact splitter
    node_parser = TokenTextSplitter(chunk_size=512, chunk_overlap=64)
    
    # 4. Initialize the Qdrant client and Vector Store
    client = qdrant_client.QdrantClient(url=settings.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=settings.QDRANT_COLLECTION_NAME
    )
    
    # 5. Build storage context and index the data
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"Parsing, embedding, and indexing {len(documents)} pages into Qdrant...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[node_parser]
    )
    
    print(f"Successfully indexed directory into collection: {settings.QDRANT_COLLECTION_NAME}")
    return index

if __name__ == "__main__":
    # Ingest everything in the data folder
    ingest_directory_with_framework("data")
    
# python -m phase1_vector_rag.ingest_llamaindex