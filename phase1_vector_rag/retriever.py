import qdrant_client
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
import llama_index.core
from shared.config import settings, get_llamaindex_settings

# We only need the embedding model here to vectorize the user's query
_, embed_model = get_llamaindex_settings()
llama_index.core.Settings.embed_model = embed_model

def retrieve_chunks(user_query: str, top_k: int = 8):
    """
    Queries Qdrant and returns the retrieved nodes (including text and scores).
    """
    client = qdrant_client.QdrantClient(url=settings.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=settings.QDRANT_COLLECTION_NAME
    )

    # Connect to the existing Qdrant DB
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Use as_retriever to get the underlying nodes
    retriever = index.as_retriever(similarity_top_k=top_k)

    # This executes the similarity search and returns NodeWithScore objects
    nodes = retriever.retrieve(user_query)

    return nodes

if __name__ == "__main__":
    # --- CHANGE YOUR QUERY HERE ---
    test_query = "Quem é Harry Potter?"
    # ------------------------------

    print(f"\nSearching for: '{test_query}' in {settings.QDRANT_COLLECTION_NAME}...\n")

    results = retrieve_chunks(test_query, top_k=8)

    if not results:
        print("No relevant chunks found.")
    else:
        for i, node in enumerate(results):
            print("-" * 30)
            print(f"Chunk {i+1} | Score: {node.score:.4f}")
            print("-" * 30)
            print(f"{node.text[:500]}...") # Print first 500 chars
            print("\n")
            
            
# python -m phase1_vector_rag.retriever