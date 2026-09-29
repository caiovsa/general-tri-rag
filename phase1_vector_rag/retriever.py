import qdrant_client
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
import llama_index.core
from shared.config import dataset_config, settings, get_llamaindex_settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import QueryBundle

# We only need the embedding model here to vectorize the user's query
_, embed_model = get_llamaindex_settings()
llama_index.core.Settings.embed_model = embed_model

def retrieve_chunks(user_query: str, top_k: int = None, node_threshold: float = None):
    """
    Queries Qdrant and returns the retrieved nodes (including text and scores).

    Store, collection and default top_k come from the active DATASET
    (shared/config.py) so HotpotQA and MetaQA never share a collection.
    """
    dataset = dataset_config(phase=1)
    if not user_query or not user_query.strip():
        print("Warning: empty query — returning no results.")
        return []
    if top_k is None:
        top_k = dataset.top_k
    if node_threshold is None:
        node_threshold = settings.DEFAULT_SIMILARITY_THRESHOLD

    # Preflight: is Qdrant reachable?
    try:
        client = qdrant_client.QdrantClient(url=settings.QDRANT_URL)
        client.get_collections()  # cheap health check
    except Exception as e:
        print(f"Error: Qdrant not reachable at {settings.QDRANT_URL}: {e}")
        print("Hint: run `docker compose up -d qdrant` and ensure the collection was ingested.")
        return []

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=dataset.qdrant_collection
    )

    # Connect to the existing Qdrant DB
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Use as_retriever to get the underlying nodes
    retriever = index.as_retriever(similarity_top_k=top_k)

    # This executes the similarity search and returns NodeWithScore objects
    nodes = retriever.retrieve(user_query)
    postprocessor = SimilarityPostprocessor(similarity_cutoff=node_threshold)
    filtered_nodes = postprocessor.postprocess_nodes(nodes, query_bundle=QueryBundle(user_query))
        
    return filtered_nodes

if __name__ == "__main__":
    # --- CHANGE YOUR QUERY HERE ---
    test_query = "The 41st International 500-Mile Sweepstakes was held at which location?" #The Coca-Cola system sold how many unit cases of products in 2019, 2018 and 2017, respectively." #"Onde o Sr. Dursley trabalha?" #"Quem é Mike Tyson?"

    print(f"\nSearching for: '{test_query}' in {dataset_config(phase=1).qdrant_collection}...\n")

    results = retrieve_chunks(test_query, top_k=8, node_threshold=0.4)

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