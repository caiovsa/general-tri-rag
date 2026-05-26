import qdrant_client
import llama_index.core
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Import your unified settings
from shared.config import settings, get_llamaindex_settings

# 1. Apply global models (We need the embed model to vectorize the user's query)
llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model

def retrieve_graph_chunks(user_query: str, top_k: int = 3) -> list[str]:
    """
    Connects to both Neo4j and Qdrant. 
    Uses vectors to find the entry node, then walks the graph to get relationships.
    Returns the combined raw text and structural context.
    """
    print("Connecting to Neo4j...")
    graph_store = Neo4jPropertyGraphStore(
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        url=settings.NEO4J_URI,
    )
    
    # Make sure you are using the Phase 2 isolated collection!
    print(f"Connecting to Qdrant collection: {settings.QDRANT_COLLECTION_NAME_PHASE2}...")
    client = qdrant_client.QdrantClient(url=settings.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=settings.QDRANT_COLLECTION_NAME_PHASE2
    )
    
    # 2. Load the Index from the existing databases (NO INGESTION HERE)
    # This automatically detects that you have both a graph and a vector store
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        vector_store=vector_store,
    )
    
    # 3. Create the Hybrid Retriever
    # Because both stores are present, LlamaIndex automatically defaults to a 
    # VectorContextRetriever, which does the Vector -> Graph traversal.
    retriever = index.as_retriever(similarity_top_k=top_k)
    
    print(f"\nTraversing graph and vectors for query: '{user_query}'...")
    nodes = retriever.retrieve(user_query)
    
    # 4. Extract the rich text
    chunks = []
    for node in nodes:
        # In a PropertyGraphIndex, 'get_content()' returns the original text chunk
        # PLUS a stringified list of all the relationships attached to it!
        chunks.append(node.get_content())
        
    return chunks

if __name__ == "__main__":
    # Test your Graph Retrieval!
    # Pick a question related to the subset of PDFs you just ingested.
    test_question = "Onde o Sr. Dursley trabalha?"
    
    results = retrieve_graph_chunks(test_question)
    
    print("\n" + "=" * 50)
    print("Retrieved Hybrid Context (Text + Graph Triplets):")
    print("=" * 50)
    for i, chunk in enumerate(results):
        print(f"\n[Result {i+1}]")
        # Printing the first 800 characters so you can see the text and the injected triplets
        print(f"{chunk[:800]}...\n")
        
# python -m phase2_graph_rag.retriever