from uuid import uuid4
from qdrant_client.models import Distance, VectorParams, PointStruct
from shared.config import settings, get_qdrant_client, get_openai_client

def init_collection(qdrant_client):
    """Ensures the Qdrant collection exists with the correct dimensions."""
    collections = qdrant_client.get_collections().collections
    exists = any(c.name == settings.QDRANT_COLLECTION_NAME for c in collections)
    
    if not exists:
        print(f"Creating collection: {settings.QDRANT_COLLECTION_NAME}...")
        qdrant_client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION, 
                distance=Distance.COSINE
            ),
        )
        print("Collection created successfully.")
    else:
        print(f"Collection '{settings.QDRANT_COLLECTION_NAME}' already exists.")

def custom_chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[str]:
    """
    A raw sliding-window chunker based on character count.
    Splits text into digestible pieces without external library overhead.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += (chunk_size - chunk_overlap)
    return chunks

def ingest_document(text: str, doc_id: str = None):
    """Executes the full pipeline: chunks text, embeds chunks, and upserts to Qdrant."""
    qdrant = get_qdrant_client()
    openai = get_openai_client()
    
    # Step 1: Ensure collection is alive
    init_collection(qdrant)
    
    # Step 2: Chunk the document
    chunks = custom_chunk_text(text)
    print(f"Split document into {len(chunks)} chunks.")
    
    points = []
    # Step 3: Embed and prepare payload
    for i, chunk in enumerate(chunks):
        # Direct call to OpenAI SDK
        response = openai.embeddings.create(
            input=[chunk],
            model=settings.EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        
        # Structure the data exactly how Qdrant native API expects it
        point = PointStruct(
            id=str(uuid4()), # Generate a unique ID for each chunk
            vector=embedding,
            payload={
                "text": chunk,
                "chunk_index": i,
                "source_doc_id": doc_id or "generic_doc"
            }
        )
        points.append(point)
    
    # Batch upsert points natively into Qdrant
    qdrant.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=points
    )
    print(f"Successfully ingested {len(points)} points into Qdrant.")

if __name__ == "__main__":
    # Quick sanity check test text
    sample_document = """
    Retrieval-Augmented Generation (RAG) is an architectural pattern that optimizes the output of a 
    Large Language Model by referencing an authoritative knowledge base outside of its training data 
    sources before generating a response. Traditional RAG relies heavily on dense vector embeddings 
    and vector databases like Qdrant to find semantic similarities between a user prompt and raw document 
    chunks. While highly effective for local semantic matches, it often lacks the structural connectivity 
    required to resolve complex multi-hop queries that span disparate parts of a large corpus.
    """
    
    print("Starting test ingestion...")
    ingest_document(sample_document, doc_id="test_rag_explainer")
    
# python -m phase1_vector_rag.ingest