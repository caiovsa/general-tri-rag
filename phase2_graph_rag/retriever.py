import json
import llama_index.core
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.vector_stores import VectorStoreQuery

from shared.config import settings, get_llamaindex_settings

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model


def _get_chunk_text(chunk_data: dict) -> str:
    """Extract text from a Chunk node returned by Neo4j."""
    props = chunk_data.get('properties', {})
    # Text might be stored directly or in _node_content JSON
    text = props.get('text', '')
    if not text:
        node_content = props.get('_node_content', '')
        if node_content:
            try:
                data = json.loads(node_content)
                text = data.get('text', '')
            except json.JSONDecodeError:
                pass
    return text


def _format_triplets(triplets: list) -> str:
    """Format graph triplets as readable text."""
    if not triplets:
        return ""
    lines = []
    for t in triplets:
        # Triplet format: (source_node, relation, target_node)
        src = t[0]
        rel = t[1]
        tgt = t[2]
        
        src_id = getattr(src, 'id', str(src))
        tgt_id = getattr(tgt, 'id', str(tgt))
        rel_type = getattr(rel, 'type', str(rel))
        
        lines.append(f"({src_id})-[:{rel_type}]->({tgt_id})")
    return "\n".join(lines)


def retrieve_graph_chunks(user_query: str, top_k: int = 5, max_rel_depth: int = 1) -> list[str]:
    """
    Pure Graph RAG retrieval using Neo4j only.
    
    Flow:
    1. Embed query
    2. Vector search on Chunk nodes in Neo4j
    3. For each chunk, traverse graph relationships
    4. Return combined text + triplets
    """
    print("Connecting to Neo4j...")
    graph_store = Neo4jPropertyGraphStore(
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        url=settings.NEO4J_URI,
        database=settings.NEO4J_DATABASE,
    )
    
    # 1. Embed the query
    print(f"Embedding query: '{user_query}'")
    query_embedding = embed_model.get_query_embedding(user_query)
    
    # 2. Vector search on Chunk nodes using Neo4j's vector index
    print(f"Searching for similar chunks (top_k={top_k})...")
    
    # Use direct Cypher query for Chunk nodes (bypasses the __Entity__ index)
    chunks_with_rels = graph_store.structured_query("""
        CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $embedding)
        YIELD node, score
        MATCH (node)-[r]-(related)
        WITH node, score, r, related, 
             type(r) AS rel_type,
             related.id AS related_id
        ORDER BY score DESC, node.id
        RETURN node.id AS chunk_id, 
               node.text AS chunk_text,
               node._node_content AS node_content,
               score,
               COLLECT(DISTINCT {rel: rel_type, target: related_id}) AS relationships
        LIMIT $limit
    """, param_map={
        'embedding': query_embedding,
        'limit': top_k,
    })
    
    if not chunks_with_rels:
        print("-> No chunks found.")
        return []
    
    print(f"-> Found {len(chunks_with_rels)} chunks with relationships.")
    
    # 3. Build enriched chunks
    chunks = []
    for row in chunks_with_rels:
        chunk_text = row.get('chunk_text', '')
        score = row.get('score', 0.0)
        relationships = row.get('relationships', [])
        
        # Fallback: extract from _node_content if text is empty
        if not chunk_text:
            node_content = row.get('node_content', '')
            if node_content:
                try:
                    data = json.loads(node_content)
                    chunk_text = data.get('text', '')
                except json.JSONDecodeError:
                    pass
        
        if not chunk_text:
            continue
        
        # Format relationships
        rel_text = ""
        if relationships:
            rel_lines = [f"({row['chunk_id']})-[:{r['rel']}]->({r['target']})" for r in relationships]
            rel_text = "\n\n--- Graph Relationships ---\n" + "\n".join(rel_lines)
        
        enriched = f"[Relevance: {score:.4f}]\n{chunk_text}{rel_text}"
        chunks.append(enriched)
    
    print(f"-> Returning {len(chunks)} enriched chunks.")
    return chunks


if __name__ == "__main__":
    test_question = "Qual o modelo da vassoura que harry ganhou?" #"Qual a casa de harry potter?"
    
    results = retrieve_graph_chunks(test_question, top_k=5)
    
    print("\n" + "=" * 50)
    print("Retrieved Graph RAG Context (Text + Triplets):")
    print("=" * 50)
    for i, chunk in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(f"{chunk[:800]}...\n")
        
# python -m phase2_graph_rag.retriever
