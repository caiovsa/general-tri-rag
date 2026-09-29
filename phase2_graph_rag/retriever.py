import json

import llama_index.core
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from shared.config import dataset_config, settings, get_llamaindex_settings

llm, embed_model = get_llamaindex_settings()
llama_index.core.Settings.llm = llm
llama_index.core.Settings.embed_model = embed_model

# Simple in-memory cache for entity extraction (avoids LLM call per identical query)
_ENTITY_CACHE: dict[str, list[str]] = {}


# ─────────────────────────────────────────────
# HELPER: Extract named entities from the query
# ─────────────────────────────────────────────
def extract_entities_from_query(user_query: str) -> list[str]:
    """
    Uses the LLM to extract key named entities (titles, people, places)
    from the user query. These are used as keyword anchors to guarantee
    the right seed chunks are retrieved even when vector search misses them.
    Results are cached per query string to avoid duplicate LLM calls.
    """
    if not user_query or not user_query.strip():
        return []
    # Cache hit?
    if user_query in _ENTITY_CACHE:
        print(f"-> Extracted entities (cached): {_ENTITY_CACHE[user_query]}")
        return _ENTITY_CACHE[user_query]

    response = llm.complete(
        f"""Extract the key named entities (people, places, episode titles, movie titles, show titles) 
from this question. Return ONLY a comma-separated list of entity names, nothing else.
Do not include generic words like 'episode', 'brother', 'guest', 'star'.

Question: {user_query}
Entities:"""
    )
    raw = response.text.strip()
    entities = [e.strip() for e in raw.split(",") if e.strip()]
    print(f"-> Extracted entities: {entities}")
    _ENTITY_CACHE[user_query] = entities
    return entities


# ─────────────────────────────────────────────
# HELPER: Single reusable Cypher block
# ─────────────────────────────────────────────
NEIGHBOR_QUERY = """
    WITH DISTINCT seed_chunk, score
    OPTIONAL MATCH (seed_chunk)-[:MENTIONS]->(entity)<-[:MENTIONS]-(neighbor_chunk)
    WHERE neighbor_chunk <> seed_chunk
    WITH seed_chunk, score, COLLECT(DISTINCT neighbor_chunk) AS neighbors
    RETURN seed_chunk.id             AS chunk_id,
           seed_chunk.text          AS chunk_text,
           seed_chunk._node_content AS node_content,
           score,
           [n IN neighbors | n.text][0..3] AS neighbor_texts
    ORDER BY score DESC
"""


# ─────────────────────────────────────────────
# MAIN RETRIEVER
# ─────────────────────────────────────────────
def retrieve_graph_chunks(user_query: str, top_k: int = None) -> list[str]:
    """
    Hybrid Graph RAG retrieval:
    1. Vector search  — semantic similarity to catch related chunks
    2. Keyword anchor — entity-based exact match to guarantee seed chunks
    3. 1-hop traversal — neighbor chunks via shared entity mentions
    4. Deduplication  — clean, non-redundant context for the LLM
    """
    dataset = dataset_config(phase=2)
    if top_k is None:
        top_k = dataset.top_k
    if not user_query or not user_query.strip():
        print("Warning: empty query — returning no results.")
        return []

    # ── Connect ──────────────────────────────
    print(f"Connecting to Neo4j at {dataset.neo4j_uri} (database={dataset.neo4j_database})...")
    try:
        graph_store = Neo4jPropertyGraphStore(
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            url=dataset.neo4j_uri,
            database=dataset.neo4j_database,
        )
        # cheap preflight
        graph_store.structured_query("RETURN 1 AS ok")
    except Exception as e:
        print(f"Error: Neo4j not reachable at {dataset.neo4j_uri} (db={dataset.neo4j_database}): {e}")
        print("Hint: run `docker compose up -d neo4j` or check Neo4j Desktop and NEO4J_DATABASE in .env")
        return []

    # ── Step 1: Vector search ─────────────────
    print(f"Embedding query: '{user_query}'")
    query_embedding = embed_model.get_query_embedding(user_query)

    print(f"Running vector search (top_k={top_k})...")
    vector_results = graph_store.structured_query(
        f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $embedding)
        YIELD node AS seed_chunk, score
        {NEIGHBOR_QUERY}
        """,
        param_map={
            "embedding": query_embedding,
            "limit": top_k,
        },
    )
    print(f"-> Vector search returned {len(vector_results or [])} chunks.")

    # ── Step 2: Keyword anchor search ─────────
    entities = extract_entities_from_query(user_query)

    keyword_results = []
    for entity in entities:
        print(f"Running keyword anchor search for: '{entity}'...")
        # Prefer full-text index (fast) if available; fallback to CONTAINS scan
        try:
            results = graph_store.structured_query(
                f"""
                CALL db.index.fulltext.queryNodes('chunk_text_fulltext', $keyword)
                YIELD node AS seed_chunk, score
                {NEIGHBOR_QUERY}
                LIMIT $limit
                """,
                param_map={
                    "keyword": entity,
                    "limit": 3,
                },
            )
            # If index missing, Neo4j returns error — caught below
            if results is None:
                raise RuntimeError("fulltext returned None")
        except Exception:
            # Fallback: CONTAINS scan (correct but slower)
            results = graph_store.structured_query(
                f"""
                MATCH (seed_chunk)
                WHERE seed_chunk.text IS NOT NULL
                  AND toLower(seed_chunk.text) CONTAINS toLower($keyword)
                WITH seed_chunk, 1.0 AS score
                {NEIGHBOR_QUERY}
                LIMIT $limit
                """,
                param_map={
                    "keyword": entity,
                    "limit": 3,
                },
            )
        found = len(results or [])
        print(f"-> Keyword anchor '{entity}' returned {found} chunks.")
        keyword_results.extend(results or [])

    # ── Step 3: Merge + deduplicate ───────────
    # Keyword results go FIRST so the LLM sees the most relevant anchor chunk
    # at the top of the context, before the semantic results.
    all_results = keyword_results + (vector_results or [])

    if not all_results:
        print("-> No chunks found.")
        return []

    chunks = []
    seen_texts = set()

    for row in all_results:
        # Get chunk text (with _node_content fallback)
        chunk_text = row.get("chunk_text", "")
        if not chunk_text:
            node_content = row.get("node_content", "")
            if node_content:
                try:
                    chunk_text = json.loads(node_content).get("text", "")
                except json.JSONDecodeError:
                    pass

        # Skip empty or already-seen chunks
        if not chunk_text or chunk_text in seen_texts:
            continue
        seen_texts.add(chunk_text)

        score = row.get("score", 0.0)
        neighbor_texts = row.get("neighbor_texts", [])

        # Deduplicate neighbors too
        valid_neighbors = [
            t for t in neighbor_texts
            if t and t not in seen_texts
        ]
        neighbor_text = ""
        if valid_neighbors:
            neighbor_text = (
                "\n\n--- Connected Context ---\n"
                + "\n---\n".join(valid_neighbors)
            )
            for t in valid_neighbors:
                seen_texts.add(t)

        chunks.append(f"[Relevance: {score:.4f}]\n{chunk_text}{neighbor_text}")

    print(f"-> Returning {len(chunks)} unique enriched chunks.")
    return chunks


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_question = "Who is the younger brother of the episode guest stars of The Hard Easy?"

    results = retrieve_graph_chunks(test_question, top_k=5)

    print("\n" + "=" * 50)
    print("Retrieved Graph RAG Context:")
    print("=" * 50)
    for i, chunk in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(chunk[:1000])

# python -m phase2_graph_rag.retriever