"""
Phase 3 — Ontology RAG Retrieval
=================================

TODO: Implement ontology-guided retrieval.

Design:
  1. Embed the user query (reuse shared embed_model).
  2. Vector search on Chunk nodes in Neo4j (same as Phase 2).
  3. For each matched chunk, traverse the graph BUT use ontology types:
     a. Walk typed relationships (e.g., Person->worksFor->Organization).
     b. Use ontology class hierarchy for generalization/specialization
        (e.g., if query asks about "people", match all :Person nodes).
     c. Optionally use OWL reasoning to infer implicit relationships
        (e.g., if A is part of B, and B is part of C, infer A is part of C).
  4. Return chunks enriched with typed relationship triplets:
     "(Harry Potter:Person)-[:WORKS_FOR]->(Grifinória:Organization)"

Key differences from Phase 2:
  - Relationships are typed and constrained by the ontology schema.
  - Retrieval can use class hierarchy (e.g., "all Organizations" matches
    Company, University, GovernmentAgency subclasses).
  - Reasoning can infer implicit connections not explicitly extracted.
  - Better for complex multi-hop queries that require type awareness.

Cypher approach:
  MATCH (c:Chunk)-[:MENTIONS]->(e)
  WHERE e:Person OR e:Organization  // ontology-typed filtering
  OPTIONAL MATCH (e)-[r]->(related)
  RETURN c, e, type(r) AS rel_type, related

Run: python -m phase3_ontology_rag.retriever
"""
