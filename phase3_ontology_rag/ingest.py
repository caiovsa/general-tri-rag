"""
Phase 3 — Ontology RAG Ingestion
=================================

TODO: Implement ontology-aware ingestion.

Design:
  1. Load the OWL ontology from `phase3_ontology_rag/ontology/domain.owl`
     using owlready2 or rdflib.
  2. Parse documents (PDFs/TXTs) and chunk them (reuse shared chunker).
  3. For each chunk:
     a. Extract entities using the LLM, but typed according to the ontology
        (e.g., Person, Organization, Location, Event).
     b. Extract relationships using ontology-defined properties
        (e.g., worksFor, locatedIn, relatedTo, partOf).
     c. Store chunks in Neo4j with embeddings (like Phase 2) BUT also
        tag each entity node with its ontology class.
  4. Create a vector index on Chunk nodes (same as Phase 2).
  5. Create constraints for each ontology class (Person, Organization, etc.)
     to enforce typed nodes.

Key differences from Phase 2:
  - Entities are typed (have an ontology class label), not just generic __Entity__.
  - Relationships use ontology-defined predicates, not free-form LLM extraction.
  - The ontology provides a schema that constrains what triples are valid.

Dependencies to add:
  - owlready2 (for OWL loading and reasoning)
  - rdflib (for RDF/SPARQL if needed)

Neo4j schema:
  - (:Person {id, name, embedding})
  - (:Organization {id, name, embedding})
  - (:Location {id, name, embedding})
  - (:Event {id, name, embedding})
  - (:Chunk {id, text, embedding})
  - (c:Chunk)-[:MENTIONS]->(e:Entity)
  - (e1:Entity)-[:WORKS_FOR]->(e2:Organization)
  - (e1:Entity)-[:LOCATED_IN]->(e2:Location)
  - etc.

Run: python -m phase3_ontology_rag.ingest
"""
