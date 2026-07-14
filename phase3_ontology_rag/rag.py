"""
Phase 3 — Ontology RAG Pipeline
================================

TODO: Implement ontology-aware RAG pipeline.

Design:
  1. Call retrieve_graph_chunks() from phase3_ontology_rag.retriever.
  2. Build a prompt that includes:
     a. Retrieved chunk text.
     b. Typed graph relationships (e.g., "Harry Potter is a Person who
        works_for Grifinória which is an Organization").
     c. Ontology class context (e.g., "Person: an individual human.
        Organization: a structured group of people.").
  3. The prompt should instruct the LLM to use the ontology types for
     reasoning: "Use the entity types and their relationships to answer
     questions that require understanding what kind of thing an entity is."

Key differences from Phase 2:
  - Prompt includes entity type information (Person, Organization, etc.).
  - LLM can reason about entity categories, not just named relationships.
  - Better for questions like "What organizations is X part of?" where
    the ontology defines what counts as an organization.

Example prompt structure:
  "The context below includes typed entities and relationships.
   Entity types are defined by the ontology:
   - Person: an individual human
   - Organization: a structured group
   - Location: a geographic place
   Use the types to answer questions that require category reasoning."

Run: python -m phase3_ontology_rag.rag
"""
