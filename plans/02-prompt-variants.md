# Task 02 — Prompt Variants

## Goal
Create domain-specific prompt variants for Phase 1 (Vector RAG) and Phase 2 (Graph RAG): one generalist and one FinanceBench-specific for each.

## Scope
- `phase1_vector_rag/rag.py` (prompt builder)
- `phase2_graph_rag/rag.py` (prompt builder)
- Possibly `phase1_vector_rag/retriever.py` and `phase2_graph_rag/retriever.py` if prompt logic lives there

## Implementation Steps

### Phase 1 (Vector RAG) Prompts
- [ ] **Generalist prompt**: Context-only, strict "I don't know" when missing, no speculation
  - Annotate chunks with relevance scores
  - Instruct model to prefer higher-scored chunks
  - Explicit uncertainty guidance when all scores are low
- [ ] **FinanceBench prompt**: Finance-focused, table/statement-aware
  - Instruct model to prioritize balance sheet, cash flow, income statement data
  - Handle numerical queries carefully (units, years, currency)
  - Reference specific financial document structure when applicable

### Phase 2 (Graph RAG) Prompts
- [ ] **Generalist GraphRAG prompt**: Must follow graph relations, no hallucinated edges
  - Explicitly tell model to use graph triplets for entity connections
  - Never invent relationships not present in the context
  - Combine text chunks and triplet evidence
- [ ] **FinanceBench GraphRAG prompt**: Finance-focused, graph + text alignment
  - Same as generalist but with finance domain instructions
  - Prioritize graph edges that connect financial entities (companies, years, metrics)
  - Handle multi-part queries by tracing graph paths

### Integration
- [ ] Add a `prompt_variant` parameter to `run_rag_pipeline` and `run_graph_rag_pipeline`
- [ ] Store prompt templates in a shared location or as separate functions
- [ ] Update `if __name__ == "__main__"` blocks to test both variants
- [ ] Document which variant to use for which dataset

## Verification
- Run each variant with a known query and inspect the generated prompt
- Compare answers across variants for the same query
- Verify FinanceBench prompts handle numerical/financial queries better than generalist

## Notes
- Phase 1 already has score annotation in `build_prompt()` — extend, don't replace
- Phase 2 already has graph relationship instructions in `build_hybrid_prompt()` — refine, don't rewrite
- Keep prompts in sync with RAGAS evaluation format if benchmarking later
