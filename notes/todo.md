# TODO / Plans

> Track high-level tasks. Each item links to a detailed plan in `plans/`.

## Active Tasks
- [ ] [Prompt Variants](plans/02-prompt-variants.md) — Add generalist + finance prompts to Phase 1 and Phase 2 (Still need it)
- [ ] Create the phase_1_all_questions and phase_2_all_questions (Need to save the answers from all of them!)
- [ ] Create the bench mark (Maybe RAGAS or something like this)
- [ ] Organize the repo! See how to make this a project

## Completed Tasks
- [X] Changed from finance_bench to hotpot!
- [X] Phase1 is already using hotpot!
- [X] [Docker Neo4j Multi](plans/03-docker-neo4j-multi.md) — Using Neo4j Desktop with separate databases instead of containers
- [X] [Phase 2 Retriever Debug](plans/01-phase2-audit.md) — Identified ID mismatch between Qdrant and Neo4j; led to decision for pure Graph RAG
- [X] Fix phase2_graph_rag - Fix ingestion (Need to update the ingestion.py) and see if retriever and rag are fine