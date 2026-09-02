# RAG Trilogy — common commands
# Usage: make <target>  (prefix with `uv run` is handled inside)

PYTHON := uv run python
PIP := uv run pip

.PHONY: help install sync prepare ingest-hot ingest-pdf ingest-graph retrieve-p1 retrieve-p2 rag-p1 rag-p2 hotpot-p1 hotpot-p2 bench-p1 bench-p2 bench-all docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install deps (uv sync)
	uv sync

sync: install  ## Alias for install

prepare:  ## Download HotpotQA (100 Qs → data_hotpot + hotpot_eval.jsonl)
	$(PYTHON) -m prepare_hotpot

ingest-hot:  ## Phase 1: ingest HotpotQA TXT into Qdrant (chunk 200/20)
	$(PYTHON) -m phase1_vector_rag.ingest_hot

ingest-pdf:  ## Phase 1: ingest finance PDFs into Qdrant (chunk 512/64)
	$(PYTHON) -m phase1_vector_rag.ingest_pdf

ingest-graph:  ## Phase 2: ingest HotpotQA into Neo4j (chunk 256/20, batch 2000)
	$(PYTHON) -m phase2_graph_rag.ingest

retrieve-p1:  ## Phase 1: test vector retrieval
	$(PYTHON) -m phase1_vector_rag.retriever

retrieve-p2:  ## Phase 2: test graph retrieval (vector + traversal)
	$(PYTHON) -m phase2_graph_rag.retriever

rag-p1:  ## Phase 1: full RAG (retrieve → prompt → generate)
	$(PYTHON) -m phase1_vector_rag.rag

rag-p2:  ## Phase 2: full RAG (vector + graph → prompt → generate)
	$(PYTHON) -m phase2_graph_rag.rag

hotpot-p1:  ## Phase 1: generate answers for 100 HotpotQA Qs
	$(PYTHON) -m phase1_vector_rag.hotpot_answers

hotpot-p2:  ## Phase 2: generate answers for 100 HotpotQA Qs
	$(PYTHON) -m phase2_graph_rag.hotpot_answers

bench-p1:  ## Phase 1: evaluate (EM, F1, Semantic, Judge, RAGAS)
	$(PYTHON) -m phase1_vector_rag.benchmark

bench-p2:  ## Phase 2: evaluate
	$(PYTHON) -m phase2_graph_rag.benchmark

bench-all: bench-p1 bench-p2  ## Evaluate both phases

docker-up:  ## Start Qdrant + Neo4j
	docker compose up -d qdrant neo4j

docker-down:  ## Stop Qdrant + Neo4j
	docker compose down

clean:  ## Remove generated HotpotQA artifacts (keeps source PDFs)
	rm -rf data_hotpot/*.txt hotpot_eval.jsonl
	rm -f phase1_vector_rag/hotpot_*.json phase1_vector_rag/hotpot_*.txt
	rm -f phase2_graph_rag/hotpot_*.json phase2_graph_rag/hotpot_*.txt
	rm -f checkpoint_lightrag.json digest.txt
