# RAG Trilogy — common commands
# Usage: make <target> [DATASET=hotpot|metaqa|metaqa_mini]
# `uv run` is handled inside the commands.

PYTHON := uv run python
SHA256SUM := $(shell command -v sha256sum 2>/dev/null || echo "shasum -a 256")
DATASET ?= hotpot
LIMIT ?=
LIMIT_FLAG := $(if $(LIMIT),--limit $(LIMIT),)
DS := DATASET=$(DATASET)

.PHONY: help install sync prepare ingest-hot ingest-pdf ingest-graph retrieve-p1 retrieve-p2 rag-p1 rag-p2 \
        ingest-p1 ingest-p2 answers-p1 answers-p2 bench-p1 bench-p2 bench-all check-config freeze-verify \
        answers-cb bench-cb clean-results clean-hotpot-DANGEROUS hotpot-p1 hotpot-p2 metaqa-check metaqa-ingest-p1 \
        metaqa-ingest-p2 metaqa-answers-p1 metaqa-answers-p2 metaqa-bench-p1 metaqa-bench-p2 metaqa-up docker-up docker-down

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

install:  ## Install deps (uv sync)
	uv sync

sync: install  ## Alias for install

prepare:  ## Download HotpotQA (100 Qs → data_hotpot + hotpot_eval.jsonl)
	$(PYTHON) -m prepare_hotpot

# ── Active-dataset pipeline (DATASET=hotpot by default) ──────────────────────

ingest-p1:  ## Ingest the active DATASET into Qdrant
	$(DS) $(PYTHON) -m phase1_vector_rag.ingest_hot

ingest-p2:  ## Ingest the active DATASET into Neo4j (vector index + graph)
	$(DS) $(PYTHON) -m phase2_graph_rag.ingest

answers-p1:  ## Phase 1: answers for the active DATASET (LIMIT=N for a smoke run)
	$(DS) $(PYTHON) -m phase1_vector_rag.hotpot_answers $(LIMIT_FLAG)

answers-p2:  ## Phase 2: answers for the active DATASET (LIMIT=N for a smoke run)
	$(DS) $(PYTHON) -m phase2_graph_rag.hotpot_answers $(LIMIT_FLAG)

bench-p1:  ## Phase 1: evaluate the active DATASET (EM, F1, Semantic, Judge, RAGAS)
	$(DS) $(PYTHON) -m phase1_vector_rag.benchmark

bench-p2:  ## Phase 2: evaluate the active DATASET
	$(DS) $(PYTHON) -m phase2_graph_rag.benchmark

bench-all: bench-p1 bench-p2  ## Evaluate both phases on the active DATASET

answers-cb:  ## Closed book: MetaQA answers with no retrieval (shared prompt, temp 0)
	$(DS) $(PYTHON) -m shared.closed_book answers

bench-cb:  ## Closed book: evaluate the MetaQA closed-book answers
	$(DS) $(PYTHON) -m shared.closed_book bench

check-config:  ## Show the resolved config for the active DATASET
	$(DS) $(PYTHON) -m shared.config

freeze-verify:  ## Verify the frozen MetaQA artifacts (full v1 + mini) against sha256
	$(SHA256SUM) -c MetaQA/FROZEN.sha256
	$(SHA256SUM) -c MetaQA/FROZEN_mini.sha256

clean-results:  ## Delete results/<DATASET> after a y/N prompt (metaqa | metaqa_mini)
	@case "$(DATASET)" in \
	  metaqa|metaqa_mini) \
	    printf "Delete results/$(DATASET)? [y/N] "; read ans; \
	    case "$$ans" in \
	      y|Y) rm -rf results/$(DATASET); echo "removed results/$(DATASET)" ;; \
	      *) echo "aborted" ;; \
	    esac ;; \
	  *) echo "clean-results needs DATASET=metaqa or DATASET=metaqa_mini."; \
	     echo "HotpotQA results live in the phase dirs — use 'make clean-hotpot-DANGEROUS'."; \
	     exit 1 ;; \
	esac

clean-hotpot-DANGEROUS:  ## DANGEROUS: remove generated HotpotQA artifacts (requires typing 'yes')
	@printf "This deletes data_hotpot/*.txt, hotpot_eval.jsonl and the hotpot_* phase results.\nType 'yes' to continue: "; read ans; \
	if [ "$$ans" != "yes" ]; then echo "aborted"; exit 1; fi
	rm -rf data_hotpot/*.txt hotpot_eval.jsonl
	rm -f phase1_vector_rag/hotpot_*.json phase1_vector_rag/hotpot_*.txt
	rm -f phase2_graph_rag/hotpot_*.json phase2_graph_rag/hotpot_*.txt
	rm -f checkpoint_lightrag.json

# ── Historical direct helpers ────────────────────────────────────────────────

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

metaqa-up:  ## MetaQA: start the isolated Neo4j service (ports 7475/7688)
	docker compose up -d neo4j-metaqa

docker-up:  ## Start Qdrant + Neo4j
	docker compose up -d qdrant neo4j
docker-down:  ## Stop Qdrant + Neo4j
	docker compose down

# ── Deprecated aliases ───────────────────────────────────────────────────────

hotpot-p1:
	@echo "deprecated, use: make answers-p1"
	@$(MAKE) --no-print-directory answers-p1 DATASET=hotpot

hotpot-p2:
	@echo "deprecated, use: make answers-p2"
	@$(MAKE) --no-print-directory answers-p2 DATASET=hotpot

metaqa-check:
	@echo "deprecated, use: make check-config DATASET=metaqa"
	@$(MAKE) --no-print-directory check-config DATASET=metaqa

metaqa-ingest-p1:
	@echo "deprecated, use: make ingest-p1 DATASET=metaqa"
	@$(MAKE) --no-print-directory ingest-p1 DATASET=metaqa

metaqa-ingest-p2:
	@echo "deprecated, use: make ingest-p2 DATASET=metaqa"
	@$(MAKE) --no-print-directory ingest-p2 DATASET=metaqa

metaqa-answers-p1:
	@echo "deprecated, use: make answers-p1 DATASET=metaqa"
	@$(MAKE) --no-print-directory answers-p1 DATASET=metaqa

metaqa-answers-p2:
	@echo "deprecated, use: make answers-p2 DATASET=metaqa"
	@$(MAKE) --no-print-directory answers-p2 DATASET=metaqa

metaqa-bench-p1:
	@echo "deprecated, use: make bench-p1 DATASET=metaqa"
	@$(MAKE) --no-print-directory bench-p1 DATASET=metaqa

metaqa-bench-p2:
	@echo "deprecated, use: make bench-p2 DATASET=metaqa"
	@$(MAKE) --no-print-directory bench-p2 DATASET=metaqa
