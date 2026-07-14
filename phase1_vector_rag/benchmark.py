"""
HotpotQA Evaluation Script for Phase 1 (Vector RAG)
====================================================
Metrics computed:
  - Exact Match (EM)         — official HotpotQA metric
  - Relaxed EM               — GT is a substring of the answer
  - Token F1                 — official HotpotQA metric
  - Extracted EM             — EM after stripping conversational filler
  - Semantic Similarity      — cosine similarity via OpenAI embeddings
  - LLM-as-Judge             — LLM grades if answer is correct (0-1)
  - RAGAS Answer Correctness — semantic + factual overlap
  - RAGAS Faithfulness       — is the answer grounded in retrieved contexts?
  - RAGAS Context Recall     — did the contexts cover the ground truth?

Usage:
  python -m phase1_vector_rag.benchmark
"""

from pathlib import Path

from shared.benchmark import run_benchmark

DATA_FILE = Path("phase1_vector_rag/hotpot_results_detailed.json")
OUTPUT_FILE = Path("phase1_vector_rag/hotpot_eval_results.json")
REPORT_FILE = Path("phase1_vector_rag/hotpot_eval_report.txt")


if __name__ == "__main__":
    run_benchmark(
        phase_name="Phase 1 (Vector RAG)",
        data_file=DATA_FILE,
        output_file=OUTPUT_FILE,
        report_file=REPORT_FILE,
        runner_command="python -m phase1_vector_rag.hotpot_answers",
    )

# python -m phase1_vector_rag.benchmark
