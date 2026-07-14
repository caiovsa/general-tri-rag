"""
Generate answers for all HotpotQA questions using Phase 1 (Vector RAG).

Usage:
  python -m phase1_vector_rag.hotpot_answers
"""

from pathlib import Path

from shared.hotpot_runner import run_hotpot_answers
from phase1_vector_rag.rag import run_rag_pipeline

RESULTS_FILE = Path("phase1_vector_rag/hotpot_results.json")
DETAILED_FILE = Path("phase1_vector_rag/hotpot_results_detailed.json")


if __name__ == "__main__":
    run_hotpot_answers(
        pipeline_func=run_rag_pipeline,
        phase_name="Phase 1 (Vector RAG)",
        results_file=RESULTS_FILE,
        detailed_file=DETAILED_FILE,
    )

# python -m phase1_vector_rag.hotpot_answers
