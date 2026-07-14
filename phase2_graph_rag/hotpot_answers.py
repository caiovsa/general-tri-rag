"""
Generate answers for all HotpotQA questions using Phase 2 (Graph RAG).

Usage:
  python -m phase2_graph_rag.hotpot_answers
"""

from pathlib import Path

from shared.hotpot_runner import run_hotpot_answers
from phase2_graph_rag.rag import run_graph_rag_pipeline

RESULTS_FILE = Path("phase2_graph_rag/hotpot_results.json")
DETAILED_FILE = Path("phase2_graph_rag/hotpot_results_detailed.json")


if __name__ == "__main__":
    run_hotpot_answers(
        pipeline_func=run_graph_rag_pipeline,
        phase_name="Phase 2 (Graph RAG)",
        results_file=RESULTS_FILE,
        detailed_file=DETAILED_FILE,
    )

# python -m phase2_graph_rag.hotpot_answers
