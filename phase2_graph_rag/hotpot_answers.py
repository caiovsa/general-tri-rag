"""
Generate answers for the active dataset with Phase 2 (Graph RAG).

Usage:
  python -m phase2_graph_rag.hotpot_answers                 # DATASET=hotpot (default)
  DATASET=metaqa python -m phase2_graph_rag.hotpot_answers  # MetaQA track
  python -m phase2_graph_rag.hotpot_answers --limit 3 --no-resume
"""

from shared.config import dataset_config
from shared.hotpot_runner import run_phase
from phase2_graph_rag.rag import run_graph_rag_pipeline

# Legacy constants (hotpot paths), kept so older callers keep working.
RESULTS_FILE = dataset_config(2).results_file
DETAILED_FILE = dataset_config(2).detailed_file


if __name__ == "__main__":
    run_phase(run_graph_rag_pipeline, phase=2, phase_name="Phase 2 (Graph RAG)")

# python -m phase2_graph_rag.hotpot_answers
