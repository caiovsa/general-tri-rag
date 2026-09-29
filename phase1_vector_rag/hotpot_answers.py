"""
Generate answers for the active dataset with Phase 1 (Vector RAG).

Usage:
  python -m phase1_vector_rag.hotpot_answers                 # DATASET=hotpot (default)
  DATASET=metaqa python -m phase1_vector_rag.hotpot_answers  # MetaQA track
  python -m phase1_vector_rag.hotpot_answers --limit 3 --no-resume
"""

from shared.config import dataset_config
from shared.hotpot_runner import run_phase
from phase1_vector_rag.rag import run_rag_pipeline

# Legacy constants (hotpot paths), kept so older callers keep working.
RESULTS_FILE = dataset_config(1).results_file
DETAILED_FILE = dataset_config(1).detailed_file


if __name__ == "__main__":
    run_phase(run_rag_pipeline, phase=1, phase_name="Phase 1 (Vector RAG)")

# python -m phase1_vector_rag.hotpot_answers
