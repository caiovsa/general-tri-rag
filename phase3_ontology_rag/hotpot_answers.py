"""
Generate answers for the active dataset with Phase 3 (Ontology RAG).

Usage:
  DATASET=metaqa_mini30 python -m phase3_ontology_rag.hotpot_answers
  DATASET=metaqa python -m phase3_ontology_rag.hotpot_answers
"""

from shared.config import dataset_config
from shared.hotpot_runner import run_phase
from phase3_ontology_rag.rag import run_ontology_rag_pipeline

RESULTS_FILE = dataset_config(3).results_file
DETAILED_FILE = dataset_config(3).detailed_file


if __name__ == "__main__":
    run_phase(run_ontology_rag_pipeline, phase=3, phase_name="Phase 3 (Ontology RAG)")

# python -m phase3_ontology_rag.hotpot_answers
