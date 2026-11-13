"""Generate answers for Phase 3b (schema-constrained extraction)."""

from shared.config import phase3b_dataset
from shared.hotpot_runner import run_phase
from phase3b_ontology_rag.rag import run_schema_rag_pipeline

RESULTS_FILE = phase3b_dataset(3).results_file
DETAILED_FILE = phase3b_dataset(3).detailed_file


if __name__ == "__main__":
    run_phase(run_schema_rag_pipeline, phase=3,
              phase_name="Phase 3b (schema-constrained extraction)",
              dataset_override=phase3b_dataset(3))
