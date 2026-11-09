"""
Phase 3 (Ontology RAG) evaluation.

Reuses the shared benchmark engine, so the metrics, grouping and per-question
file are identical to Phase 1/2. The only Phase 3-specific metric is recorded
retrieval provenance (topic entity, candidate movies, fallback) captured by the
retriever in `results/<dataset>/phase3/results_detailed.json`.

Usage:
  DATASET=metaqa_mini30 python -m phase3_ontology_rag.benchmark
  DATASET=metaqa python -m phase3_ontology_rag.benchmark
"""

from shared.benchmark import run_benchmark
from shared.config import dataset_config

DATASET = dataset_config(phase=3)
DATA_FILE = DATASET.detailed_file
OUTPUT_FILE = DATASET.eval_results_file
REPORT_FILE = DATASET.eval_report_file


if __name__ == "__main__":
    run_benchmark(
        phase_name="Phase 3 (Ontology RAG)",
        data_file=DATA_FILE,
        output_file=OUTPUT_FILE,
        report_file=REPORT_FILE,
        runner_command="python -m phase3_ontology_rag.hotpot_answers",
        dataset=DATASET,
        phase=3,
        per_question_file=DATASET.per_question_file,
    )

# python -m phase3_ontology_rag.benchmark
