"""Evaluate Phase 3b (schema-constrained extraction)."""

from shared.benchmark import run_benchmark
from shared.config import phase3b_dataset

DATASET = phase3b_dataset(3)
DATA_FILE = DATASET.detailed_file
OUTPUT_FILE = DATASET.eval_results_file
REPORT_FILE = DATASET.eval_report_file


if __name__ == "__main__":
    run_benchmark(
        phase_name="Phase 3b (schema-constrained extraction)",
        data_file=DATA_FILE,
        output_file=OUTPUT_FILE,
        report_file=REPORT_FILE,
        runner_command="python -m phase3b_ontology_rag.hotpot_answers",
        dataset=DATASET,
        phase="3b",
        per_question_file=DATASET.per_question_file,
    )
