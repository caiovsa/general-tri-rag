"""
Evaluation Script for Phase 1 (Vector RAG)
==========================================
Metrics computed:
  - Exact Match (EM)         — official HotpotQA metric
  - Relaxed EM               — GT is a substring of the answer
  - Token F1                 — official HotpotQA metric
  - Extracted EM             — EM after stripping conversational filler
  - Gold Recall / All Found  — set answers: every gold item present (MetaQA)
  - Evidence Recall          — [Movie: title] tags found in retrieved contexts (MetaQA)
  - Semantic Similarity      — cosine similarity via OpenAI embeddings
  - LLM-as-Judge             — LLM grades if answer is correct (0-1)
  - RAGAS Answer Correctness — semantic + factual overlap
  - RAGAS Faithfulness       — is the answer grounded in retrieved contexts?
  - RAGAS Context Recall     — did the contexts cover the ground truth?

Grouped by hop and evidence size when the dataset provides them; per-question
metrics are written to a separate file for paired comparisons.

Usage:
  python -m phase1_vector_rag.benchmark                 # DATASET=hotpot (default)
  DATASET=metaqa python -m phase1_vector_rag.benchmark  # MetaQA track
"""

from shared.benchmark import run_benchmark
from shared.config import dataset_config

DATASET = dataset_config(phase=1)
DATA_FILE = DATASET.detailed_file
OUTPUT_FILE = DATASET.eval_results_file
REPORT_FILE = DATASET.eval_report_file


if __name__ == "__main__":
    run_benchmark(
        phase_name="Phase 1 (Vector RAG)",
        data_file=DATA_FILE,
        output_file=OUTPUT_FILE,
        report_file=REPORT_FILE,
        runner_command="python -m phase1_vector_rag.hotpot_answers",
        dataset=DATASET,
        phase=1,
        per_question_file=DATASET.per_question_file,
    )

# python -m phase1_vector_rag.benchmark
