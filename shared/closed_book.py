"""Closed-book control for the MetaQA tracks (DATASET=metaqa|metaqa_mini).

Same shared prompt and model as the retrieval phases, with the context replaced by
"(no context)" and no retrieval. Results land in results/<dataset>/closed_book/ in
the same format as a phase run.

Usage:
  DATASET=metaqa uv run python -m shared.closed_book answers
  DATASET=metaqa uv run python -m shared.closed_book bench
"""

import sys

from shared.benchmark import run_benchmark
from shared.config import closed_book_dataset
from shared.hotpot_runner import run_phase
from shared.llm import generate_completion
from shared.prompts import CLOSED_BOOK_CONTEXT, build_metaqa_prompt


def run_closed_book_pipeline(user_query: str) -> dict:
    """Shared MetaQA prompt, no retrieval, same model at temperature 0."""
    prompt = build_metaqa_prompt(user_query, CLOSED_BOOK_CONTEXT)
    answer = generate_completion(prompt, temperature=0)
    return {
        "answer": answer,
        "contexts": [],
        "retrieval": {"chunks_in_prompt": 0, "seeds": 0, "anchors": 0, "neighbours": 0},
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "answers"
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # let the shared runner parse its own flags

    if mode == "answers":
        run_phase(run_closed_book_pipeline, phase=1, phase_name="Closed book (no retrieval)",
                  closed_book=True)
        return

    if mode == "bench":
        dataset = closed_book_dataset(1)
        run_benchmark(
            phase_name="Closed book (no retrieval)",
            data_file=dataset.detailed_file,
            output_file=dataset.eval_results_file,
            report_file=dataset.eval_report_file,
            runner_command="make answers-cb",
            dataset=dataset,
            phase=0,
            per_question_file=dataset.per_question_file,
        )
        return

    raise SystemExit(f"Unknown mode {mode!r} (expected 'answers' or 'bench').")


if __name__ == "__main__":
    main()
