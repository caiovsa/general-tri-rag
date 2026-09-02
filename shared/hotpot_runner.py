"""
Shared HotpotQA answer generation runner.
Used by phase1_vector_rag.hotpot_answers and phase2_graph_rag.hotpot_answers.
"""

import json
from pathlib import Path

from tqdm import tqdm

EVAL_FILE = Path("hotpot_eval.jsonl")


def load_questions():
    questions = []
    if not EVAL_FILE.exists():
        print(f"File not found: {EVAL_FILE}")
        print("Run 'python -m prepare_hotpot' first. (legacy: python -m preapre_hotpot)")
        return questions
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def save_results(results, detailed_results, results_file, detailed_file):
    simple = [{"question": r["question"], "answer": r["answer"]} for r in results]
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(simple, f, indent=2, ensure_ascii=False)
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {results_file}")
    print(f"Saved {len(detailed_results)} detailed results to {detailed_file}")


def run_hotpot_answers(pipeline_func, phase_name, results_file, detailed_file):
    """
    Run HotpotQA answer generation for a phase.

    Args:
        pipeline_func: Function that takes a question string and returns
                       {"answer": str, "contexts": list[str]}
        phase_name: Display name (e.g., "Phase 1 (Vector RAG)")
        results_file: Path to save simple results
        detailed_file: Path to save detailed results
    """
    questions = load_questions()
    if not questions:
        print(f"No questions loaded from {EVAL_FILE}")
        return

    print(f"Loaded {len(questions)} questions from {EVAL_FILE}")
    print(f"Running {phase_name} benchmark...\n")

    results = []
    detailed_results = []

    for i, q in enumerate(tqdm(questions, desc=f"{phase_name}")):
        question = q["question"]
        ground_truth = q["answer"]
        supporting_facts = q.get("supporting_facts_titles", [])

        try:
            pipeline_result = pipeline_func(question)
            answer = pipeline_result.get("answer", "")
            contexts = pipeline_result.get("contexts", [])

            results.append({
                "question": question,
                "answer": answer,
            })

            detailed_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "supporting_facts": supporting_facts,
                "contexts": contexts,
            })

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n  [Question {i+1}] {error_msg}")

            results.append({
                "question": question,
                "answer": error_msg,
            })

            detailed_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": error_msg,
                "supporting_facts": supporting_facts,
                "contexts": [],
                "error": str(e),
            })

    save_results(results, detailed_results, results_file, detailed_file)
    print(f"\nDone! Processed {len(results)}/{len(questions)} questions.")
