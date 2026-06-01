import json
import time
from pathlib import Path
from tqdm import tqdm

from phase2_graph_rag.rag import run_graph_rag_pipeline

EVAL_FILE = Path("hotpot_eval.jsonl")
RESULTS_FILE = Path("phase2_graph_rag/hotpot_results.json")
DETAILED_FILE = Path("phase2_graph_rag/hotpot_results_detailed.json")


def load_questions():
    questions = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions


def save_results(results, detailed_results):
    simple = [{"question": r["question"], "answer": r["answer"]} for r in results]
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(simple, f, indent=2, ensure_ascii=False)
    with open(DETAILED_FILE, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {RESULTS_FILE}")
    print(f"Saved {len(detailed_results)} detailed results to {DETAILED_FILE}")


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions from {EVAL_FILE}")
    print(f"Running Phase 2 (Graph RAG) benchmark...\n")

    results = []
    detailed_results = []

    for i, q in enumerate(tqdm(questions, desc="Phase 2 RAG")):
        question = q["question"]
        ground_truth = q["answer"]
        supporting_facts = q.get("supporting_facts_titles", [])

        try:
            pipeline_result = run_graph_rag_pipeline(question)
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

        save_results(results, detailed_results)

    print(f"\nDone! Processed {len(results)}/{len(questions)} questions.")


if __name__ == "__main__":
    main()

# python -m phase2_graph_rag.hotpot_answers
