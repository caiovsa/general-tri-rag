"""
Shared answer-generation runner for a dataset's eval file.
Used by phase1_vector_rag.hotpot_answers and phase2_graph_rag.hotpot_answers.

The eval file, results paths and retrieval depth come from the active DATASET
(shared/config.py). HotpotQA keeps its historical paths and file shape; MetaQA
adds id/hop/qtype/answers to the detailed results so runs can be compared paired.

Usage from a phase module:
    run_phase(run_rag_pipeline, phase=1, phase_name="Phase 1 (Vector RAG)")
CLI: --limit N (new questions to answer), --no-resume, --top-k K
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from shared.config import dataset_config, set_top_k

EVAL_FILE = Path("hotpot_eval.jsonl")  # legacy default, kept for backwards compatibility


def load_questions(eval_file: Path | None = None):
    """Read the eval file (JSONL). Defaults to the legacy hotpot_eval.jsonl."""
    eval_file = Path(eval_file or EVAL_FILE)
    questions = []
    if not eval_file.exists():
        print(f"File not found: {eval_file}")
        print("Run 'python -m prepare_hotpot' first (legacy: python -m preapre_hotpot).")
        return questions
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def question_key(row: dict) -> str:
    """Stable identity for resume: dataset id when present, else the question text."""
    return str(row.get("id") or row.get("question"))


def _read_json_list(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Warning: could not parse {path} — starting fresh.")
        return []
    return data if isinstance(data, list) else []


def save_results(results, detailed_results, results_file, detailed_file):
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    simple = [{"question": r["question"], "answer": r["answer"]} for r in results]
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(simple, f, indent=2, ensure_ascii=False)
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {results_file}")
    print(f"Saved {len(detailed_results)} detailed results to {detailed_file}")


def detailed_row(question: dict, answer: str, contexts: list, extra_keys=("id", "hop", "qtype", "answers")) -> dict:
    """Detailed result row. HotpotQA rows keep their exact shape; MetaQA adds the
    fields that let the benchmark group by hop and compare runs paired."""
    row = {
        "question": question["question"],
        "ground_truth": question.get("answer", ""),
        "answer": answer,
        "supporting_facts": question.get("supporting_facts_titles", []),
        "contexts": contexts,
    }
    for key in extra_keys:
        if key in question:
            row[key] = question[key]
    return row


def run_hotpot_answers(pipeline_func, phase_name, results_file, detailed_file, limit: int = None,
                       resume: bool = True, eval_file: Path | None = None, run_meta: dict | None = None):
    """
    Run answer generation for a phase.

    Args:
        pipeline_func: Function that takes a question string and returns
                       {"answer": str, "contexts": list[str]}
        phase_name: Display name (e.g., "Phase 1 (Vector RAG)")
        results_file: Path to save simple results
        detailed_file: Path to save detailed results
        limit: answer at most N *new* questions (already-answered ones are skipped)
        resume: keep and skip questions already present in detailed_file
        eval_file: eval JSONL to read (defaults to hotpot_eval.jsonl)
        run_meta: metadata dict written next to the results (dataset, phase, top_k)
    """
    questions = load_questions(eval_file)
    if not questions:
        print(f"No questions loaded from {eval_file or EVAL_FILE}")
        return

    previous = _read_json_list(detailed_file) if resume else []
    answered = {question_key(row) for row in previous}
    pending = [q for q in questions if question_key(q) not in answered]
    if limit:
        pending = pending[:limit]

    print(f"Loaded {len(questions)} questions from {eval_file or EVAL_FILE}")
    if previous:
        print(f"Resume: {len(previous)} already answered, {len(pending)} pending")
    if not pending:
        print("Nothing to do — every question is already answered.")
        return
    print(f"Running {phase_name} benchmark...\n")

    results = []
    detailed_results = []

    for i, q in enumerate(tqdm(pending, desc=f"{phase_name}")):
        question = q["question"]
        try:
            pipeline_result = pipeline_func(question)
            answer = pipeline_result.get("answer", "")
            contexts = pipeline_result.get("contexts", [])
        except Exception as e:
            answer, contexts = f"Error: {str(e)}", []
            print(f"\n  [Question {i+1}] {answer}")

        results.append({"question": question, "answer": answer})
        detailed_results.append(detailed_row(q, answer, contexts))

    # Merge with previous answers and write in eval order, so a resumed run never
    # drops rows answered by an earlier one.
    merged = {question_key(row): row for row in previous}
    merged.update({question_key(row): row for row in detailed_results})
    ordered = [merged[question_key(q)] for q in questions if question_key(q) in merged]

    save_results(ordered, ordered, results_file, detailed_file)

    if run_meta is not None:
        meta_path = Path(detailed_file).parent / "run_meta.json"
        meta = {**run_meta, "questions": len(questions), "answered": len(ordered),
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Run metadata saved to {meta_path}")
    print(f"\nDone! Processed {len(pending)}/{len(questions)} questions.")


def parse_args(phase: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Generate answers (dataset={dataset_config(phase).name})")
    parser.add_argument("--limit", type=int, default=None, help="answer at most N new questions")
    parser.add_argument("--no-resume", action="store_true", help="ignore previous answers and start over")
    parser.add_argument("--top-k", type=int, default=None, help="retrieval depth (overrides TOP_K env)")
    return parser.parse_args()


def run_phase(pipeline_func, phase: int, phase_name: str):
    """Resolve the active dataset, parse CLI flags and run the phase."""
    args = parse_args(phase)
    if args.top_k:
        set_top_k(args.top_k)
    dataset = dataset_config(phase)

    print(f"Dataset: {dataset.name} | phase: {phase}")
    print(f"  eval file  : {dataset.eval_file}")
    print(f"  results dir: {dataset.results_file.parent}")
    print(f"  top_k      : {dataset.top_k}")
    if dataset.doc_list:
        print(f"  corpus     : {dataset.doc_list}")
    print()

    run_hotpot_answers(
        pipeline_func=pipeline_func,
        phase_name=phase_name,
        results_file=dataset.results_file,
        detailed_file=dataset.detailed_file,
        limit=args.limit,
        resume=not args.no_resume,
        eval_file=dataset.eval_file,
        run_meta={"dataset": dataset.name, "phase": phase, "top_k": dataset.top_k,
                  "eval_file": str(dataset.eval_file), "limit": args.limit},
    )
