"""
Shared answer-generation runner for a dataset's eval file.
Used by phase1_vector_rag.hotpot_answers and phase2_graph_rag.hotpot_answers.

The eval file, results paths and retrieval depth come from the active DATASET
(shared/config.py). HotpotQA keeps its historical paths and file shape; MetaQA
adds id/hop/qtype/answers to the detailed results so runs can be compared paired.

A run carries a fingerprint (dataset, eval/corpus hashes, chunking, top_k,
models, temperature, prompt hash, ablation switches). Resuming a run whose
fingerprint changed is refused: use --no-resume to start over explicitly.

Usage from a phase module:
    run_phase(run_rag_pipeline, phase=1, phase_name="Phase 1 (Vector RAG)")
CLI: --limit N (new questions to answer), --no-resume, --top-k K,
     --max-errors N (0 = strict), --no-neighbours (Phase 2 ablation)
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from shared.config import (
    dataset_config,
    closed_book_dataset,
    phase3_database,
    settings,
    set_top_k,
    set_include_neighbours,
)
from shared.prompts import metaqa_prompt_template_hash

EVAL_FILE = Path("hotpot_eval.jsonl")  # legacy default, kept for backwards compatibility


def load_questions(eval_file: Path | None = None):
    """Read the eval file (JSONL). Defaults to the legacy hotpot_eval.jsonl."""
    eval_file = Path(eval_file or EVAL_FILE)
    questions = []
    if not eval_file.exists():
        print(f"File not found: {eval_file}")
        print("Run 'python -m prepare_hotpot' first.")
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


def sha256_file(path: Path | str) -> str | None:
    """sha256 of a file, or None when it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_signature(dataset) -> dict:
    """Cheap, stable identity of the corpus a run reads (manifest or directory)."""
    if dataset.doc_list:
        paths = Path(dataset.doc_list)
        return {
            "kind": "manifest",
            "path": str(paths),
            "sha256": sha256_file(paths),
            "files": len([line for line in paths.read_text(encoding="utf-8").splitlines() if line.strip()])
            if paths.exists() else None,
        }
    if dataset.docs_dir:
        directory = Path(dataset.docs_dir)
        files = sorted(directory.glob(dataset.file_pattern)) if directory.exists() else []
        signature = "\n".join(f"{path.name}:{path.stat().st_size}" for path in files)
        return {
            "kind": "directory",
            "path": str(directory),
            "pattern": dataset.file_pattern,
            "sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
            "files": len(files),
        }
    return {"kind": "none", "path": None, "sha256": None, "files": 0}


def run_fingerprint(dataset, phase: int, closed_book: bool = False) -> dict:
    """Everything that must stay identical for a resume to be valid."""
    shared_prompt = dataset.prompt_style == "shared"
    return {
        "dataset": dataset.name,
        "phase": phase,
        "closed_book": bool(closed_book),
        "eval_file": str(dataset.eval_file),
        "eval_sha256": sha256_file(dataset.eval_file),
        "corpus": corpus_signature(dataset),
        "chunk_size": dataset.chunk_size,
        "chunk_overlap": dataset.chunk_overlap,
        "top_k": dataset.top_k,
        "tag_movies": dataset.tag_movies,
        "prompt_style": dataset.prompt_style,
        "prompt_sha256": metaqa_prompt_template_hash() if shared_prompt else None,
        "generation_model": settings.GENERATION_MODEL,
        "generation_provider": "openrouter" if settings.GENERATION_MODEL.startswith("openrouter/") else "default",
        "openrouter_provider_only": (settings.OPENROUTER_PROVIDER_ONLY
                                     if settings.GENERATION_MODEL.startswith("openrouter/") else None),
        "openrouter_allow_fallbacks": (settings.OPENROUTER_ALLOW_FALLBACKS
                                       if settings.GENERATION_MODEL.startswith("openrouter/") else None),
        "generation_reasoning_effort": settings.GENERATION_REASONING_EFFORT,
        "judge_model": settings.JUDGE_MODEL,
        "extraction_model": settings.EXTRACTION_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "temperature": settings.GENERATION_TEMPERATURE if settings.GENERATION_TEMPERATURE is not None else "default",
        "extraction_temperature": settings.EXTRACTION_TEMPERATURE if settings.EXTRACTION_TEMPERATURE is not None else "default",
        "include_neighbours": settings.INCLUDE_NEIGHBOURS if phase == 2 else None,
        "graph_database": phase3_database(dataset) if phase == 3 else dataset.neo4j_database,
        **({"ontology_sha256": _ontology_hash()} if phase == 3 else {}),
    }


def _ontology_hash() -> str:
    """Phase 3 only: hash of the typed schema (imported lazily to avoid cycles)."""
    from phase3_ontology_rag.schema import ontology_hash
    return ontology_hash()


def _read_json_list(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Warning: could not parse {path} — starting fresh.")
        return []
    return data if isinstance(data, list) else []


def read_ingest_meta(dataset) -> dict | None:
    """ingest_meta.json written next to the results by a phase's ingest."""
    path = Path(dataset.results_file).parent / "ingest_meta.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def read_run_meta(detailed_file: Path) -> dict | None:
    """run_meta.json sitting next to the detailed results, when present."""
    path = Path(detailed_file).parent / "run_meta.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _check_resume(detailed_file: Path, fingerprint: dict | None, previous: list[dict]) -> None:
    """Refuse to mix rows produced by a different configuration."""
    if not previous or fingerprint is None:
        return
    meta = read_run_meta(detailed_file)
    if meta is None:
        raise SystemExit(
            f"Refusing to resume: {Path(detailed_file).parent / 'run_meta.json'} is missing "
            "(legacy run, no fingerprint). Re-run with --no-resume to start over."
        )
    old = meta.get("fingerprint")
    if old != fingerprint:
        old = old or {}
        changed = [key for key in sorted(set(old) | set(fingerprint))
                   if old.get(key) != fingerprint.get(key)]
        raise SystemExit(
            "Refusing to resume: the run configuration changed "
            f"({', '.join(changed) or 'unknown fields'}). Results in "
            f"{Path(detailed_file).parent} belong to a different run. "
            "Use --no-resume to replace them, or point the results dir elsewhere."
        )


def save_results(results, detailed_results, results_file, detailed_file):
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    simple = [{"question": r["question"], "answer": r["answer"]} for r in results]
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(simple, f, indent=2, ensure_ascii=False)
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {results_file}")
    print(f"Saved {len(detailed_results)} detailed results to {detailed_file}")


def detailed_row(question: dict, answer: str, contexts: list, extra_keys=("id", "hop", "qtype", "answers"),
                 retrieval: dict | None = None, error: str | None = None) -> dict:
    """Detailed result row. HotpotQA rows keep their exact shape; MetaQA adds the
    fields that let the benchmark group by hop and compare runs paired, plus the
    per-question retrieval budget/provenance when the phase provides it."""
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
    if retrieval is not None:
        row["retrieval"] = retrieval
    if error is not None:
        row["error"] = error
    return row


def retrieval_summary(rows: list[dict]) -> dict | None:
    """Aggregate per-question retrieval counts (chunks_in_prompt + provenance)."""
    retrieval_rows = [row.get("retrieval") for row in rows if isinstance(row.get("retrieval"), dict)]
    if not retrieval_rows:
        return None
    chunks = [r.get("chunks_in_prompt") or 0 for r in retrieval_rows]
    return {
        "questions": len(retrieval_rows),
        "chunks_in_prompt_mean": round(sum(chunks) / len(chunks), 4),
        "chunks_in_prompt_max": max(chunks),
        "seeds_total": sum((r.get("seeds") or 0) for r in retrieval_rows),
        "anchors_total": sum((r.get("anchors") or 0) for r in retrieval_rows),
        "neighbours_total": sum((r.get("neighbours") or 0) for r in retrieval_rows),
        "entities_total": sum(len(r.get("entities") or ()) for r in retrieval_rows),
    }


def run_hotpot_answers(pipeline_func, phase_name, results_file, detailed_file, limit: int = None,
                       resume: bool = True, eval_file: Path | None = None, run_meta: dict | None = None,
                       fingerprint: dict | None = None, allow_empty_contexts: bool = False,
                       max_errors: int = 0):
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
        fingerprint: identity of this run; a resume with a different one is refused
        allow_empty_contexts: closed-book runs have no contexts by design
        max_errors: allowed failed questions before the run exits non-zero (0 = strict)
    """
    questions = load_questions(eval_file)
    if not questions:
        print(f"No questions loaded from {eval_file or EVAL_FILE}")
        return {"answered": 0, "errors": 0, "empty_contexts": 0, "ok": False}

    previous = _read_json_list(detailed_file) if resume else []
    _check_resume(Path(detailed_file), fingerprint, previous)
    answered = {question_key(row) for row in previous}
    pending = [q for q in questions if question_key(q) not in answered]
    if limit:
        pending = pending[:limit]

    print(f"Loaded {len(questions)} questions from {eval_file or EVAL_FILE}")
    if previous:
        print(f"Resume: {len(previous)} already answered, {len(pending)} pending")
    if not pending:
        print("Nothing to do — every question is already answered.")
        return {"answered": len(previous), "errors": 0, "empty_contexts": 0, "ok": True}
    print(f"Running {phase_name} benchmark...\n")

    results = []
    detailed_results = []
    failures: list[dict] = []
    empty_contexts = 0

    for i, q in enumerate(tqdm(pending, desc=f"{phase_name}")):
        question = q["question"]
        error = None
        try:
            pipeline_result = pipeline_func(question)
            answer = pipeline_result.get("answer", "")
            contexts = pipeline_result.get("contexts", [])
            retrieval = pipeline_result.get("retrieval")
            if not answer or not str(answer).strip():
                raise RuntimeError("pipeline returned an empty answer")
            if not contexts and not allow_empty_contexts:
                empty_contexts += 1
        except Exception as e:  # noqa: BLE001 - every failure is recorded and the run fails
            error = f"{type(e).__name__}: {e}"
            answer, contexts, retrieval = f"Error: {error}", [], None
            failures.append({"question": question, "error": error})
            print(f"\n  [Question {i + 1}] FAILED: {error}")

        results.append({"question": question, "answer": answer})
        detailed_results.append(detailed_row(q, answer, contexts, retrieval=retrieval, error=error))

    # Merge with previous answers and write in eval order, so a resumed run never
    # drops rows answered by an earlier one.
    merged = {question_key(row): row for row in previous}
    merged.update({question_key(row): row for row in detailed_results})
    ordered = [merged[question_key(q)] for q in questions if question_key(q) in merged]

    save_results(ordered, ordered, results_file, detailed_file)

    if run_meta is not None or fingerprint is not None:
        meta_path = Path(detailed_file).parent / "run_meta.json"
        meta = {**(run_meta or {}), "fingerprint": fingerprint,
                "questions": len(questions), "answered": len(ordered),
                "errors": len(failures), "empty_contexts": empty_contexts,
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        retrieval = retrieval_summary(ordered)
        if retrieval is not None:
            meta["retrieval"] = retrieval
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Run metadata saved to {meta_path}")
    print(f"\nRun summary: {len(ordered)}/{len(questions)} answered | "
          f"{len(failures)} error(s) | {empty_contexts} empty context(s)")
    if failures:
        for failure in failures[:5]:
            print(f"  - {failure['error']}")
    print(f"\nDone! Processed {len(pending)}/{len(questions)} questions.")

    if failures:
        print(f"FAILED: {len(failures)} question(s) raised an exception.")
    if empty_contexts > max_errors:
        print(f"FAILED: {empty_contexts} question(s) returned no context.")
    ok = len(failures) == 0 and empty_contexts <= max_errors
    return {"answered": len(ordered), "errors": len(failures),
            "empty_contexts": empty_contexts, "ok": ok}


def parse_args(phase: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Generate answers (dataset={dataset_config(phase).name})")
    parser.add_argument("--limit", type=int, default=None, help="answer at most N new questions")
    parser.add_argument("--no-resume", action="store_true", help="ignore previous answers and start over")
    parser.add_argument("--top-k", type=int, default=None, help="retrieval depth (overrides TOP_K env)")
    parser.add_argument("--max-errors", type=int, default=0,
                        help="allowed failed questions before exiting non-zero (default: 0 = strict)")
    parser.add_argument("--no-neighbours", action="store_true",
                        help="Phase 2 ablation: drop 1-hop neighbour chunks from MetaQA context")
    return parser.parse_args()


def run_phase(pipeline_func, phase: int, phase_name: str, closed_book: bool = False,
              dataset_override=None):
    """Resolve the active dataset, parse CLI flags and run the phase.

    ``closed_book=True`` routes the run to results/<dataset>/closed_book/ (MetaQA
    tracks only) and uses the same runner/format as a normal phase.
    """
    args = parse_args(phase)
    if args.top_k:
        set_top_k(args.top_k)
    if args.no_neighbours:
        set_include_neighbours(False)
    dataset = dataset_override or (closed_book_dataset(phase) if closed_book else dataset_config(phase))

    print(f"Dataset: {dataset.name} | phase: {phase}")
    print(f"  eval file  : {dataset.eval_file}")
    print(f"  results dir: {dataset.results_file.parent}")
    print(f"  top_k      : {dataset.top_k}")
    temp = settings.GENERATION_TEMPERATURE if settings.GENERATION_TEMPERATURE is not None else "default (1)"
    print(f"  models     : gen={settings.GENERATION_MODEL} judge={settings.JUDGE_MODEL} "
          f"extract={settings.EXTRACTION_MODEL} | temperature={temp} "
          f"| reasoning={settings.GENERATION_REASONING_EFFORT or 'default'}")
    if dataset.doc_list:
        print(f"  corpus     : {dataset.doc_list}")
    print()

    fingerprint = run_fingerprint(dataset, phase, closed_book)
    run_meta = {"dataset": dataset.name, "phase": phase, "top_k": dataset.top_k,
                "eval_file": str(dataset.eval_file), "limit": args.limit,
                "closed_book": closed_book}
    if dataset.results_file.parent.name == "phase3b":
        run_meta["variant"] = "3b"
    ingest_meta = read_ingest_meta(dataset)
    if ingest_meta is not None:
        run_meta["ingest"] = ingest_meta
    if dataset.prompt_style == "shared":
        run_meta.update({
            "generation_model": settings.GENERATION_MODEL,
            "generation_provider": "openrouter" if settings.GENERATION_MODEL.startswith("openrouter/") else "default",
            "openrouter_provider_only": (settings.OPENROUTER_PROVIDER_ONLY
                                         if settings.GENERATION_MODEL.startswith("openrouter/") else None),
            "openrouter_allow_fallbacks": (settings.OPENROUTER_ALLOW_FALLBACKS
                                           if settings.GENERATION_MODEL.startswith("openrouter/") else None),
            "generation_reasoning_effort": settings.GENERATION_REASONING_EFFORT,
            "judge_model": settings.JUDGE_MODEL,
            "extraction_model": settings.EXTRACTION_MODEL,
            "temperature": settings.GENERATION_TEMPERATURE if settings.GENERATION_TEMPERATURE is not None else "default",
            "prompt_template_hash": metaqa_prompt_template_hash(),
        })

    summary = run_hotpot_answers(
        pipeline_func=pipeline_func,
        phase_name=phase_name,
        results_file=dataset.results_file,
        detailed_file=dataset.detailed_file,
        limit=args.limit,
        resume=not args.no_resume,
        eval_file=dataset.eval_file,
        run_meta=run_meta,
        fingerprint=fingerprint,
        allow_empty_contexts=closed_book,
        max_errors=args.max_errors,
    )
    if summary and not summary.get("ok", True):
        raise SystemExit(1)
