"""
Phase 3 — Ontology RAG pipeline (typed KB graph, Phase 3a).

Retrieval: `phase3_ontology_rag.retriever` (entity match -> typed BFS -> chunk
ranking). Generation: the same frozen shared prompt and model as the MetaQA
tracks, so Phase 1 / Phase 2 / Phase 3 are comparable at the same top_k.

Run: DATASET=metaqa_mini30 python -m phase3_ontology_rag.rag
"""

from phase3_ontology_rag.retriever import retrieve_ontology_chunks
from shared.config import dataset_config, settings
from shared.llm import generate_completion
from shared.prompts import build_metaqa_prompt, format_context


def run_ontology_rag_pipeline(user_query: str):
    dataset = dataset_config(phase=3)
    print(f"Executing Ontology RAG for: '{user_query}'\n")

    print("1. Matching the topic entity and walking the typed KB graph...")
    entries, counts = retrieve_ontology_chunks(user_query)
    contexts = [entry["text"] for entry in entries]
    print(f"-> Retrieved {len(contexts)} chunks "
          f"({counts.get('entity')!r}, candidates={counts.get('candidates')}).\n")

    prompt = build_metaqa_prompt(user_query, format_context(contexts))
    print(f"2. Generating answer using model: {settings.GENERATION_MODEL}...\n")
    answer = generate_completion(prompt)

    print("=" * 50)
    print("Final Ontology RAG Answer:")
    print("=" * 50)
    print(answer)
    return {"answer": answer, "contexts": contexts, "retrieval": counts}


if __name__ == "__main__":
    run_ontology_rag_pipeline("what movies did Cher act in")

# python -m phase3_ontology_rag.rag
