"""
Phase 3b — schema-constrained ontology RAG pipeline.

Retrieval reuses the typed Phase 3a retriever (entity match -> typed BFS ->
chunk ranking) against the 3b graph, which was built by gpt-6-luna from the
documents under the ontology schema. Same prompt/model/top_k as every phase.
"""

from phase3_ontology_rag.retriever import retrieve_ontology_chunks
from shared.config import phase3b_dataset, settings
from shared.llm import generate_completion
from shared.prompts import build_metaqa_prompt, format_context


def run_schema_rag_pipeline(user_query: str):
    dataset = phase3b_dataset(3)
    print(f"Executing Phase 3b (schema-constrained) RAG for: '{user_query}'\n")

    print("1. Retrieving from the extracted typed graph...")
    entries, counts = retrieve_ontology_chunks(user_query, dataset=dataset)
    contexts = [entry["text"] for entry in entries]
    print(f"-> Retrieved {len(contexts)} chunks "
          f"({counts.get('entity')!r}, candidates={counts.get('candidates')}).\n")

    prompt = build_metaqa_prompt(user_query, format_context(contexts))
    print(f"2. Generating answer using model: {settings.GENERATION_MODEL}...\n")
    answer = generate_completion(prompt)

    print("=" * 50)
    print("Final Phase 3b Answer:")
    print("=" * 50)
    print(answer)
    return {"answer": answer, "contexts": contexts, "retrieval": counts}


if __name__ == "__main__":
    run_schema_rag_pipeline("who directed the movies starred by Cher")
