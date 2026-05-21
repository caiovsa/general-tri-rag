from phase1_vector_rag.retriever import retrieve_chunks
from shared.llm import generate_completion
from shared.config import settings

def extract_chunk_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "node"):
        node = chunk.node
        if hasattr(node, "get_content"):
            return node.get_content()
        if hasattr(node, "text"):
            return node.text if isinstance(node.text, str) else str(node.text)
    if hasattr(chunk, "get_content"):
        return chunk.get_content()
    if hasattr(chunk, "text"):
        return chunk.text if isinstance(chunk.text, str) else str(chunk.text)
    return str(chunk)


def build_prompt(query: str, chunks: list[str], scores: list[float] = None) -> str:
    if scores:
        # Annotate each chunk with its confidence score
        annotated = [
            f"[Relevance: {score:.2f}]\n{chunk}"
            for chunk, score in zip(chunks, scores)
        ]
        context_str = "\n\n---\n\n".join(annotated)
        confidence_note = (
            "Note: Each context chunk is annotated with a relevance score (0 to 1). "
            "Prefer higher-scored chunks. If all scores are low, be explicit about uncertainty."
        )
    else:
        context_str = "\n\n---\n\n".join(chunks)
        confidence_note = ""

    prompt = f"""You are an expert academic research assistant.
                Use the following pieces of retrieved context to answer the user's question.
                If the answer is not explicitly contained in the context, say "I don't know based on the provided context."
                {confidence_note}

                Context:
                {context_str}

                Question: {query}
                Answer:"""
    return prompt

def run_rag_pipeline(user_query: str):
    print(f"Executing Vector RAG for: '{user_query}'\n")

    # 1. Retrieve
    print("1. Retrieving chunks from Qdrant...")
    nodes = retrieve_chunks(user_query)
    chunks = [extract_chunk_text(node) for node in nodes]
    scores = [node.score for node in nodes]
    print(f"-> Retrieved {len(chunks)} chunks.\n")

    # 2. Early exit if nothing passed the threshold
    if not chunks:
        print("-> No relevant chunks found above threshold. Aborting generation.")
        return {
            "answer": "I don't know based on the provided context.",
            "contexts": []
        }

    # 3. Build Prompt
    print("2. Building prompt...")
    prompt = build_prompt(user_query, chunks, scores=scores)

    # 4. Generate
    print(f"3. Generating answer using model: {settings.GENERATION_MODEL}...\n")
    answer = generate_completion(prompt)

    print("=" * 50)
    print("Final Answer:")
    print("=" * 50)
    print(answer)

    return {"answer": answer, "contexts": chunks}


if __name__ == "__main__":
    run_rag_pipeline("Onde o Sr. Dursley trabalha?")

# python -m phase1_vector_rag.rag