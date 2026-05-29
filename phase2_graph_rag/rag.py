from phase2_graph_rag.retriever import retrieve_graph_chunks
from shared.llm import generate_completion
from shared.config import settings

def build_graph_prompt(query: str, chunks: list[str]) -> str:
    """
    Constructs the prompt for pure Graph RAG.
    Context includes both document text and graph relationships.
    """
    context_str = "\n\n---\n\n".join(chunks)
    
    prompt = f"""You are an expert research assistant.
Use the following pieces of retrieved context to answer the user's question.
The context contains document chunks enriched with graph relationships (triplets) that connect entities.
Use the graph relationships silently to accurately connect entities and resolve complex queries.

Rules:
- Base your answer strictly on the provided context.
- If the answer is not in the context, say "I don't know based on the provided context."
- Do not invent relationships or facts not present in the context.
- Give a natural, conversational answer. Do NOT expose raw triplets, node IDs, or graph syntax in your response.
- Use the graph relationships as internal evidence only — never mention them explicitly.

Context:
{context_str}

Question: {query}
Answer:"""
    return prompt

def run_graph_rag_pipeline(user_query: str):
    """Main entry point: Graph Retrieve -> Prompt -> Generate"""
    print(f"Executing Graph RAG for: '{user_query}'\n")
    
    print("1. Retrieving from Neo4j (vector + graph traversal)...")
    chunks = retrieve_graph_chunks(user_query)
    print(f"-> Retrieved {len(chunks)} graph-enriched chunks.\n")
    
    if not chunks:
        print("-> No relevant chunks found. Aborting generation.")
        return {
            "answer": "I don't know based on the provided context.",
            "contexts": []
        }
    
    print("2. Building graph prompt...")
    prompt = build_graph_prompt(user_query, chunks)
    
    print(f"3. Generating answer using model: {settings.GENERATION_MODEL}...\n")
    answer = generate_completion(prompt)
    
    print("=" * 50)
    print("Final Graph RAG Answer:")
    print("=" * 50)
    print(answer)
    
    return {"answer": answer, "contexts": chunks}

if __name__ == "__main__":
    test_question = "Quando se tornou proibido criar Dragões?"#"Qual o modelo da vassoura que harry ganhou?" #"Qual a casa de harry potter?"
    run_graph_rag_pipeline(test_question)
    
# python -m phase2_graph_rag.rag
