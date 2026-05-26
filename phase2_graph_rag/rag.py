from phase2_graph_rag.retriever import retrieve_graph_chunks
from shared.llm import generate_completion
from shared.config import settings

def build_hybrid_prompt(query: str, chunks: list[str]) -> str:
    """
    Constructs the prompt window. 
    Notice we explicitly tell the LLM to look for the graph relationships.
    """
    context_str = "\n\n---\n\n".join(chunks)
    
    prompt = f"""You are an expert financial research assistant.
            Use the following pieces of retrieved context to answer the user's question.
            The context contains both raw text from financial documents AND structured graph relationships (Triplets). 
            Use the graph relationships to accurately connect entities and resolve complex multi-part queries.

            If the answer is not explicitly contained in the context, say "I don't know based on the provided context."

            Context:
            {context_str}

            Question: {query}
            Answer:"""
    return prompt

def run_graph_rag_pipeline(user_query: str):
    """Main entry point: Hybrid Retrieve -> Prompt -> Generate"""
    print(f"Executing Graph RAG for: '{user_query}'\n")
    
    # 1. Retrieve (This hits both Qdrant and Neo4j via our retriever script)
    print("1. Traversing Graph and Vector stores...")
    chunks = retrieve_graph_chunks(user_query)
    print(f"-> Retrieved {len(chunks)} hybrid chunks.\n")
    
    # 2. Build Prompt
    print("2. Building hybrid prompt...")
    prompt = build_hybrid_prompt(user_query, chunks)
    
    # 3. Generate Answer
    print(f"3. Generating answer using LiteLLM model: {settings.GENERATION_MODEL}...\n")
    answer = generate_completion(prompt)
    
    print("=" * 50)
    print("Final Graph RAG Answer:")
    print("=" * 50)
    print(answer)
    
    # Returning the chunks alongside the answer is exactly what RAGAS needs for evaluation
    return {"answer": answer, "contexts": chunks}

if __name__ == "__main__":
    # Test your Phase 2 pipeline!
    test_question = "Onde o Sr. Dursley trabalha?"
    run_graph_rag_pipeline(test_question)
    
# python -m phase2_graph_rag.rag