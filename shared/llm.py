from litellm import completion
from shared.config import settings

def generate_completion(prompt: str, system_message: str = "You are a helpful assistant.") -> str:
    """Invokes the LLM using LiteLLM for universal model switching."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    # LiteLLM automatically uses the OPENAI_API_KEY from the environment
    response = completion(
        model=settings.GENERATION_MODEL, # e.g., "openai/gpt-4o-mini"
        messages=messages,
    )
    
    return response.choices[0].message.content