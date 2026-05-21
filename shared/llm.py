from litellm import completion
from litellm.exceptions import APIError, AuthenticationError, RateLimitError
from shared.config import settings

def generate_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.3,  # Low temp for RAG — you want factual, not creative
    #max_tokens: int = 1024,
) -> str:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    try:
        response = completion(
            model=settings.GENERATION_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    except AuthenticationError:
        raise RuntimeError(f"Invalid API key for model: {settings.GENERATION_MODEL}")
    except RateLimitError:
        raise RuntimeError("Rate limit hit — consider retry logic or switching models.")
    except APIError as e:
        raise RuntimeError(f"LLM API error: {e}")