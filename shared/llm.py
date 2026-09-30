import time

from litellm import completion
from litellm.exceptions import APIError, AuthenticationError, RateLimitError
from shared.config import settings


def generate_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    temperature: float = 1,  # Low temp for RAG — you want factual, not creative
    max_retries: int = 3,
    #max_tokens: int = 1024,
) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=settings.GENERATION_MODEL,
                messages=messages,
                temperature=temperature,
                #max_tokens=max_tokens,
                api_key=settings.MARITACA_API_KEY or None,
                api_base=settings.MARITACA_API_BASE or None,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("LLM returned no content.")
            return content

        except AuthenticationError as e:
            raise RuntimeError(f"Invalid API key for model: {settings.GENERATION_MODEL}") from e
        except (RateLimitError, APIError) as e:
            last_exc = e
            if attempt >= max_retries:
                break
            backoff = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(backoff)
            continue
        except Exception as e:
            # Non-retryable or unexpected — bubble up
            raise RuntimeError(f"LLM API error: {e}") from e

    # Exhausted retries
    if isinstance(last_exc, RateLimitError):
        raise RuntimeError("Rate limit hit — retries exhausted. Try again later or switch models.") from last_exc
    raise RuntimeError(f"LLM API error after {max_retries} retries: {last_exc}") from last_exc