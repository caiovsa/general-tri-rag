import time

from litellm import completion
from litellm.exceptions import APIError, AuthenticationError, RateLimitError
from shared.config import llm_route, settings



def openrouter_extra_body(reasoning_effort: str | None = None,
                         provider_only: str | None = None,
                         allow_fallbacks: bool | None = None) -> dict:
    """Build OpenRouter's unified reasoning/provider payload.

    Provider pinning is opt-in. ``allow_fallbacks=False`` means OpenRouter must
    use one of the listed providers or fail, rather than silently routing to a
    different quantization/provider.
    """
    body: dict = {}
    effort = reasoning_effort if reasoning_effort is not None else settings.GENERATION_REASONING_EFFORT
    if effort:
        body["reasoning"] = {"effort": effort}
    only = provider_only if provider_only is not None else settings.OPENROUTER_PROVIDER_ONLY
    providers = [item.strip() for item in (only or "").split(",") if item.strip()]
    if providers:
        body["provider"] = {
            "only": providers,
            "allow_fallbacks": (settings.OPENROUTER_ALLOW_FALLBACKS
                                if allow_fallbacks is None else allow_fallbacks),
        }
    return body

def generate_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    temperature: float | None = None,  # None -> settings.GENERATION_TEMPERATURE
    reasoning_effort: str | None = None,  # None -> settings.GENERATION_REASONING_EFFORT
    max_retries: int = 3,
    #max_tokens: int = 1024,
) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if temperature is None:
        temperature = settings.GENERATION_TEMPERATURE

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    # Luna models accept only the provider default; omit the parameter entirely
    # instead of sending a value that would 400.
    temperature_kwargs = {} if temperature is None else {"temperature": temperature}
    reasoning_effort = reasoning_effort or settings.GENERATION_REASONING_EFFORT
    reasoning_kwargs = {}
    if reasoning_effort:
        if settings.GENERATION_MODEL.startswith("openrouter/"):
            # OpenRouter's unified reasoning field; ``exclude`` only hides the
            # trace, while ``effort: none`` actually disables reasoning.
            reasoning_kwargs = {"extra_body": openrouter_extra_body(reasoning_effort)}
        else:
            # LiteLLM's OpenAI mapping rejects reasoning_effort before sending,
            # even though the OpenAI-compatible endpoint accepts it. Pass it via
            # extra_body so the provider receives the actual request field.
            reasoning_kwargs = {"extra_body": {"reasoning_effort": reasoning_effort}}
    elif settings.GENERATION_MODEL.startswith("openrouter/"):
        # Provider pinning can apply even when reasoning is left at its default.
        extra_body = openrouter_extra_body(reasoning_effort)
        if extra_body:
            reasoning_kwargs = {"extra_body": extra_body}

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=settings.GENERATION_MODEL,
                messages=messages,
                **temperature_kwargs,
                **reasoning_kwargs,
                #max_tokens=max_tokens,
                **llm_route(settings.GENERATION_MODEL),
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