"""litellm-backed extraction LLM for LlamaIndex property-graph extractors.

Luna models:
  * only accept the provider default temperature (sending 0 raises 400),
  * reject ``max_tokens`` (use ``max_completion_tokens``),
  * are not in LlamaIndex's static OpenAI model list.

This wrapper therefore sends no temperature and no max-token parameter, uses
litellm's async completion for real concurrency, and counts calls/errors/policy
skips for the ingest report.
"""

from __future__ import annotations

from typing import Any

import litellm
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.llms.custom import CustomLLM

from shared.config import settings


class LitellmExtractionLLM(CustomLLM):
    """CustomLLM that talks to litellm directly (no model validation, no temp)."""

    calls: int = 0
    errors: int = 0
    policy_skips: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            is_chat_model=False,
            model_name=settings.EXTRACTION_MODEL,
            context_window=922_000,
            num_output=128_000,
        )

    @classmethod
    def class_name(cls) -> str:
        return "litellm_extraction_llm"

    @staticmethod
    def _is_policy_error(error: Exception) -> bool:
        text = str(error).lower()
        return (getattr(error, "code", None) == "invalid_prompt"
                or "usage policy" in text
                or "invalid prompt" in text)

    def _record(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    @staticmethod
    def _content(response) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        return choices[0].message.content or ""

    def _kwargs(self, prompt: str) -> dict[str, Any]:
        # Deliberately no temperature and no max_tokens: Luna rejects both.
        return {
            "model": settings.EXTRACTION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "api_key": settings.OPENAI_API_KEY,
        }

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.calls += 1
        try:
            response = litellm.completion(**self._kwargs(prompt))
        except Exception as error:  # noqa: BLE001 - classified below
            if self._is_policy_error(error):
                self.policy_skips += 1
                return CompletionResponse(text="")
            self.errors += 1
            raise
        self._record(response)
        return CompletionResponse(text=self._content(response))

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.calls += 1
        try:
            response = await litellm.acompletion(**self._kwargs(prompt))
        except Exception as error:  # noqa: BLE001 - classified below
            if self._is_policy_error(error):
                self.policy_skips += 1
                return CompletionResponse(text="")
            self.errors += 1
            raise
        self._record(response)
        return CompletionResponse(text=self._content(response))

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        # Streaming is not used by the graph extractors.
        yield self.complete(prompt, formatted=formatted, **kwargs)
