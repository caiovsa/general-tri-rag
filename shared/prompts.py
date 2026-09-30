"""Shared prompt for the MetaQA tracks.

One template is used by Phase 1, Phase 2, the closed-book control and (later)
Phase 3. Only the context block differs by phase; the template itself is frozen
and its sha256 is recorded in run_meta.json and the eval metadata.

HotpotQA keeps its phase-specific prompts (`phaseN_*/rag.py`), so nothing here
is imported on DATASET=hotpot.
"""

import hashlib

METAQA_PROMPT_TEMPLATE = """You are a helpful assistant answering questions from the provided context.

Rules:
- Answer only from the provided context.
- If the question asks for several items (movies, people, genres, languages, years), list ALL matching items found in the context, comma-separated.
- Reply with the answer only, without extra commentary.
- If the context contains no answer, reply "I don't know".

Context:
{context}

Question: {question}
Answer:"""

# Context placeholders (the template's context block is the only phase-specific part).
METAQA_EMPTY_CONTEXT = "(no context retrieved)"
CLOSED_BOOK_CONTEXT = "(no context)"


def build_metaqa_prompt(question: str, context: str) -> str:
    """Fully built MetaQA prompt. `context` is the phase's context block."""
    return METAQA_PROMPT_TEMPLATE.format(context=context, question=question)


def format_context(chunks: list[str]) -> str:
    """Context block shared by the retrieval phases (chunk text only, no scores)."""
    return "\n\n---\n\n".join(chunks) if chunks else METAQA_EMPTY_CONTEXT


def metaqa_prompt_template_hash() -> str:
    """sha256 of the frozen template (recorded with every MetaQA run)."""
    return hashlib.sha256(METAQA_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
