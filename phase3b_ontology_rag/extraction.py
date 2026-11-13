"""Schema-constrained fact extraction for Phase 3b.

One call per chunk, one JSON object per movie. The ontology schema is the
allowed contract: actors / directors / writers / genres / languages / year.
No temperature and no max_tokens (Luna only accepts the provider default).
"""

from __future__ import annotations

import json
import re

from shared.extraction_llm import LitellmExtractionLLM

LIST_KEYS = ("actors", "directors", "writers", "genres", "languages")
MAX_ITEMS = 40
MAX_NAME_LENGTH = 120
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")


def build_prompt(movie: str, text: str) -> str:
    return f"""Extract facts about the movie "{movie}" from the text below.

Allowed fields and types:
- actors, directors, writers, genres, languages: lists of names (strings)
- year: a 4-digit release year (string) or null

Return ONLY a JSON object with these keys, for example:
{{"actors": [], "directors": [], "writers": [], "genres": [], "languages": [], "year": null}}

Rules:
- Use only facts explicitly stated in the text about this movie.
- Use the exact names as written; omit anything you are not sure about.
- Omit unknown fields or use [] / null.
- No commentary, no markdown, no code fences.

Text:
{text}
"""


def _clean_list(values, movie: str) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned, seen = [], set()
    for value in values:
        if not isinstance(value, (str, int)):
            continue
        name = str(value).strip().strip('"').strip()
        if not name or len(name) > MAX_NAME_LENGTH:
            continue
        if name.casefold() == (movie or "").casefold():
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(name)
        if len(cleaned) >= MAX_ITEMS:
            break
    return cleaned


def parse_facts(raw: str, movie: str = "") -> tuple[dict, str | None]:
    """Parse a JSON fact object. Returns (facts, error). Never raises."""
    text = (raw or "").strip()
    if not text:
        return {}, "empty response"
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}, "no JSON object"
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        return {}, f"invalid JSON: {error}"
    if not isinstance(data, dict):
        return {}, "JSON is not an object"

    facts = {key: _clean_list(data.get(key), movie) for key in LIST_KEYS}
    year = data.get("year", data.get("release_year"))
    match = YEAR_RE.search(str(year)) if year is not None else None
    facts["year"] = match.group(0) if match else None
    return facts, None


async def extract_facts(llm: LitellmExtractionLLM, movie: str, text: str):
    response = await llm.acomplete(build_prompt(movie, text))
    return parse_facts(response.text, movie=movie)
