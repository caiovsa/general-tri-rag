"""MetaQA ontology schema (Phase 3).

The RDF file `ontology/metaqa.ttl` documents the same contract for humans and
other tools; this module is the executable version used by ingest/retrieval.
`ontology_hash()` is recorded in every Phase 3 run's `run_meta.json`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ONTOLOGY_FILE = Path(__file__).resolve().parent / "ontology" / "metaqa.ttl"

CLASSES = ("Movie", "Person", "Actor", "Director", "Writer", "Genre", "Language", "Year", "Tag", "Rating", "Votes")

# KB relation -> (subject label, Cypher edge type, object label)
RELATIONS: dict[str, tuple[str, str, str]] = {
    "starred_actors": ("Movie", "STARRED_ACTORS", "Person"),
    "directed_by": ("Movie", "DIRECTED_BY", "Person"),
    "written_by": ("Movie", "WRITTEN_BY", "Person"),
    "has_genre": ("Movie", "HAS_GENRE", "Genre"),
    "in_language": ("Movie", "IN_LANGUAGE", "Language"),
    "release_year": ("Movie", "RELEASED_IN", "Year"),
    "has_tags": ("Movie", "HAS_TAG", "Tag"),
    "has_imdb_rating": ("Movie", "HAS_IMDB_RATING", "Rating"),
    "has_imdb_votes": ("Movie", "HAS_IMDB_VOTES", "Votes"),
}
PERSON_RELATIONS = ("starred_actors", "directed_by", "written_by")
PERSON_EDGE_TYPES = tuple(RELATIONS[rel][1] for rel in PERSON_RELATIONS)
ROLE_LABELS = {"starred_actors": "Actor", "directed_by": "Director", "written_by": "Writer"}

# Only these edge types are traversed to collect evidence movies: genre/language/
# tag/year edges lead to literal nodes, not to other movies on a gold path.
TRAVERSABLE_EDGE_TYPES = PERSON_EDGE_TYPES


def ontology_hash() -> str:
    """sha256 of the RDF file plus the executable schema (stable across runs)."""
    digest = hashlib.sha256()
    if ONTOLOGY_FILE.exists():
        digest.update(ONTOLOGY_FILE.read_bytes())
    digest.update(json.dumps({"classes": CLASSES, "relations": RELATIONS}, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()
