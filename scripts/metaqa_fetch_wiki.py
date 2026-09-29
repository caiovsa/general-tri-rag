#!/usr/bin/env python3
"""Fetch Wikipedia pages for the MetaQA corpus and write RAG documents.

Run: uv run python -m scripts.metaqa_fetch_wiki [--dry-run]

Corpus = every movie in any question's `evidence_movies` (must-have) plus safe
distractors, up to --total-docs:
  * hard   — share a director, writer or actor with an evidence movie
  * random — uniform from the remaining safe movies
Unsafe rules are imported from scripts.metaqa_sample (reused titles, missing
release_year, person-name collisions).

Fetching uses the MediaWiki action=parse API (one title per request, <=1 req/s,
User-Agent from $WIKI_CONTACT); every raw response is cached under <base>/cache/
so re-runs cost no requests.

Outputs: <base>/docs/<safe_title>.txt and <base>/fetch_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from scripts.metaqa_sample import PERSON_RELS, load_kb, resolve_root, unsafe_movies

# Same .env the phases use, so WIKI_CONTACT can live there instead of the shell
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

API = "https://en.wikipedia.org/w/api.php"
INFOBOX_FIELDS = ("directed by", "written by", "screenplay by", "story by",
                  "based on", "starring", "language", "release date")
RULE_YEAR, RULE_FILM, RULE_PLAIN, RULE_SEARCH = (
    "<title> (<year> film)", "<title> (film)", "<title>", "search")
MAX_RETRIES, MIN_INTERVAL, YEAR_WINDOW, CHARS_PER_CHUNK = 5, 1.0, 1500, 256 * 4
JUNK_SELECTORS = ("sup.reference", ".mw-editsection", "style", "script", "table",
                  ".navbox", ".reflist", "ol.references", ".metadata", ".hatnote")


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()
    return re.sub(r"\s+([)\]};,.;:])", r"\1", text).replace("( ", "(")


def strip_notes(text: str) -> str:
    return clean(re.sub(r"\[\d+\]", "", text))


def safe_name(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9 ,'()\-]", "_", title).strip() or "untitled"


# ── corpus ────────────────────────────────────────────────────────────────
def load_candidates(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_corpus(rows, kb, unsafe, total_docs: int, hard_frac: float, rng) -> dict:
    movies, forward, backward, _ = kb
    evidence = {m for row in rows for m in row["evidence_movies"]}
    related = set()
    for movie in evidence:
        for rel in PERSON_RELS:
            for person in forward[rel].get(movie, ()):
                related |= backward[rel].get(person, set())
    hard_pool = sorted((related - evidence) & (movies - unsafe))
    random_pool = sorted(movies - unsafe - evidence - related)

    budget = max(0, total_docs - len(evidence))
    hard_pick = rng.sample(hard_pool, min(len(hard_pool), int(round(budget * hard_frac))))
    random_pick = rng.sample(random_pool, min(len(random_pool), budget - len(hard_pick)))
    shortfall = budget - len(hard_pick) - len(random_pick)
    if shortfall:  # one pool exhausted — top up from the other
        spare = [m for m in hard_pool if m not in set(hard_pick)]
        hard_pick += rng.sample(spare, min(len(spare), shortfall))

    kind = {m: "must-have" for m in evidence}
    kind.update({m: "hard" for m in hard_pick})
    kind.update({m: "random" for m in random_pick})
    return {"docs": sorted(evidence) + sorted(hard_pick) + sorted(random_pick), "kind": kind,
            "must_have": len(evidence), "hard": len(hard_pick), "random": len(random_pick),
            "hard_pool": len(hard_pool), "random_pool": len(random_pool)}


def print_corpus(corpus, total_docs):
    print("Corpus")
    print(f"  must-have (evidence movies) : {corpus['must_have']:,}")
    print(f"  hard distractors            : {corpus['hard']:,}  "
          f"(pool {corpus['hard_pool']:,}: share director/writer/actor with an evidence movie)")
    print(f"  random distractors          : {corpus['random']:,}  (pool {corpus['random_pool']:,})")
    print(f"  total documents             : {len(corpus['docs']):,}  (--total-docs {total_docs})")
    if corpus["must_have"] > total_docs:
        print(f"  WARNING: must-have evidence movies ({corpus['must_have']:,}) exceed "
              f"--total-docs ({total_docs:,}); keeping all of them and adding no distractors.")


def print_estimates(corpus, max_chars):
    docs = len(corpus["docs"])
    chunks_per_doc = -(-max_chars // CHARS_PER_CHUNK)
    chunks = docs * chunks_per_doc
    print(f"\nEstimates (upper bound: every doc reaches --max-chars {max_chars:,})")
    print(f"  chars per chunk             : {CHARS_PER_CHUNK:,}  (256 tokens x ~4 chars/token)")
    print(f"  chunks per doc              : {chunks_per_doc}  ->  total chunks {chunks:,}")
    print(f"  Phase 2 extraction calls    : ~{chunks:,}  (SimpleLLMPathExtractor: 1 call/chunk, "
          "max_paths_per_chunk=12)")
    print(f"  fetch requests (worst case) : {docs * 4:,}  (up to 4 title lookups per doc)")
    print(f"  wall clock at <=1 req/s     : ~{docs * 4 / 60:.0f} min worst case, "
          f"~{docs / 60:.0f} min if every title resolves on the first rule")


# ── fetching ──────────────────────────────────────────────────────────────
class Fetcher:
    """MediaWiki API client: cached, rate limited, retried."""

    def __init__(self, api: str, cache_dir: Path, user_agent: str, min_interval: float = MIN_INTERVAL):
        self.api, self.cache_dir, self.user_agent, self.min_interval = api, cache_dir, user_agent, min_interval
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last = 0.0
        self.stats = Counter()

    def get(self, params: dict) -> dict:
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            self.stats["cache_hits"] += 1
            return json.loads(cached.read_text(encoding="utf-8"))["response"]

        url = f"{self.api}?{urllib.parse.urlencode(params)}"
        last_error = None
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.loads(response.read().decode("utf-8"))
                cached.write_text(json.dumps({"params": params, "response": data}, ensure_ascii=False),
                                  encoding="utf-8")
                self.stats["requests"] += 1
                return data
            except urllib.error.HTTPError as error:
                last_error = error
                retry_after = float(error.headers.get("Retry-After") or 0)
                backoff = max(retry_after, 2 ** attempt)
                if error.code in (429, 503):  # rate limited / unavailable — back off harder
                    backoff = max(backoff, 5.0 * (attempt + 1))
                time.sleep(backoff)
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
                last_error = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API request failed after {MAX_RETRIES} attempts: {url} ({last_error})")

    def _throttle(self):
        delta = time.monotonic() - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


def fetch_page(fetcher: Fetcher, title: str):
    """Return (html, None) or (None, error_code)."""
    data = fetcher.get({"action": "parse", "page": title, "prop": "text", "redirects": 1,
                        "format": "json", "formatversion": 2})
    page = data.get("parse")
    if not page:
        return None, (data.get("error") or {}).get("code", "missing")
    return page.get("text", ""), None


def search_titles(fetcher: Fetcher, query: str, limit: int = 3) -> list[str]:
    data = fetcher.get({"action": "query", "list": "search", "srsearch": query, "srlimit": limit,
                        "format": "json", "formatversion": 2})
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


# ── parsing ───────────────────────────────────────────────────────────────
def infobox_lines(soup) -> list[str]:
    """Emit "Label: value" for the whitelisted infobox fields, in a fixed order."""
    table = soup.find("table", class_="infobox")
    if table is None:
        return []
    found = {}
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) < 2:
            continue
        label = strip_notes(cells[0].get_text(" ", strip=True)).rstrip(":").lower()
        field = next((f for f in INFOBOX_FIELDS if label.startswith(f)), None)
        if field is None or field in found:
            continue
        value = cells[-1]
        for br in value.find_all("br"):  # <br> separated values are separate names
            br.replace_with("; ")
        items = [strip_notes(li.get_text(" ", strip=True)) for li in value.find_all("li")]
        text = "; ".join(i for i in items if i) or strip_notes(value.get_text(" ", strip=True))
        if text:
            found[field] = f"{field.title()}: {text}"
    return [found[f] for f in INFOBOX_FIELDS if f in found]


def body_sections(soup) -> tuple[list[str], list[str], list[str]]:
    """(lead, cast, plot) text blocks, in document order."""
    content = soup.find("div", class_="mw-parser-output") or soup
    lead, cast, plot = [], [], []
    section, past_first_heading = None, False
    for element in content.find_all(["h2", "h3", "h4", "p", "li"]):
        if element.name in ("h2", "h3", "h4"):
            heading = strip_notes(element.get_text(" ", strip=True)).lower()
            past_first_heading = True
            section = ("cast" if heading.startswith("cast")
                       else "plot" if heading.startswith("plot") else None)
            continue
        text = strip_notes(element.get_text(" ", strip=True))
        if not text:
            continue
        if section == "cast":
            cast.append(text)
        elif section == "plot":
            plot.append(text)
        elif not past_first_heading:
            lead.append(text)
    return lead, cast, plot


def build_doc(title: str, infobox: list[str], lead: list[str], cast: list[str], plot: list[str],
              max_chars: int):
    """Return (doc, full_text). The infobox block is never truncated."""
    header = f"Title: {title}\n\nInfobox:\n" + "\n".join(infobox) + "\n\n"
    body = list(lead)
    if cast:
        body += ["Cast"] + cast
    if plot:
        body += ["Plot"] + plot

    doc = header
    for block in body:
        if len(doc) + len(block) + 2 > max_chars:  # truncate at a paragraph boundary
            break
        doc += f"{block}\n\n"
    full = header + "".join(f"{block}\n\n" for block in body)
    return doc.rstrip() + "\n", full


def parse_page(html: str, title: str, max_chars: int):
    soup = BeautifulSoup(html, "lxml")
    infobox = infobox_lines(soup)
    for selector in JUNK_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    lead, cast, plot = body_sections(soup)
    return build_doc(title, infobox, lead, cast, plot, max_chars)


def verify(doc: str, full_text: str, persons, year: str):
    """(a) mentions a KB person linked to the movie, (b) KB year early in the doc."""
    haystack = full_text.lower()
    person = next((p for p in persons if p.lower() in haystack), None)
    if person is None:
        return False, "no linked KB person mentioned"
    if year and year not in doc[:YEAR_WINDOW]:
        return False, f"release_year {year} not within first {YEAR_WINDOW} chars"
    return True, f"person: {person}"


def resolve(fetcher: Fetcher, title: str, year: str, persons, max_chars: int):
    """Try the title rules in order, then the search fallback.

    Returns (hit, tried, failures, pages): `pages` non-empty means at least one
    page was retrieved, so a total miss is 'unresolved' and a page that fails
    verification is 'unverified'.
    """
    attempts = []
    if year:
        attempts.append((RULE_YEAR, f"{title} ({year} film)"))
    attempts += [(RULE_FILM, f"{title} (film)"), (RULE_PLAIN, title)]

    tried, failures, pages = [], [], []
    for rule, page in attempts:
        tried.append(page)
        html, error = fetch_page(fetcher, page)
        if html is None:
            failures.append(f"{page}: {error}")
            continue
        pages.append(page)
        doc, full = parse_page(html, title, max_chars)
        ok, why = verify(doc, full, persons, year)
        if ok:
            return {"page": page, "rule": rule, "doc": doc, "verify": why}, tried, failures, pages
        failures.append(f"{page}: {why}")

    query = f"{title} {year} film".strip()
    for page in search_titles(fetcher, query):
        tried.append(f"search: {page}")
        html, error = fetch_page(fetcher, page)
        if html is None:
            failures.append(f"{page}: {error}")
            continue
        pages.append(page)
        doc, full = parse_page(html, title, max_chars)
        ok, why = verify(doc, full, persons, year)
        if ok:
            return {"page": page, "rule": RULE_SEARCH, "doc": doc, "verify": why}, tried, failures, pages
        failures.append(f"{page}: {why}")
    return None, tried, failures, pages


# ── main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--total-docs", type=int, default=800)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="only fetch the first N docs (testing)")
    ap.add_argument("--dry-run", action="store_true", help="print the corpus and estimates, no network")
    ap.add_argument("--api", default=API)
    args = ap.parse_args()

    root = resolve_root()
    base = root.parent if root.name == "raw" else root
    kb = load_kb(root)
    rows = load_candidates(base / "metaqa_candidates.jsonl")
    unsafe = set().union(*unsafe_movies(kb[0], kb[1], kb[2]).values())
    corpus = build_corpus(rows, kb, unsafe, args.total_docs, args.hard_frac, random.Random(args.seed))
    print(f"root: {root} | candidates: {len(rows):,} questions | kb: {len(kb[0]):,} movies\n")
    print_corpus(corpus, args.total_docs)
    print_estimates(corpus, args.max_chars)
    if args.dry_run:
        print("\n--dry-run: no network requests made.")
        return

    contact = os.getenv("WIKI_CONTACT", "")
    if not contact:
        raise SystemExit("Set WIKI_CONTACT (e.g. WIKI_CONTACT=you@example.com) for the User-Agent "
                         "required by the Wikimedia API.")
    user_agent = f"rag-trilogy-metaqa/0.1 (MetaQA corpus build; contact: {contact})"
    docs_dir = base / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(args.api, base / "cache", user_agent)

    docs = corpus["docs"][:args.limit] if args.limit else corpus["docs"]
    forward = kb[1]
    report = {"counts": {"candidates": len(rows), "must_have": corpus["must_have"],
                         "hard": corpus["hard"], "random": corpus["random"], "total_docs": len(docs)},
              "rules": Counter(), "resolved": [], "unresolved": [], "unverified": []}
    print(f"\nFetching {len(docs):,} documents ...")
    for number, title in enumerate(docs, 1):
        year = next(iter(forward["release_year"].get(title, ())), "")
        persons = sorted({p for rel in PERSON_RELS for p in forward[rel].get(title, ())},
                         key=len, reverse=True)
        hit, tried, failures, pages = resolve(fetcher, title, year, persons, args.max_chars)
        prefix = f"[{number:>4}/{len(docs)}]"
        if hit:
            path = docs_dir / f"{safe_name(title)}.txt"
            path.write_text(hit["doc"], encoding="utf-8")
            report["rules"][hit["rule"]] += 1
            report["resolved"].append({"title": title, "kind": corpus["kind"][title], "page": hit["page"],
                                       "rule": hit["rule"], "chars": len(hit["doc"]), "verify": hit["verify"],
                                       "file": str(path)})
            print(f"{prefix} OK         {corpus['kind'][title]:<9} {title} -> {hit['page']} "
                  f"({len(hit['doc']):,} chars, {hit['rule']})")
        elif pages:
            report["unverified"].append({"title": title, "kind": corpus["kind"][title],
                                         "tried": tried, "failures": failures, "pages": pages})
            print(f"{prefix} UNVERIFIED {corpus['kind'][title]:<9} {title} ({failures[-1]})")
        else:
            report["unresolved"].append({"title": title, "kind": corpus["kind"][title], "tried": tried})
            print(f"{prefix} UNRESOLVED {corpus['kind'][title]:<9} {title}")

    report["rules"] = dict(report["rules"])
    report["counts"].update({"resolved": len(report["resolved"]), "unresolved": len(report["unresolved"]),
                             "unverified": len(report["unverified"]),
                             "api_requests": fetcher.stats["requests"], "cache_hits": fetcher.stats["cache_hits"]})
    (base / "fetch_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResolved {len(report['resolved']):,} | unverified {len(report['unverified']):,} | "
          f"unresolved {len(report['unresolved']):,}")
    print("Resolution rules: " + ", ".join(f"{k}={v}" for k, v in sorted(report["rules"].items())))
    print(f"API requests: {fetcher.stats['requests']:,} (cache hits: {fetcher.stats['cache_hits']:,})")
    print(f"Docs: {docs_dir} | report: {base / 'fetch_report.json'}")


if __name__ == "__main__":
    main()
