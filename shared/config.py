import os
from dataclasses import dataclass, replace
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

REPO_ROOT = Path(__file__).parent.parent
PHASE_DIRS = {1: "phase1_vector_rag", 2: "phase2_graph_rag", 3: "phase3_ontology_rag"}


def _optional_float(name: str) -> float | None:
    """Env float, or None when unset (models that only accept the default)."""
    raw = os.getenv(name)
    return None if raw is None or not raw.strip() else float(raw)

class Settings:
    # Database Configs
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "testpass123")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "hotpot-graph")
    
    # Models & Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # Optional OpenRouter route. Use a model prefixed with "openrouter/".
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_API_BASE: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gpt-6-luna")
    # Judge must differ from the generator; pinned in .env and every run_meta.
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-5.6-luna")
    EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "gpt-6-luna")
    # gpt-5 / gpt-6 families accept only the provider default (1); when unset we
    # send no temperature parameter at all and record "default".
    GENERATION_TEMPERATURE: float | None = _optional_float("GENERATION_TEMPERATURE")
    EXTRACTION_TEMPERATURE: float | None = _optional_float("EXTRACTION_TEMPERATURE")
    # RAGAS injects temperature=1e-8, which gpt-5/gpt-6 models reject. Keep this
    # on a model that supports it (recorded in run_meta / eval metadata).
    RAGAS_MODEL: str = os.getenv("RAGAS_MODEL", "gpt-4o-mini")

    # RAG parameters
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_phase_1_baseline")
    # Standardized RAG retrieval defaults (shared across phases)
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "8"))
    DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.3"))
    # Canonical chunk sizes (override per-phase only if experiment requires)
    CHUNK_SIZE_HOTPOT: int = int(os.getenv("CHUNK_SIZE_HOTPOT", "200"))
    CHUNK_OVERLAP_HOTPOT: int = int(os.getenv("CHUNK_OVERLAP_HOTPOT", "20"))
    CHUNK_SIZE_PDF: int = int(os.getenv("CHUNK_SIZE_PDF", "512"))
    CHUNK_OVERLAP_PDF: int = int(os.getenv("CHUNK_OVERLAP_PDF", "64"))
    CHUNK_SIZE_GRAPH: int = int(os.getenv("CHUNK_SIZE_GRAPH", "256"))
    CHUNK_OVERLAP_GRAPH: int = int(os.getenv("CHUNK_OVERLAP_GRAPH", "20"))
    
    #MARITACA CONFIGS
    MARITACA_API_KEY: str = os.getenv("MARITACA_API_KEY", "")
    MARITACA_API_BASE: str = os.getenv("MARITACA_API_BASE", "")

    # Dataset selection: hotpot (default) | metaqa
    DATASET: str = os.getenv("DATASET", "hotpot").strip().lower()
    # Retrieval depth: TOP_K overrides the per-phase default (phase 1 -> 8, phase 2 -> 5)
    TOP_K: int = int(os.getenv("TOP_K", "0"))

    # MetaQA track (frozen artifacts, see README -> MetaQA Track)
    METAQA_DIR: Path = REPO_ROOT / "MetaQA"
    METAQA_EVAL_FILE: Path = Path(os.getenv("METAQA_EVAL_FILE", str(METAQA_DIR / "metaqa_eval_v1.jsonl")))
    METAQA_DOC_LIST: Path = Path(os.getenv("METAQA_DOC_LIST", str(METAQA_DIR / "corpus_manifest.txt")))
    METAQA_DOCS_DIR: Path = Path(os.getenv("METAQA_DOCS_DIR", str(METAQA_DIR / "docs")))
    # MetaQA mini track: a hop-balanced slice of the same corpus, built by
    # scripts/metaqa_make_mini.py, for fast baselines on its own isolated stores.
    METAQA_MINI_EVAL_FILE: Path = Path(os.getenv("METAQA_MINI_EVAL_FILE", str(METAQA_DIR / "metaqa_eval_mini.jsonl")))
    METAQA_MINI_DOC_LIST: Path = Path(os.getenv("METAQA_MINI_DOC_LIST", str(METAQA_DIR / "corpus_manifest_mini.txt")))
    NEO4J_METAQA_MINI_DATABASE: str = os.getenv("NEO4J_METAQA_MINI_DATABASE", "metaqa-mini-graph")
    # MetaQA mini30: the 30-question / 100-doc plumbing corpus built by
    # scripts/metaqa_make_mini30.py (additive track; the 12/50 mini stays frozen).
    METAQA_MINI30_EVAL_FILE: Path = Path(os.getenv("METAQA_MINI30_EVAL_FILE", str(METAQA_DIR / "metaqa_eval_mini30.jsonl")))
    METAQA_MINI30_DOC_LIST: Path = Path(os.getenv("METAQA_MINI30_DOC_LIST", str(METAQA_DIR / "corpus_manifest_mini30.txt")))
    NEO4J_METAQA_MINI30_DATABASE: str = os.getenv("NEO4J_METAQA_MINI30_DATABASE", "metaqa-mini30-graph")
    # Phase 3 (ontology) must never share a database with its Phase 2 track.
    NEO4J_PHASE3_DATABASE: str = os.getenv("NEO4J_PHASE3_DATABASE", "hotpot-ontology-graph")
    NEO4J_METAQA_PHASE3_DATABASE: str = os.getenv("NEO4J_METAQA_PHASE3_DATABASE", "metaqa-ontology-graph")
    NEO4J_METAQA_MINI_PHASE3_DATABASE: str = os.getenv("NEO4J_METAQA_MINI_PHASE3_DATABASE", "metaqa-mini-ontology-graph")
    NEO4J_METAQA_MINI30_PHASE3_DATABASE: str = os.getenv("NEO4J_METAQA_MINI30_PHASE3_DATABASE", "metaqa-mini30-ontology-graph")
    # Phase 3b (schema-constrained extraction) must not overwrite the 3a oracle graph.
    NEO4J_PHASE3B_DATABASE: str = os.getenv("NEO4J_PHASE3B_DATABASE", "hotpot-ontology-b-graph")
    NEO4J_METAQA_PHASE3B_DATABASE: str = os.getenv("NEO4J_METAQA_PHASE3B_DATABASE", "metaqa-ontology-b-graph")
    NEO4J_METAQA_MINI_PHASE3B_DATABASE: str = os.getenv("NEO4J_METAQA_MINI_PHASE3B_DATABASE", "metaqa-mini-ontology-b-graph")
    NEO4J_METAQA_MINI30_PHASE3B_DATABASE: str = os.getenv("NEO4J_METAQA_MINI30_PHASE3B_DATABASE", "metaqa-mini30-ontology-b-graph")

    # Generic local track (DATASET=local): any TXT/PDF/MD corpus + an eval JSONL.
    LOCAL_EVAL_FILE: str = os.getenv("LOCAL_EVAL_FILE", "")
    LOCAL_DOCS_DIR: str = os.getenv("LOCAL_DOCS_DIR", "")
    LOCAL_COLLECTION: str = os.getenv("LOCAL_COLLECTION", "rag_local")
    LOCAL_NEO4J_DATABASE: str = os.getenv("LOCAL_NEO4J_DATABASE", "local-graph")
    LOCAL_NEO4J_PHASE3_DATABASE: str = os.getenv("LOCAL_NEO4J_PHASE3_DATABASE", "local-ontology-graph")
    LOCAL_NEO4J_PHASE3B_DATABASE: str = os.getenv("LOCAL_NEO4J_PHASE3B_DATABASE", "local-ontology-b-graph")
    LOCAL_PATTERN: str = os.getenv("LOCAL_PATTERN", "**/*.txt")
    LOCAL_CHUNK_SIZE: int = int(os.getenv("LOCAL_CHUNK_SIZE", "256"))
    LOCAL_CHUNK_OVERLAP: int = int(os.getenv("LOCAL_CHUNK_OVERLAP", "32"))

    # Phase 2 MetaQA ablation: include 1-hop neighbour chunks (default on).
    INCLUDE_NEIGHBOURS: bool = os.getenv("INCLUDE_NEIGHBOURS", "1").strip().lower() not in ("0", "false", "no")
    # Optional suffix for results/<dataset>/phaseN<tag> (e.g. _no_neighbours),
    # so an ablation never overwrites the baseline results directory.
    RESULTS_TAG: str = os.getenv("RESULTS_TAG", "").strip().strip("_")
    # One chunk size for every MetaQA phase so the phases stay comparable
    METAQA_CHUNK_SIZE: int = int(os.getenv("METAQA_CHUNK_SIZE", "256"))
    METAQA_CHUNK_OVERLAP: int = int(os.getenv("METAQA_CHUNK_OVERLAP", "32"))
    # MetaQA graph must never touch the HotpotQA database; override to point at a
    # multi-database server (e.g. NEO4J_METAQA_DATABASE=metaqa-graph on NEO4J_URI)
    # or at the isolated docker service `neo4j-metaqa` (neo4j://127.0.0.1:7688).
    NEO4J_METAQA_URI: str = os.getenv("NEO4J_METAQA_URI", "") or os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_METAQA_DATABASE: str = os.getenv("NEO4J_METAQA_DATABASE", "metaqa-graph")

# Instantiate settings
settings = Settings()


@dataclass(frozen=True)
class Dataset:
    """Everything that differs between the HotpotQA and MetaQA tracks."""

    name: str
    eval_file: Path
    docs_dir: Path | None          # directory to glob (None -> use doc_list)
    doc_list: Path | None          # manifest with one doc path per line (metaqa)
    results_file: Path
    detailed_file: Path
    eval_results_file: Path
    eval_report_file: Path
    per_question_file: Path
    qdrant_collection: str
    neo4j_uri: str
    neo4j_database: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    tag_movies: bool = False       # prepend "[Movie: <title>]" to every chunk
    # "phase" = historical HotpotQA prompts; "shared" = one frozen template
    # (MetaQA tracks and the generic local track).
    prompt_style: str = "phase"
    # Directory ingest pattern (Phase 1/2/3); ignored when doc_list is set.
    file_pattern: str = "**/*.txt"
    # Phase 3 graph database; never the Phase 2 database.
    neo4j_ontology_database: str | None = None


def metaqa_database(name: str) -> str:
    """Configured Phase 2 graph database for a MetaQA variant (never HotpotQA)."""
    if name == "metaqa_mini":
        return settings.NEO4J_METAQA_MINI_DATABASE
    if name == "metaqa_mini30":
        return settings.NEO4J_METAQA_MINI30_DATABASE
    if name == "metaqa":
        return settings.NEO4J_METAQA_DATABASE
    raise ValueError(f"Unknown MetaQA track {name!r}")


def phase3_database(dataset: "Dataset") -> str:
    """Graph database a Phase 3 run must use (always distinct from Phase 2)."""
    return dataset.neo4j_ontology_database or dataset.neo4j_database


def phase3b_database(name: str) -> str:
    """Phase 3b graph database for a track (never the 3a or Phase 2 database)."""
    mapping = {
        "hotpot": settings.NEO4J_PHASE3B_DATABASE,
        "metaqa": settings.NEO4J_METAQA_PHASE3B_DATABASE,
        "metaqa_mini": settings.NEO4J_METAQA_MINI_PHASE3B_DATABASE,
        "metaqa_mini30": settings.NEO4J_METAQA_MINI30_PHASE3B_DATABASE,
        "local": settings.LOCAL_NEO4J_PHASE3B_DATABASE,
    }
    if name not in mapping:
        raise ValueError(f"Unknown Phase 3b track {name!r}")
    return mapping[name]


def phase3b_dataset(phase: int = 3) -> "Dataset":
    """Phase 3b paths: same corpus/eval/chunks as 3a, separate DB and results dir."""
    base = dataset_config(phase)
    if not base.tag_movies:
        raise SystemExit("Phase 3b is MetaQA-only for now (DATASET=metaqa|metaqa_mini|metaqa_mini30).")
    directory = REPO_ROOT / "results" / base.name / "phase3b"
    return replace(
        base,
        results_file=directory / "results.json",
        detailed_file=directory / "results_detailed.json",
        eval_results_file=directory / "eval_results.json",
        eval_report_file=directory / "eval_report.txt",
        per_question_file=directory / "per_question.jsonl",
        neo4j_ontology_database=phase3b_database(base.name),
    )


def dataset_config(phase: int = 1) -> Dataset:
    """Resolve paths and parameters for the active DATASET.

    DATASET=hotpot (default) returns exactly the historical paths, collections,
    chunk sizes and databases, so existing runs are unchanged. DATASET=metaqa
    switches to the frozen MetaQA corpus; DATASET=metaqa_mini is the same track on
    the hop-balanced mini slice (scripts/metaqa_make_mini.py), isolated stores.
    """
    phase_dir = PHASE_DIRS[phase]
    phase_default_top_k = 8 if phase == 1 else 5  # 8 = Phase 1 today, 5 = Phase 2 today

    if settings.DATASET in ("metaqa", "metaqa_mini", "metaqa_mini30"):
        # Experiment protocol: top_k=20 for BOTH phases on the MetaQA tracks.
        top_k = settings.TOP_K or 20
        name = settings.DATASET
        eval_file, doc_list = {
            "metaqa": (settings.METAQA_EVAL_FILE, settings.METAQA_DOC_LIST),
            "metaqa_mini": (settings.METAQA_MINI_EVAL_FILE, settings.METAQA_MINI_DOC_LIST),
            "metaqa_mini30": (settings.METAQA_MINI30_EVAL_FILE, settings.METAQA_MINI30_DOC_LIST),
        }[name]
        tag = f"_{settings.RESULTS_TAG}" if settings.RESULTS_TAG else ""
        results_dir = REPO_ROOT / "results" / name / f"phase{phase}{tag}"
        return Dataset(
            name=name,
            eval_file=eval_file,
            docs_dir=settings.METAQA_DOCS_DIR,
            doc_list=doc_list,
            results_file=results_dir / "results.json",
            detailed_file=results_dir / "results_detailed.json",
            eval_results_file=results_dir / "eval_results.json",
            eval_report_file=results_dir / "eval_report.txt",
            per_question_file=results_dir / "per_question.jsonl",
            qdrant_collection=f"{settings.QDRANT_COLLECTION_NAME}_{name}",
            neo4j_uri=settings.NEO4J_METAQA_URI,
            neo4j_database=metaqa_database(name),
            chunk_size=settings.METAQA_CHUNK_SIZE,
            chunk_overlap=settings.METAQA_CHUNK_OVERLAP,
            top_k=top_k,
            tag_movies=True,
            prompt_style="shared",
            neo4j_ontology_database={
                "metaqa": settings.NEO4J_METAQA_PHASE3_DATABASE,
                "metaqa_mini": settings.NEO4J_METAQA_MINI_PHASE3_DATABASE,
                "metaqa_mini30": settings.NEO4J_METAQA_MINI30_PHASE3_DATABASE,
            }[name],
        )

    if settings.DATASET == "local":
        if not settings.LOCAL_EVAL_FILE or not settings.LOCAL_DOCS_DIR:
            raise SystemExit(
                "DATASET=local needs LOCAL_DOCS_DIR and LOCAL_EVAL_FILE "
                "(eval JSONL in the hotpot/metaqa shape: question, answer, optional answers/"
                "supporting_facts_titles)."
            )
        tag = f"_{settings.RESULTS_TAG}" if settings.RESULTS_TAG else ""
        results_dir = REPO_ROOT / "results" / "local" / f"phase{phase}{tag}"
        return Dataset(
            name="local",
            eval_file=Path(settings.LOCAL_EVAL_FILE),
            docs_dir=Path(settings.LOCAL_DOCS_DIR),
            doc_list=None,
            results_file=results_dir / "results.json",
            detailed_file=results_dir / "results_detailed.json",
            eval_results_file=results_dir / "eval_results.json",
            eval_report_file=results_dir / "eval_report.txt",
            per_question_file=results_dir / "per_question.jsonl",
            qdrant_collection=settings.LOCAL_COLLECTION,
            neo4j_uri=settings.NEO4J_URI,
            neo4j_database=settings.LOCAL_NEO4J_DATABASE,
            chunk_size=settings.LOCAL_CHUNK_SIZE,
            chunk_overlap=settings.LOCAL_CHUNK_OVERLAP,
            top_k=settings.TOP_K or 8,
            tag_movies=False,
            prompt_style="shared",
            file_pattern=settings.LOCAL_PATTERN,
            neo4j_ontology_database=settings.LOCAL_NEO4J_PHASE3_DATABASE,
        )

    if settings.DATASET != "hotpot":
        raise ValueError(f"Unknown DATASET={settings.DATASET!r} "
                         "(expected 'hotpot', 'metaqa', 'metaqa_mini', 'metaqa_mini30' or 'local')")

    top_k = settings.TOP_K or phase_default_top_k
    chunk_size, chunk_overlap = (
        (settings.CHUNK_SIZE_HOTPOT, settings.CHUNK_OVERLAP_HOTPOT) if phase == 1
        else (settings.CHUNK_SIZE_GRAPH, settings.CHUNK_OVERLAP_GRAPH)
    )
    return Dataset(
        name="hotpot",
        # Relative paths on purpose: SimpleDirectoryReader stores file_path in node
        # metadata, and the HotpotQA embeddings must stay byte-identical to before.
        eval_file=Path("hotpot_eval.jsonl"),
        docs_dir=Path("data_hotpot"),
        doc_list=None,
        results_file=Path(phase_dir) / "hotpot_results.json",
        detailed_file=Path(phase_dir) / "hotpot_results_detailed.json",
        eval_results_file=Path(phase_dir) / "hotpot_eval_results.json",
        eval_report_file=Path(phase_dir) / "hotpot_eval_report.txt",
        per_question_file=Path(phase_dir) / "hotpot_per_question.jsonl",
        qdrant_collection=settings.QDRANT_COLLECTION_NAME,
        neo4j_uri=settings.NEO4J_URI,
        neo4j_database=settings.NEO4J_DATABASE,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        prompt_style="phase",
        file_pattern="**/*.txt",
        neo4j_ontology_database=settings.NEO4J_PHASE3_DATABASE,
    )


def closed_book_dataset(phase: int = 1) -> Dataset:
    """Dataset paths for the closed-book control: results/<dataset>/closed_book/.

    Only the MetaQA tracks support it (the shared prompt + no retrieval); HotpotQA
    keeps its historical behavior and raises here.
    """
    dataset = dataset_config(phase)
    if not dataset.tag_movies:
        raise SystemExit(
            "Closed-book mode is only implemented for the MetaQA tracks "
            "(DATASET=metaqa or DATASET=metaqa_mini)."
        )
    directory = REPO_ROOT / "results" / dataset.name / "closed_book"
    return replace(
        dataset,
        results_file=directory / "results.json",
        detailed_file=directory / "results_detailed.json",
        eval_results_file=directory / "eval_results.json",
        eval_report_file=directory / "eval_report.txt",
        per_question_file=directory / "per_question.jsonl",
    )


def set_top_k(value: int) -> None:
    """Override retrieval depth for this process (used by the runner's --top-k)."""
    settings.TOP_K = int(value)


def set_include_neighbours(value: bool) -> None:
    """Phase 2 ablation switch (used by the runner's --no-neighbours)."""
    settings.INCLUDE_NEIGHBOURS = bool(value)


def write_ingest_meta(dataset, phase: int, **payload) -> Path:
    """Persist what an ingest actually built (read back by the answer runner)."""
    import json
    from datetime import datetime, timezone

    path = Path(dataset.results_file).parent / "ingest_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": dataset.name,
        "phase": phase,
        "top_k": dataset.top_k,
        "chunk_size": dataset.chunk_size,
        "chunk_overlap": dataset.chunk_overlap,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Ingest metadata saved to {path}")
    return path


def check_neo4j_target(uri: str, database: str, user: str | None = None, password: str | None = None) -> None:
    """Print SHOW DATABASES and refuse to run MetaQA against the HotpotQA database."""
    from neo4j import GraphDatabase

    user = user or settings.NEO4J_USER
    password = password or settings.NEO4J_PASSWORD
    if database == settings.NEO4J_DATABASE and uri == settings.NEO4J_URI:
        raise SystemExit(
            f"Refusing to run: MetaQA is pointed at the HotpotQA database "
            f"({uri} / {database}). Set NEO4J_METAQA_URI/NEO4J_METAQA_DATABASE, or start the "
            f"isolated service with `docker compose up -d neo4j-metaqa` (neo4j://127.0.0.1:7688)."
        )
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        with driver.session(database="system") as session:
            names = [record["name"] for record in session.run("SHOW DATABASES YIELD name")]
        driver.close()
    except Exception as error:  # noqa: BLE001 - surfaced as a clear hint below
        raise SystemExit(
            f"Neo4j not reachable at {uri}: {error}\n"
            f"MetaQA needs its own database. Either create it on your server "
            f"(NEO4J_METAQA_DATABASE={database}) or use the isolated service: "
            f"`docker compose up -d neo4j-metaqa` and NEO4J_METAQA_URI=neo4j://127.0.0.1:7688."
        ) from error
    print(f"Neo4j {uri}: databases = {names}")
    if database not in names:
        raise SystemExit(
            f"Database {database!r} does not exist on {uri} (available: {names}). "
            f"Create it (multi-database servers: `CREATE DATABASE {database}`), set "
            f"NEO4J_METAQA_DATABASE to one of the existing databases, or use the isolated "
            f"service `neo4j-metaqa` (neo4j://127.0.0.1:7688, database `neo4j`)."
        )

# Global Client initializations for Phase 1
def get_qdrant_client() -> QdrantClient:
    """Returns an active Qdrant client instance."""
    return QdrantClient(url=settings.QDRANT_URL)

def get_openai_client() -> OpenAI:
    """Returns an active OpenAI client instance using the loaded API key."""
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing from the environment configuration.")
    return OpenAI(api_key=settings.OPENAI_API_KEY)

def llm_route(model: str | None = None) -> dict[str, str | None]:
    """Return the API key/base for a model, routing OpenRouter by model prefix.

    A model is routed to OpenRouter only when it starts with ``openrouter/``.
    All existing OpenAI/Luna and Maritaca paths keep their previous behavior.
    Key values are never printed or written to run metadata.
    """
    model = model or settings.GENERATION_MODEL
    if model.startswith("openrouter/"):
        if not settings.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; cannot route "
                f"{model!r} through OpenRouter."
            )
        return {"api_key": settings.OPENROUTER_API_KEY,
                "api_base": settings.OPENROUTER_API_BASE}
    return {"api_key": settings.MARITACA_API_KEY or None,
            "api_base": settings.MARITACA_API_BASE or None}


def get_llamaindex_settings():
    """Configures LlamaIndex global settings for LLM and Embeddings."""
    route = llm_route(settings.GENERATION_MODEL)
    if settings.GENERATION_MODEL.startswith("openrouter/"):
        # LlamaIndex's OpenAI wrapper expects the model without the provider prefix.
        # The routed LLM is only used by LlamaIndex internals; answer generation
        # goes through shared.llm.generate_completion.
        model = settings.GENERATION_MODEL.split("/", 1)[1]
        llm = LlamaOpenAI(model=model, api_key=route["api_key"],
                          api_base=route["api_base"])
    else:
        llm = LlamaOpenAI(model=settings.GENERATION_MODEL, api_key=settings.OPENAI_API_KEY)
    embed_model = OpenAIEmbedding(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY, embed_batch_size=100)
    return llm, embed_model


def describe_datasets() -> None:
    """Print the resolved config for every phase (python -m shared.config)."""
    for phase in (1, 2, 3):
        dataset = dataset_config(phase)
        if dataset.doc_list and dataset.doc_list.exists():
            docs = f"{len([l for l in dataset.doc_list.read_text(encoding='utf-8').splitlines() if l.strip()])} (manifest)"
        elif dataset.docs_dir and dataset.docs_dir.exists():
            docs = f"{len(list(dataset.docs_dir.glob('**/*.txt')))} (glob)"
        else:
            docs = "n/a"
        rows = (f"{len([l for l in dataset.eval_file.read_text(encoding='utf-8').splitlines() if l.strip()])}"
                if dataset.eval_file.exists() else "MISSING")
        print(f"phase {phase}: dataset={dataset.name}")
        print(f"  eval       : {dataset.eval_file} ({rows} rows)")
        print(f"  docs       : {docs}")
        print(f"  qdrant     : {dataset.qdrant_collection}")
        print(f"  neo4j      : {dataset.neo4j_uri} / {dataset.neo4j_database}")
        print(f"  neo4j (p3) : {dataset.neo4j_uri} / {phase3_database(dataset)}")
        if phase == 3 and dataset.tag_movies:
            print(f"  neo4j (p3b): {dataset.neo4j_uri} / {phase3b_database(dataset.name)}")
        print(f"  chunk      : {dataset.chunk_size}/{dataset.chunk_overlap} | top_k={dataset.top_k} | "
              f"tag_movies={dataset.tag_movies} | prompt={dataset.prompt_style}")
        print(f"  results    : {dataset.results_file.parent}")


if __name__ == "__main__":
    describe_datasets()