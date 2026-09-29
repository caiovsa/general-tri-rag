import os
from dataclasses import dataclass
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

class Settings:
    # Database Configs
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "testpass123")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "hotpot-graph")
    
    # Models & Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
    EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "gpt-5-mini")

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


def dataset_config(phase: int = 1) -> Dataset:
    """Resolve paths and parameters for the active DATASET.

    DATASET=hotpot (default) returns exactly the historical paths, collections,
    chunk sizes and databases, so existing runs are unchanged. DATASET=metaqa
    switches to the frozen MetaQA corpus, isolated stores and results/ dir.
    """
    phase_dir = PHASE_DIRS[phase]
    phase_default_top_k = 8 if phase == 1 else 5  # 8 = Phase 1 today, 5 = Phase 2 today
    top_k = settings.TOP_K or phase_default_top_k

    if settings.DATASET == "metaqa":
        results_dir = REPO_ROOT / "results" / "metaqa" / f"phase{phase}"
        return Dataset(
            name="metaqa",
            eval_file=settings.METAQA_EVAL_FILE,
            docs_dir=settings.METAQA_DOCS_DIR,
            doc_list=settings.METAQA_DOC_LIST,
            results_file=results_dir / "results.json",
            detailed_file=results_dir / "results_detailed.json",
            eval_results_file=results_dir / "eval_results.json",
            eval_report_file=results_dir / "eval_report.txt",
            per_question_file=results_dir / "per_question.jsonl",
            qdrant_collection=f"{settings.QDRANT_COLLECTION_NAME}_metaqa",
            neo4j_uri=settings.NEO4J_METAQA_URI,
            neo4j_database=settings.NEO4J_METAQA_DATABASE,
            chunk_size=settings.METAQA_CHUNK_SIZE,
            chunk_overlap=settings.METAQA_CHUNK_OVERLAP,
            top_k=top_k,
            tag_movies=True,
        )

    if settings.DATASET != "hotpot":
        raise ValueError(f"Unknown DATASET={settings.DATASET!r} (expected 'hotpot' or 'metaqa')")

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
    )


def set_top_k(value: int) -> None:
    """Override retrieval depth for this process (used by the runner's --top-k)."""
    settings.TOP_K = int(value)


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

def get_llamaindex_settings():
    """Configures LlamaIndex global settings for LLM and Embeddings."""
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
        print(f"  chunk      : {dataset.chunk_size}/{dataset.chunk_overlap} | top_k={dataset.top_k} | tag_movies={dataset.tag_movies}")
        print(f"  results    : {dataset.results_file.parent}")


if __name__ == "__main__":
    describe_datasets()