import os
import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Set
from dotenv import load_dotenv

# Keep using LlamaIndex just for fast PDF text extraction
from llama_index.core import SimpleDirectoryReader

# LightRAG Imports
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Load your existing .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ----------------------------------------------------------------------------
# Setup Neo4j for LightRAG (It reads these from env vars automatically)
# ----------------------------------------------------------------------------
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
os.environ["NEO4J_USER"] = os.getenv("NEO4J_USER", "neo4j")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USER", "neo4j")  # Fallback for some versions
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD", "testpass123")
os.environ["NEO4J_DATABASE"] = os.getenv("NEO4J_DATABASE", "finance-bench")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

WORKING_DIR = "./lightrag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

DATA_DIR = Path("data_finance")
CHECKPOINT_FILE = Path("checkpoint_lightrag.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Checkpoint Helpers
# ----------------------------------------------------------------------------
def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data.get("processed", [])), set(data.get("failed", []))
    return set(), set()


def save_checkpoint(processed: Set[str], failed: Set[str]):
    CHECKPOINT_FILE.write_text(
        json.dumps(
            {
                "processed": sorted(processed),
                "failed": sorted(failed),
                "last_saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )


# ----------------------------------------------------------------------------
# Main Ingestion
# ----------------------------------------------------------------------------
async def run_ingestion():
    setup_logger("lightrag", level="INFO")

    # 1. Initialize LightRAG with Neo4j and OpenAI
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="Neo4JStorage",
        chunk_token_size=4000,
        chunk_overlap_token_size=200,
        llm_model_max_async=16,
        embedding_func_max_async=32,
    )

    # Required initialization steps for LightRAG
    await rag.initialize_storages()
    await initialize_pipeline_status()

    processed, failed = set(), set()
    try:
        # 2. Discover files
        all_files = sorted(str(p) for p in DATA_DIR.glob("*.pdf"))
        processed, failed = load_checkpoint()
        pending = [f for f in all_files if f not in processed and f not in failed]

        log.info(
            f"Total: {len(all_files)} | Done: {len(processed)} | Remaining: {len(pending)} | Failed: {len(failed)}"
        )
        if not pending:
            log.info("All files already ingested!")
            return

        # 3. Process sequentially (LightRAG handles internal batching safely without triggering 429s)
        for i, file_path in enumerate(pending):
            try:
                log.info(f"Processing [{i+1}/{len(pending)}]: {Path(file_path).name}")

                # Use LlamaIndex just for fast PDF text extraction
                docs = await asyncio.to_thread(
                    lambda fp=file_path: SimpleDirectoryReader(input_files=[fp]).load_data()
                )
                if not docs:
                    log.warning(f"No text extracted from {Path(file_path).name}")
                    processed.add(file_path)
                    continue

                text = "\n".join([doc.text for doc in docs])

                # LightRAG handles chunking, entity extraction, and Neo4j insertion internally
                await rag.ainsert(text)
                processed.add(file_path)
                log.info(f"✓ Success: {Path(file_path).name}")

            except Exception as e:
                log.error(f"✗ Failed {Path(file_path).name}: {e}")
                failed.add(file_path)

            # Checkpoint every 5 files
            if (i + 1) % 5 == 0:
                save_checkpoint(processed, failed)

    finally:
        # Always save final state and cleanly close connections
        save_checkpoint(processed, failed)
        await rag.finalize_storages()
        log.info("=== INGESTION COMPLETE ===")
        log.info(f"Processed: {len(processed)} | Failed: {len(failed)}")


if __name__ == "__main__":
    asyncio.run(run_ingestion())