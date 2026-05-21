import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    # Database Configs
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "testpassword123")
    
    # Models & Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gpt-5-mini")
    
    # RAG parameters
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_phase_1_baseline")

# Instantiate settings
settings = Settings()

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
    embed_model = OpenAIEmbedding(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
    return llm, embed_model