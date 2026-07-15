"""
Configuration loader for the RAG Demo Application.
Reads settings from .env file and exposes them as module-level constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load .env ───
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)

# --- LLM Config ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")          # "ollama" or "groq"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "gpt-oss-120b")

# ─── Embedding ───
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Lajavaness/bilingual-embedding-large")

# ─── Qdrant ───
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_system")

# ─── RAG parameters ───
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))

# ─── Paths ───
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
ASSETS_DIR = PROJECT_DIR / "assets"
