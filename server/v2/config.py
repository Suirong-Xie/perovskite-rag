"""Configuration for PerovskiteGPT v2 - all settings in one place."""

import os
from pathlib import Path

# ── Paths ──
BASE_DIR = Path("/data1/perovskite-rag")
DATA_DIR = BASE_DIR / "data"
SESSION_DIR = DATA_DIR / "perovskite_sessions"
QDRANT_PATH = str(DATA_DIR / "qdrant_data")
CHUNKED_DATA_PATH = str(DATA_DIR / "chunked_data" / "chunks.jsonl")
STATIC_DIR = BASE_DIR / "static"
LOG_DIR = Path("/var/log/perovskitegpt")

# ── Ollama ──
OLLAMA_HOST_GENERATION = "http://127.0.0.1:11434"
OLLAMA_HOST_EMBEDDING = "http://127.0.0.1:11435"

GENERATION_MODEL = "llama3-70b-gpu"
EMBEDDING_MODEL = "mxbai-embed-large"

# ── Qdrant ──
QDRANT_COLLECTION = "perovskite_papers"
QDRANT_TOP_K_MIN = 0
QDRANT_TOP_K_MAX = 20
QDRANT_TOP_K_DEFAULT = 10

# ── RAG Parameters ──
LLM_TEMPERATURE = 0.7
LLM_REPEAT_PENALTY = 1.05
LLM_NUM_PREDICT = -1  # unlimited

# ── Session ──
MAX_HISTORY_ROUNDS = 10
LRU_CACHE_SIZE = 50

# ── Server ──
HOST = "0.0.0.0"
PORT = 8000

# ── Logging ──
LOG_LEVEL = "INFO"

# ── Ensure dirs exist ──
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
