"""Configuration for PerovskiteGPT v3 - Multi-query + MMR diversity."""

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
QDRANT_TOP_K_DEFAULT = 15

# ── RAG Parameters ──
LLM_TEMPERATURE = 0.7
LLM_REPEAT_PENALTY = 1.05
LLM_NUM_PREDICT = -1  # unlimited

# ── Session ──
MAX_HISTORY_ROUNDS = 10
LRU_CACHE_SIZE = 50
# ── v3: Multi-query ──
MULTI_QUERY_ENABLED = False         # 是否启用多 query 扩展
MULTI_QUERY_COUNT = 3               # 单个问题拆成几个子 query
MAX_CONTEXT_SOURCES = 20            # 最终喂给 LLM 的最大来源数

# ── v3: MMR diversity ──
MMR_ENABLED = True                  # 是否启用 MMR 多样性重排
MMR_LAMBDA = 0.6                    # MMR 平衡参数：0.6 = 相关度优先, 0.3 = 多样性优先
MMR_CANDIDATES = 30                 # MMR 候选池大小

# ── v3: Context ──
DEFAULT_TOP_K = 15                  # 默认检索数量（从 10 提升到 15）
ABSTRACT_EXPAND_COUNT = 5           # Auto-expand 读取的论文摘要数量（从 3 提升到 5）
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
