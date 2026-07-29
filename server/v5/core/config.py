"""
PerovskiteGPT V5 — 统一配置管理
所有配置从环境变量读取，支持 .env 文件
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env（如果存在）
_dotenv_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)

# ── 路径 ──
BASE_DIR = Path(os.getenv("PEROVSKITE_RAG_BASE", "/data1/perovskite-rag"))
V5_DIR = BASE_DIR / "server/v5"
VECTOR_DB_DIR = BASE_DIR / "data/vector_db"
SESSIONS_FILE = V5_DIR / "sessions.json"
SESSIONS_DIR = V5_DIR / "sessions"
ANNOTATED_DIR = V5_DIR / "annotated_pdfs"

# ── PDF 论文目录 ──
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", "/data/data/pkb/01_raw_data/papers_pdf"))
JOURNALS_PDF_DIR = Path(os.getenv("JOURNALS_PDF_DIR", "/data/data/pkb/01_raw_data/journals_pdf"))

# ── Ollama 嵌入 ──
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://127.0.0.1:11435/api/embed")
OLLAMA_EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")

# ── LLM 后端选择 ──
# "deepseek"  → DeepSeek API（默认）
# "openclaw"  → OpenClaw Gateway（sunny agent，需本地部署）
LLM_BACKEND = os.getenv("LLM_BACKEND", "deepseek")

# ── OpenClaw Gateway ──
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
OPENCLAW_MODEL = os.getenv("OPENCLAW_MODEL", "openclaw/sunny")

# ── DeepSeek API ──
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── 检索配置 ──
SEARCH_DEFAULT_TOP_K = int(os.getenv("SEARCH_DEFAULT_TOP_K", "5"))

# ── Semantic Scholar ──
S2_API_KEY = os.getenv("S2_API_KEY", "s2k-O06rEcAcYIxe89rzq7TekcIymBh6XrbMmevynf2y")

# ── S2 向量库 (Phase 3.5-4) ──
S2_VECTOR_DB_DIR = BASE_DIR / "data/s2_vector_db"
S2_CORPUS_DIR = BASE_DIR / "data/s2_corpus"
S2_ENABLED = os.path.isdir(S2_VECTOR_DB_DIR) and os.path.isfile(S2_VECTOR_DB_DIR / "vectors.npy")
S2_RESULT_WEIGHT = float(os.getenv("S2_RESULT_WEIGHT", "0.85"))
S2_CITATION_BOOST_FACTOR = float(os.getenv("S2_CITATION_BOOST_FACTOR", "0.1"))

# ── Agent 配置 ──
AGENT_MAX_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "10"))  # 安全阀，状态机下不太会触达

# Agent 状态机预算
AGENT_STATE_BUDGETS = {
    "retrieve_llm": int(os.getenv("AGENT_RETRIEVE_LLM", "2")),        # RETRIEVE 阶段最多 LLM 决策轮
    "retrieve_search": int(os.getenv("AGENT_RETRIEVE_SEARCH", "2")),  # RETRIEVE 阶段最多搜索次数
    "quick_read": int(os.getenv("AGENT_QUICK_READ", "4")),            # QUICK_READ 最多读 4 篇
    "deep_read": int(os.getenv("AGENT_DEEP_READ", "4")),              # DEEP_READ 最多读 4 篇
}

# RETRIEVE → READ 最少相关论文数
AGENT_MIN_RELEVANT_PAPERS = int(os.getenv("AGENT_MIN_RELEVANT_PAPERS", "3"))
