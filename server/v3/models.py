"""LLM and embedding clients — thin wrappers around Ollama."""

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from config import (
    OLLAMA_HOST_GENERATION,
    OLLAMA_HOST_EMBEDDING,
    GENERATION_MODEL,
    EMBEDDING_MODEL,
    LLM_TEMPERATURE,
    LLM_REPEAT_PENALTY,
    LLM_NUM_PREDICT,
)

# ── Embedding model (mxbai-embed-large on GPU 1, port 11435) ──
embed_model = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_HOST_EMBEDDING,
)

# ── Generation model (llama3:70b on GPU 0, port 11434) ──
llm = OllamaLLM(
    model=GENERATION_MODEL,
    base_url=OLLAMA_HOST_GENERATION,
    temperature=LLM_TEMPERATURE,
    repeat_penalty=LLM_REPEAT_PENALTY,
    num_predict=LLM_NUM_PREDICT,
)
