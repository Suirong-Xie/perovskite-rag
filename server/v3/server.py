"""PerovskiteGPT v3 — FastAPI server entry point.

Modular Agentic RAG system with tool-using orchestration.

Files:
  config.py       → Configuration
  models.py       → LLM + Embedding clients
  vector_store.py → Qdrant + retriever
  sessions.py     → Session persistence + LRU cache
  tools.py        → Tool definitions + execution
  agent.py        → Agent orchestration loop
  prompts.py      → System prompts
  web_ui.html     → Standalone HTML UI (edit independently!)
  server.py       → This file: FastAPI app + API routes
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import HOST, PORT, STATIC_DIR, LOG_DIR, LOG_LEVEL
from sessions import list_all_sessions, load_session, delete_session, rename_session
from agent import run_agent, run_agent_stream

# ── Logging ──
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "perovskitegpt.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Pydantic models ──

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class RenameRequest(BaseModel):
    title: str


# ── Load HTML (separate file for easy editing) ──

def load_html() -> str:
    """Load HTML UI from external file."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_ui.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()





# ── FastAPI app ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PerovskiteGPT v3 starting up...")
    yield
    logger.info("PerovskiteGPT v3 shutting down.")

app = FastAPI(title="PerovskiteGPT v3 — Agentic RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Routes ──

@app.get("/")
async def get_web_ui():
    return HTMLResponse(content=load_html())


@app.post("/ask")
async def ask_expert(req: QueryRequest):
    """Non-streaming Q&A."""
    session_id = req.session_id or f"session_{os.urandom(8).hex()}"
    logger.info(f"POST /ask | session={session_id} | q={req.question[:80]}")
    result = run_agent(req.question, session_id)
    return {
        "answer": result.get("result", ""),
        "sources": result.get("sources", []),
        "session_id": session_id,
    }


@app.post("/ask/stream")
async def ask_expert_stream(req: QueryRequest):
    """Streaming Q&A via SSE."""
    session_id = req.session_id or f"session_{os.urandom(8).hex()}"
    logger.info(f"POST /ask/stream | session={session_id} | q={req.question[:80]}")
    return StreamingResponse(
        run_agent_stream(req.question, session_id),
        media_type="text/event-stream",
    )


@app.get("/sessions")
async def list_sessions_api():
    """List all conversation sessions."""
    raw = list_all_sessions()
    sessions = []
    for s in raw:
        sessions.append({
            "session_id": s["id"],
            "title": s["title"],
            "history": [{"question": m.get("content","")} for m in (load_session(s["id"]) if len(load_session(s["id"])) > 0 else [])],
            "last_accessed": s["mtime"],
        })
    return {"sessions": sessions}


@app.get("/session/{session_id}/history")
async def get_session_history_api(session_id: str):
    """Get conversation history."""
    history = load_session(session_id)
    turns = []
    i = 0
    while i < len(history):
        entry = history[i]
        if entry.get("_type") == "title":
            i += 1
            continue
        if entry.get("role") == "user":
            turn = {"question": entry.get("content", "")}
            if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                turn["answer"] = history[i + 1].get("content", "")
                turn["sources"] = history[i + 1].get("sources", [])
                i += 1
            else:
                turn["answer"] = ""
                turn["sources"] = []
            turns.append(turn)
        i += 1
    return {"history": turns}


@app.delete("/session/{session_id}")
async def delete_session_api(session_id: str):
    """Delete a session."""
    delete_session(session_id)
    return {"status": "deleted"}


@app.post("/session/{session_id}/rename")
async def rename_session_api(session_id: str, req: RenameRequest):
    """Rename a session."""
    rename_session(session_id, req.title)
    return {"status": "renamed"}


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting PerovskiteGPT v3 on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
