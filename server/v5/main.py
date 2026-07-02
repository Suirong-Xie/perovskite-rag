#!/data1/perovskite-rag/.RAGenv/bin/python3
"""
PerovskiteGPT V5 — Sunny-RAG 科研 Agent（模块化架构）
架构:
  - core/    → 配置、LLM 抽象、数据模型
  - services/ → 会话、检索、翻译、标注、Agent 循环
  - routers/  → API 路由（chat, sessions, papers）
  - web_ui.html → 前端

与 v4（server/v4/server.py）兼容并存，v5 默认端口 8002。
"""
import os
import sys
from contextlib import asynccontextmanager

# 确保 server/ 在 sys.path，使 v5 可作为包导入
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from v5.core.config import V5_DIR as V5_PATH, PAPERS_DIR
from v5.services.session_store import store
from v5.routers import chat, sessions, papers


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield
    store._save()


app = FastAPI(
    title="PerovskiteGPT V5",
    version="5.0.0",
    description="Sunny-RAG 科研 Agent — 模块化架构",
    lifespan=lifespan,
)

# ── 注册路由 ──
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(papers.router)


# ── 前端 ──
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = V5_PATH / "web_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>PerovskiteGPT V5</h1><p>web_ui.html 尚未创建</p>")


# ── 健康检查 ──
@app.get("/api/health")
async def health():
    return {
        "version": "5.0.0",
        "status": "ok",
        "sessions": len(store.sessions),
    }


# ── 启动 ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PerovskiteGPT V5 Server")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"🚀 PerovskiteGPT V5 starting on http://{args.host}:{args.port}")
    print(f"📂 Config from: .env, core/config.py")
    print(f"📚 Papers directory: {PAPERS_DIR}")
    print(f"🧠 LLM Gateway: openclaw/sunny")
    uvicorn.run(app, host=args.host, port=args.port)
