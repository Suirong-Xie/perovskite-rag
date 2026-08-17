#!/data1/perovskite-rag/.RAGenv/bin/python3
"""
PerovskiteGPT v1.5 — Sunny-RAG 科研 Agent（模块化架构）
架构:
  - core/    → 配置、LLM 抽象、数据模型
  - services/ → 会话、检索、翻译、标注、Agent 循环
  - routers/  → API 路由（chat, sessions, papers）
  - web_ui.html → 前端

与 v4（server/v4/server.py）兼容并存，app 默认端口 8002。
"""
import os
import sys
from contextlib import asynccontextmanager

# 确保 server/ 在 sys.path，使 app 可作为包导入
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import APP_DIR as APP_PATH, PAPERS_DIR
from app.services.session_store import store
from app.routers import chat, sessions, papers


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield
    store._save()


app = FastAPI(
    title="PerovskiteGPT v1.5",
    version="1.5.0",
    description="Sunny-RAG 科研 Agent — 模块化架构",
    lifespan=lifespan,
)

# ── 注册路由 ──
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(papers.router)

# 静态资源
app.mount("/static", StaticFiles(directory=str(APP_PATH / "static")), name="static")


# ── 前端 ──
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = APP_PATH / "web_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>PerovskiteGPT v1.5</h1><p>web_ui.html 尚未创建</p>")


# ── 健康检查 ──
@app.get("/api/health")
async def health():
    return {
        "version": "1.5.0",
        "status": "ok",
        "sessions": len(store.sessions),
    }


# ── 启动 ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PerovskiteGPT v1.5 Server")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"🚀 PerovskiteGPT v1.5 starting on http://{args.host}:{args.port}")
    print(f"📂 Config from: .env, core/config.py")
    print(f"📚 Papers directory: {PAPERS_DIR}")
    print(f"🧠 LLM Gateway: openclaw/sunny")
    uvicorn.run(app, host=args.host, port=args.port)
