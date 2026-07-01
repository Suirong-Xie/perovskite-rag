#!/data1/perovskite-rag/.RAGenv/bin/python3
"""
PerovskiteGPT V4 - Sunny-RAG Web Server
架构:
  - 生成任务(job)和输出通道(SSE)完全解耦
  - /api/chat 异步启动生成任务,立即返回 task_id
  - /api/chat/{task_id}/stream?offset=N 从任意位置订阅输出
  - 多个 SSE 连接可以同时消费同一个任务
"""
import json, time, uuid, asyncio, subprocess, re, os
from pathlib import Path
from collections import defaultdict
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
import httpx

# ── 路径 ──
BASE_DIR = Path("/data1/perovskite-rag")
V4_DIR = BASE_DIR / "server/v4"
SUNNY_RAG_DIR = BASE_DIR / "sunny-rag"
PAPERS_DIR = Path("/data/data/pkb/01_raw_data/papers_pdf")
DATA_DIR = SUNNY_RAG_DIR / "data"
OLLAMA_EMBED_URL = "http://127.0.0.1:11435/api/embed"
EMBED_MODEL = "mxbai-embed-large"
SESSIONS_FILE = V4_DIR / "sessions.json"
SESSIONS_DIR = V4_DIR / "sessions"

# ── 会话存储 ──
sessions = {}
session_order = []

# ── 搜索缓存 ──
_SEARCH_CACHE = {}  # cleared on reload
def _column_idx(lb, page_width):
    """Assign a rect to a column index (0=left, 1=right) based on bbox midpoint."""
    mid = (lb[0] + lb[2]) / 2
    col_w = page_width / 2
    return int(mid // col_w)

def _same_column(lb1, lb2, page_width):
    """Check if two line bboxes are in the same column."""
    return _column_idx(lb1, page_width) == _column_idx(lb2, page_width)

# Annotated PDF storage (deprecated, kept for cleanup)
ANNOTATED_DIR = V4_DIR / "annotated_pdfs"

def annotate_pdf(pdf_path: str, chunk_texts: list) -> str:
    """在 PDF 文件中搜索 chunk 文本并写入原生高亮标注。
    策略：对一个 chunk，用其文本中所有可能的子句和片段搜索 PDF。
    对找到的所有匹配矩形，按页面分组，合并重叠矩形，
    最终只保留合并后的连续大矩形（宽度>80pt, 高度合理的）。
    不修改原始 PDF，标注后的副本保存到 annotated_pdfs/ 目录。"""
    import fitz
    import shutil
    import tempfile
    import re
    from pathlib import Path
    
    ANNOTATED_DIR = V4_DIR / "annotated_pdfs"
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    
    stem = Path(pdf_path).stem
    out_path = str(ANNOTATED_DIR / f"{stem}_annotated.pdf")
    
    if os.path.exists(out_path):
        log(f"[ANNOTATE] Cache hit for {stem}")
        return out_path
    
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    shutil.copy2(pdf_path, tmp_path)
    
    doc = fitz.open(tmp_path)
    total_annotations = 0
    
    def normalize(s):
        return " ".join(s.replace("\\n", " ").replace("\\r", " ").split())
    
    for chunk in chunk_texts:
        text = chunk.strip()
        if not text:
            continue
        
        flat = normalize(text)
        words = flat.split()
        
        # 按页收集所有匹配矩形
        page_rects = {}  # pg -> [(y0, y1, x0, x1), ...]
        
        # 生成候选查询（短句到长句）
        candidates = set()
        # 10 词滑动窗口
        for i in range(0, len(words), max(1, len(words)//12)):
            frag = " ".join(words[i:i+min(14, len(words)-i)])
            if len(frag) > 20:
                candidates.add(frag[:180])
        # 每个完整句
        for sent in re.split(r'[.!?]', flat):
            s = sent.strip()
            if len(s) > 30:
                candidates.add(s[:200])
        # 完整文本
        candidates.add(flat[:250])
        
        for q in candidates:
            qq = " ".join(q.split())
            if len(qq) < 20:
                continue
            for pg in range(len(doc)):
                for r in doc[pg].search_for(qq):
                    page_rects.setdefault(pg, []).append((r.y0, r.y1, r.x0, r.x1))
        
        if not page_rects:
            continue
        
        # 对每页合并重叠/相邻矩形
        for pg, rects in page_rects.items():
            page_h = doc[pg].rect.height
            rects.sort()
            
            # 用 y 轴交集来合并相邻行
            merged = []
            for y0, y1, x0, x1 in rects:
                # 过滤页眉页脚
                if y0 < 100 or y1 > page_h - 80:
                    continue
                found = False
                for i, (my0, my1, mx0, mx1) in enumerate(merged):
                    # y 重叠或间距 < 15pt
                    if not (y1 + 15 < my0 or y0 - 15 > my1):
                        merged[i] = (min(my0, y0), max(my1, y1), min(mx0, x0), max(mx1, x1))
                        found = True
                        break
                if not found:
                    merged.append((y0, y1, x0, x1))
            
            # 标注合并结果
            for y0, y1, x0, x1 in merged:
                w = x1 - x0
                if w < 60:
                    continue
                rect = fitz.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1)
                doc[pg].add_highlight_annot(rect)
                total_annotations += 1
    
    if total_annotations == 0:
        doc.close()
        os.remove(tmp_path)
        log(f"[ANNOTATE] No text found for {stem}")
        return ""
    
    doc.save(out_path, incremental=False, garbage=4, deflate=True)
    doc.close()
    os.remove(tmp_path)
    log(f"[ANNOTATE] Saved annotated PDF: {out_path} ({total_annotations} merged highlights)")
    return out_path

# ── 文章库缓存 ──
_papers_cache = None

# ── 模型配置 ──

OPENCLAW_GATEWAY = "http://localhost:18789"
OPENCLAW_GATEWAY_TOKEN = "363d7724aa2028011a2f28a2abb534517cce347060c8515a"
OPENCLAW_MODEL = "openclaw/sunny"  # Sunny agent
# ── 生成任务管理 ──
# task_id → { "sid": str, "chunks": [str], "done": bool, "error": str|None }
_tasks = {}
_tasks_lock = asyncio.Lock()

def log(msg: str):
    print(f"[V4] {msg}", flush=True)

# ── Session 工具函数 ──

def load_sessions():
    global sessions, session_order
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if SESSIONS_FILE.exists():
        with open(SESSIONS_FILE) as f:
            data = json.load(f)
        sessions_in = data.get("sessions", {})
        session_order = data.get("order", [])
        migrated = False
        sessions = {}
        for sid, s in sessions_in.items():
            if "messages" in s and s["messages"]:
                session_dir = SESSIONS_DIR / sid
                session_dir.mkdir(parents=True, exist_ok=True)
                history_file = session_dir / "history.json"
                if not history_file.exists():
                    with open(history_file, "w") as f:
                        json.dump(s["messages"], f, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                sessions[sid] = {"title": s.get("title", ""), "message_count": len(s["messages"])}
                migrated = True
            else:
                session_dir = SESSIONS_DIR / sid
                history_file = session_dir / "history.json"
                mc = s.get("message_count", 0)
                if history_file.exists():
                    with open(history_file) as f:
                        mc = len(json.load(f))
                sessions[sid] = {"title": s.get("title", ""), "message_count": mc}
        if migrated:
            save_sessions()
    else:
        sessions = {}

def save_sessions():
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for sid, s in sessions.items():
        summary[sid] = {"title": s.get("title", ""), "message_count": s.get("message_count", 0)}
    with open(SESSIONS_FILE, "w") as f:
        json.dump({"sessions": summary, "order": session_order}, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

def append_message(sid: str, role: str, content: str):
    session_dir = SESSIONS_DIR / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    history_file = session_dir / "history.json"
    msgs = []
    if history_file.exists():
        with open(history_file) as f:
            msgs = json.load(f)
    msgs.append({"role": role, "content": content})
    with open(history_file, "w") as f:
        json.dump(msgs, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    if sid not in sessions:
        sessions[sid] = {"title": content[:50] if role == "user" else "", "message_count": 0}
    sessions[sid]["message_count"] = len(msgs)
    save_sessions()

def get_history(sid: str) -> list:
    history_file = SESSIONS_DIR / sid / "history.json"
    msgs = []
    if history_file.exists():
        with open(history_file) as f:
            msgs = json.load(f)
    return msgs

def search_papers(query: str, top_k: int = 5):
    cache_key = f"{query}:{top_k}"
    if cache_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[cache_key]
    cmd = [
        "python3", str(SUNNY_RAG_DIR / "scripts/search_tool.py"),
        query, "--top_k", str(top_k),
        "--data-version", "v3",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        log(f"search_papers FAILED: {proc.stderr[:200]}")
        return []
    results = json.loads(proc.stdout)
    _SEARCH_CACHE[cache_key] = results
    return results

def read_pdf_text(pdf_path: str) -> str:
    abs_path = pdf_path
    if not abs_path.startswith("/"):
        abs_path = str(PAPERS_DIR / pdf_path)
    if not os.path.exists(abs_path):
        return "[PDF not found]"
    proc = subprocess.run(["pdftotext", abs_path, "-"], capture_output=True, text=True, timeout=30)
    return proc.stdout if proc.returncode == 0 else f"[PDF read error: {proc.stderr[:200]}]"

def find_pdf_path(source: str) -> Optional[str]:
    for year_dir in sorted(PAPERS_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            pdf_file = month_dir / source
            if pdf_file.exists():
                return str(pdf_file)
    return None

def file_id_from_source(source: str) -> str:
    return source.replace(".pdf", "")

# ── 生成任务核心函数(后台运行) ──

async def run_generation(task_id: str, sid: str, user_message: str,
                          search_query: str, search_results: list,
                          paper_context: str, messages_to_send: list):
    """后台运行 Sunny 生成,结果持续写入 _tasks[task_id].chunks"""
    full_content = ""
    try:
        # 检查是否已被取消
        async with _tasks_lock:
            if _tasks[task_id].get("cancelled"):
                log(f"TASK {task_id} cancelled before start")
                return

        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": OPENCLAW_MODEL,
                "messages": messages_to_send,
                "stream": True,
            }
            headers = {
                "Authorization": f"Bearer {OPENCLAW_GATEWAY_TOKEN}",
                "Content-Type": "application/json",
            }
            async with client.stream("POST", f"{OPENCLAW_GATEWAY}/v1/chat/completions",
                                     json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    log(f"TASK {task_id} Sunny API error: HTTP {resp.status_code} {error_body[:200]}")
                    async with _tasks_lock:
                        _tasks[task_id]["error"] = f"HTTP {resp.status_code}"
                        _tasks[task_id]["done"] = True
                    return

                async for line in resp.aiter_lines():
                    # 检查是否被取消
                    async with _tasks_lock:
                        if _tasks[task_id].get("cancelled"):
                            log(f"TASK {task_id} cancelled mid-stream")
                            return
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                chunk = delta["content"]
                                full_content += chunk
                                async with _tasks_lock:
                                    _tasks[task_id]["chunks"].append(chunk)
                        except (json.JSONDecodeError, KeyError):
                            pass

        log(f"TASK {task_id} done: sid={sid} total_chars={len(full_content)}")

        # 完成任务:写入 history
        append_message(sid, "assistant", full_content)

        # 自动生成标题
        history = get_history(sid)
        user_msgs = [m for m in history if m["role"] == "user"]
        if user_msgs:
            title = user_msgs[0]["content"][:40]
            if len(title) == 40:
                title += "......"
            if sid in sessions:
                sessions[sid]["title"] = title
                save_sessions()

        # ── 标注 PDF：在引用的 PDF 文件中直接写入高亮注释 ──
        cited_pdfs = set(re.findall(r"/api/pdf/([\w-]+)", full_content))
        log(f"TASK {task_id} ANNOTATE: Sunny cited {len(cited_pdfs)} PDFs: {cited_pdfs}")
        
        annot_count = 0
        for r in search_results:
            source = r.get("source", "")
            pdf_name = source.replace(".pdf", "")
            if pdf_name not in cited_pdfs:
                continue
            pdf_path_str = find_pdf_path(source)
            if not pdf_path_str:
                log(f"TASK {task_id} ANNOTATE: PDF not found for {source}")
                continue
            try:
                result = annotate_pdf(pdf_path_str, [r["content"][:2000]])
                if result:
                    annot_count += 1  # marking as success
                    log(f"TASK {task_id} ANNOTATE: {pdf_name} → annotated")
                else:
                    log(f"TASK {task_id} ANNOTATE: {pdf_name} → no text found")
            except Exception as e:
                log(f"TASK {task_id} ANNOTATE error for {pdf_name}: {e}")
        
        log(f"TASK {task_id} ANNOTATE: total {annot_count} highlights added")

        async with _tasks_lock:
            _tasks[task_id]["done"] = True

    except Exception as e:
        log(f"TASK {task_id} error: {e}")
        async with _tasks_lock:
            _tasks[task_id]["error"] = str(e)
            _tasks[task_id]["done"] = True
        # 异常时也保存已有内容
        if full_content:
            append_message(sid, "assistant", full_content + "\n\n_(生成中断)_")
            async with _tasks_lock:
                _tasks[task_id]["done"] = True


# ── FastAPI App ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_sessions()
    yield
    save_sessions()

app = FastAPI(title="PerovskiteGPT V4", version="4.0.0", lifespan=lifespan)

# ── API: 文章库 ──

@app.get("/api/papers")
def list_papers(
    category: Optional[str] = Query(None, description="期刊分类"),
    year: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    global _papers_cache
    if _papers_cache is None:
        import sqlite3, pickle
        sqlite_path = BASE_DIR / "data/qdrant_data/collection/perovskite_papers/storage.sqlite"
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.execute("SELECT point FROM points")
        papers = {}
        prefix_journal = {
            "Nature_": "Nature", "NatEnergy_": "Nature Energy",
            "NatMater_": "Nature Materials", "NatPhoton_": "Nature Photonics",
            "NatNanotech_": "Nature Nanotechnology", "NatComm_": "Nature Communications",
            "arXiv_": "arXiv / Preprint",
        }
        for row in cursor:
            p = pickle.loads(row[0])
            meta = p.payload.get("metadata", {})
            source = meta.get("source", "")
            path = meta.get("path", "")
            content = p.payload.get("page_content", "")
            if not source:
                continue
            journal = "Other"
            for prefix, name in prefix_journal.items():
                if source.startswith(prefix):
                    journal = name
                    break
            if source not in papers:
                papers[source] = {
                    "id": file_id_from_source(source),
                    "source": source,
                    "path": path,
                    "journal": journal,
                    "title_preview": "",
                    "chunk_count": 0,
                    "year": None,
                }
                if path:
                    m = re.search(r'/(\d{4})/', path)
                    if m:
                        papers[source]["year"] = int(m.group(1))
            papers[source]["chunk_count"] += 1
            if not papers[source]["title_preview"] and len(content) > 20:
                papers[source]["title_preview"] = content[:150].replace("\n", " ").strip()
        conn.close()
        _papers_cache = sorted(papers.values(), key=lambda x: -(x["year"] or 0))
    sorted_papers = _papers_cache
    if year:
        sorted_papers = [p for p in sorted_papers if p["year"] == year]
    total = len(sorted_papers)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "papers": sorted_papers[start:end],
        "categories": ["Nature", "Nature Energy", "Nature Materials",
                       "Nature Photonics", "Nature Nanotechnology",
                       "Nature Communications", "arXiv / Preprint", "Other"],
    }

# ── API: PDF ──

@app.get("/api/pdf/{file_id}")
def get_pdf(file_id: str, page: Optional[int] = Query(None)):
    # 优先返回 annotated 版本（带高亮标注）
    annotated_path = os.path.join(str(ANNOTATED_DIR), f"{file_id}_annotated.pdf")
    if os.path.exists(annotated_path):
        return FileResponse(annotated_path, media_type="application/pdf",
                            filename=f"{file_id}.pdf",
                            headers={"Content-Disposition": f'inline; filename="{file_id}.pdf"'})
    pdf_path = find_pdf_path(f"{file_id}.pdf")
    if not pdf_path:
        raise HTTPException(404, "PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=f"{file_id}.pdf",
                        headers={"Content-Disposition": f'inline; filename="{file_id}.pdf"'})

# ── API: 会话管理 ──

@app.get("/api/sessions")
def list_sessions():
    ordered = [s for s in session_order if s in sessions]
    return {"sessions": [{"id": sid, "title": sessions[sid].get("title", ""), "message_count": sessions[sid].get("message_count", 0)}
                         for sid in ordered], "order": ordered}

@app.post("/api/sessions")
def create_session():
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = {"title": f"会话 {len(sessions) + 1}", "message_count": 0}
    session_order.insert(0, sid)
    save_sessions()
    return {"id": sid, "title": sessions[sid]["title"]}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        if session_id in session_order:
            session_order.remove(session_id)
        session_dir = SESSIONS_DIR / session_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
        save_sessions()
    return {"ok": True}

@app.put("/api/sessions/{session_id}")
def rename_session(session_id: str, data: dict):
    title = data.get("title", "").strip()
    if session_id in sessions and title:
        sessions[session_id]["title"] = title[:60]
        save_sessions()
    return {"ok": True}

@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    if session_id not in sessions:
        return {"messages": []}
    msgs = get_history(session_id)
    # 检查这个 session 有没有活跃的 task
    for tid, t in _tasks.items():
        if t.get("sid") == session_id and not t.get("done"):
            full = "".join(t["chunks"])
            if full:
                msgs = msgs + [{"role": "assistant", "content": full + "\n\n_(回答未完成)_", "_task_id": tid}]
            break
    return {"messages": msgs}


# ── API: 聊天(解耦版) ──

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    paper_id: Optional[str] = None

@app.post("/api/chat")
async def chat_start(req: ChatRequest):
    """启动生成任务,立即返回 task_id + session_id。不阻塞。"""
    sid = req.session_id
    if not sid or sid not in sessions:
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = {"title": req.message[:50], "message_count": 0}
        session_order.insert(0, sid)

    task_id = uuid.uuid4().hex[:16]

    async with _tasks_lock:
        _tasks[task_id] = {"sid": sid, "chunks": [], "done": False, "error": None}

    # 保存用户消息
    append_message(sid, "user", req.message)

    # 在后台启动生成
    asyncio.create_task(_run_chat(task_id, sid, req))

    return {"task_id": task_id, "session_id": sid}


async def _run_chat(task_id: str, sid: str, req: ChatRequest):
    """后台完整流程:翻译 → 搜索 → 构建 prompt → 生成"""
    # 附加文章上下文
    paper_context = ""
    if req.paper_id:
        pdf_path = find_pdf_path(f"{req.paper_id}.pdf")
        if pdf_path:
            paper_context = read_pdf_text(pdf_path)[:3000]

    # 1. 翻译(中文 → 英文)
    search_query = req.message
    if any('\u4e00' <= c <= '\u9fff' for c in search_query[:10]):
        translated = None
        try:
            translate_prompt = [
                {"role": "system", "content": "You are a translator. Translate the following Chinese perovskite research question to English. Output ONLY the English translation, nothing else."},
                {"role": "user", "content": search_query}
            ]
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-be9caa332fd34a69a92d3d13c661e0aa", "Content-Type": "application/json"},
                    json={"model": "deepseek-chat", "messages": translate_prompt, "stream": False, "max_tokens": 100}
                )
                if r.status_code == 200:
                    translated = r.json()["choices"][0]["message"]["content"].strip().strip('"')
                    if translated:
                        log(f"TASK {task_id} TRANSLATE: '{search_query}' → '{translated}'")
                        search_query = translated
        except Exception as exc:
            log(f"TASK {task_id} TRANSLATE error: {exc}")

    log(f"TASK {task_id} SEARCH: query='{search_query}'")
    search_results = search_papers(search_query, top_k=5)
    if search_results:
        journals = [r['journal_name'] for r in search_results]
        log(f"TASK {task_id} SEARCH: got {len(search_results)} results, journals={journals}")
    else:
        log(f"TASK {task_id} SEARCH: no results")

    # 2. 构建 prompt
    system_prompt = (
        "你是 Sunny,钙钛矿太阳能电池领域的 AI 研究助手。\n"
        "你的回答风格:专业、直接、基于数据。先给框架,再给具体数据。\n\n"
        "## 引用规则(非常重要)\n"
        "你回答时**必须在每个数据点后面直接附加引用链接**,格式:`[📄](/api/pdf/文件名)`\n"
        "示例:『PCE 达到 26.1% [📄](/api/pdf/Nature_2021_s41467-021-26121-1)』\n"
        "不要在末尾额外追加来源列表,引用在正文中就够。\n"
        "**不要说你查了什么、搜了什么**,直接给答案。\n"
    )

    user_rag = f"{req.message}\n\n"
    if search_results:
        user_rag += "参考以下文献来回答(必须引用):\n"
        for i, r in enumerate(search_results):
            pdf_link = f"/api/pdf/{file_id_from_source(r['source'])}"
            user_rag += f"[{i+1}] {r['journal_name']} - {r['source']}\n"
            user_rag += f"    链接: {pdf_link}\n"
            user_rag += f"    内容: {r['content'][:500]}\n\n"

    if paper_context:
        user_rag += f"\n以下是要解读的文章全文:\n{paper_context}\n"

    messages_to_send = [{"role": "system", "content": system_prompt}]
    recent_msgs = get_history(sid)[-10:]
    for msg in recent_msgs:
        if msg.get("_task_id"):
            continue
        messages_to_send.append({"role": msg["role"], "content": msg["content"]})
    if messages_to_send and messages_to_send[-1]["role"] == "user":
        messages_to_send[-1]["content"] = user_rag
    else:
        messages_to_send.append({"role": "user", "content": user_rag})

    # 3. 运行生成
    await run_generation(task_id, sid, req.message, search_query,
                         search_results, paper_context, messages_to_send)


@app.get("/api/sessions/{session_id}/refs")
async def get_session_refs(session_id: str):
    """Note: refs.json is deprecated. PDF highlights are embedded in the PDF files directly."""
    return {}


@app.get("/api/chat/tasks/active")
async def active_tasks():
    """返回当前 session 的活跃 task ids"""
    global _cleanup_time
    now = time.time()
    # 每 60s 清理一次已完成的任务
    async with _tasks_lock:
        if now - _cleanup_time > 60:
            stale = [tid for tid, t in _tasks.items() if t.get("done")]
            for tid in stale:
                del _tasks[tid]
            _cleanup_time = now
        # 按 session 分组
        by_sid = {}
        for tid, t in _tasks.items():
            sid = t["sid"]
            if sid not in by_sid:
                by_sid[sid] = []
            by_sid[sid].append({"task_id": tid, "done": t["done"], "error": t.get("error")})
    return {"tasks": by_sid}

@app.get("/api/chat/{task_id}/status")
async def task_status(task_id: str):
    """查询任务状态"""
    async with _tasks_lock:
        if task_id not in _tasks:
            raise HTTPException(404, "Task not found")
        t = _tasks[task_id]
        return {
            "task_id": task_id,
            "session_id": t["sid"],
            "done": t["done"],
            "error": t.get("error"),
            "total_chars": sum(len(c) for c in t["chunks"]),
        }

@app.post("/api/chat/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消正在生成的任务"""
    async with _tasks_lock:
        if task_id not in _tasks:
            raise HTTPException(404, "Task not found")
        _tasks[task_id]["cancelled"] = True
        _tasks[task_id]["done"] = True
    return {"ok": True, "task_id": task_id}


@app.get("/api/chat/{task_id}/stream")
async def chat_stream(task_id: str, offset: int = Query(0, ge=0)):
    """SSE 流:从指定 offset 开始推送任务输出。支持任意数量连接同时订阅。"""
    async with _tasks_lock:
        if task_id not in _tasks:
            raise HTTPException(404, "Task not found")
        task = _tasks[task_id]

    async def event_stream():
        nonlocal task
        seen_offset = offset

        # 先推已有内容(如果有)
        async with _tasks_lock:
            existing = "".join(task["chunks"])
        if existing and seen_offset < len(existing):
            new_part = existing[seen_offset:]
            seen_offset = len(existing)
            yield f"data: {json.dumps({'type': 'text', 'content': new_part})}\n\n"

        if task["done"]:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # 轮询等待新 chunks
        while True:
            await asyncio.sleep(0.1)

            async with _tasks_lock:
                if task_id not in _tasks:
                    break
                task = _tasks[task_id]
                joined = "".join(task["chunks"])

            if seen_offset < len(joined):
                new_part = joined[seen_offset:]
                seen_offset = len(joined)
                yield f"data: {json.dumps({'type': 'text', 'content': new_part})}\n\n"

            if task["done"]:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            if task.get("error"):
                yield "data: " + json.dumps({"type": "text", "content": "❌ 错误: " + task.get("error", "")}) + "\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")




# ── 清理已完成的 stale tasks(可定期调,或者每次 list 时清理) ──
_cleanup_time = 0




# ── 前端 ──

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = V4_DIR / "web_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>PerovskiteGPT V4</h1><p>web_ui.html 尚未创建</p>")

# ── 启动 ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()
    if args.api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.api_key
    print(f"🚀 PerovskiteGPT V4 starting on http://{args.host}:{args.port}")
    print(f"📚 Papers directory: {PAPERS_DIR}")
    print(f"🧠 Sunny agent via OpenClaw Gateway: {OPENCLAW_GATEWAY}")
    uvicorn.run(app, host=args.host, port=args.port)
