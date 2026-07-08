"""
PerovskiteGPT V5 — 聊天 API Router
POST /api/chat → 启动生成任务 → SSE stream
"""
import json
import re
import time
import uuid
import subprocess
import asyncio
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from ..core.schemas import ChatRequest, TaskInfo
from ..core.config import AGENT_MAX_ROUNDS
from ..services.session_store import store
from ..services.translator import translate_to_english
from ..services.annotator import extract_highlight_meta
from ..services.agent import run_agent_loop, find_pdf_fast

router = APIRouter()

# ── 任务管理 ──
_tasks: dict[str, TaskInfo] = {}
_tasks_lock = asyncio.Lock()
_cleanup_time = 0


def log(msg: str):
    print(f"[V5] {msg}", flush=True)


# ── 核心 Agent 生成任务 ──


async def run_agent_generation(task_id: str, sid: str, user_message: str):
    """后台运行 Agent 循环，将事件转为 chunks 写入 _tasks[task_id].chunks。

    新增：累积 search_results 事件到 TaskInfo.sources（系统链路），
    Agent 完成后验证 PDF、做高亮标注、推送 sources 事件到前端。
    """
    full_content = ""
    try:
        async with _tasks_lock:
            if _tasks[task_id].cancelled:
                log(f"TASK {task_id} cancelled before start")
                return

        history = store.get_history(sid)

        async for event in run_agent_loop(task_id, sid, user_message, history):
            async with _tasks_lock:
                if _tasks[task_id].cancelled:
                    log(f"TASK {task_id} cancelled mid-agent")
                    return

            if event.type == "thinking":
                async with _tasks_lock:
                    _tasks[task_id].chunks.append(
                        f"💭 {event.data['content']}\n\n"
                    )
            elif event.type == "tool_call":
                # 工具调用开始：之前实时推送的 text 是 thinking，
                # 重置 full_content 避免将思考过程存入历史
                full_content = ""
                name = event.data["name"]
                args = event.data.get("arguments", {})
                async with _tasks_lock:
                    _tasks[task_id].chunks.append(
                        f"🔧 **{name}**({json.dumps(args, ensure_ascii=False)})\n\n"
                    )
            elif event.type == "tool_result":
                if event.data.get("error"):
                    async with _tasks_lock:
                        _tasks[task_id].chunks.append(
                            f"⚠️ {event.data['error']}\n\n"
                        )
            elif event.type == "text":
                chunk = event.data["content"]
                full_content += chunk
                async with _tasks_lock:
                    _tasks[task_id].chunks.append(chunk)
            elif event.type == "search_results":
                # 系统链路：累积搜索结果到 TaskInfo.sources（去重）
                raw_results = event.data.get("results", [])
                async with _tasks_lock:
                    seen_sources = {s.get("source", "") for s in _tasks[task_id].sources}
                    for r in raw_results:
                        src = r.get("source", "")
                        if src and src not in seen_sources:
                            seen_sources.add(src)
                            _tasks[task_id].sources.append(r)
                log(f"TASK {task_id} SOURCES: accumulated "
                    f"{len(raw_results)} new, total {len(seen_sources)} unique")
            elif event.type == "done":
                break
            elif event.type == "error":
                log(f"TASK {task_id} agent error: {event.data['message']}")
                async with _tasks_lock:
                    _tasks[task_id].error = event.data["message"]
                    _tasks[task_id].done = True
                return

        log(f"TASK {task_id} agent done: sid={sid} total_chars={len(full_content)}")

        # ── 系统链路：从 TaskInfo.sources 验证 PDF、做高亮、构建 sources 列表 ──
        async with _tasks_lock:
            task_sources = list(_tasks[task_id].sources)
        log(f"TASK {task_id} SOURCES: processing {len(task_sources)} unique sources")

        validated_sources = []
        annot_count = 0
        for s in task_sources:
            source = s.get("source", "")
            if not source:
                continue
            file_id = source.replace(".pdf", "")
            journal_name = s.get("journal_name", "")

            # 用 find_pdf_fast 做 O(1) 查找（复用 journal_name）
            pdf_path_str = find_pdf_fast(source, journal_name)
            if not pdf_path_str:
                log(f"TASK {task_id} SOURCES: PDF not found for {source}")
                continue

            async with _tasks_lock:
                _tasks[task_id].pdfs_validated.add(file_id)

            # PDF 高亮元数据提取（不修改 PDF，只出 metadata 给前端 CSS overlay）
            content_preview = (s.get("content", "") or "")[:200].replace("\n", " ").strip()
            highlight_meta = {}
            chunk_text = s.get("content", "")[:2000]
            if chunk_text:
                try:
                    highlight_meta = extract_highlight_meta(pdf_path_str, [chunk_text])
                    if highlight_meta.get("chunks"):
                        annot_count += 1
                        log(f"TASK {task_id} ANNOTATE: {file_id} → meta extracted")
                except Exception as e:
                    log(f"TASK {task_id} ANNOTATE error for {file_id}: {e}")

            # 构建验证过的 source 条目（含高亮元数据）
            validated_sources.append({
                "file_id": file_id,
                "journal_name": journal_name or "Unknown",
                "source": source,
                "content_preview": content_preview,
                "pdf_url": f"/api/pdf/{file_id}",
                "highlight": highlight_meta,  # 高亮页码和 chunk 信息
            })

        log(f"TASK {task_id} SOURCES: {len(validated_sources)} validated, "
            f"{annot_count} annotated")

        # ── 引用幻觉校验：检查回答中引用的 File ID 是否都经过 PDF 验证 ──
        if full_content:
            import re as _re
            cited_ids = set(_re.findall(r'\[📄\]\(/api/pdf/([^)]+)\)', full_content))
            async with _tasks_lock:
                validated_ids = set(_tasks[task_id].pdfs_validated)
            fake_ids = cited_ids - validated_ids
            if fake_ids:
                log(f"TASK {task_id} HALLUCINATION: {len(fake_ids)} fake citations: {fake_ids}")
                for fid in fake_ids:
                    # 把假引用替换为醒目的警告标记
                    full_content = full_content.replace(
                        f"[📄](/api/pdf/{fid})",
                        f"[⚠️ 未验证引用](/api/pdf/{fid})"
                    )
            else:
                log(f"TASK {task_id} CITATION CHECK: all {len(cited_ids)} citations valid")

        # 持久化：消息 + 参考来源一起写入 session 历史
        if full_content:
            store.append_message(sid, "assistant", full_content, validated_sources)

        # 存储验证后的 sources 到 task，供 SSE 端点推送
        if validated_sources:
            async with _tasks_lock:
                _tasks[task_id].sources_json = json.dumps(
                    validated_sources, ensure_ascii=False
                )

        async with _tasks_lock:
            _tasks[task_id].done = True

    except Exception as e:
        log(f"TASK {task_id} error: {e}")
        async with _tasks_lock:
            if task_id in _tasks:
                _tasks[task_id].error = str(e)
                _tasks[task_id].done = True
        if full_content:
            store.append_message(sid, "assistant", full_content + "\n\n_(生成中断)_")


async def run_chat_pipeline(task_id: str, sid: str, req: ChatRequest):
    """Agent 驱动流水线：翻译 → 系统预搜索 → Agent 自主搜索/阅读/回答

    关键设计：系统在 Agent 循环前强制做一次初始搜索，将真实存在的
    File ID 注入 Agent 上下文。这样即使 LLM 跳过搜索直接回答，它看到
    的也是真实 File ID，不会凭空编造。
    """
    from ..services.retrieval import search_papers

    # 1. 翻译（中文 → 英文）
    search_hint = await translate_to_english(req.message)
    log(f"TASK {task_id} TRANSLATE: '{req.message[:50]}' → '{search_hint[:80]}'")

    # 2. 系统预搜索：查询扩展 + BM25 混合检索，结果存入 TaskInfo.sources
    initial_results = search_papers(search_hint, top_k=5, expand=True, hybrid=True)
    if initial_results:
        async with _tasks_lock:
            if task_id in _tasks:
                seen = {s.get("source", "") for s in _tasks[task_id].sources}
                for r in initial_results:
                    src = r.get("source", "")
                    if src and src not in seen:
                        seen.add(src)
                        _tasks[task_id].sources.append(r)
        log(f"TASK {task_id} PRE-SEARCH: {len(initial_results)} results, "
            f"total {len(seen)} unique sources")

    # 3. 构建用户消息：注入预搜索结果（含真实 File ID 和 PDF 链接）
    user_message = req.message
    if initial_results:
        user_message += "\n\n系统已为你预检索了以下文献（这些文件的 PDF 链接都是真实存在的）：\n"
        for i, r in enumerate(initial_results):
            file_id = r.get("source", "").replace(".pdf", "")
            pdf_link = f"/api/pdf/{file_id}"
            user_message += (
                f"[{i+1}] {r.get('journal_name', 'Unknown')} | "
                f"File ID: `{file_id}` | "
                f"PDF: {pdf_link}\n"
                f"    内容: {r.get('content', '')[:400]}\n\n"
            )
    elif search_hint != req.message:
        user_message += f"\n\n[搜索提示] 你可以用以下英文关键词来搜索: {search_hint}"

    if req.paper_id:
        pdf_path = find_pdf_fast(f"{req.paper_id}.pdf")
        if pdf_path:
            try:
                proc = subprocess.run(
                    ["pdftotext", pdf_path, "-"],
                    capture_output=True, text=True, timeout=30,
                )
                if proc.returncode == 0:
                    paper_context = proc.stdout[:3000]
                    user_message += f"\n\n以下是要解读的文章全文：\n{paper_context}"
                    log(f"TASK {task_id} injected paper context: {len(paper_context)} chars")
            except Exception:
                pass

    # 4. Agent 循环：LLM 自主搜索/阅读/回答（预搜索结果已在上下文中）
    await run_agent_generation(task_id, sid, user_message)


# ── API 端点 ──


@router.post("/api/chat")
async def chat_start(req: ChatRequest):
    """启动生成任务，立即返回 task_id + session_id。不阻塞。"""
    sid = req.session_id
    if not sid or not store.exists(sid):
        sid = store.create(req.message[:50])

    task_id = uuid.uuid4().hex[:16]
    async with _tasks_lock:
        _tasks[task_id] = TaskInfo(sid)

    store.append_message(sid, "user", req.message)
    asyncio.create_task(run_chat_pipeline(task_id, sid, req))

    return {"task_id": task_id, "session_id": sid}


@router.get("/api/chat/{task_id}/stream")
async def chat_stream(task_id: str, offset: int = Query(0, ge=0)):
    """SSE 流：从指定 offset 开始推送任务输出"""
    async with _tasks_lock:
        if task_id not in _tasks:
            raise HTTPException(404, "Task not found")
        task = _tasks[task_id]

    async def event_stream():
        nonlocal task
        seen_offset = offset
        sources_sent = False

        # 先推送已有内容
        async with _tasks_lock:
            existing = "".join(task.chunks)
        if existing and seen_offset < len(existing):
            new_part = existing[seen_offset:]
            seen_offset = len(existing)
            yield f"data: {json.dumps({'type': 'text', 'content': new_part})}\n\n"

        if task.done:
            # 推送 sources（如果有）
            if task.sources_json and not sources_sent:
                sources_sent = True
                yield f"data: {json.dumps({'type': 'sources', 'data': json.loads(task.sources_json)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # 轮询新 chunks
        while True:
            await asyncio.sleep(0.1)
            async with _tasks_lock:
                if task_id not in _tasks:
                    break
                task = _tasks[task_id]
                joined = "".join(task.chunks)

            if seen_offset < len(joined):
                new_part = joined[seen_offset:]
                seen_offset = len(joined)
                yield f"data: {json.dumps({'type': 'text', 'content': new_part})}\n\n"

            if task.done:
                # 推送 sources（如果有）
                if task.sources_json and not sources_sent:
                    sources_sent = True
                    yield f"data: {json.dumps({'type': 'sources', 'data': json.loads(task.sources_json)})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            if task.error:
                yield "data: " + json.dumps({"type": "text", "content": "❌ 错误: " + task.error}) + "\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/api/chat/{task_id}/status")
async def task_status(task_id: str):
    async with _tasks_lock:
        if task_id not in _tasks:
            raise HTTPException(404, "Task not found")
        t = _tasks[task_id]
        return {
            "task_id": task_id,
            "session_id": t.sid,
            "done": t.done,
            "error": t.error,
            "total_chars": sum(len(c) for c in t.chunks),
        }


@router.post("/api/chat/{task_id}/cancel")
async def cancel_task(task_id: str):
    async with _tasks_lock:
        if task_id not in _tasks:
            raise HTTPException(404, "Task not found")
        _tasks[task_id].cancelled = True
        _tasks[task_id].done = True
    return {"ok": True, "task_id": task_id}


@router.get("/api/chat/tasks/active")
async def active_tasks():
    global _cleanup_time
    now = time.time()
    async with _tasks_lock:
        if now - _cleanup_time > 60:
            stale = [tid for tid, t in _tasks.items() if t.done]
            for tid in stale:
                del _tasks[tid]
            _cleanup_time = now
        by_sid = {}
        for tid, t in _tasks.items():
            sid = t.sid
            if sid not in by_sid:
                by_sid[sid] = []
            by_sid[sid].append({"task_id": tid, "done": t.done, "error": t.error})
    return {"tasks": by_sid}
