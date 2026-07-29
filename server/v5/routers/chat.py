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
from ..core.schemas import AgentEvent, ChatRequest, TaskInfo
from ..core.config import AGENT_MAX_ROUNDS
from ..services.session_store import store
from ..services.translator import translate_to_english
from ..services.annotator import extract_highlight_meta
from ..services.agent import run_agent_loop
from ..services.tools.paper_utils import find_pdf_fast

router = APIRouter()

# ── 任务管理 ──
_tasks: dict[str, TaskInfo] = {}
_tasks_lock = asyncio.Lock()
_cleanup_time = 0
_task_queues: dict[str, asyncio.Queue] = {}  # task_id → asyncio.Queue (SSE push)


def log(msg: str):
    print(f"[V5] {msg}", flush=True)


# ── 思考链路构建 ──

def _build_thinking_chain(chunks: list[str]) -> str:
    """从 Agent 中间产物中提取思考链路（💭 思考 + 🔧 工具调用 + ⚠️ 错误）。
    过滤掉最终的 Markdown 回答文本。
    """
    chain_parts = []
    for c in chunks:
        stripped = c.strip()
        if not stripped:
            continue
        if stripped.startswith("💭") or stripped.startswith("🔧") or stripped.startswith("⚠️"):
            chain_parts.append(stripped)
    return "\n\n".join(chain_parts)


# ── 后续建议解析 ──

_SUGGESTIONS_PATTERN = re.compile(
    r'---\s*\n💡\s*\*?\*?继续深入\*?\*?[：:]\s*\n((?:\d+\.\s*[^\n]+\n?)+)',
    re.DOTALL,
)


def _parse_suggestions(text: str) -> list[str]:
    """从回答中提取后续建议列表。"""
    suggestions = []
    match = _SUGGESTIONS_PATTERN.search(text)
    if match:
        block = match.group(1)
        for line in block.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 去掉编号前缀 "1. " "2. " 等
            cleaned = re.sub(r'^\d+\.\s*', '', line).strip()
            if cleaned and len(cleaned) > 5:
                suggestions.append(cleaned)
    # Fallback: 如果没匹配到格式，尝试匹配问号结尾的行
    if not suggestions:
        for line in text.split("\n"):
            line = line.strip()
            if line.endswith("?") and len(line) > 10:
                suggestions.append(line)
    return suggestions[:3]


def _strip_suggestions_block(text: str) -> str:
    """移除回答中的建议块，保持回答干净。"""
    return _SUGGESTIONS_PATTERN.sub("", text).rstrip()


# ── 核心 Agent 生成任务 ──


async def run_agent_generation(task_id: str, sid: str, user_message: str, pool=None,
                               pool_existed: bool = False):
    """后台运行 Agent 循环，将事件转为 chunks 写入 _tasks[task_id].chunks。

    Args:
        pool: SessionPaperPool | None — 会话级论文池（跨轮复用）
        pool_existed: 论文池是否从磁盘加载（vs 本次预搜索新建）
    """
    full_content = ""
    in_respond = False  # 只有 RESPOND 阶段的 text 才累积到最终回答
    queue = _task_queues.get(task_id)
    try:
        async with _tasks_lock:
            if _tasks[task_id].cancelled:
                log(f"TASK {task_id} cancelled before start")
                return

        history = store.get_history(sid)

        # 判断是否为跟进问题（只有从磁盘加载的已有论文池才算跟进）
        is_followup = pool is not None and pool_existed and pool.unread_count > 0
        # 清理用户消息（去掉预搜索注入文本）
        clean_msg = user_message.split("系统已为你预检索")[0].strip() if "系统已为你预检索" in user_message else user_message
        followup_q = clean_msg if is_followup else ""

        async for event in run_agent_loop(task_id, sid, user_message, history,
                                           pool=pool, followup_question=followup_q):
            async with _tasks_lock:
                if _tasks[task_id].cancelled:
                    log(f"TASK {task_id} cancelled mid-agent")
                    return

            # 推入 SSE queue（done 事件延后到 sources 处理完毕再发）
            if queue and event.type != "done":
                await queue.put(event)

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
                if in_respond:
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
            elif event.type == "state":
                # 状态机状态切换事件 — 更新 TaskInfo 并推送前端
                current = event.data.get('current_state', '')
                async with _tasks_lock:
                    _tasks[task_id].agent_state = event.data
                log(f"TASK {task_id} STATE: {current} "
                    f"(papers: {event.data.get('papers_found', 0)}, "
                    f"read: {event.data.get('papers_read', 0)})")
                # 进入 RESPOND 阶段后才开始累积最终回答
                if current == "respond":
                    full_content = ""
                    in_respond = True
            elif event.type == "error":
                log(f"TASK {task_id} agent error: {event.data['message']}")
                async with _tasks_lock:
                    _tasks[task_id].error = event.data["message"]
                    _tasks[task_id].done = True
                return

        log(f"TASK {task_id} agent done: sid={sid} total_chars={len(full_content)}")

        # ── 解析后续建议（从回答末尾提取）──
        if full_content:
            suggestions = _parse_suggestions(full_content)
            if suggestions:
                log(f"TASK {task_id} SUGGESTIONS: {len(suggestions)} found")
                async with _tasks_lock:
                    _tasks[task_id].suggestions_json = json.dumps(
                        suggestions, ensure_ascii=False
                    )
                if queue:
                    await queue.put(AgentEvent.suggestions(suggestions))
            # 从 full_content 中移除建议块（保持回答干净）
            full_content = _strip_suggestions_block(full_content)

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
            has_pdf = bool(pdf_path_str)

            if has_pdf:
                async with _tasks_lock:
                    _tasks[task_id].pdfs_validated.add(file_id)

                # PDF 高亮元数据提取
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

                validated_sources.append({
                    "file_id": file_id,
                    "journal_name": journal_name or "Unknown",
                    "source": source,
                    "content_preview": content_preview,
                    "pdf_url": f"/api/pdf/{file_id}",
                    "highlight": highlight_meta,
                    "has_pdf": True,
                })
            else:
                # 无本地 PDF — 作为补充来源，仅提供 DOI 链接
                doi = s.get("_s2_doi", "")
                content_preview = (s.get("content", "") or "")[:200].replace("\n", " ").strip()
                log(f"TASK {task_id} SOURCES: supplementary (no PDF) — {source}, doi={doi}")

                validated_sources.append({
                    "file_id": file_id,
                    "journal_name": journal_name or "Unknown",
                    "source": source,
                    "content_preview": content_preview or "(摘要)",
                    "doi_url": f"https://doi.org/{doi}" if doi else "",
                    "has_pdf": False,
                })

        log(f"TASK {task_id} SOURCES: {len(validated_sources)} validated, "
            f"{annot_count} annotated")

        # ── 引用幻觉校验：检查回答中引用的 File ID 是否都经过 PDF 验证 ──
        if full_content:
            import re as _re
            # 多种引用格式（LLM 可能用不同形式）：
            #   [📄](/api/pdf/FileID)       — 标准格式
            #   [📄](/api/pdf/FileID.pdf)   — 带 .pdf 后缀
            #   [📄](FileID)                — 无路径的裸引用
            cited_ids = set()
            for pattern in [
                r'\[📄\]\(/api/pdf/([^)]+)\)',       # standard path
                r'\[📄\]\(([A-Za-z][A-Za-z0-9_\-]{8,})\)',  # bare FileID in link
                r'\[⚠️ 未验证引用\]\(/api/pdf/([^)]+)\)',  # already marked as fake
            ]:
                for cid in _re.findall(pattern, full_content):
                    cited_ids.add(cid.replace('.pdf', ''))
            async with _tasks_lock:
                validated_ids = set(_tasks[task_id].pdfs_validated)
            fake_ids = cited_ids - validated_ids
            if fake_ids:
                log(f"TASK {task_id} HALLUCINATION: {len(fake_ids)} fake citations: {fake_ids}")
                for fid in fake_ids:
                    # 把假引用替换为醒目的警告标记（覆盖多种格式变体）
                    for fmt in [
                        f"[📄](/api/pdf/{fid})",
                        f"[📄](/api/pdf/{fid}.pdf)",
                        f"[📄]({fid})",
                        f"[📄]({fid}.pdf)",
                    ]:
                        full_content = full_content.replace(
                            fmt,
                            f"[⚠️ 未验证引用](/api/pdf/{fid})"
                        )
            else:
                log(f"TASK {task_id} CITATION CHECK: all {len(cited_ids)} citations valid")

        # 持久化：消息 + 参考来源一起写入 session 历史
        # 过滤掉裸 <tool_calls> 残骸（Agent 未正常完成时可能残留在 full_content 中）
        clean_content = re.sub(r'<(?:anth:)?tool_calls>.*?</(?:anth:)?tool_calls>', '', full_content, flags=re.DOTALL).strip()
        if not clean_content or len(clean_content) < 50:
            # Agent 未产出有效回答 → 用来源生成兜底摘要
            if validated_sources:
                n_primary = sum(1 for s in validated_sources if s.get("has_pdf") != False)
                n_supp = len(validated_sources) - n_primary
                parts = [f"已检索到 {len(validated_sources)} 篇相关文献"]
                if n_primary:
                    parts.append(f"{n_primary} 篇有全文")
                if n_supp:
                    parts.append(f"{n_supp} 篇补充参考")
                parts.append("，请查看下方来源列表获取详细信息。")
                full_content = "".join(parts)
                log(f"TASK {task_id} FALLBACK: generated summary from {len(validated_sources)} sources")
            else:
                full_content = "抱歉，未能完成本次检索。请重试或更换关键词。"
                log(f"TASK {task_id} FALLBACK: no sources, generic error message")
        else:
            full_content = clean_content

        # 构建思考链路（清洗掉裸 XML）
        thinking_chain = _build_thinking_chain(_tasks[task_id].chunks)

        if full_content:
            suggestions = None
            if _tasks[task_id].suggestions_json:
                suggestions = json.loads(_tasks[task_id].suggestions_json)
            store.append_message(sid, "assistant", full_content, validated_sources,
                                 thinking_chain=thinking_chain,
                                 suggestions=suggestions)

        # 存储验证后的 sources 到 task，供 SSE 端点推送
        if validated_sources:
            async with _tasks_lock:
                _tasks[task_id].sources_json = json.dumps(
                    validated_sources, ensure_ascii=False
                )

        # 推送 done 事件（在 sources 和 suggestions 都准备好之后）
        if queue:
            if _tasks[task_id].suggestions_json:
                await queue.put(AgentEvent.suggestions(
                    json.loads(_tasks[task_id].suggestions_json)))
            await queue.put(AgentEvent.done())

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
    from ..services.paper_pool import SessionPaperPool

    # 0. 加载会话级论文池
    pool = None
    pool_existed = False  # 区分 "从磁盘加载的已有池" vs "本次新建的"
    pool_data = store.get_paper_pool(sid)
    if pool_data:
        pool = SessionPaperPool.from_dict(pool_data)
        pool_existed = True
        log(f"TASK {task_id} POOL: loaded {pool.total_count} papers "
            f"(read={pool.read_count}, unread={pool.unread_count})")

    # 1. 翻译（中文 → 英文）
    search_hint = await translate_to_english(req.message)
    log(f"TASK {task_id} TRANSLATE: '{req.message[:50]}' → '{search_hint[:80]}'")

    # 话题漂移检测：如果新问题和论文池不相关，清空池
    if pool and pool.total_count > 0:
        if pool.detect_topic_drift(search_hint):
            log(f"TASK {task_id} POOL: topic drift detected, resetting pool")
            pool = None

    # 2. 系统预搜索：查询扩展 + BM25 混合检索，结果存入 TaskInfo.sources
    initial_results = search_papers(search_hint, top_k=5, expand=True, hybrid=True)
    if initial_results:
        # 标记每个结果是否有本地 PDF 全文
        for r in initial_results:
            src = r.get("source", "")
            r["has_pdf"] = bool(src and find_pdf_fast(src, r.get("journal_name", "")))
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

        # 预搜索结果加入论文池
        if pool is None:
            pool = SessionPaperPool(session_id=sid)
        pool.add_from_search(initial_results, query=search_hint)
        log(f"TASK {task_id} POOL: after pre-search → {pool.total_count} papers")

    # 3. 构建用户消息：注入预搜索结果（精简版，减少 LLM prompt 负担）
    user_message = req.message
    if initial_results:
        primary = [r for r in initial_results if r.get("has_pdf")]
        supplementary = [r for r in initial_results if not r.get("has_pdf")]

        user_message += "\n\n系统预检索文献：\n"
        if primary:
            user_message += "📄 全文可用: "
            parts = []
            for r in primary:
                file_id = r.get("source", "").replace(".pdf", "")
                parts.append(f"`{file_id}` ({r.get('journal_name', '?')})")
            user_message += ", ".join(parts) + "\n"
        if supplementary:
            user_message += "🔗 仅摘要: "
            parts = []
            for r in supplementary:
                parts.append(r.get("journal_name", "?"))
            user_message += ", ".join(parts) + "\n"
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
    await run_agent_generation(task_id, sid, user_message, pool, pool_existed=pool_existed)

    # 5. 保存论文池（跨轮复用）
    if pool and pool.total_count > 0:
        store.save_paper_pool(sid, pool.to_dict())
        log(f"TASK {task_id} POOL: saved {pool.total_count} papers "
            f"(read={pool.read_count}, unread={pool.unread_count})")


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
        _task_queues[task_id] = asyncio.Queue()

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

        # 从 queue 推播事件（零延迟）fallback 到轮询
        queue = _task_queues.get(task_id)
        if queue:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    break

                if event.type == "text":
                    yield f"data: {json.dumps({'type': 'text', 'content': event.data['content']})}\n\n"
                elif event.type == "state":
                    yield f"data: {json.dumps({'type': 'state', 'data': event.data})}\n\n"
                elif event.type == "tool_call":
                    yield f"data: {json.dumps({'type': 'tool_call', 'data': event.data})}\n\n"
                elif event.type == "tool_result":
                    yield f"data: {json.dumps({'type': 'tool_result', 'data': event.data})}\n\n"
                elif event.type == "suggestions":
                    yield f"data: {json.dumps({'type': 'suggestions', 'data': event.data['items']})}\n\n"
                elif event.type == "done":
                    if task.suggestions_json:
                        yield f"data: {json.dumps({'type': 'suggestions', 'data': json.loads(task.suggestions_json)})}\n\n"
                    if task.sources_json and not sources_sent:
                        sources_sent = True
                        yield f"data: {json.dumps({'type': 'sources', 'data': json.loads(task.sources_json)})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
                elif event.type == "error":
                    yield f"data: {json.dumps({'type': 'text', 'content': '❌ 错误: ' + event.data['message']})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Fallback: 轮询（旧客户端兼容）
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
            "agent_state": t.agent_state,
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
