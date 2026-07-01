import warnings
import traceback
import time
import uuid
import json
import os
from typing import List, Optional, Dict, Any
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate

warnings.filterwarnings("ignore")

print("=== 初始化 RAG 系统（持久化会话至 /data） ===")

# ========== 0. 持久化配置 ==========
SESSION_DIR = "/data/perovskite_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
CACHE_MAX_SIZE = 50

_session_cache: OrderedDict[str, Any] = OrderedDict()

def _get_session_path(session_id: str) -> str:
    return os.path.join(SESSION_DIR, f"{session_id}.json")

def load_session(session_id: str) -> List[Dict[str, str]]:
    path = _get_session_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['last_accessed'] = time.time()
                with open(path, 'w', encoding='utf-8') as f2:
                    json.dump(data, f2, ensure_ascii=False, indent=2)
                return data.get('history', [])
        except Exception as e:
            print(f"加载会话 {session_id} 失败: {e}")
            return []
    return []

def save_session(session_id: str, history: List[Dict[str, str]]):
    path = _get_session_path(session_id)
    data = {
        "session_id": session_id,
        "history": history,
        "last_accessed": time.time()
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_session_history(session_id: str) -> List[Dict[str, str]]:
    if session_id in _session_cache:
        history = _session_cache[session_id]
        _session_cache.move_to_end(session_id)
        return history
    history = load_session(session_id)
    _session_cache[session_id] = history
    _session_cache.move_to_end(session_id)
    if len(_session_cache) > CACHE_MAX_SIZE:
        oldest_id, _ = _session_cache.popitem(last=False)
        print(f"缓存淘汰会话: {oldest_id}")
    return history

def update_session_history(session_id: str, history: List[Dict[str, str]]):
    _session_cache[session_id] = history
    _session_cache.move_to_end(session_id)
    save_session(session_id, history)

# ========== 1. 嵌入模型 ==========
embed_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11435",
)

# ========== 2. 加载 Qdrant 向量数据库 ==========
client = QdrantClient(path="./data/qdrant_data")
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="perovskite_papers",
    embedding=embed_model,
)

# ========== 3. 检索器 ==========
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ========== 4. 生成模型 ==========
llm = OllamaLLM(
    model="llama3-70b-gpu",
    base_url="http://127.0.0.1:11434",
    temperature=0.7,
    top_p=0.95,
    num_predict=-1,
    repeat_penalty=1.05,
)

# ========== 5. 格式化文档（带来源） ==========
def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        if "/" in source:
            source = source.split("/")[-1]
        content = doc.page_content.strip().replace("\n", " ")
        if len(content) > 800:
            content = content[:800] + "..."
        formatted.append(f"[{i}] Source: {source}\nContent: {content}\n")
    return "\n".join(formatted)

# ========== 6. 提示词模板（更自然，不机械） ==========
template = """You are PerovskiteGPT, a world-class materials science professor specialized in perovskite solar cells, optoelectronics, and device physics.

**Your expertise includes:**
- Semiconductor physics (bandgap, doping, recombination, carrier transport)
- Optical engineering (light trapping, photon confinement, plasmonics, photonic crystals)
- Thermodynamics of solar cells (Shockley-Queisser limit, detailed balance, radiative and non-radiative losses)
- Perovskite chemistry and stability

**Task:** Answer the user's question thoroughly and insightfully.
- Use the provided **Context** (scientific literature) as evidence. Cite sources with number and file name, e.g., "[2, arXiv_2106.04391v1.pdf]".
- Draw on your **general scientific knowledge** to fill gaps or explain underlying physics, but clearly mark when you're doing so (e.g., "From basic semiconductor physics..." or "Generally...").
- If the context is insufficient, say so, then give your best answer based on general knowledge.

**Conversation history:**
{history}

**Current question:**
{question}

**Retrieved context from papers:**
{context}

**Guidelines for your answer:**
- Think step by step. Analyze the question from first principles before answering.
- Provide depth: explain mechanisms, trade-offs, and the reasoning behind your conclusions.
- Apply your scientific judgment — don't just regurgitate context. Evaluate, compare, and synthesize.
- If there are competing viewpoints in the literature, discuss them.
- Organize your answer naturally — paragraphs, comparisons, or structured lists as appropriate.
- Do not artificially shorten your answer. Be as thorough as the question demands.
- Avoid vague hedging. If you have enough information, give a clear, actionable answer.
- End with a forward-looking insight, open question, or practical takeaway when appropriate.

Answer:"""

PROMPT = PromptTemplate(template=template, input_variables=["history", "question", "context"])

# ========== 7. 对话历史处理 ==========
MAX_HISTORY = 10

def get_history_text(session_id: str) -> str:
    history = get_session_history(session_id)
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history[-MAX_HISTORY:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)

def add_to_history(session_id: str, question: str, answer: str, sources: List[str] = None):
    history = get_session_history(session_id)
    history.append({"question": question, "answer": answer, "sources": sources or []})
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    update_session_history(session_id, history)

# ========== 8. RAG 核心函数 ==========
def run_rag_with_history(question: str, session_id: str):
    print(f"\n[DEBUG] Session: {session_id}")
    print(f"[DEBUG] Question: {question[:100]}...")
    try:
        try:
            docs = base_retriever.invoke(question)
        except AttributeError:
            docs = base_retriever.get_relevant_documents(question)
        print(f"[DEBUG] Retrieved {len(docs)} docs")
        
        context = format_docs_with_sources(docs)
        history = get_history_text(session_id)
        
        prompt_text = PROMPT.format(
            history=history,
            question=question,
            context=context
        )
        
        try:
            answer = llm.invoke(prompt_text)
        except AttributeError:
            answer = llm(prompt_text)
        
        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:200].replace("\\n", " ")
            sources.append(f"{src}: {snippet}...")
        
        add_to_history(session_id, question, answer, sources)
        
        print(f"[DEBUG] Answer generated, length {len(answer)}")
        return {"result": answer, "source_documents": docs, "sources": sources}
    except Exception as e:
        print("[ERROR] RAG failed:")
        traceback.print_exc()
        raise


async def run_rag_stream(question: str, session_id: str):
    """Streaming version of RAG. Yields SSE-formatted strings."""
    import asyncio
    print(f"\n[DEBUG STREAM] Session: {session_id}")
    print(f"[DEBUG STREAM] Question: {question[:100]}...")
    try:
        docs = base_retriever.invoke(question)
        print(f"[DEBUG STREAM] Retrieved {len(docs)} docs")
        
        context = format_docs_with_sources(docs)
        history = get_history_text(session_id)
        
        prompt_text = PROMPT.format(
            history=history,
            question=question,
            context=context
        )
        
        # 先发 sources
        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:200].replace("\\n", " ")
            sources.append(f"{src}: {snippet}...")
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        
        # 流式输出
        full_answer = ""
        for chunk in llm.stream(prompt_text):
            if chunk:
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
            await asyncio.sleep(0)  # 让出事件循环
        
        # 保存历史
        add_to_history(session_id, question, full_answer, sources)
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
        
    except Exception as e:
        print("[ERROR STREAM] RAG failed:")
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

# ========== 9. FastAPI 应用 ==========
app = FastAPI(title="Perovskite RAG Expert with Persistent Memory")

# 静态文件
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 数据模型 ----------
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

# ========== 10. 前端页面（带历史会话侧边栏） ==========
@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PerovskiteGPT-钙钛矿专家</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; display: flex; overflow: hidden; }
        
        /* ---- 侧边栏 ---- */
        .sidebar { width: 280px; min-width: 280px; background: #1e2a3a; color: #ccc; display: flex; flex-direction: column; }
        .sidebar-header { padding: 20px; border-bottom: 1px solid #2a3a4a; }
        .sidebar-header h2 { font-size: 1.1rem; color: #fff; margin-bottom: 12px; }
        .sidebar-header button { width: 100%; background: #4CAF50; color: #fff; border: none; border-radius: 8px; padding: 10px; font-size: 0.9rem; cursor: pointer; transition: background 0.2s; }
        .sidebar-header button:hover { background: #45a049; }
        .session-list { flex: 1; overflow-y: auto; padding: 8px; }
        .session-item { padding: 10px 12px; border-radius: 8px; cursor: pointer; margin-bottom: 4px; transition: background 0.15s; word-break: break-all; font-size: 0.85rem; }
        .session-item:hover { background: #2a3a4a; }
        .session-item.active { background: #007aff; color: #fff; }
        .session-item .title { font-weight: 600; margin-bottom: 2px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; padding-right: 48px; }
        .session-item .preview { color: #999; font-size: 0.75rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .session-item.active .preview { color: #cce5ff; }
        .sidebar-footer { padding: 12px 20px; border-top: 1px solid #2a3a4a; font-size: 0.75rem; color: #666; text-align: center; }
        .session-item { position: relative; }
        .session-actions { display: none; position: absolute; right: 8px; top: 8px; gap: 4px; }
        .session-item:hover .session-actions { display: flex; }
        .session-actions button { background: none; border: none; cursor: pointer; font-size: 0.75rem; padding: 2px 4px; border-radius: 4px; color: #999; }
        .session-actions button:hover { background: rgba(255,255,255,0.15); }
        .session-item.active .session-actions button { color: #cce5ff; }
        .rename-input { width: 100%; background: #2a3a4a; border: 1px solid #007aff; border-radius: 4px; color: #fff; padding: 4px 8px; font-size: 0.85rem; outline: none; }
        .welcome-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; color: #555; padding: 40px; text-align: center; gap: 20px; }
        .welcome-screen .logo { font-size: 3rem; font-weight: 700; color: #2c3e50; letter-spacing: 2px; }
        .welcome-screen .logo span { color: #007aff; }
        .welcome-screen p { font-size: 0.95rem; color: #888; max-width: 400px; line-height: 1.6; }
        .welcome-screen .examples { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
        .welcome-screen .examples button { background: #e9ecef; color: #555; border: 1px solid #dee2e6; border-radius: 20px; padding: 8px 18px; font-size: 0.85rem; cursor: pointer; transition: all 0.15s; }
        .welcome-screen .examples button:hover { background: #d0d4d9; border-color: #adb5bd; }
        .welcome-screen .input-area-welcome { width: 100%; max-width: 600px; display: flex; gap: 12px; align-items: center; margin-top: 8px; }
        .welcome-screen .input-area-welcome textarea { flex: 1; padding: 12px 20px; border: 1px solid #ced4da; border-radius: 28px; font-family: inherit; font-size: 1rem; resize: none; outline: none; max-height: 120px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        .welcome-screen .input-area-welcome textarea:focus { border-color: #007aff; box-shadow: 0 2px 12px rgba(0,122,255,0.15); }
        .welcome-screen .input-area-welcome button { background: #007aff; color: white; border: none; border-radius: 40px; padding: 12px 32px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; white-space: nowrap; }
        .welcome-screen .input-area-welcome button:hover { background: #005fc1; }

        /* ---- 主聊天区 ---- */
        .main { flex: 1; display: flex; flex-direction: column; min-height: 0; }
        .chat-container { flex: 1; display: flex; flex-direction: column; min-height: 0; }
        .chat-header { background: #2c3e50; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
        .chat-header h1 { margin: 0; font-size: 1.3rem; }
        .chat-header .subtitle { font-size: 0.8rem; color: #aaa; }
        .chat-header .home-btn { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; color: #ccc; padding: 6px 12px; font-size: 0.8rem; cursor: pointer; margin-left: auto; }
        .chat-header .home-btn:hover { background: rgba(255,255,255,0.2); color: #fff; }
        .messages-area { flex: 1; overflow-y: auto; padding: 20px; background: #f9f9fc; display: flex; flex-direction: column; gap: 16px; min-height: 0; }
        .message { display: flex; flex-direction: column; position: relative; overflow: visible; }
        .message-body { display: flex; align-items: flex-start; gap: 0; max-width: 1080px; }
        .user-message .message-body { flex-direction: row-reverse; align-self: flex-end; }
        .assistant-message .message-body { align-self: flex-start; }
        .bubble-wrapper { display: flex; flex-direction: column; min-width: 0; flex: 0 1 auto; max-width: 1080px; position: relative; }
        .msg-actions { position: absolute; opacity: 0; transition: opacity 0.15s; bottom: 2px; }
        .message:hover .msg-actions { opacity: 1; }
        .msg-actions button { background: none; border: none; cursor: pointer; font-size: 0.75rem; padding: 2px; color: #aaa; line-height: 1; border-radius: 4px; }
        .msg-actions button:hover { color: #e74c3c; background: rgba(0,0,0,0.05); }
        .assistant-message .bubble-wrapper .msg-actions { right: -22px; }
        .user-message .bubble-wrapper .msg-actions { left: -22px; }
        .user-message { align-items: flex-end; }
        .assistant-message { align-items: flex-start; }
        .bubble { max-width: 100%; padding: 12px 18px; border-radius: 20px; line-height: 1.6; word-wrap: break-word; overflow-wrap: break-word; white-space: pre-wrap; }
        .user-bubble { background: #007aff; color: white; border-bottom-right-radius: 4px; }
        .assistant-bubble { background: #e9ecef; color: #1e2a3a; border-bottom-left-radius: 4px; }
        .sources { font-size: 0.75rem; color: #6c757d; margin-top: 6px; max-width: 80%; }
        .input-area { padding: 16px 24px; background: white; border-top: 1px solid #dee2e6; display: flex; gap: 12px; align-items: center; }
        textarea { flex: 1; padding: 12px 16px; border: 1px solid #ced4da; border-radius: 28px; font-family: inherit; font-size: 1rem; resize: none; outline: none; max-height: 120px; }
        textarea:focus { border-color: #007aff; }
        .input-area button { background: #007aff; color: white; border: none; border-radius: 40px; padding: 10px 28px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; white-space: nowrap; }
        .input-area button:hover { background: #005fc1; }
        .input-area button:disabled { background: #86b7fe; cursor: not-allowed; }
        .input-area .stop-btn { background: #e74c3c; }
        .input-area .stop-btn:hover { background: #c0392b; }
        .loading { display: flex; align-items: center; gap: 8px; color: #007aff; font-style: italic; margin-top: 8px; }
        .spinner { width: 16px; height: 16px; border: 2px solid #e9ecef; border-top-color: #007aff; border-radius: 50%; animation: spin 0.6s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .welcome { text-align: center; color: #aaa; padding: 40px; font-style: italic; }
        .empty-state { text-align: center; color: #aaa; padding: 20px; }
        .cursor-blink::after { content: '▊'; animation: blink 0.8s step-end infinite; color: #007aff; }
        @keyframes blink { 50% { opacity: 0; } }
    </style>
</head>
<body>
<div class="sidebar" id="sidebar">
    <div class="sidebar-header">
        <h2>💬 历史会话</h2>
        <button id="newChatBtn">➕ 新建聊天</button>
    </div>
    <div class="session-list" id="sessionList">
        <div class="empty-state">暂无历史会话</div>
    </div>
    <div class="sidebar-footer">PerovskiteGPT v2</div>
</div>

<div class="main">
<div class="chat-container">
    <div class="chat-header">
        <h1 style="cursor:pointer" onclick="showWelcome()"><img src="/static/logo.webp" style="height:32px;vertical-align:middle;margin-right:8px">PerovskiteGPT</h1>
        <div class="subtitle">钙钛矿太阳能电池 · 材料科学 · 光电子学</div>
    </div>
    <div id="welcomeScreen" class="welcome-screen" style="display:none">
        <div class="logo">Perovskite<span>GPT</span></div>
        <p>钙钛矿太阳能电池 · 材料科学 · 光电子学<br>基于 RAG 检索增强的钙钛矿领域专家</p>
        <div class="examples">
            <button onclick="quickAsk('请介绍钙钛矿太阳能电池的基本结构')">钙钛矿电池结构</button>
            <button onclick="quickAsk('钙钛矿的主要缺陷类型有哪些？')">缺陷类型</button>
            <button onclick="quickAsk('如何提高钙钛矿电池的稳定性？')">提高稳定性</button>
        </div>
        <div class="input-area-welcome">
            <textarea id="welcomeInput" rows="1" placeholder="输入你的问题..." style="height:auto"></textarea>
            <button onclick="sendFromWelcome()">发送</button>
        </div>
    </div>
    <div class="messages-area" id="messagesArea" style="display:none"></div>
    <div class="input-area" id="inputArea">
        <textarea id="questionInput" rows="1" placeholder="输入你的问题... (Ctrl+Enter 发送)"></textarea>
        <button id="sendBtn" style="display:none">发送</button>
        <button id="stopBtn" class="stop-btn" style="display:none">⏹ 停止</button>
    </div>
</div>
</div>

<script>
    // ===== 会话管理 =====
    let currentSessionId = localStorage.getItem('perovskite_session_id');
    let allSessions = {};

    function generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 10);
    }

    // ===== 加载所有会话列表 =====
    async function loadSessionList() {
        try {
            const resp = await fetch('/sessions');
            const data = await resp.json();
            // 保留临时条目
            const tempSessions = {};
            Object.keys(allSessions).forEach(k => {
                if (allSessions[k]._temp) tempSessions[k] = allSessions[k];
            });
            allSessions = tempSessions;
            data.sessions.forEach(s => {
                allSessions[s.session_id] = s;
            });
            renderSessionList();
        } catch(e) {
            console.log('获取会话列表失败', e);
        }
    }

    function getSessionTitle(s) {
        if (s.title) return s.title;
        if (s.history && s.history.length > 0) {
            return s.history[0].question.substring(0, 40) || '新聊天';
        }
        return '新聊天';
    }

    function renderSessionList() {
        const list = document.getElementById('sessionList');
        const ids = Object.keys(allSessions);
        if (ids.length === 0) {
            list.innerHTML = '<div class="empty-state">暂无历史会话</div>';
            return;
        }
        ids.sort((a, b) => (allSessions[b].last_accessed || 0) - (allSessions[a].last_accessed || 0));
        list.innerHTML = ids.map(id => {
            const s = allSessions[id];
            const title = getSessionTitle(s);
            const isActive = id === currentSessionId ? 'active' : '';
            return `<div class="session-item ${isActive}" data-session="${id}">
                <div class="title">${escapeHtml(title)}</div>
                <div class="session-actions">
                    <button onclick="event.stopPropagation();renameSession('${id}')" title="重命名">✏️</button>
                    <button onclick="event.stopPropagation();deleteSession('${id}')" title="删除">🗑️</button>
                </div>
            </div>`;
        }).join('');

        document.querySelectorAll('.session-item').forEach(el => {
            el.addEventListener('click', function() {
                const sid = this.dataset.session;
                switchSession(sid);
            });
        });
    }

    async function renameSession(sessionId) {
        const item = document.querySelector(`.session-item[data-session="${sessionId}"]`);
        if (!item) return;
        // 隐藏操作按钮
        const actionsDiv = item.querySelector('.session-actions');
        if (actionsDiv) actionsDiv.style.display = 'none';
        const titleDiv = item.querySelector('.title');
        const oldTitle = titleDiv.textContent.replace(/^💬 /, '');
        const input = document.createElement('input');
        input.className = 'rename-input';
        input.value = oldTitle;
        input.autofocus = true;
        titleDiv.innerHTML = '';
        titleDiv.appendChild(input);
        input.focus();
        input.select();
        const finish = async () => {
            const newTitle = input.value.trim() || oldTitle;
            try {
                await fetch('/session/' + sessionId + '/rename', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: newTitle })
                });
                if (allSessions[sessionId]) allSessions[sessionId].title = newTitle;
            } catch(e) {}
            renderSessionList();
        };
        input.addEventListener('blur', finish);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
            if (e.key === 'Escape') { e.preventDefault(); input.blur(); }
        });
    }

    async function deleteSession(sessionId) {
        if (!confirm('确定要删除这个会话吗？')) return;
        try {
            await fetch('/session/' + sessionId, { method: 'DELETE' });
            delete allSessions[sessionId];
            if (sessionId === currentSessionId) {
                showWelcome();
            }
            renderSessionList();
        } catch(e) {
            console.log('删除失败', e);
        }
    }

    async function switchSession(sessionId) {
        if (sessionId === currentSessionId) return;
        currentSessionId = sessionId;
        localStorage.setItem('perovskite_session_id', sessionId);
        hideWelcome();
        loadSessionList();
        await loadMessages(sessionId);
    }

    const MAX_VISIBLE_TURNS = 10;

    async function loadMessages(sessionId) {
        const area = document.getElementById('messagesArea');
        area.style.display = 'flex';
        area.innerHTML = '<div class="loading"><div class="spinner"></div><span>加载会话中...</span></div>';
        try {
            const resp = await fetch('/session/' + sessionId + '/history');
            const data = await resp.json();
            area.innerHTML = '';
            if (data.history && data.history.length > 0) {
                const total = data.history.length;
                const showCount = Math.min(total, MAX_VISIBLE_TURNS);
                const hiddenCount = total - showCount;
                const startIdx = total - showCount;

                if (hiddenCount > 0) {
                    const expandDiv = document.createElement('div');
                    expandDiv.id = 'expandCollapse';
                    expandDiv.style.textAlign = 'center';
                    expandDiv.style.margin = '12px 0';
                    const btn = document.createElement('button');
                    btn.textContent = '📜 查看更早的 ' + hiddenCount + ' 轮对话';
                    btn.style.background = '#6c757d';
                    btn.style.padding = '8px 20px';
                    btn.style.fontSize = '0.85rem';
                    btn.addEventListener('click', function() {
                        area.innerHTML = '';
                        for (let i = 0; i < total; i++) {
                            const turn = data.history[i];
                            addMessage('user', turn.question, [], false);
                            addMessage('assistant', turn.answer, turn.sources || [], false);
                        }
                        area.scrollTop = area.scrollHeight;
                    });
                    expandDiv.appendChild(btn);
                    area.appendChild(expandDiv);
                }

                for (let i = startIdx; i < total; i++) {
                    const turn = data.history[i];
                    addMessage('user', turn.question, [], false);
                    addMessage('assistant', turn.answer, turn.sources || [], false);
                }
            } else {
                area.innerHTML = '<div class="message assistant-message"><div class="bubble assistant-bubble">这个会话还没有消息。开始提问吧！</div></div>';
            }
            area.scrollTop = area.scrollHeight;
            showSendButton();
        } catch(e) {
            area.innerHTML = '<div class="message assistant-message"><div class="bubble assistant-bubble">加载会话失败：' + e.message + '</div></div>';
            showSendButton();
        }
    }

    // ===== 起始页 / 欢迎页 =====
    function showWelcome() {
        stopGeneration();
        document.getElementById('messagesArea').style.display = 'none';
        document.getElementById('welcomeScreen').style.display = 'flex';
        document.getElementById('inputArea').style.display = 'none';
    }
    function hideWelcome() {
        document.getElementById('messagesArea').style.display = 'flex';
        document.getElementById('welcomeScreen').style.display = 'none';
        document.getElementById('inputArea').style.display = 'flex';
    }

    // ===== 新建聊天 = 回到起始页 =====
    function newChat() {
        showWelcome();
    }

    function quickAsk(question) {
        // 从起始页快速提问：先创建会话再发送
        currentSessionId = generateSessionId();
        localStorage.setItem('perovskite_session_id', currentSessionId);
        document.getElementById('messagesArea').innerHTML = '';
        createTempSession();
        hideWelcome();
        showSendButton();
        loadSessionList();
        setTimeout(() => {
            document.getElementById('questionInput').value = question;
            sendQuestion();
        }, 100);
    }

    // ===== 发送消息（流式版） =====
    const messagesArea = document.getElementById('messagesArea');
    const questionInput = document.getElementById('questionInput');
    const sendBtn = document.getElementById('sendBtn');
    const stopBtn = document.getElementById('stopBtn');
    const newChatBtn = document.getElementById('newChatBtn');

    let abortController = null;  // 用于中断请求

    newChatBtn.addEventListener('click', newChat);

    // 输入框自动调整高度
    function autoResize(el) {
        el.style.height = 'auto';
        el.style.height = Math.min(120, el.scrollHeight) + 'px';
    }
    questionInput.addEventListener('input', function() { autoResize(this); });

    // 欢迎页输入框同样处理
    document.addEventListener('DOMContentLoaded', function() {
        const wi = document.getElementById('welcomeInput');
        if (wi) wi.addEventListener('input', function() { autoResize(this); });
    });

    function getActiveInput() {
        const ws = document.getElementById('welcomeScreen');
        if (ws.style.display !== 'none') {
            return document.getElementById('welcomeInput');
        }
        return questionInput;
    }

    function handleKeydown(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            sendQuestion();
        }
        // Enter 发送（不按 Shift）
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    }
    questionInput.addEventListener('keydown', handleKeydown);
    // 欢迎页输入框也绑定
    setTimeout(() => {
        const wi = document.getElementById('welcomeInput');
        if (wi) wi.addEventListener('keydown', handleKeydown);
    }, 0);

    sendBtn.addEventListener('click', sendQuestion);
    stopBtn.addEventListener('click', stopGeneration);

    function showSendButton() {
        sendBtn.style.display = '';
        stopBtn.style.display = 'none';
    }
    function showStopButton() {
        sendBtn.style.display = 'none';
        stopBtn.style.display = '';
    }

    function sendFromWelcome() {
        const wi = document.getElementById('welcomeInput');
        const q = wi.value.trim();
        if (!q) return;
        wi.value = '';
        // 创建新会话并输入
        currentSessionId = generateSessionId();
        localStorage.setItem('perovskite_session_id', currentSessionId);
        document.getElementById('messagesArea').innerHTML = '';
        createTempSession();
        hideWelcome();
        showSendButton();
        loadSessionList();
        questionInput.value = q;
        autoResize(questionInput);
        sendQuestion();
    }

    function stopGeneration() {
        if (abortController) {
            abortController.abort();
            abortController = null;
        }
        showSendButton();
        const cursorEl = document.querySelector('.cursor-blink');
        if (cursorEl) cursorEl.classList.remove('cursor-blink');
    }

    // 在起始页输入时，先插入一个临时会话到侧边栏（不等后端）
    function createTempSession() {
        const sid = currentSessionId;
        allSessions[sid] = {
            session_id: sid,
            last_accessed: Date.now() / 1000,
            title: '新聊天',
            history: [],
            _temp: true
        };
        renderSessionList();
    }

    async function sendQuestion() {
        let input = getActiveInput();
        const question = input.value.trim();
        if (!question) return;
        input.value = '';
        autoResize(input);

        // 如果在起始页，自动创建新会话再发送
        const ws = document.getElementById('welcomeScreen');
        if (ws.style.display !== 'none') {
            currentSessionId = generateSessionId();
            localStorage.setItem('perovskite_session_id', currentSessionId);
            document.getElementById('messagesArea').innerHTML = '';
            createTempSession();
            hideWelcome();
            showSendButton();
            loadSessionList();  // 后台刷新，等后端数据
            input = questionInput;
        }

        addMessage('user', question);
        showStopButton();

        // 创建 assistant 消息气泡（空白，后面逐字填充）
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant-message';
        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'message-body';
        const wrapperDiv = document.createElement('div');
        wrapperDiv.className = 'bubble-wrapper';
        const bubble = document.createElement('div');
        bubble.className = 'bubble assistant-bubble';
        bubble.classList.add('cursor-blink');
        bubble.textContent = '';
        wrapperDiv.appendChild(bubble);

        // 删除按钮（移到 wrapperDiv 里，以气泡为参照定位）
        const actions = document.createElement('div');
        actions.className = 'msg-actions';
        const delBtn = document.createElement('button');
        delBtn.textContent = '🗑️';
        delBtn.title = '删除此条消息';
        delBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            msgDiv.remove();
        });
        actions.appendChild(delBtn);
        wrapperDiv.appendChild(actions);
        bodyDiv.appendChild(wrapperDiv);
        msgDiv.appendChild(bodyDiv);

        // sources 容器占位
        const sourcesContainer = document.createElement('div');
        sourcesContainer.style.maxWidth = '80%';
        sourcesContainer.style.marginTop = '6px';
        sourcesContainer.style.display = 'none';
        msgDiv.appendChild(sourcesContainer);
        messagesArea.appendChild(msgDiv);
        messagesArea.scrollTop = messagesArea.scrollHeight;

        // 发起流式请求
        abortController = new AbortController();
        try {
            const response = await fetch('/ask/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    session_id: currentSessionId
                }),
                signal: abortController.signal
            });
            if (!response.ok) throw new Error('HTTP ' + response.status);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let fullAnswer = '';
            let receivedSources = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split(String.fromCharCode(10));
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.type === 'chunk') {
                            fullAnswer += data.text;
                            bubble.innerHTML = fullAnswer.replace(new RegExp(String.fromCharCode(10), 'g'), '<br>');
                            messagesArea.scrollTop = messagesArea.scrollHeight;
                        } else if (data.type === 'sources') {
                            receivedSources = data.sources;
                        } else if (data.type === 'done') {
                            if (data.session_id) {
                                currentSessionId = data.session_id;
                                localStorage.setItem('perovskite_session_id', currentSessionId);
                            }
                            // 添加 sources
                            if (receivedSources.length > 0) {
                                sourcesContainer.style.display = '';
                                const toggle = document.createElement('span');
                                toggle.style.cssText = 'font-size:0.75rem;color:#007aff;cursor:pointer;user-select:none;';
                                toggle.textContent = '📚 展开来源 (' + receivedSources.length + ' 篇)';
                                const sourcesDiv = document.createElement('div');
                                sourcesDiv.className = 'sources';
                                sourcesDiv.style.display = 'none';
                                sourcesDiv.innerHTML = receivedSources.map((s, idx) => {
                                    return '<span style="font-weight:600;color:#007aff">[' + (idx+1) + ']</span> ' + escapeHtml(s.substring(0, 200));
                                }).join('<br>');
                                let expanded = false;
                                toggle.addEventListener('click', function() {
                                    expanded = !expanded;
                                    sourcesDiv.style.display = expanded ? 'block' : 'none';
                                    toggle.textContent = expanded ? '📚 收起来源' : '📚 展开来源 (' + receivedSources.length + ' 篇)';
                                });
                                sourcesContainer.appendChild(toggle);
                                sourcesContainer.appendChild(sourcesDiv);
                            }
                            bubble.classList.remove('cursor-blink');
                            loadSessionList();
                        } else if (data.type === 'error') {
                            bubble.classList.remove('cursor-blink');
                            bubble.innerHTML = '<span style="color:#e74c3c">错误：' + escapeHtml(data.text) + '</span>';
                        }
                    } catch(e) {
                        console.log('SSE parse error:', e);
                    }
                }
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                bubble.innerHTML = (bubble.textContent || '') + '<br><span style="color:#999;font-style:italic;font-size:0.85rem">⏹ 已停止</span>';
            } else {
                bubble.innerHTML = '<span style="color:#e74c3c">请求失败：' + escapeHtml(error.message) + '</span>';
            }
        } finally {
            abortController = null;
            showSendButton();
            bubble.classList.remove('cursor-blink');
        }
    }

    function addMessage(role, text, sources = [], scroll = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + (role === 'user' ? 'user-message' : 'assistant-message');
        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'message-body';
        const wrapperDiv = document.createElement('div');
        wrapperDiv.className = 'bubble-wrapper';
        const bubble = document.createElement('div');
        bubble.className = 'bubble ' + (role === 'user' ? 'user-bubble' : 'assistant-bubble');
        bubble.innerHTML = text.replace(/\\n/g, '<br>');
        wrapperDiv.appendChild(bubble);

        // 消息操作按钮（删除）
        const actions = document.createElement('div');
        actions.className = 'msg-actions';
        const delBtn = document.createElement('button');
        delBtn.textContent = '🗑️';
        delBtn.title = '删除此条消息';
        delBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            messageDiv.remove();
        });
        actions.appendChild(delBtn);
        wrapperDiv.appendChild(actions);
        bodyDiv.appendChild(wrapperDiv);
        messageDiv.appendChild(bodyDiv);

        // Sources 折叠区域：带序号
        if (role === 'assistant' && sources && sources.length > 0) {
            const container = document.createElement('div');
            container.style.maxWidth = '80%';
            container.style.marginTop = '6px';
            const toggle = document.createElement('span');
            toggle.style.cssText = 'font-size:0.75rem;color:#007aff;cursor:pointer;user-select:none;';
            toggle.textContent = '📚 展开来源 (' + sources.length + ' 篇)';
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            sourcesDiv.style.display = 'none';
            // 给每条来源加序号 [1], [2] ...
            sourcesDiv.innerHTML = sources.map((s, idx) => {
                const num = idx + 1;
                return '<span style="font-weight:600;color:#007aff">[' + num + ']</span> ' + escapeHtml(s.substring(0, 200));
            }).join('<br>');
            let expanded = false;
            toggle.addEventListener('click', function() {
                expanded = !expanded;
                sourcesDiv.style.display = expanded ? 'block' : 'none';
                toggle.textContent = expanded ? '📚 收起来源' : '📚 展开来源 (' + sources.length + ' 篇)';
            });
            container.appendChild(toggle);
            container.appendChild(sourcesDiv);
            messageDiv.appendChild(container);
        }
        messagesArea.appendChild(messageDiv);
        if (scroll) messagesArea.scrollTop = messagesArea.scrollHeight;
    }

    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = id;
        loadingDiv.className = 'message assistant-message';
        loadingDiv.innerHTML = '<div class="bubble assistant-bubble"><div class="loading"><div class="spinner"></div><span>正在检索文献并生成回答...</span></div></div>';
        messagesArea.appendChild(loadingDiv);
        messagesArea.scrollTop = messagesArea.scrollHeight;
        return id;
    }

    function removeLoadingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/[&<>]/g, function(m) {
            if (m === '&') return '&amp;';
            if (m === '<') return '&lt;';
            if (m === '>') return '&gt;';
            return m;
        });
    }

    // ===== 启动时加载 =====
    // 尝试恢复上次的会话
    (async function init() {
        await loadSessionList();
        if (currentSessionId && allSessions[currentSessionId]) {
            // 上次的会话存在，加载它
            hideWelcome();
            await loadMessages(currentSessionId);
            // 刷新高亮
            renderSessionList();
        } else if (Object.keys(allSessions).length > 0) {
            // 有历史会话但 currentSessionId 无效，选中最近的一个
            const ids = Object.keys(allSessions);
            ids.sort((a, b) => (allSessions[b].last_accessed || 0) - (allSessions[a].last_accessed || 0));
            currentSessionId = ids[0];
            localStorage.setItem('perovskite_session_id', currentSessionId);
            hideWelcome();
            await loadMessages(currentSessionId);
            renderSessionList();
        } else {
            // 没有会话，显示起始页
            showWelcome();
        }
    })();
</script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

# ========== 11. 列出所有会话 ==========
@app.get("/sessions")
async def list_sessions():
    sessions = []
    if not os.path.exists(SESSION_DIR):
        return {"sessions": sessions}
    for fname in os.listdir(SESSION_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(SESSION_DIR, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 只保留摘要信息
                    summary = {
                        "session_id": data.get("session_id", fname.replace(".json", "")),
                        "last_accessed": data.get("last_accessed", 0),
                        "title": data.get("title", ""),
                        "history": [
                            {"question": h["question"]} for h in data.get("history", [])
                        ]
                    }
                    # 取最后一条问题的预览
                    sessions.append(summary)
            except:
                pass
    # 按 last_accessed 降序
    sessions.sort(key=lambda s: s.get("last_accessed", 0), reverse=True)
    return {"sessions": sessions}

# ========== 12. 获取单个会话详情 ==========
@app.get("/session/{session_id}/history")
async def get_session_history_api(session_id: str):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return {"history": [], "session_id": session_id}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            "history": data.get("history", []),
            "session_id": session_id
        }
    except:
        return {"history": [], "session_id": session_id}

# ========== 13. /ask 接口 ==========
@app.post("/ask", response_model=QueryResponse)
async def ask_expert(req: QueryRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())
        result = run_rag_with_history(req.question, session_id)
        answer = result["result"]
        sources = result.get("sources", [])
        return QueryResponse(answer=answer, sources=sources, session_id=session_id)
    except Exception as e:
        print("[API] 异常:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== 13b. /ask/stream SSE 端点 ==========
@app.post("/ask/stream")
async def ask_expert_stream(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        run_rag_stream(req.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# ========== 14. 删除与重命名会话 ==========
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)
    if session_id in _session_cache:
        del _session_cache[session_id]
    return {"status": "deleted"}

class RenameRequest(BaseModel):
    title: str

@app.post("/session/{session_id}/rename")
async def rename_session(session_id: str, req: RenameRequest):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Session not found")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data["title"] = req.title
    data["last_accessed"] = time.time()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"status": "renamed", "title": req.title}

# ========== 启动 ==========
if __name__ == "__main__":
    import uvicorn
    print(f"=== 服务启动，会话存储路径: {SESSION_DIR} ===")
    print("=== 访问 http://0.0.0.0:8000 使用对话系统 ===")
    uvicorn.run(app, host="0.0.0.0", port=8000)
