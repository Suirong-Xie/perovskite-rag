import warnings
import traceback
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# LangChain 相关导入（使用你现有的版本）
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate

warnings.filterwarnings("ignore")

print("=== 正在初始化 RAG 系统 ===")

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

# ========== 3. 基础检索器 ==========
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# ========== 4. 生成模型 ==========
llm = OllamaLLM(
    model="llama3-70b-gpu",
    base_url="http://127.0.0.1:11434",
    temperature=0.3,
    top_p=0.9,
    num_predict=4096,
    repeat_penalty=1.1,
)

# ========== 5. 辅助函数：格式化文档并添加来源 ==========
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

# ========== 6. 提示词模板 ==========
template = """You are PerovskiteGPT, a world-class professor of materials science and optoelectronics, specialized in perovskite solar cells, light management, and device physics.

**Your expertise includes:**
- Semiconductor physics (bandgap, doping, recombination, carrier transport)
- Optical engineering (light trapping, photon confinement, plasmonics, photonic crystals)
- Thermodynamics of solar cells (Shockley-Queisser limit, detailed balance, radiative and non-radiative losses)
- Perovskite chemistry and stability

**Task:** Answer the user's question in a **detailed, mechanistic way**. 
- Use the provided **Context** (scientific literature) as the primary source of evidence. Cite each claim with the source number and file name, e.g., "[2, arXiv_2106.04391v1.pdf]".
- Complement the context with your **general scientific knowledge** (from your training) to explain the underlying physics, especially if the context only gives results without mechanisms. When you use general knowledge, state it clearly (e.g., "From basic semiconductor physics..." or "Generally, photon confinement...").
- If the context lacks critical information to answer the question, say so and then provide the best possible answer based on your general knowledge.

**Answer format requirements:**
- Start with a concise answer (1 sentence).
- Then provide a **step-by-step mechanistic explanation** (at least 4-6 sentences or bullet points). 
- Include the **physical principles** (e.g., density of optical states, Purcell effect, absorption enhancement, voltage increase, recombination reduction).
- End with a brief conclusion that ties back to the PCE improvement.

**Important:** 
- Be specific, not vague. Use terms like "local density of optical states (LDOS)", "radiative recombination rate", "open-circuit voltage (Voc)", "short-circuit current (Jsc)", "fill factor (FF)".
- If the question asks "how does X raise PCE", you must explain the causal chain from X to efficiency improvement.

Context:
{context}

Question: {question}

Answer (with citations and integrated general knowledge):"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# ========== 7. 手动实现 RAG 链（兼容所有版本，带详细日志） ==========
def run_rag(query: str):
    print(f"\n[DEBUG] 收到问题: {query[:100]}...")
    try:
        # 检索文档（兼容不同版本的 LangChain）
        print("[DEBUG] 正在检索文档...")
        try:
            docs = base_retriever.invoke(query)
        except AttributeError:
            docs = base_retriever.get_relevant_documents(query)
        print(f"[DEBUG] 检索到 {len(docs)} 个文档")
        
        # 格式化上下文
        print("[DEBUG] 格式化文档...")
        context = format_docs_with_sources(docs)
        
        # 生成 prompt
        print("[DEBUG] 构建 prompt...")
        prompt_text = PROMPT.format(context=context, question=query)
        
        # 调用 LLM（兼容不同版本）
        print("[DEBUG] 调用 LLM...")
        try:
            answer = llm.invoke(prompt_text)
        except AttributeError:
            answer = llm(prompt_text)
        
        print("[DEBUG] LLM 回答已生成")
        return {"result": answer, "source_documents": docs}
    except Exception as e:
        print("[ERROR] RAG 链执行失败:")
        traceback.print_exc()
        raise

qa_chain = run_rag

# ========== 8. FastAPI 应用 ==========
app = FastAPI(title="Perovskite RAG Expert with Web UI")

# CORS 中间件
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

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]

# ========== 9. 根路径网页 ==========
@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PerovskiteGPT 钙钛矿专家问答</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; margin: 0; padding: 20px; }
        .chat-container { max-width: 1000px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; height: 90vh; }
        .chat-header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .chat-header h1 { margin: 0; font-size: 1.8rem; }
        .chat-header p { margin: 8px 0 0; opacity: 0.8; }
        .messages-area { flex: 1; overflow-y: auto; padding: 20px; background: #f9f9fc; }
        .message { margin-bottom: 20px; display: flex; flex-direction: column; }
        .user-message { align-items: flex-end; }
        .assistant-message { align-items: flex-start; }
        .bubble { max-width: 80%; padding: 12px 18px; border-radius: 20px; line-height: 1.5; word-wrap: break-word; }
        .user-bubble { background: #007aff; color: white; border-bottom-right-radius: 4px; }
        .assistant-bubble { background: #e9ecef; color: #1e2a3a; border-bottom-left-radius: 4px; }
        .sources { font-size: 0.75rem; color: #6c757d; margin-top: 6px; max-width: 80%; }
        .input-area { padding: 20px; background: white; border-top: 1px solid #dee2e6; display: flex; gap: 12px; }
        textarea { flex: 1; padding: 12px; border: 1px solid #ced4da; border-radius: 28px; font-family: inherit; font-size: 1rem; resize: none; outline: none; }
        textarea:focus { border-color: #007aff; }
        button { background: #007aff; color: white; border: none; border-radius: 40px; padding: 0 24px; font-size: 1rem; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #005fc1; }
        button:disabled { background: #86b7fe; cursor: not-allowed; }
        .loading { display: flex; align-items: center; gap: 8px; color: #007aff; font-style: italic; margin-top: 8px; }
        .spinner { width: 16px; height: 16px; border: 2px solid #e9ecef; border-top-color: #007aff; border-radius: 50%; animation: spin 0.6s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
<div class="chat-container">
    <div class="chat-header">
        <h1>🌱 PerovskiteGPT</h1>
        <p>钙钛矿材料专家 · RAG 增强问答</p>
    </div>
    <div class="messages-area" id="messagesArea">
        <div class="message assistant-message">
            <div class="bubble assistant-bubble">
                你好！我是钙钛矿材料领域的专家助手。你可以问我关于钙钛矿太阳能电池、发光材料、晶体结构、合成方法等任何科学问题。<br>我会从论文库中检索并结合知识回答，并给出引用来源。
            </div>
        </div>
    </div>
    <div class="input-area">
        <textarea id="questionInput" rows="1" placeholder="输入你的问题... (Ctrl+Enter 发送)"></textarea>
        <button id="sendBtn">发送</button>
    </div>
</div>
<script>
    const messagesArea = document.getElementById('messagesArea');
    const questionInput = document.getElementById('questionInput');
    const sendBtn = document.getElementById('sendBtn');
    questionInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(120, this.scrollHeight) + 'px';
    });
    questionInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            sendQuestion();
        }
    });
    sendBtn.addEventListener('click', sendQuestion);
    async function sendQuestion() {
        const question = questionInput.value.trim();
        if (!question) return;
        questionInput.value = '';
        questionInput.style.height = 'auto';
        addMessage('user', question);
        const loadingId = addLoadingIndicator();
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question }),
            });
            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errText}`);
            }
            const data = await response.json();
            removeLoadingIndicator(loadingId);
            addMessage('assistant', data.answer, data.sources);
        } catch (error) {
            removeLoadingIndicator(loadingId);
            addMessage('assistant', `抱歉，请求失败：${error.message}。请查看后端日志。`, []);
        }
    }
    function addMessage(role, text, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
        const bubble = document.createElement('div');
        bubble.className = `bubble ${role === 'user' ? 'user-bubble' : 'assistant-bubble'}`;
        bubble.innerHTML = text.replace(/\\n/g, '<br>');
        messageDiv.appendChild(bubble);
        if (role === 'assistant' && sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            sourcesDiv.innerHTML = '<strong>📚 引用来源：</strong><br>' + sources.map(s => `• ${escapeHtml(s)}`).join('<br>');
            messageDiv.appendChild(sourcesDiv);
        }
        messagesArea.appendChild(messageDiv);
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }
    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = id;
        loadingDiv.className = 'message assistant-message';
        loadingDiv.innerHTML = `<div class="bubble assistant-bubble"><div class="loading"><div class="spinner"></div><span>正在检索文献并生成回答...</span></div></div>`;
        messagesArea.appendChild(loadingDiv);
        messagesArea.scrollTop = messagesArea.scrollHeight;
        return id;
    }
    function removeLoadingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
    function escapeHtml(str) {
        return str.replace(/[&<>]/g, function(m) {
            if (m === '&') return '&amp;';
            if (m === '<') return '&lt;';
            if (m === '>') return '&gt;';
            return m;
        });
    }
</script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

# ========== 10. /ask 端点（带详细日志） ==========
@app.post("/ask", response_model=QueryResponse)
async def ask_expert(req: QueryRequest):
    try:
        print(f"\n[API] 收到请求: {req.question[:100]}...")
        result = qa_chain(req.question)   # 我们的函数直接接收字符串
        answer = result["result"]
        sources = []
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:200].replace("\n", " ")
            sources.append(f"{src}: {snippet}...")
        print(f"[API] 回答成功，长度 {len(answer)} 字符")
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        print("[API] 发生异常:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

# ========== 11. OpenAI 兼容端点 ==========
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "perovskite-expert",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "me"
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    user_message = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    try:
        result = qa_chain(user_message)
        answer = result["result"]
        response = ChatCompletionResponse(
            id=f"rag-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=answer),
                    finish_reason="stop"
                )
            ]
        )
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== 启动脚本 ==========
if __name__ == "__main__":
    import uvicorn
    print("=== 服务启动，监听 0.0.0.0:8000 ===")
    uvicorn.run(app, host="0.0.0.0", port=8000)
