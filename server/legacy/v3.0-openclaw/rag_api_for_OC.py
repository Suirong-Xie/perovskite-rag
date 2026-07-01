import warnings
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from fastapi.responses import StreamingResponse
import time
import uuid
import logging
import sys
import json

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 忽略警告
warnings.filterwarnings("ignore")

# 日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("rag_api_openclaw")

# ========== 1. 初始化 RAG 组件（与你原来一致） ==========
embed_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11435",
)

client = QdrantClient(path="./data/qdrant_data")
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="perovskite_papers",
    embedding=embed_model,
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = OllamaLLM(
    model="llama3-70b-gpu",
    base_url="http://127.0.0.1:11434",
    temperature=0.1,
    top_p=0.9,
    num_predict=1024,
)

template = """You are a world-class expert in perovskite materials and optoelectronics.
Use the following pieces of context from scientific literature to answer the question.
If the answer cannot be found in the context, say so explicitly and do not fabricate.
Always cite your sources by giving the paper title or DOI/file name when possible.

Context:
{context}

Question: {question}
Answer with citations:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=base_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

app = FastAPI(title="Perovskite RAG Expert (OpenClaw Compatible)")

# ---------- 原有的 /ask 端点（保持不变） ----------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=QueryResponse)
async def ask_expert(req: QueryRequest):
    try:
        result = qa_chain.invoke({"query": req.question})
        answer = result["result"]
        sources = []
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:200].replace("\n", " ")
            sources.append(f"{src}: {snippet}...")
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- /v1/models 端点 ----------
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

# ---------- 完全兼容 OpenClaw 的 /v1/chat/completions 端点 ----------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 1. 获取原始 JSON 请求体
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    logger.info(f"Received request body size: {len(json.dumps(body))} bytes")
    
    # 2. 提取最后一条用户消息（OpenClaw 可能在 messages 数组最后包含用户消息）
    messages = body.get("messages", [])
    user_content = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            # 处理 content 可能是字符串或数组的情况
            content = msg.get("content")
            if isinstance(content, str):
                user_content = content
            elif isinstance(content, list):
                # 提取文本部分
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_content = part.get("text")
                        break
            if user_content:
                break
    
    if not user_content:
        raise HTTPException(status_code=400, detail="No user message found")
    
    logger.info(f"Extracted user message: {user_content[:100]}...")
    
    # 3. 调用 RAG 链
    try:
        result = qa_chain.invoke({"query": user_content})
        answer = result["result"]
    except Exception as e:
        logger.error(f"RAG chain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # 4. 构造 OpenAI 兼容的响应（非流式）
    response = {
        "id": f"rag-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "perovskite-expert"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    return response

# 健康检查
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
