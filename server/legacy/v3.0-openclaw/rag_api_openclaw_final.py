#!/usr/bin/env python3
import warnings
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import time
import uuid
import logging
import sys
import json
import asyncio

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 忽略警告
warnings.filterwarnings("ignore")

# 日志配置
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("rag_api_openclaw")

# ========== 1. 初始化 RAG 组件（与之前一致） ==========
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
        logger.error(f"/ask error: {e}")
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

# ========== 支持流式响应的 /v1/chat/completions ==========
async def generate_stream(answer: str, model: str, request_id: str, created: int):
    """生成 OpenAI 兼容的流式响应（将完整答案作为单个 chunk）"""
    # 第一个 chunk：包含实际内容
    chunk1 = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": answer},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(chunk1)}\n\n"
    # 第二个 chunk：结束标记
    chunk2 = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(chunk2)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 1. 读取原始 JSON 请求体
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    logger.info(f"Received request: model={body.get('model')}, stream={body.get('stream')}, messages={len(body.get('messages', []))}")
    
    # 2. 提取最后一条用户消息（兼容 content 为字符串或数组）
    messages = body.get("messages", [])
    user_content = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_content = content
                break
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_content = part.get("text")
                        break
                if user_content:
                    break
    if not user_content:
        raise HTTPException(status_code=400, detail="No user message found")
    
    logger.info(f"User question: {user_content[:100]}...")
    
    # 3. 调用 RAG 链获得答案
    try:
        result = qa_chain.invoke({"query": user_content})
        answer = result["result"]
    except Exception as e:
        logger.error(f"RAG chain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    logger.info(f"Generated answer length: {len(answer)}")
    
    # 4. 判断是否流式输出
    stream = body.get("stream", False)
    request_id = f"rag-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model = body.get("model", "perovskite-expert")
    
    if stream:
        return StreamingResponse(
            generate_stream(answer, model, request_id, created),
            media_type="text/event-stream"
        )
    else:
        # 非流式响应（标准 OpenAI 格式，含 usage）
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
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

# ---------- 健康检查 ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
