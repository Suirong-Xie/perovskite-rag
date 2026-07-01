#!/usr/bin/env python3
import warnings
import time
import uuid
import logging
import sys
import json
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("rag_api_openclaw")

# ========== 初始化 RAG 组件 ==========
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3-70b-gpu"

# ========== FastAPI 应用 ==========
app = FastAPI(title="Perovskite RAG Expert (Ollama Cleaned)")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=QueryResponse)
async def ask_expert(req: QueryRequest):
    docs = await retriever.ainvoke(req.question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are PerovskiteGPT, an expert in perovskite materials.
Use the following context to answer the question. If the answer is not in the context, say so.

Context: {context}

Question: {req.question}
Answer:"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30.0,
        )
        answer = resp.json().get("response", "No response")
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    return QueryResponse(answer=answer, sources=sources)

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

# ========== 辅助函数 ==========
async def retrieve_documents(query: str) -> List[str]:
    docs = await retriever.ainvoke(query)
    return [doc.page_content for doc in docs]

def clean_messages_for_ollama(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    将 OpenAI 格式的消息转换为 Ollama 兼容格式：
    - 只保留 role: system / user / assistant
    - 移除 role: tool 的消息（或将其内容合并到 user 消息中，此处简单丢弃）
    - 确保每条消息都有 content 字段（字符串）
    """
    cleaned = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("system", "user", "assistant"):
            # 忽略 tool 等角色，或者你可以将 tool 的结果附加到上一条 assistant 消息后
            # 这里简单丢弃，但为了不丢失信息，可以将内容附加上一条 assistant 消息
            if role == "tool" and cleaned and cleaned[-1]["role"] == "assistant":
                # 把 tool 调用结果作为 assistant 消息的一段补充（不完美，但至少保留信息）
                tool_content = msg.get("content", "")
                if tool_content:
                    cleaned[-1]["content"] += f"\n[tool result: {tool_content}]"
            continue
        content = msg.get("content")
        if content is None:
            content = ""
        elif isinstance(content, list):
            # 处理 content 为数组的情况（例如 OpenAI 的 multimodal 格式）
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = " ".join(text_parts)
        cleaned.append({"role": role, "content": str(content)})
    return cleaned

def inject_context_into_messages(messages: List[Dict[str, str]], context: str) -> List[Dict[str, str]]:
    """在 system 消息后插入包含上下文的 system 消息"""
    if messages and messages[0]["role"] == "system":
        # 复制第一个 system 消息，并在其后添加一个 context system 消息
        new_messages = [messages[0]]
        context_msg = {
            "role": "system",
            "content": f"Relevant scientific context:\n{context}"
        }
        new_messages.append(context_msg)
        new_messages.extend(messages[1:])
    else:
        context_msg = {
            "role": "system",
            "content": f"You are PerovskiteGPT, a world-class expert in perovskite materials and optoelectronics. Use the following context to inform your answers.\n\nRelevant context:\n{context}"
        }
        new_messages = [context_msg] + messages
    return new_messages

async def stream_ollama_chat(messages: List[Dict[str, str]], model: str):
    """流式调用 Ollama chat API 并转换为 OpenAI SSE 格式"""
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
                timeout=60.0,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"Ollama stream error {response.status_code}: {error_text}")
                    yield {
                        "error": f"Ollama API error: {response.status_code}",
                        "details": error_text.decode()
                    }
                    return
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "message" in data:
                            content = data["message"].get("content", "")
                            if content:
                                yield {
                                    "id": f"rag-{uuid.uuid4().hex[:8]}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None
                                    }]
                                }
                        if data.get("done", False):
                            yield {
                                "id": f"rag-{uuid.uuid4().hex[:8]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            break
                    except Exception as e:
                        logger.error(f"Error parsing Ollama stream line: {e}")
        except Exception as e:
            logger.error(f"Ollama stream connection error: {e}")
            yield {"error": str(e)}

async def nonstream_ollama_chat(messages: List[Dict[str, str]], model: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=60.0,
        )
        if resp.status_code != 200:
            error_text = resp.text
            logger.error(f"Ollama non-stream error {resp.status_code}: {error_text}")
            raise HTTPException(status_code=502, detail=f"Ollama error: {error_text}")
        data = resp.json()
        return data["message"]["content"]

# ========== 核心端点 ==========
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    stream = body.get("stream", False)
    messages = body.get("messages", [])
    model = body.get("model", OLLAMA_MODEL)
    logger.info(f"Request: stream={stream}, raw_messages={len(messages)}")
    
    # 提取用户消息用于检索
    user_query = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_query = content
                break
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_query = part.get("text")
                        break
                if user_query:
                    break
    if not user_query:
        user_query = "Hello"
    
    # 检索上下文
    try:
        docs = await retrieve_documents(user_query)
        context = "\n\n".join(docs[:3]) if docs else ""
        logger.info(f"Retrieved context length: {len(context)} chars")
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        context = "Context retrieval temporarily unavailable."
    
    # 清理消息格式以兼容 Ollama
    cleaned_messages = clean_messages_for_ollama(messages)
    # 注入上下文
    enhanced_messages = inject_context_into_messages(cleaned_messages, context)
    logger.info(f"Cleaned messages count: {len(enhanced_messages)}")
    
    request_id = f"rag-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    ollama_model = OLLAMA_MODEL
    if stream:
        async def generate():
            async for chunk in stream_ollama_chat(enhanced_messages, ollama_model):
                if "error" in chunk:
                    # 发生错误，发送错误信息并停止
                    error_msg = chunk.get("error")
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    break
                chunk["id"] = request_id
                chunk["created"] = created
                chunk["model"] = model
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        answer = await nonstream_ollama_chat(enhanced_messages, ollama_model)
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        return response

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
