import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List, Optional
import time
import uuid
#from langchain_classic.retrievers import ContextualCompressionRetriever
#from flashrank import Ranker

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ========== 1. 嵌入模型 (GPU 1, 端口 11435) ==========
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

# ========== 3. 基础检索器 (返回 top-10) ==========
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ========== 4. 重排序器 (FlashRank, 压缩为 top-3) ==========
#ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./reranker_cache")

#def flashrank_compress(docs, query):
#    """用 FlashRank 对文档重排序并返回 Top-3 相关文档"""
#    passages = [doc.page_content for doc in docs]
#    # 执行重排序
#    results = ranker.rerank(query, passages)
#    # 取前 3 个索引
#    top_indices = [item["id"] for item in results[:3]]
#    return [docs[i] for i in top_indices]

# 压缩检索器：先检索 10 篇，再重排序取 3 篇
#compression_retriever = ContextualCompressionRetriever(
#    base_compressor=flashrank_compress,
#    base_retriever=base_retriever,
#)

# ========== 5. 生成模型 (GPU 0, 70B, 端口 11434) ==========
llm = OllamaLLM(
    model="llama3-70b-gpu",
    base_url="http://127.0.0.1:11434",
    temperature=0.1,
    top_p=0.9,
    num_predict=1024,
)

# ========== 6. 自定义提示词 ==========
template = """You are PerovskiteGPT, a world-class expert in perovskite materials and optoelectronics.
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

# ========== 7. 构建 RAG 问答链 ==========
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=base_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

# ========== 8. FastAPI 接口 ==========
app = FastAPI(title="Perovskite RAG Expert")

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
            # 截取内容片段作为来源参考
            snippet = doc.page_content[:200].replace("\n", " ")
            sources.append(f"{src}: {snippet}...")
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- OpenAI 兼容数据模型 ----------
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

# ---------- /v1/chat/completions 端点 ----------
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    # 提取最后一条用户消息
    user_message = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    # 调用现有 RAG 链
    result = qa_chain.invoke({"query": user_message})
    answer = result["result"]
    
    # 构造 OpenAI 风格响应
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
