# PerovskiteGPT v4 — Architecture Document

> 当前运行版本：**v4（Sunny-RAG）**
> 历史版本：v2（Agentic RAG, llama3:70b）→ v1（朴素 RAG, 已归档）

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                   用户浏览器 (Web UI)                         │
│              http://localhost:8001/                          │
│              web_ui.html (825 lines)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ POST /api/chat  {message, session_id}
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              V4 FastAPI Server (:8001)                       │
│              server.py (~750 lines)                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. 接收用户请求                                       │   │
│  │ 2. 创建 task_id（生成任务 vs SSE 解耦）               │   │
│  │ 3. 后台 asyncio.create_task 启动 run_generation()     │   │
│  │ 4. POST 到 OpenClaw Gateway → Sunny agent             │   │
│  │ 5. Sunny 推理 + 检索 + 生成回答                        │   │
│  │ 6. 后处理：提取 📄 链接 → PDF 高亮区域搜索             │   │
│  │ 7. SSE 流式返回（支持多个客户端并发消费同一 task）      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────┬────────────────────────────────────────┬─────────┘
           │                                        │
           ▼                                        ▼
┌─────────────────────┐              ┌─────────────────────────┐
│  Ollama (GPU 1)      │              │  OpenClaw Gateway       │
│  127.0.0.1:11435     │              │  localhost:18789        │
│  mxbai-embed-large   │              │                         │
│  (334M, 1024维)      │              │  → Sunny agent           │
│                     │              │    (deepseek-v4-flash)   │
│  用途：query 向量化   │              │    + DeepSeek API       │
└──────────┬──────────┘              └─────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              Sunny-RAG 检索层                                 │
│              sunny-rag/scripts/search_tool.py                │
│                                                              │
│  1. 接收 query + top_k                                        │
│  2. POST 到 Ollama 获取 query 向量                            │
│  3. numpy 向量检索（余弦相似度）                               │
│  4. journal_rank 重排序                                       │
│  5. 返回 top_k 结果 JSON                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  sunny-rag/data/        │
              │  vectors.npy    (722MB) │
              │  texts.jsonl   (119MB)  │
              │  id_to_index.json (8MB) │
              │  188,214 chunks         │
              └─────────────────────────┘
```

## 生成任务与 SSE 解耦

v4 的核心架构改进是生成任务与输出通道分离：

```
POST /api/chat {message, session_id}
  │
  ├→ 创建 task_id
  ├→ 启动 async def run_generation(task_id, sid, msg)
  └→ 立即返回 {task_id, session_id}

run_generation() 后台运行：
  1. POST Sunny agent (OpenClaw Gateway)
  2. Sunny 内部:
     a. 分析问题
     b. 调用 search_tool.py 检索相关论文
     c. 综合推理生成回答
     d. 输出中包含 [📄](/api/pdf/xxx.pdf) 引用
  3. 后处理:
     a. 提取 📄 链接
     b. 用 PyMuPDF 在对应 PDF 中搜索 chunk 文本位置
     c. 对称句号扩展（找上下句号边界）
     d. 跨页支持（收集前后页行数据）
     e. 写入 refs.json → 前端获取高亮区域坐标
  4. 逐 chunk 推送到 _tasks[task_id]["chunks"]
  5. 标记 done = True

SSE consumer:
  GET /api/chat/{task_id}/stream?offset=N
  → 从 offset N 开始读取 chunks
  → 长轮询等待新数据
  → 支持多个客户端从不同 offset 同时消费
```

## PDF 引用与高亮机制

- Sunny 回答中的 `[📄](/api/pdf/文件名.pdf)` 链接
- 后端后处理：
  1. 提取 `/api/pdf/xxx` → 确定 PDF 文件
  2. 用 PyMuPDF 在 PDF 中搜索 chunk 文本
  3. 找到位置 → 对称句号扩展（找上下句号）
  4. 跨页支持 → 收集前后页行数据
  5. 写入 `refs.json` → 前端点击 📄 获取高亮区域坐标
- 前端渲染：content 先替换 `[📄](url)` → `<a>` 标签，再传给 marked.parse

## 检索流程（search_tool.py）

```
Sunny agent → search_tool.py(query, top_k=5)
  │
  ├→ POST Ollama 11435 → query vector (1024d)
  ├→ 加载 vectors.npy (N×1024, numpy memmap)
  ├→ 余弦相似度排序
  ├→ 取 top_k*3 候选 → journal_rank 加权
  ├→ 最终 top_k
  └→ JSON 输出 {content, metadata[source, journal, rank]}
```

## 启动流程（start_v4.sh）

```
1. 清理旧进程：pkill ollama serve/runner
2. 启动 Ollama 11435（mxbai-embed-large, GPU 1）
3. 等待嵌入模型就绪（最长 15s）
4. 清理端口 8001（sudo fuser -k）
5. 启动 V4 server
6. 最终状态：
   - Ollama 嵌入: 127.0.0.1:11435
   - V4 Server:   localhost:8001
   - 注意：llama3:70b (11434) 不再启动
```

## Qdrant（备用索引）

Qdrant 本地模式位于 `data/qdrant_data/`，collection 名 `perovskite_papers`（1024d, COSINE）。
**当前未在生产中使用**——生产检索走 numpy 向量矩阵（sunny-rag/data/）。

## 数据管道

### 当前状态

| 数据集 | chunks | 质量 | 状态 |
|--------|--------|------|------|
| 旧库（chunks.jsonl） | ~587k | ❌ 500字符硬截断 | 旧生产数据 |
| 旧库（向量化后） | 188k | ⚠️ 部分清理 | 当前线上检索 |
| 🌟 Journals 新库 | 4,573 | ✅ 语义分块 | 已分块，未向量化 |
| arXiv 新库 | — | — | 待处理 |

### 分块策略（LLM Semantic Chunking）

- **模型**：DeepSeek Chat API
- **chunk 大小**：500-2000 字符
- **Nature 系列期刊**：跳过 relevance 判断，直接分块
- **arXiv 论文**：先判断 PSC 相关性，通过后再分块
- **输出格式**：
  ```json
  {"content": "...", "metadata": {"source": "...", "journal": "...", "journal_rank": N}}
  ```

### PDF 源

| 源 | 路径 | 数量 |
|----|------|------|
| Nature 系列期刊 | `/data/data/pkb/01_raw_data/journals_pdf/` | ~511 篇 |
| arXiv 论文 | `/data/data/pkb/01_raw_data/papers_pdf/` | ~4083 篇 |

## 端口一览

| 端口 | 服务 | 说明 |
|------|------|------|
| 8001 | V4 Server | 生产 Web UI + API |
| 8000 | V2 Server | 旧版（可并存） |
| 18789 | OpenClaw Gateway | Sunny agent 入口 |
| 11435 | Ollama | mxbai-embed-large 嵌入 |

## 版本历史

| 版本 | 时期 | 架构 | LLM | 检索 |
|------|------|------|-----|------|
| v1 | ~2025 | 朴素 RAG | 本地 | Qdrant |
| v2 | ~2026 Q1 | Agentic RAG | llama3:70b | Qdrant + langchain |
| **v4** | **当前** | **Sunny-RAG** | **DeepSeek API** | **numpy 检索** |
