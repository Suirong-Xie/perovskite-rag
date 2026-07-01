# PerovskiteGPT — Sunny-RAG

面向钙钛矿太阳能电池研究的 **Agentic RAG** 智能问答系统。
当前运行版本：**v4（Sunny-RAG）**

## 架构概览（v4）

```
用户浏览器 (web_ui.html :8001)
         │ POST /api/chat
         ▼
┌─────────────────────────────────────────────┐
│         V4 FastAPI Server (server.py)        │
│  端口 8001                                    │
│  ┌───────────────────────────────────────┐  │
│  │ 1. 接收用户问题                        │  │
│  │ 2. 创建 task_id, 后台异步启动生成任务    │  │
│  │ 3. POST 到 OpenClaw Gateway (:18789)   │  │
│  │    → Sunny agent (DeepSeek V4) 处理     │  │
│  │ 4. Sunny 调用 search_tool.py 检索向量库  │  │
│  │ 5. Sunny 生成回答 + 📄 引用路径         │  │
│  │ 6. 后处理：PDF 高亮区域搜索             │  │
│  │ 7. SSE 流式返回给前端                   │  │
│  └───────────────────────────────────────┘  │
└──────────────┬──────────────────┬───────────┘
               │                  │
               ▼                  ▼
    ┌─────────────────┐   ┌─────────────────┐
    │  Ollama (11435)  │   │   numpy 向量库    │
    │ mxbai-embed-large│   │  sunny-rag/data/  │
    │ query 向量化     │   │  vectors.npy      │
    │                 │   │  texts.jsonl      │
    │                 │   │  188,214 chunks   │
    └─────────────────┘   └─────────────────┘
               │
               ▼
    ┌─────────────────────────────────┐
    │   OpenClaw Gateway (:18789)     │
    │   → Sunny agent (deepseek-v4)   │
    │   → 推理 + 生成回答              │
    └─────────────────────────────────┘
```

### 关键变化（vs v2）

| 方面 | v2 | v4（当前） |
|------|-----|-----------|
| **LLM** | 本地 llama3:70b (GPU) | Sunny agent (DeepSeek V4 API) |
| **检索** | Qdrant + langchain | numpy 向量检索（search_tool.py） |
| **生成控制** | 同步 SSE | 异步 task + 多消费者 SSE |
| **PDF 引用** | 纯文字 | 📄 链接 + PDF 高亮区域 |
| **架构** | 8 文件模块化 | server.py + search_tool.py |

## 目录结构

```
perovskite-rag/
├── server/
│   ├── v2/              # v2 历史版本（Agentic RAG，已归档）
│   │   ├── server.py    # FastAPI 应用
│   │   ├── agent.py     # Agent 编排
│   │   ├── vector_store.py
│   │   ├── models.py    # LLM + Embedding
│   │   ├── tools.py
│   │   ├── sessions.py
│   │   ├── prompts.py
│   │   └── config.py
│   ├── v4/              # ✅ 当前运行版本（Sunny-RAG）
│   │   ├── server.py    # FastAPI 应用（核心文件，~750行）
│   │   ├── web_ui.html  # 独立 Web UI（825行）
│   │   ├── start_v4.sh  # 启动脚本
│   │   └── sessions/    # 会话持久化存储
│   └── legacy/          # v1 历史版本归档
│
├── sunny-rag/           # Sunny agent 检索工具
│   ├── data/
│   │   ├── vectors.npy       # 向量矩阵（188214 x 1024）
│   │   ├── texts.jsonl       # 文本数据（18.8 万条）
│   │   └── id_to_index.json  # chunk ID → 索引映射
│   └── scripts/
│       ├── search_tool.py    # 检索入口（Sunny agent 调用）
│       ├── 01_extract_vectors.py
│       ├── 02_search.py
│       └── fix_journal_rank.py
│
├── pipeline/            # 数据管道
│   ├── fetch_top_papers.py        # Nature 系列论文搜索 & 下载
│   ├── daily_incremental.py       # 每日增量更新
│   ├── build_vectordb_qdrant.py   # Qdrant 重建（旧）
│   ├── llm_semantic_chunker.py    # LLM 语义分块（v2/v4 共用）
│   └── ...
│
├── data/
│   ├── chunked_data/    # 分块文件（JSONL）
│   ├── qdrant_data/     # Qdrant 向量库（旧/备用）
│   └── pipeline/        # pipeline 状态追踪
│
├── logs/                # 运行日志
├── .RAGenv/             # Python 虚拟环境
├── README.md            # 本文档
└── ARCHITECTURE.md      # 详细架构文档
```

## 启动方式

```bash
# v4（当前生产版本，端口 8001）
bash /data1/perovskite-rag/server/v4/start_v4.sh
```

## 数据管道

### 当前向量库
- **来源**：Nature 系列期刊论文（`/data/data/pkb/01_raw_data/journals_pdf/`）
- **向量数量**：188,214 chunks
- **维度**：1024（mxbai-embed-large）
- **存储**：numpy `.npy` 矩阵文件 + JSONL 文本（`sunny-rag/data/`）
- **journal_rank**：按期刊等级加权（Nature=1 → NatComm=6 → Other=7）

### 分块策略
- v2 版本：500 字符硬截断（质量差，已废弃）
- v4 版本：LLM 语义分块（500-2000 字符），使用 DeepSeek API
  - Nature 系列期刊：直接分块，跳过 relevance 判断
  - arXiv 论文：先 relevance 判断，符合条件的再分块

## PDF 源目录

| 目录 | 内容 | 数量 |
|------|------|------|
| `/data/data/pkb/01_raw_data/journals_pdf/` | Nature 系列期刊 | ~511 篇 |
| `/data/data/pkb/01_raw_data/papers_pdf/` | arXiv %2B 其他 | ~4083 篇 |

## v4 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web UI 页面 |
| `/api/chat` | POST | 发送消息（返回 task_id） |
| `/api/chat/{task_id}/stream` | GET | SSE 流式输出（支持 offset） |
| `/api/chat/{task_id}/status` | GET | 任务状态查询 |
| `/api/chat/tasks/active` | GET | 活跃任务列表 |
| `/api/pdf/{file_id}` | GET | 获取 PDF（可指定 page） |
| `/api/pdf/{file_id}/refs` | GET | 高亮区域坐标 |
| `/api/papers` | GET | 论文库列表（分页/搜索/过滤） |
| `/api/sessions` | GET | 会话列表 |
| `/api/sessions` | POST | 创建新会话 |

## 已知问题 / TODO

- [ ] arXiv 论文的语义分块尚未完成
- [ ] 向量库（sunny-rag/data/）需从旧 188k 更新为 4,573 新 semantic chunks
- [ ] PDF 高亮搜索有时因文本层问题失败
