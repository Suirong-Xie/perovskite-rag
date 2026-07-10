# PerovskiteGPT V5 — 系统架构文档

> 最后更新: 2026-07-10
> 状态: P1 完成 (Semantic Scholar 接入) → P2 待启动

---

## 1. 项目概览

PerovskiteGPT 是一个钙钛矿太阳能电池领域的 AI 研究助手。核心能力：

- **文献搜索**：本地 RAG (504 篇 Nature 全文) + arXiv API (16 万+ 预印本) + Semantic Scholar (2 亿+ 论文)
- **深度阅读**：PDF 下载 → pdftotext 提取 → 自动清洗参考文献/致谢噪声
- **材料计算**：Pymatgen 容忍因子分析 + Materials Project DFT 查询 + Gaussian 16 第一性原理计算
- **前端交互**：SSE 流式输出 + PDF 侧边栏 + 段落级高亮定位

技术栈：FastAPI + DeepSeek API (ReAct Agent) + Ollama embedding + NumPy 向量检索 + pdftotext + Pymatgen + Gaussian 16

---

## 2. 目录结构

```
perovskite-rag/
├── server/v5/                    # ← 当前主版本
│   ├── main.py                   # FastAPI 入口, 端口 8002
│   ├── web_ui.html               # 前端单页 (~900 行)
│   ├── core/
│   │   ├── config.py             # 全局配置 (路径/LLM/检索/Agent)
│   │   ├── llm.py                # LLM 抽象层 (DeepSeek + OpenClaw)
│   │   └── schemas.py            # Pydantic 数据模型 + SSE 事件定义
│   ├── routers/
│   │   ├── chat.py               # 核心聊天 API + Agent 编排 (~530行)
│   │   ├── sessions.py           # 会话 CRUD API
│   │   └── papers.py             # PDF 服务 + 高亮 API
│   ├── services/
│   │   ├── agent.py              # ★ ReAct Agent 循环引擎 (~980行)
│   │   ├── retrieval.py          # 本地 RAG 检索 (语义 + BM25 混合)
│   │   ├── vector_search.py      # 向量检索核心 (Ollama embed + NumPy cosine)
│   │   ├── arxiv_service.py      # arXiv API 客户端 (搜索/下载/清洗)
│   │   ├── semantic_scholar_service.py  # Semantic Scholar API (2亿+ 论文)
│   │   ├── annotator.py          # PDF 段落级高亮元数据提取
│   │   ├── materials_service.py  # Pymatgen 钙钛矿结构分析
│   │   ├── gaussian_service.py   # Gaussian 16 DFT 计算提交/监控
│   │   ├── translator.py         # 中文→英文查询翻译
│   │   └── session_store.py      # 会话持久化 (JSON 文件)
│   └── static/
│       └── pdf-reader.html       # PDF 阅读器组件 (pdf.js + CSS overlay)
│
├── data/
│   └── vector_db/                  # 向量库数据
│       ├── texts.jsonl             # 文本数据
│       └── vectors.npy             # mxbai-embed-large (1024维) 向量矩阵
│
├── pipeline/                     # 数据摄入管道 (离线)
│   ├── pdf_text_extractor.py     # PyMuPDF 双栏感知文本提取
│   ├── llm_chunker_v5.py         # LLM 语义分块 (当前主力)
│   ├── llm_semantic_chunker.py   # 实验性语义分块 (句子嵌入相似度)
│   ├── update_vectors.py         # Ollama 向量化 (texts.jsonl → vectors.npy)
│   ├── ingest_with_rank.py       # 基础分块 + 期刊排名元数据
│   ├── fetch_top_papers.py       # Nature 系列论文批量获取
│   ├── sync_nature.py            # Nature 期刊 PDF 同步
│   ├── add_journal_rank.py       # 期刊排名元数据补充
│   └── daily_incremental.py      # 每日增量更新管道
│
└── .claude/skills/               # Claude Code 技能
    ├── search-papers.md          # 自定义论文搜索 skill
    └── manage-server.md          # 服务管理 skill
```

---

## 3. 数据管道

### 3.1 论文来源

```
Nature 期刊官网
  │
  ▼
sync_nature.py → 下载 PDF → journals_pdf/{Nature,NatEnergy,...}/
  │
  ▼
pdf_text_extractor.py → PyMuPDF 双栏感知文本提取
  │
  ▼
llm_chunker_v5.py → DeepSeek API 语义分块
  │  - 排除 References / Acknowledgments
  │  - chunk_size ~500-1500 chars
  │
  ▼
update_vectors.py → Ollama mxbai-embed-large → vectors.npy
  │
  ▼
data/vector_db/texts.jsonl + vectors.npy  ← 最终向量库
```

### 3.2 数据规模

| 指标 | 数值 |
|------|------|
| 论文数 | 504 篇 |
| chunks | 11,779 |
| 向量维度 | 1024 (mxbai-embed-large) |
| 来源 | Nature 系列 6 刊 |
| 存储大小 | ~61 MB |

### 3.3 期刊分布 & 搜索权重

| 期刊 | 权重 |
|------|------|
| Nature | 1.5x |
| NatEnergy | 1.4x |
| NatMater | 1.3x |
| NatPhoton | 1.2x |
| NatNanotech | 1.15x |
| NatComm | 1.05x |

---

## 4. 服务端架构

### 4.1 API 路由

```
FastAPI (:8002)
├── GET  /                    → web_ui.html 前端
├── GET  /api/health          → 健康检查
├── POST /api/chat            → 启动 Agent 任务 → {task_id, session_id}
├── GET  /api/chat/{id}/stream → SSE 事件流 (实时推送 Agent 进度)
├── GET  /api/chat/{id}/status → 任务状态查询
├── POST /api/chat/{id}/cancel → 取消任务
├── GET  /api/sessions        → 会话列表
├── GET  /api/pdf/{file_id}   → PDF 文件服务
├── GET  /api/pdf/{file_id}/highlights → 高亮元数据
└── /static/pdf-reader.html   → PDF 阅读器组件
```

### 4.2 请求生命周期 (chat.py)

```
POST /api/chat {message, session_id?}
  │
  ├─ 1. 中文→英文翻译 (translator.py)
  ├─ 2. 预搜索: search_papers(query, top_k=5)
  │      → 注入用户消息 (让 Agent 一开就看到真实的 File ID)
  ├─ 3. 创建 TaskInfo + async 后台任务
  ├─ 4. 同步返回 {task_id, session_id}
  │
  └─ 后台: run_agent_generation()
       │
       ├─ Agent 循环 (agent.py: run_agent_loop)
       │   ├─ Round 1..N: LLM ↔ Tool 交替
       │   └─ 累积 sources (PDF 寻回)
       │
       ├─ 后处理:
       │   ├─ PDF 存在性验证
       │   ├─ 高亮元数据提取 (annotator.py)
       │   ├─ 引用幻觉校验 (正则提取 File ID → 交叉检查)
       │   └─ 推送 sources_json 到前端
       │
       └─ TaskInfo.done = True
```

---

## 5. Agent 循环 ★ 核心

### 5.1 循环结构

```
run_agent_loop(task_id, session_id, user_message, history)
  │
  ├─ 构建 messages:
  │   [system_prompt, ...history(最近10条), user_message(含预搜索结果)]
  │
  ├─ for round in 1..AGENT_MAX_ROUNDS (当前=5):
  │   │
  │   ├─ 状态注入:
  │   │   Round 4: "⚠️ 只剩最后一轮工具调用机会"
  │   │   Round 5: "🚨 工具已被禁用, 必须给出完整回答"
  │   │
  │   ├─ LLM 调用:
  │   │   ├─ DeepSeek: 原生 Function Calling
  │   │   │   └─ Round 5: force_answer=True → tools=[]
  │   │   └─ OpenClaw: 文本 <tool_call> XML 解析 (fallback)
  │   │
  │   ├─ 检测 tool_call:
  │   │   ├─ 有 → 执行工具 → 结果追加到 messages → 继续
  │   │   └─ 无 → 最终回答 → break (yield done)
  │   │
  │   └─ Context 压缩 (round≥2: 旧结果>500字 → 摘要)
  │
  └─ 上限 → yield AgentEvent.error
```

### 5.2 当前控制机制

| 机制 | 实现方式 | 可靠性 |
|------|---------|--------|
| Prompt 约束 | "最多 4 次工具调用后必须回答" | ❌ LLM 经常无视 |
| 状态注入消息 | system message 警告 | ⚠️ 部分有效 |
| **硬关闭工具** | Round 5: `force_answer=True, tools=[]` | ✅ 100% |
| Context 压缩 | 旧结果截断为摘要 | ✅ 防 token 爆炸 |

### 5.3 已知问题

| # | 问题 | 根因 |
|---|------|------|
| 1 | Agent 连续搜索 4 次不读不答 | 无阶段约束，同类型工具可无限调用 |
| 2 | 每次搜索只是微调关键词 | 无去重/覆盖率追踪 |
| 3 | prompt "4次后必须答" 被无视 | 纯文本约束，LLM 在 function calling 模式下不遵守 |
| 4 | 答案质量不稳定 (284 chars, 0 citations) | 最后一轮粗暴截断，Agent 被迫用搜索摘要拼凑 |
| 5 | search_arxiv 调了 3 次才切到 search_papers | 无搜索源切换逻辑 |

---

## 6. 工具矩阵 (当前 10 个)

| 工具 | 数据源 | 延迟 | 返回内容 |
|------|--------|------|---------|
| `search_papers` | 本地 504 篇全文 RAG | ~1s | 排名/期刊/相似度/File ID/片段 |
| `search_arxiv` | arXiv API (16万+) | ~2s | 标题/摘要/作者/PDF链接/arXiv ID |
| `search_semantic_scholar` | Semantic Scholar (2亿+) | ~2s | 标题/摘要/作者/期刊/年份/引用数/DOI |
| `read_paper` | 本地 PDF → 清洗 | ~3s | 正文 (去 References/Ack 等尾部噪声) |
| `read_arxiv_paper` | arXiv PDF 下载 → 清洗 | ~10s | 正文 (去噪声) |
| `extract_data` | 本地 PDF → DeepSeek LLM | ~5s | PCE/Voc/Jsc/FF/结构 (JSON) |
| `analyze_perovskite` | Pymatgen | <0.1s | 容忍因子 t / 八面体因子 μ / 晶体系统 |
| `search_materials` | Materials Project API | ~3s | 带隙/形成能/晶体结构 (需 MP_API_KEY) |
| `run_gaussian` | Gaussian 16 集群 | 小时级 | job_id |
| `check_gaussian` | 本地 fs | <0.1s | 能量/偶极矩/状态 |

### 工具调用优先级链

```
analyze_perovskite → search_materials → search_papers + search_arxiv + search_semantic_scholar → read_paper → run_gaussian
  (快速筛选)         (已知DFT)          (文献发现, 三路并行)                              (深度阅读)    (精确计算)
```

---

## 7. 检索服务 (retrieval.py)

### 7.1 本地 RAG 管道

```
search_papers(query, top_k=5)
  │
  ├─ 1. 查询扩展 (_expand_queries)
  │   ├─ 原始查询
  │   ├─ 去停用词
  │   ├─ 同义词替换 (stability→degradation, PCE↔efficiency...)
  │   └─ 截短版 (长问题→前5词)
  │
  ├─ 2. 语义搜索 (server/v5/services/vector_search.py)
  │   ├─ Ollama mxbai-embed-large → query vec (1024维)
  │   ├─ NumPy cosine: vectors.npy @ query_vec
  │   └─ 期刊加权 (Nature 1.5x → NatComm 1.05x)
  │
  ├─ 3. BM25 关键词搜索
  │   ├─ 惰性构建索引 (texts.jsonl)
  │   ├─ k1=1.2, b=0.75
  │   └─ 融合: 0.7×语义 + 0.3×BM25
  │
  └─ 4. 全查询去重 → top_k
```

### 7.2 Semantic Scholar (semantic_scholar_service.py)

```
search_semantic_scholar(query, max_results=5)
  → https://api.semanticscholar.org/graph/v1/paper/search
  → 免费 tier: 1 req/sec (with API key)
  → 返回: 标题/摘要/作者/期刊/年份/引用数/DOI

优势: 2亿+ 论文, 覆盖 Science/ACS/Wiley/RSC 等非 Nature 期刊
      引用数作为论文质量信号
```

### 7.3 arXiv API (arxiv_service.py)

```
search_arxiv(query, max_results=5)
  → http://export.arxiv.org/api/query (Atom XML)
  → 免费, 无 rate limit

read_arxiv_paper(arxiv_id)
  → https://arxiv.org/pdf/{id}.pdf
  → pdftotext → clean_paper_text()

clean_paper_text():
  ├─ 截断: References / Bibliography / Acknowledgments /
  │         Author Contributions / Conflict of Interest /
  │         Supplementary / Data Availability / Funding
  └─ 过滤: 页码行 / arXiv ID 行 / 版权行 / DOI 行
```

---

## 8. 前端 (web_ui.html ~900行)

### 8.1 组件

```
web_ui.html
├─ 聊天区
│   ├─ SSE EventSource 实时流
│   ├─ 工具调用 badge (🔧 search_arxiv...)
│   ├─ 引用链接 [📄](/api/pdf/xxx)
│   └─ 参考来源面板 (sources_json)
├─ PDF 侧边栏
│   └─ iframe → pdf-reader.html
│       ├─ pdf.js 渲染
│       ├─ CSS overlay 段落级高亮
│       └─ postMessage 选中文本 → 追问
└─ 会话管理
    ├─ 列表 / 切换 / 删除
    └─ 历史持久化 (server sessions.json)
```

### 8.2 PDF 高亮链路

```
Agent 引用 File ID
  → server 验证 PDF 存在
    → annotator.py: chunk→page 定位 + 坐标归一化
      → sources_json → SSE 推送
        → 前端 pdf.js 加载 + CSS overlay 画矩形
```

---

## 9. 部署

```
Python:  3.11 (.RAGenv)
Server:  uvicorn :8002
LLM:     DeepSeek API (deepseek-chat) / OpenClaw fallback
Embed:   Ollama mxbai-embed-large :11435
PDF:     pdftotext (poppler) + PyMuPDF (pipeline)
计算:    Gaussian 16 @ 10.28.0.147
```

### 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BACKEND` | deepseek | deepseek / openclaw |
| `DEEPSEEK_API_KEY` | - | DeepSeek API key |
| `AGENT_MAX_ROUNDS` | 5 | Agent 最大工具调用轮数 |
| `MP_API_KEY` | - | Materials Project (可选) |
| `JOURNALS_PDF_DIR` | /data/data/pkb/01_raw_data/journals_pdf | Nature PDF 目录 |

---

## 10. 下一步

| 优先级 | 任务 | 动机 |
|--------|------|------|
| ~~P1~~ | ~~Semantic Scholar API 接入~~ | ✅ 完成 (2026-07-10) |
| P0 | **Agent 状态机** (PLAN→SEARCH→READ→SYNTHESIZE) | 解决 Agent 不可预测的搜索行为 |
| P2 | 引文追踪 (forward/backward citations) | 论文发现链 |
| P3 | 扩展本地论文库 (Science/ACS/RSC/Wiley) | 当前只覆盖 Nature 系列 |
| P4 | 前端搜索进度可视化 | 用户可见 Agent 的搜索过程 |

## 11. 变更记录

| 日期 | 内容 |
|------|------|
| 2026-07-10 | P1: Semantic Scholar API 接入 (2亿+ 论文搜索) |
| 2026-07-10 | P3.5: arXiv API 整合 (search_arxiv + read_arxiv_paper) |
| 2026-07-10 | 项目清理: 消除 sunny-rag/ 历史目录, 删除 800MB 冗余 |
| 2026-07-08 | P3: Pymatgen 材料工具整合 (analyze_perovskite + search_materials) |
| 2026-07-02 | Phase 2: Agent Loop + DeepSeek + 前端重设计 |
| 2026-06-28 | Phase 1: V5 模块化架构搭建 |
