# PerovskiteGPT V5 — Sunny

面向钙钛矿太阳能电池研究的 AI 研究助手。支持文献搜索、深度阅读、材料计算。

**当前版本：V5**（FastAPI + DeepSeek ReAct Agent）

## 核心能力

| 能力 | 实现 | 覆盖范围 |
|------|------|----------|
| 文献搜索 | numpy 向量检索 + BM25 混合搜索 | 504 篇 Nature 全文 |
| arXiv 搜索 | arXiv API | 16 万+ 预印本 |
| 深度阅读 | pdftotext + 自动清洗 | 参考文献/致谢截断 |
| 材料计算 | Pymatgen + Gaussian 16 | 容忍因子 / DFT |
| 前端 | SSE 流式 + PDF 侧边栏高亮 | 段落级定位 |

## 目录结构

```
perovskite-rag/
├── server/v5/                    # ✅ 当前版本
│   ├── main.py                   # FastAPI 入口, 端口 8002
│   ├── web_ui.html               # 前端单页
│   ├── core/                     # 基础层
│   │   ├── config.py             # 全局配置
│   │   ├── llm.py                # LLM 调用 (DeepSeek API)
│   │   └── schemas.py            # Pydantic 模型
│   ├── routers/                  # API 路由
│   │   ├── chat.py               # 对话 + SSE 流式
│   │   ├── sessions.py           # 会话管理
│   │   └── papers.py             # PDF 查看
│   ├── services/                 # 业务服务
│   │   ├── agent.py              # ReAct Agent 编排
│   │   ├── retrieval.py          # 语义搜索 + BM25 混合
│   │   ├── vector_search.py      # 向量检索核心
│   │   ├── arxiv_service.py      # arXiv API 集成
│   │   ├── materials_service.py  # Pymatgen 材料分析
│   │   ├── gaussian_service.py   # Gaussian 16 计算
│   │   ├── annotator.py          # PDF 段落高亮
│   │   └── translator.py         # 翻译服务
│   ├── static/                   # 静态资源
│   └── sessions/                 # 会话持久化
│
├── data/
│   └── vector_db/                # 向量库数据
│       ├── vectors.npy           # 向量矩阵
│       └── texts.jsonl           # 文本数据
│
├── pipeline/                     # 离线数据处理
│   ├── llm_chunker_v5.py         # ✅ 主力 LLM 语义分块
│   ├── llm_semantic_chunker.py   # 🧪 实验性分块
│   ├── update_vectors.py         # 向量化
│   ├── ingest_with_rank.py       # 基础分块（带期刊排名）
│   ├── pdf_text_extractor.py     # PDF 文本提取
│   ├── fetch_top_papers.py       # 论文获取
│   ├── sync_nature.py            # Nature 同步
│   ├── add_journal_rank.py       # 期刊排名补充
│   └── daily_incremental.py      # 每日增量
│
├── backups/                      # 历史备份
├── logs/                         # 运行日志
├── .RAGenv/                      # Python 虚拟环境
├── README.md                     # 本文档
└── ARCHITECTURE.md               # 详细架构文档
```

## 启动

```bash
cd /data1/perovskite-rag
source .RAGenv/bin/activate
python server/v5/main.py
# → http://localhost:8002
```

## V5 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web UI |
| `/api/chat` | POST | 发送消息（返回 task_id） |
| `/api/chat/{task_id}/stream` | GET | SSE 流式输出 |
| `/api/chat/{task_id}/status` | GET | 任务状态 |
| `/api/pdf/{file_id}` | GET | 获取 PDF |
| `/api/pdf/{file_id}/refs` | GET | 高亮坐标 |
| `/api/papers` | GET | 论文库列表 |
| `/api/sessions` | GET/POST | 会话管理 |

## 数据管道

- **向量库**: `data/vector_db/` — numpy 矩阵 + JSONL，1024 维 mxbai-embed-large
- **PDF 源**: `/data/data/pkb/01_raw_data/journals_pdf/` (Nature 系列 ~500 篇)
- **分块**: LLM 语义分块（llm_chunker_v5.py），自动清洗参考文献/致谢

## 已知问题

- [ ] arXiv 论文语义分块尚未完成
- [ ] 向量库需要从旧 chunk 迁移到新 semantic chunk
- [ ] PDF 高亮搜索偶因文本层问题失败
