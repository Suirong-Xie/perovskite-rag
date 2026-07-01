# pipeline — 数据处理管道

## 流程

PDF 论文 → ingest.py → chunks.jsonl → build_vectordb_qdrant.py → Qdrant

### ingest.py
PDF 文本提取 + 分块。使用 PyMuPDF + RecursiveCharacterTextSplitter。
输入: /data/data/pkb/01_raw_data/papers_pdf/
输出: data/chunked_data/chunks.jsonl
参数: chunk_size=500, chunk_overlap=100

### build_vectordb_qdrant.py ✅ 当前使用
读取 chunks.jsonl → mxbai-embed-large 向量化 → Qdrant 本地存储
集合: perovskite_papers | 维度: 1024 | 距离: COSINE

### built_vectordb.py ❌ 已弃用
旧版 Milvus Lite 构建脚本。
