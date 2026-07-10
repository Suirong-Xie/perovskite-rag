# pipeline — 数据处理管道

## 当前流程

PDF 论文 → llm_chunker_v5.py / llm_semantic_chunker.py → texts.jsonl → update_vectors.py → vectors.npy

### pdf_text_extractor.py
PDF 文本提取：PyMuPDF → 纯文本。处理 Nature 双栏排版。

### llm_chunker_v5.py ✅ 主力分块
基于 DeepSeek LLM 的语义分块。输入：PDF 提取文本 → 输出：chunked JSONL。
V5 特性：自动去除参考文献/致谢噪声，按语义边界切分。

### llm_semantic_chunker.py 🧪 实验
语义分块的另一种策略，基于句子嵌入相似度切分。

### update_vectors.py ✅ 向量化
读取 texts.jsonl → Ollama mxbai-embed-large → vectors.npy (1024维)。

### ingest_with_rank.py ✅ 备选
PyMuPDF + RecursiveCharacterTextSplitter 基础分块，带期刊排名元数据。

### sync_nature.py
Nature 期刊 PDF 同步脚本。

### add_journal_rank.py
为已有 chunks 补充期刊排名元数据。

### fetch_top_papers.py
批量获取 Nature 系列期刊论文 PDF。

### daily_incremental.py
每日增量更新管道。
