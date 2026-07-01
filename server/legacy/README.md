# server/legacy — 历史版本

## 版本演进

v1.0-simple → v2.0-html-ui → v3.0-openclaw → v3.1-html-memory-bak

### v1.0-simple (rag_api.py)
最早版本：LangChain RetrievalQA 链，纯 API，无 UI。

### v2.0-html-ui (rag_api_html.py)
加 Web UI，单会话对话，检索 k=15。

### v3.0-openclaw
为 OpenClaw 平台对接的 OpenAI 兼容 API：
- rag_api_for_OC.py — 基础版
- rag_api_openclaw_final.py — 最终版
- rag_api_openclaw_fullfunction.py — 完整版（httpx 直连 Ollama）

### v3.1-html-memory-bak
rag_api_html_memory.py 的旧备份。
