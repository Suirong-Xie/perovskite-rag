# server/current — 当前运行版本

## rag_api_html_memory.py

**正在运行** 的 PerovskiteGPT 服务端。

### 功能特性

- ✅ 内置 Web UI（多会话聊天界面）
- ✅ RAG 问答（检索 + 生成）
- ✅ 流式输出（SSE）
- ✅ 会话持久化（JSON 文件 → /data/perovskite_sessions/）
- ✅ LRU 缓存（最多 50 个会话）
- ✅ 会话重命名 / 删除
- ✅ 自动标题生成
- ✅ 来源标注 + 折叠展开

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | / | Web UI |
| POST | /ask | 非流式问答 |
| POST | /ask/stream | 流式问答 (SSE) |
| GET | /sessions | 会话列表 |
| GET | /session/{id}/history | 会话历史 |
| DELETE | /session/{id} | 删除会话 |
| POST | /session/{id}/rename | 重命名会话 |

### 启动

```bash
cd /data1/perovskite-rag
.RAGenv/bin/python server/current/rag_api_html_memory.py
```
