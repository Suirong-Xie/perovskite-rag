"""Tool definitions for PerovskiteGPT v2 Agent — with multi-context retrieval."""

import json
import os
import subprocess
from collections import defaultdict

from vector_store import get_retriever
from sessions import load_session
from config import QDRANT_TOP_K_DEFAULT, QDRANT_TOP_K_MIN, QDRANT_TOP_K_MAX, CHUNKED_DATA_PATH, DATA_DIR


# ── Tool definitions (JSON schema for function calling) ──

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_papers",
            "description": "从钙钛矿论文数据库中检索相关文献信息。当你需要引用具体论文、数据、实验方法、表征结果、效率数值等客观信息时使用。如果用户问的是主观分析、概念解释或编程问题，不要使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词，应当提取用户问题中的核心技术术语和关键概念"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "要检索的文献数量（0~20）。窄问题1-5，标准5-10，综述10-20。默认10。",
                        "minimum": 0,
                        "maximum": 20,
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_paper_fulltext",
            "description": "IMPORTANT: When retrieved paper snippets from retrieve_papers don't contain enough detail (missing efficiency numbers, device structure, or test conditions), use this tool to read the original paper's full abstract, conclusion, or first pages. This gives you the complete picture from the paper, not just fragments. Source is the filename from search results (e.g. arXiv_2402.10286v1.pdf).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "论文文件名，如 arXiv_2402.10286v1.pdf"
                    },
                    "section": {
                        "type": "string",
                        "description": "要获取的部分：abstract（摘要）, conclusion（结论）, 或 full（全文前几页）",
                        "enum": ["abstract", "conclusion", "full"]
                    }
                },
                "required": ["source", "section"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_session_history",
            "description": "在当前会话的历史记录中搜索之前的讨论内容。当用户问「我刚才说了什么」、「之前讨论过X吗」、「关于X我们之前怎么说的」等时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "要查找的关键词"
                    }
                },
                "required": ["keywords"]
            }
        }
    }
]


def _build_index() -> dict:
    """Build a source -> list of chunks index from the JSONL file for context expansion."""
    idx = defaultdict(list)
    path = CHUNKED_DATA_PATH
    if not os.path.exists(path):
        return idx
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                src = d["metadata"].get("source", "unknown")
                idx[src].append(d["content"])
    except:
        pass
    return idx


# Lazy-loaded index
_INDEX = None

def _get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    return _INDEX


def format_docs_with_sources(docs) -> str:
    """Format retrieved documents into context string with source citations."""
    lines = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.replace("\n", " ")
        lines.append(f"[Source {i}] ({source})\n{content}\n")
    return "\n".join(lines)


def _expand_context(docs, max_extra_chunks=3) -> str:
    """For the top most-relevant sources, fetch additional chunks for richer context."""
    # Count how many chunks each source contributes
    source_counts = defaultdict(int)
    source_chunks = defaultdict(list)
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        source_counts[src] += 1
        source_chunks[src].append(doc.page_content)
    
    # Identify top sources (those with the most matches)
    top_sources = sorted(source_counts.keys(), key=lambda s: source_counts[s], reverse=True)
    
    idx = _get_index()
    if not idx:
        return ""
    
    extra_parts = []
    for src in top_sources[:2]:  # expand top 2 sources
        all_chunks = idx.get(src, [])
        already_have = set(source_chunks[src])
        extra = 0
        for chunk_text in all_chunks:
            # Skip if we already have this chunk
            normalized_chunk = chunk_text.replace("\n", " ").strip()
            already_normalized = {c.replace("\n", " ").strip() for c in already_have}
            if normalized_chunk not in already_normalized and extra < max_extra_chunks:
                extra_parts.append(f"[Extra from {src}]\n{chunk_text}")
                extra += 1
    
    return "\n\n".join(extra_parts)


def tool_retrieve_papers(query: str, top_k: int = QDRANT_TOP_K_DEFAULT) -> dict:
    """检索论文文献，Agent 可指定检索数量（0~20篇）。返回丰富的上下文片段。"""
    if top_k <= 0:
        return {
            "context": "",
            "sources": [],
            "result": "未检索文献（top_k=0）。请直接基于自身知识回答。"
        }
    
    top_k = max(QDRANT_TOP_K_MIN, min(QDRANT_TOP_K_MAX, top_k))
    
    # Stage 1: retrieve chunks
    retriever = get_retriever(top_k)
    docs = retriever.invoke(query)
    
    # Format initial context
    context = format_docs_with_sources(docs)
    
    # Stage 2: expand context from top sources
    extra = _expand_context(docs, max_extra_chunks=3)
    if extra:
        context += "\n\n--- Additional context from top papers ---\n" + extra
    
    # Build sources
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:300].replace("\n", " ")
        sources.append("%s: %s..." % (src, snippet))
    
    result_text = context if context else "未检索到相关文献。"
    return {
        "context": context,
        "sources": sources,
        "result": result_text
    }


def tool_search_history(keywords: str, session_id: str) -> dict:
    """在会话历史中搜索关键词。"""
    history = load_session(session_id)
    matches = []
    for msg in history:
        if msg.get("_type") == "title":
            continue
        content = msg.get("content", "")
        if keywords.lower() in content.lower():
            role = msg.get("role", "unknown")
            snippet = content[:300]
            matches.append("[%s]: %s" % (role, snippet))
    
    return {
        "result": "\n\n".join(matches) if matches else "未在历史中找到与'%s'相关的内容。" % keywords
    }


# ── PDF reader: fetch original paper text from disk ──
# Build a filename → path index lazily
_PDF_INDEX = None

def _build_pdf_index():
    idx = {}
    base = "/data/data/pkb/01_raw_data/papers_pdf"
    try:
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.endswith(".pdf"):
                    idx[f] = os.path.join(root, f)
    except:
        pass
    return idx

def _get_pdf_path(source: str) -> str:
    global _PDF_INDEX
    if _PDF_INDEX is None:
        _PDF_INDEX = _build_pdf_index()
    return _PDF_INDEX.get(source, "")

def tool_read_paper_fulltext(source: str, section: str = "abstract") -> dict:
    """从原始 PDF 文件中读取论文内容（需安装 pdftotext 或使用 python 库）。"""
    pdf_path = _get_pdf_path(source)
    if not pdf_path:
        return {"result": "未找到论文文件: %s" % source}
    
    try:
        # Try pdftotext first (fast)
        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            text = result.stdout
        else:
            # Fallback: try python PyPDF2/pypdf
            try:
                from pypdf import PdfReader
                reader = PdfReader(pdf_path)
                text = "\n".join(page.extract_text() for page in reader.pages[:5])
            except:
                # Try pdfminer
                try:
                    from pdfminer.high_level import extract_text
                    text = extract_text(pdf_path)
                except:
                    return {"result": "无法提取PDF文本，请安装 pdftotext 或 pypdf"}
    except:
        return {"result": "读取PDF超时或出错"}
    
    lines = text.split("\n")
    
    if section == "abstract":
        # Try to find Abstract and extract ~500 words after it
        abstract_start = -1
        for i, line in enumerate(lines):
            if "abstract" in line.lower().strip():
                abstract_start = i
                break
        if abstract_start >= 0:
            result_text = "\n".join(lines[abstract_start:abstract_start+50])
        else:
            result_text = "\n".join(lines[:30])
        result_text = result_text[:3000]
    
    elif section == "conclusion":
        # Try to find Conclusion
        conclusion_start = -1
        for i, line in enumerate(lines):
            if "conclusion" in line.lower().strip() or "summary" in line.lower().strip():
                conclusion_start = i
                break
        if conclusion_start >= 0:
            result_text = "\n".join(lines[conclusion_start:conclusion_start+50])
        else:
            result_text = "\n".join(lines[-80:])
        result_text = result_text[:3000]
    
    else:  # full
        result_text = "\n".join(lines[:200])
        result_text = result_text[:5000]
    
    return {
        "result": "\n=== Full text from %s (%s) ===\n%s" % (source, section, result_text)
    }


TOOL_FUNCTIONS = {
    "retrieve_papers": tool_retrieve_papers,
    "read_paper_fulltext": tool_read_paper_fulltext,
    "search_session_history": tool_search_history,
}
