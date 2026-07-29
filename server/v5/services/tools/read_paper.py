"""read_paper + read_arxiv_paper — 论文全文阅读。"""

import os
import sys
import subprocess
from ...core.schemas import ToolCall, ToolResult

# PyMuPDF4LLM 可选依赖
try:
    import pymupdf4llm  # noqa: F401
    _HAS_PYMUPDF4LLM = True
except ImportError:
    _HAS_PYMUPDF4LLM = False

# 复用 chunking 管线的 markdown 清洗函数
_clean_md = None


def _get_clean_md():
    global _clean_md
    if _clean_md is None:
        _pipeline = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'pipeline')
        if _pipeline not in sys.path:
            sys.path.insert(0, _pipeline)
        from s2_chunk_and_embed import clean_markdown_text
        _clean_md = clean_markdown_text
    return _clean_md


# ── PDF 查找 ──

from .paper_utils import find_pdf_path


# ── 文本清洗 (pdftotext fallback) ──

from ..arxiv_service import clean_paper_text


def _extract_pdf_text(pdf_path: str) -> str:
    """提取 PDF 文本: PyMuPDF4LLM (OCR 已关闭), pdftotext fallback。"""
    if _HAS_PYMUPDF4LLM:
        try:
            result = _extract_with_pymupdf4llm(pdf_path)
            if result:
                return result
        except Exception:
            pass

    # Fallback: pdftotext
    try:
        proc = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0 and proc.stdout and len(proc.stdout.strip()) >= 200:
            cleaned = clean_paper_text(proc.stdout)
            return cleaned[:5000] if len(cleaned) >= 200 else proc.stdout[:5000]
    except Exception:
        pass

    return ""


def _extract_with_pymupdf4llm(pdf_path: str) -> str:
    """PyMuPDF4LLM 结构化 Markdown 提取，OCR 关闭。"""
    import pymupdf4llm
    md_text = pymupdf4llm.to_markdown(pdf_path, use_ocr=False)
    cleaned = _get_clean_md()(md_text)
    if len(cleaned) >= 100:
        return cleaned[:5000]
    return ""




# ── read_paper ──

READ_PAPER_SCHEMA = {
    "name": "read_paper",
    "description": (
        "Read the full text of a specific paper given its source filename. "
        "Use this when search results are insufficient and you need to read "
        "a paper in detail. Source filenames look like 'Nature_2021_s41467-021-26121-1.pdf'."
    ),
    "parameters": {
        "source": "Paper source filename from search results (e.g., 'Nature_2021_xxx.pdf')",
    },
}


def execute_read_paper(arguments: dict) -> tuple:
    source = arguments.get("source", "")
    if not source:
        return (ToolResult(ToolCall("read_paper", arguments), "", error="source is required"), {})

    pdf_path = find_pdf_path(source)
    if not pdf_path:
        return (ToolResult(
            ToolCall("read_paper", arguments),
            f"⚠️ 无全文: {source} — 这篇论文在本地没有 PDF 全文，请跳过它，直接尝试下一篇。"
            f"如果连续 3 篇都没有全文，停止阅读，用已有信息直接回答。",
        ), {})

    try:
        content = _extract_pdf_text(pdf_path)
    except subprocess.TimeoutExpired:
        return (ToolResult(ToolCall("read_paper", arguments), "", error="PDF extraction timed out"), {})
    except Exception as e:
        return (ToolResult(ToolCall("read_paper", arguments), "", error=str(e)), {})

    return (ToolResult(
        ToolCall("read_paper", arguments),
        f"Content of {source} (first 5000 chars):\n\n{content}",
    ), {"source": source, "content": content[:600]})


# ── read_arxiv_paper ──

from ..arxiv_service import download_arxiv_pdf

READ_ARXIV_SCHEMA = {
    "name": "read_arxiv_paper",
    "description": (
        "Download and read the full text of an arXiv paper given its arXiv ID "
        "(e.g., '2606.13414' or '2606.13414v1'). Downloads the PDF from arXiv, "
        "extracts text, and automatically strips references, acknowledgments, "
        "and other non-content sections. Use this to deeply read a paper found "
        "via search_arxiv when the abstract alone is insufficient."
    ),
    "parameters": {
        "arxiv_id": "ArXiv paper ID from search_arxiv results (e.g., '2606.13414')",
    },
}


def execute_read_arxiv(arguments: dict) -> tuple:
    arxiv_id = arguments.get("arxiv_id", "")
    if not arxiv_id:
        return (ToolResult(ToolCall("read_arxiv_paper", arguments), "", error="arxiv_id is required"), {})

    pdf_path = download_arxiv_pdf(arxiv_id)
    if not pdf_path:
        return (ToolResult(
            ToolCall("read_arxiv_paper", arguments),
            f"Failed to download PDF for arXiv:{arxiv_id}",
        ), {})

    try:
        content = _extract_pdf_text(pdf_path)
        source = f"arXiv:{arxiv_id}"
        result = (ToolResult(
            ToolCall("read_arxiv_paper", arguments),
            f"Content of arXiv:{arxiv_id} (first 5000 chars):\n\n{content}",
        ), {"source": source, "content": content[:600]})
    except Exception as e:
        result = (ToolResult(ToolCall("read_arxiv_paper", arguments), "", error=str(e)), {})
    finally:
        try:
            os.unlink(pdf_path)
        except Exception:
            pass
    return result


# ── 统一导出 ──

# read_paper 模块导出两个 schema (read_paper + read_arxiv_paper)
# 通过 __init__.py 的特殊处理来注册它们
SCHEMAS = [READ_PAPER_SCHEMA, READ_ARXIV_SCHEMA]
EXECUTOR_MAP = {
    "read_paper": execute_read_paper,
    "read_arxiv_paper": execute_read_arxiv,
}
