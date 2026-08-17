"""
PerovskiteGPT v1.5 — arXiv API 服务

提供的 Agent 工具:
  - search_arxiv:       搜索 arXiv 预印本 (标题/摘要/作者/PDF链接)
  - download_arxiv_pdf: 下载 arXiv PDF 到临时文件
  - clean_paper_text:   清洗 pdftotext 输出 (去参考文献/致谢/页眉页脚)

arXiv API: http://export.arxiv.org/api/query
  - 免费, 无需 key
  - 钙钛矿领域 16 万+ 预印本
  - 返回 Atom XML, 包含标题/摘要/PDF链接/作者/日期
"""

import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


# ── arXiv API 搜索 ──

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def search_arxiv(query: str, max_results: int = 5,
                 sort_by: str = "relevance") -> list[dict]:
    """搜索 arXiv 预印本。

    Args:
        query: 英文搜索查询 (支持 arXiv 查询语法: ti:, au:, cat: 等)
        max_results: 返回结果数 (默认5, 最大100)
        sort_by: 排序方式 "relevance" | "lastUpdatedDate" | "submittedDate"

    Returns:
        [{title, summary, published, authors, pdf_url, arxiv_id,
          category, journal_ref, comment}, ...]
    """
    params = urllib.parse.urlencode({
        "search_query": query,
        "max_results": max_results,
        "sortBy": sort_by,
    })
    url = f"{ARXIV_API_URL}?{params}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
    except Exception as e:
        print(f"[arXiv] Search error: {e}", flush=True)
        return []

    root = ET.fromstring(data)
    results = []

    for entry in root.findall("atom:entry", ARXIV_NAMESPACES):
        title_el = entry.find("atom:title", ARXIV_NAMESPACES)
        summary_el = entry.find("atom:summary", ARXIV_NAMESPACES)
        published_el = entry.find("atom:published", ARXIV_NAMESPACES)
        journal_el = entry.find("arxiv:journal_ref", ARXIV_NAMESPACES)
        comment_el = entry.find("arxiv:comment", ARXIV_NAMESPACES)

        title = title_el.text.strip() if title_el is not None else ""
        summary = summary_el.text.strip() if summary_el is not None else ""
        published = published_el.text.strip()[:10] if published_el is not None else ""

        # 提取作者
        authors = []
        for author_el in entry.findall("atom:author", ARXIV_NAMESPACES):
            name_el = author_el.find("atom:name", ARXIV_NAMESPACES)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        # 提取 PDF 链接
        pdf_url = ""
        for link_el in entry.findall("atom:link", ARXIV_NAMESPACES):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break

        # 提取 arXiv ID (从 id URL 中)
        id_el = entry.find("atom:id", ARXIV_NAMESPACES)
        arxiv_id = ""
        if id_el is not None and id_el.text:
            # 格式: http://arxiv.org/abs/2606.13414v1
            arxiv_id = id_el.text.strip().split("/abs/")[-1]

        # 提取主分类
        cat_el = entry.find("arxiv:primary_category", ARXIV_NAMESPACES)
        category = cat_el.get("term", "") if cat_el is not None else ""

        results.append({
            "title": title.replace("\n", " "),
            "summary": summary.replace("\n", " "),
            "published": published,
            "authors": authors,
            "pdf_url": pdf_url,
            "arxiv_id": arxiv_id,
            "category": category,
            "journal_ref": journal_el.text.strip() if journal_el is not None else "",
            "comment": comment_el.text.strip() if comment_el is not None else "",
        })

    print(f"[arXiv] SEARCH: '{query[:60]}' → {len(results)} results", flush=True)
    return results


# ── PDF 下载 ──

def download_arxiv_pdf(arxiv_id: str) -> Optional[str]:
    """下载 arXiv PDF 到临时文件。

    Args:
        arxiv_id: arXiv ID (如 "2606.13414" 或 "2606.13414v1")

    Returns:
        临时文件路径, 失败返回 None
    """
    # 去掉版本号，使用最新版本
    base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"

    try:
        req = urllib.request.Request(pdf_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            pdf_data = resp.read()

        # 写入临时文件
        tmp = tempfile.NamedTemporaryFile(suffix=f"_{base_id}.pdf", delete=False)
        tmp.write(pdf_data)
        tmp.close()
        print(f"[arXiv] PDF downloaded: {base_id} ({len(pdf_data)} bytes)", flush=True)
        return tmp.name
    except Exception as e:
        print(f"[arXiv] PDF download error for {arxiv_id}: {e}", flush=True)
        return None


# ── 文本清洗 ──

# 在以下标题处截断 (大小写不敏感, 匹配行首)
CUT_SECTION_TITLES = [
    r"(?i)^\s*\d*\.?\s*references?\s*$",
    r"(?i)^\s*\d*\.?\s*bibliography\s*$",
    r"(?i)^\s*\d*\.?\s*acknowledgments?\s*$",
    r"(?i)^\s*\d*\.?\s*acknowledgements?\s*$",
    r"(?i)^\s*\d*\.?\s*author\s+contributions?\s*$",
    r"(?i)^\s*\d*\.?\s*conflict\s+of\s+interest\s*$",
    r"(?i)^\s*\d*\.?\s*competing\s+interests?\s*$",
    r"(?i)^\s*\d*\.?\s*supplementary\s+(information|materials?)\s*$",
    r"(?i)^\s*\d*\.?\s*supporting\s+information\s*$",
    r"(?i)^\s*\d*\.?\s*data\s+availability\s*(statement)?\s*$",
    r"(?i)^\s*\d*\.?\s*code\s+availability\s*$",
    r"(?i)^\s*\d*\.?\s*declarations?\s*$",
    r"(?i)^\s*\d*\.?\s*funding\s*$",
]

# 页码/噪声行
NOISE_LINE_PATTERNS = [
    r"^\s*\d{1,4}\s*$",                # 纯数字行 (页码)
    r"^arXiv:\s*\d{4}\.\d{4,5}.*$",    # arXiv ID 行
    r"^\s*©\s*20\d{2}.*$",             # 版权声明
    r"^This\s+is\s+a\s+preprint.*$",
    r"^\s*Preprint\s+submitted\s+to.*$",
    r"^\s*\d+\s*\|?\s*Page\s*$",       # "1 | Page"
    r"^\s*[A-Z][a-z]+\s+\d{1,2},\s+20\d{2}\s*$",  # "January 15, 2024"
    r"^https?://doi\.org/.*$",          # DOI 链接行
]


def clean_paper_text(text: str) -> str:
    """清洗 pdftotext 输出: 截断参考文献/致谢, 移除页眉页脚噪声。

    Returns:
        清洗后的文本, 保留正文部分
    """
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append("")
            continue

        # 检查是否遇到截断标题 → 停止
        should_stop = False
        for pattern in CUT_SECTION_TITLES:
            if re.match(pattern, stripped):
                should_stop = True
                break
        if should_stop:
            break

        # 跳过噪声行
        is_noise = False
        for pattern in NOISE_LINE_PATTERNS:
            if re.match(pattern, stripped):
                is_noise = True
                break
        if is_noise:
            continue

        clean_lines.append(line)

    result = "\n".join(clean_lines)

    # 合并连续空行
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
