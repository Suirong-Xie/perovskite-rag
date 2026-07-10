"""
PerovskiteGPT V5 — Semantic Scholar API 服务

提供的 Agent 工具:
  - search_semantic_scholar: 搜索学术论文 (2亿+ 覆盖, 含引用数/DOI)

API: https://api.semanticscholar.org/graph/v1
  - 免费 tier: 100 req / 5min (with API key), 1 req / sec
  - 覆盖: 2亿+ 论文, 全学科
  - 优势: 补 local RAG (504篇) 和 arXiv (预印本) 之间的空白
         — 已发表的期刊论文, 带引用数作为质量信号
"""

import time
import urllib.request
import urllib.parse
import json
from typing import Optional

from ..core.config import S2_API_KEY

# ── 配置 ──

S2_API_URL = "https://api.semanticscholar.org/graph/v1"

# 搜索返回字段
SEARCH_FIELDS = [
    "title",
    "abstract",
    "authors",
    "year",
    "venue",
    "citationCount",
    "externalIds",
    "fieldsOfStudy",
    "openAccessPdf",
]

# 速率限制
_MIN_INTERVAL = 1.2  # 秒, < 1 req/sec
_last_request_time = 0.0


def _rate_limit():
    """确保请求间隔 >= _MIN_INTERVAL。"""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


# ── API 调用 ──

def _s2_get(endpoint: str, params: dict) -> dict:
    """GET Semantic Scholar API, 带 key 和速率限制。"""
    _rate_limit()

    url = f"{S2_API_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    req.add_header("x-api-key", S2_API_KEY)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[S2] HTTP {e.code}: {body[:200]}", flush=True)
        return {}
    except Exception as e:
        print(f"[S2] Request error: {e}", flush=True)
        return {}


# ── 搜索 ──

def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    fields_of_study: Optional[str] = None,
) -> list[dict]:
    """搜索 Semantic Scholar 学术论文。

    Args:
        query: 英文搜索查询
        max_results: 返回结果数 (默认5, 最大20)
        year_min: 最早年份 (如 2024)
        year_max: 最晚年
        fields_of_study: 学科过滤 (如 "Materials Science", "Chemistry")

    Returns:
        [{paperId, title, abstract, authors, year, venue,
          citationCount, externalIds, fieldsOfStudy, openAccessPdf}, ...]
    """
    max_results = min(max_results, 20)

    params = {
        "query": query,
        "limit": max_results,
        "fields": ",".join(SEARCH_FIELDS),
    }
    if year_min:
        params["year"] = f"{year_min}-{year_max or ''}"
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    data = _s2_get("paper/search", params)
    papers = data.get("data", [])

    results = []
    for p in papers:
        # 提取作者名
        authors = [a.get("name", "") for a in p.get("authors", []) if a.get("name")]

        # 提取外部 ID
        ext = p.get("externalIds", {}) or {}

        # 构建结果
        r = {
            "paperId": p.get("paperId", ""),
            "title": (p.get("title") or "").replace("\n", " "),
            "abstract": (p.get("abstract") or "")[:800],  # 截断
            "authors": authors,
            "year": p.get("year"),
            "venue": (p.get("venue") or ""),
            "citationCount": p.get("citationCount", 0),
            "doi": ext.get("DOI", ""),
            "arxivId": ext.get("ArXiv", ""),
            "fieldsOfStudy": p.get("fieldsOfStudy") or [],
            "openAccessUrl": (p.get("openAccessPdf") or {}).get("url", ""),
        }
        results.append(r)

    print(f"[S2] SEARCH: '{query[:60]}' → {len(results)} papers "
          f"(total: {data.get('total', '?')})", flush=True)
    return results


# ── 论文详情 ──

def get_paper_details(paper_id: str) -> Optional[dict]:
    """获取单篇论文详情 (含 TL;DR)。

    Args:
        paper_id: Semantic Scholar paperId 或 DOI (如 "10.1038/s41560-024-01579-7")

    Returns:
        {paperId, title, abstract, tldr, authors, year, venue,
         citationCount, externalIds, ...}
    """
    data = _s2_get(f"paper/{urllib.parse.quote(paper_id, safe='')}", {
        "fields": ",".join(SEARCH_FIELDS + ["tldr"]),
    })
    if not data or not data.get("paperId"):
        return None

    authors = [a.get("name", "") for a in data.get("authors", []) if a.get("name")]
    ext = data.get("externalIds", {}) or {}

    return {
        "paperId": data.get("paperId", ""),
        "title": (data.get("title") or "").replace("\n", " "),
        "abstract": (data.get("abstract") or "")[:1200],
        "tldr": (data.get("tldr") or {}).get("text", ""),
        "authors": authors,
        "year": data.get("year"),
        "venue": (data.get("venue") or ""),
        "citationCount": data.get("citationCount", 0),
        "doi": ext.get("DOI", ""),
        "arxivId": ext.get("ArXiv", ""),
        "fieldsOfStudy": data.get("fieldsOfStudy") or [],
        "openAccessUrl": (data.get("openAccessPdf") or {}).get("url", ""),
    }
