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
# S2 免费 tier: 100 req/5min ≈ 0.33 req/sec 持续速率
# 这里设 3.1s 间隔 ≈ 0.32 req/sec, 留 5% 安全边际
_MIN_INTERVAL = 3.1  # 秒, ~19 req/min, 95 req/5min
_FAST_INTERVAL = 1.2  # 交互式搜索用, < 1 req/sec
_last_request_time = 0.0
# S2 offset 上限 (API 限制: offset + limit < 1000, 即 max offset=999)
OFFSET_MAX = 999
# 实际可用: limit=100 时 max offset=899, 共 900 条
# 最后一页需调整: offset=900 时 limit 减为 99 可满足 900+99=999 < 1000
BULK_PAGE_SIZE = 100


def _rate_limit(fast: bool = False):
    """确保请求间隔 >= _MIN_INTERVAL。
    fast=True 用于交互式搜索 (1.2s), 默认用于批量 (3.1s)。
    """
    global _last_request_time
    interval = _FAST_INTERVAL if fast else _MIN_INTERVAL
    elapsed = time.time() - _last_request_time
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_request_time = time.time()


# ── API 调用 ──

def _s2_get(endpoint: str, params: dict, retries: int = 3) -> dict:
    """GET Semantic Scholar API, 带 key、速率限制和自动重试。

    Args:
        retries: 429/5xx 时的最大重试次数
    """
    url = f"{S2_API_URL}/{endpoint}?{urllib.parse.urlencode(params)}"

    for attempt in range(retries + 1):
        _rate_limit()

        req = urllib.request.Request(url)
        req.add_header("x-api-key", S2_API_KEY)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < retries:
                wait = (attempt + 1) * 10
                print(f"[S2] 429 rate limited, retry after {wait}s (attempt {attempt+1}/{retries})...", flush=True)
                time.sleep(wait)
                continue
            print(f"[S2] HTTP {e.code}: {body[:200]}", flush=True)
            return {}
        except Exception as e:
            if attempt < retries:
                wait = (attempt + 1) * 5
                print(f"[S2] Request error: {e}, retry after {wait}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"[S2] Request error (exhausted retries): {e}", flush=True)
            return {}

    return {}


# ── 搜索 ──

def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    offset: int = 0,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    fields_of_study: Optional[str] = None,
) -> list[dict]:
    """搜索 Semantic Scholar 学术论文。

    Args:
        query: 英文搜索查询
        max_results: 返回结果数 (默认5, 最大100)
        offset: 分页偏移 (默认0, 最大9999)
        year_min: 最早年份 (如 2024)
        year_max: 最晚年
        fields_of_study: 学科过滤 (如 "Materials Science", "Chemistry")

    Returns:
        [{paperId, title, abstract, authors, year, venue,
          citationCount, externalIds, fieldsOfStudy, openAccessPdf}, ...]
    """
    max_results = min(max_results, 100)
    offset = min(offset, OFFSET_MAX)

    params = {
        "query": query,
        "limit": max_results,
        "offset": offset,
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


# ── 批量搜索 ──

def search_semantic_scholar_bulk(
    query: str,
    max_total: Optional[int] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    fields_of_study: Optional[str] = None,
) -> list[dict]:
    """翻页批量搜索, 获取某个 query 的全部结果。

    S2 API 限制 offset + limit < 1000, 因此每 query 最多拿约 1000 条。
    循环翻页直到:
      - 返回结果不足一页 (到底了)
      - 达到 max_total
      - offset 达到可用的上限 (约 900)

    Args:
        query: 英文搜索查询
        max_total: 最大返回条数 (None = 不额外限制)
        year_min/year_max: 年份范围
        fields_of_study: 学科过滤

    Returns:
        [{paperId, title, abstract, ...}, ...]
    """
    all_papers: list[dict] = []
    offset = 0
    page = 0

    while True:
        page += 1

        # 最后一页可能需要减小 limit 以遵守 offset+limit < 1000
        remaining = OFFSET_MAX + 1 - offset  # 还能拿多少
        page_size = min(BULK_PAGE_SIZE, remaining)
        if page_size <= 0:
            print(f"[S2 BULK] '{query[:60]}' offset={offset} >= limit, stopping.", flush=True)
            break

        batch = search_semantic_scholar(
            query=query,
            max_results=page_size,
            offset=offset,
            year_min=year_min,
            year_max=year_max,
            fields_of_study=fields_of_study,
        )

        if not batch:
            print(f"[S2 BULK] '{query[:60]}' page {page}: 0 results, done.", flush=True)
            break

        all_papers.extend(batch)

        print(f"[S2 BULK] '{query[:60]}' page {page}: "
              f"offset={offset} +{len(batch)} → total {len(all_papers)}", flush=True)

        # 停止条件
        if len(batch) < page_size:
            # 最后一批, 结果不足一页
            break
        if max_total and len(all_papers) >= max_total:
            break
        if offset + page_size >= OFFSET_MAX:
            print(f"[S2 BULK] '{query[:60]}' reached offset limit ({OFFSET_MAX}), stopping.", flush=True)
            break

        offset += page_size

    if max_total and len(all_papers) > max_total:
        all_papers = all_papers[:max_total]

    print(f"[S2 BULK] '{query[:60]}' DONE: {len(all_papers)} papers collected.", flush=True)
    return all_papers


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
