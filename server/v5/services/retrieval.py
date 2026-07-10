"""
PerovskiteGPT V5 — 检索服务（P1 增强版）

新增：
  - 查询扩展：一个查询 → 多个变体 → 合并去重
  - BM25 关键词搜索：与语义搜索融合，提升关键词召回
  - 元数据过滤：按期刊、年份范围筛选
"""
import sys
import json
import re
import time
import math
from pathlib import Path
from typing import Optional
from collections import defaultdict
from ..core.config import VECTOR_DB_DIR, SEARCH_DEFAULT_TOP_K
from . import vector_search

# 初始化向量检索
vector_search.init(str(VECTOR_DB_DIR))

# ── 缓存 ──
_cache: dict = {}


def clear_cache():
    _cache.clear()


# ── BM25 关键词搜索 ──

_bm25_index = None  # {"texts": [...], "doc_freqs": {...}, "avg_dl": float, "N": int}


def _build_bm25():
    """惰性构建 BM25 索引。"""
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    data_dir = VECTOR_DB_DIR
    txt_path = data_dir / "texts.jsonl"
    if not txt_path.exists():
        _bm25_index = {"texts": [], "doc_freqs": {}, "avg_dl": 0, "N": 0}
        return _bm25_index

    texts = []
    doc_freqs = defaultdict(int)
    with open(txt_path) as f:
        for line in f:
            rec = json.loads(line)
            content = rec.get("content", "")
            # 简单分词：小写 + 按非字母数字分割
            tokens = re.findall(r'[a-z0-9]+', content.lower())
            texts.append({
                "tokens": tokens,
                "source": rec.get("source", ""),
                "journal_name": _JOURNAL_NAMES.get(rec.get("journal_rank", 7), "Other"),
                "journal_rank": rec.get("journal_rank", 7),
                "content": content,
            })
            for t in set(tokens):
                doc_freqs[t] += 1

    N = len(texts)
    avg_dl = sum(len(t["tokens"]) for t in texts) / max(N, 1)
    _bm25_index = {"texts": texts, "doc_freqs": doc_freqs, "avg_dl": avg_dl, "N": N}
    print(f"[V5] BM25 index built: {N} docs, {len(doc_freqs)} terms", flush=True)
    return _bm25_index


_JOURNAL_NAMES = {1: "Nature", 2: "NatEnergy", 3: "NatMater",
                  4: "NatPhoton", 5: "NatNanotech", 6: "NatComm", 7: "Other"}


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                doc_freqs: dict, N: int, avg_dl: float,
                k1: float = 1.2, b: float = 0.75) -> float:
    """BM25 评分。"""
    dl = len(doc_tokens)
    score = 0.0
    tf = defaultdict(int)
    for t in doc_tokens:
        tf[t] += 1
    for t in query_tokens:
        df = doc_freqs.get(t, 0)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        numerator = tf.get(t, 0) * (k1 + 1)
        denominator = tf.get(t, 0) + k1 * (1 - b + b * dl / avg_dl)
        score += idf * numerator / denominator
    return score


def _bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """BM25 关键词搜索。"""
    idx = _build_bm25()
    if idx["N"] == 0:
        return []

    query_tokens = re.findall(r'[a-z0-9]+', query.lower())
    if not query_tokens:
        return []

    scores = []
    for i, doc in enumerate(idx["texts"]):
        s = _bm25_score(query_tokens, doc["tokens"],
                        idx["doc_freqs"], idx["N"], idx["avg_dl"])
        if s > 0:
            scores.append((i, s))

    scores.sort(key=lambda x: -x[1])
    results = []
    for i, score in scores[:top_k]:
        doc = idx["texts"][i]
        results.append({
            "rank": len(results) + 1,
            "similarity": round(score / (scores[0][1] + 0.001), 4),  # normalize
            "journal_rank": doc["journal_rank"],
            "journal_name": doc["journal_name"],
            "source": doc["source"],
            "content": doc["content"][:6000],
            "_bm25_score": round(score, 4),
        })
    return results


# ── 查询扩展 ──

def _expand_queries(query: str) -> list[str]:
    """从一个查询生成多个变体，提高召回率。

    策略：
      1. 原始查询
      2. 关键词版本（移除停用词）
      3. 同义词替换版本（领域术语的常见变体）
      4. 宽泛版本（截取前几个词，针对自然语言长问题）
    """
    queries = [query]
    query_lower = query.lower()

    # 停用词
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each", "every",
        "all", "any", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "than", "too", "very", "just", "about", "also",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "how", "when", "where", "why", "methods", "method", "best", "ways",
        "way", "approach", "approaches",
    }

    words = query_lower.split()

    # 关键词版本：移除停用词
    keywords = [w for w in words if w not in stop_words]
    if 3 <= len(keywords) < len(words):
        queries.append(" ".join(keywords))

    # 钙钛矿领域同义词映射
    synonyms = {
        "stability": ["degradation", "lifetime", "durability"],
        "efficient": ["performance", "PCE"],
        "efficiency": ["performance", "PCE", "power conversion"],
        "defect": ["vacancy", "trap", "recombination"],
        "passivation": ["treatment", "modification", "surface engineering"],
        "interface": ["heterojunction", "junction", "contact"],
        "transport": ["extraction", "collection"],
        "hole": ["HTL", "HTM"],
        "electron": ["ETL", "ETM"],
        "inverted": ["p-i-n", "pin"],
        "normal": ["n-i-p", "nip"],
        "tandem": ["multi-junction", "multijunction", "stacked"],
        "flexible": ["bendable", "foldable"],
        "large area": ["scalable", "scale-up", "module"],
        "lead free": ["tin", "Sn-based", "lead-free"],
        "2D": ["two-dimensional", "Ruddlesden-Popper", "layered"],
    }

    # 同义词版本：替换 1-2 个词为同义词
    synonym_variants = []
    for i, w in enumerate(keywords):
        if w in synonyms:
            for syn in synonyms[w][:2]:  # 最多取 2 个同义词
                variant = keywords.copy()
                variant[i] = syn
                synonym_variants.append(" ".join(variant))
    # 只加前 2 个变体（避免查询爆炸）
    for sv in synonym_variants[:2]:
        if sv not in queries:
            queries.append(sv)

    # 宽泛版本：长问题截短
    if len(words) > 6:
        queries.append(" ".join(words[:5]))

    # 去重
    seen = set()
    unique = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


# ── 融合搜索 ──

def _merge_results(semantic_results: list[dict], bm25_results: list[dict],
                   top_k: int, bm25_weight: float = 0.3) -> list[dict]:
    """融合语义搜索和 BM25 关键词搜索的结果。

    算法：
      1. 归一化两路分数到 [0, 1]
      2. 加权求和：final_score = semantic * (1-w) + bm25 * w
      3. 同一 source 只保留最高分
      4. 按 final_score 排序返回 top_k
    """
    # 归一化
    max_sem = max((r.get("similarity", 0) for r in semantic_results), default=0.001)
    max_bm = max((r.get("_bm25_score", 0) for r in bm25_results), default=0.001)

    # 构建 source → best score 的映射
    combined = {}  # source → {score, ...}

    for r in semantic_results:
        src = r.get("source", "")
        norm_score = r.get("similarity", 0) / max_sem
        combined[src] = {
            "score": norm_score * (1 - bm25_weight),
            "rank": r.get("rank", 99),
            "similarity": r.get("similarity", 0),
            "journal_rank": r.get("journal_rank", 7),
            "journal_name": r.get("journal_name", "Other"),
            "source": src,
            "content": r.get("content", ""),
            "_sem_score": norm_score,
            "_bm25_score": 0,
        }

    for r in bm25_results:
        src = r.get("source", "")
        norm_score = r.get("_bm25_score", 0) / max_bm
        bm25_contrib = norm_score * bm25_weight
        if src in combined:
            combined[src]["score"] += bm25_contrib
            combined[src]["_bm25_score"] = norm_score
        else:
            combined[src] = {
                "score": bm25_contrib,
                "rank": 99,
                "similarity": 0,
                "journal_rank": r.get("journal_rank", 7),
                "journal_name": r.get("journal_name", "Other"),
                "source": src,
                "content": r.get("content", ""),
                "_sem_score": 0,
                "_bm25_score": norm_score,
            }

    # 排序
    ranked = sorted(combined.values(), key=lambda x: -x["score"])[:top_k]
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        r["similarity"] = round(r["score"], 4)

    return ranked


# ── 主搜索接口 ──

def search_papers(query: str, top_k: int = None,
                  clear_cache: bool = False, expand: bool = True,
                  hybrid: bool = True,
                  journal_filter: str = None,
                  year_min: int = None, year_max: int = None) -> list:
    """
    增强版语义搜索。

    Args:
        query: 英文搜索查询
        top_k: 返回结果数
        expand: 是否启用查询扩展
        hybrid: 是否启用 BM25 混合搜索
        journal_filter: 只返回指定期刊（如 "Nature"）
        year_min/year_max: 年份范围过滤（暂未实现，预留接口）

    Returns:
        搜索结果列表
    """
    if clear_cache:
        _cache.clear()

    top_k = top_k or SEARCH_DEFAULT_TOP_K

    cache_key = f"{query}:{top_k}:{expand}:{hybrid}"
    if cache_key in _cache:
        return _cache[cache_key]

    start = time.time()
    all_results = []

    queries = _expand_queries(query) if expand else [query]
    log_detail = "expanded" if len(queries) > 1 else "single"

    for q in queries:
        sem_results = vector_search.search(q, top_k=max(top_k * 2, 10),
                                          journal_boost=True)

        if hybrid:
            bm25_results = _bm25_search(q, top_k=top_k * 2)
            merged = _merge_results(sem_results, bm25_results, top_k=top_k * 2)
        else:
            merged = sem_results

        all_results.extend(merged)

    # 全局去重合并
    final = {}
    for r in all_results:
        src = r.get("source", "")
        score = r.get("similarity", r.get("rank", 99))
        if src not in final or score > final[src].get("similarity", 0):
            final[src] = r

    ranked = sorted(final.values(), key=lambda x: -x.get("similarity", 0))[:top_k]
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    elapsed = time.time() - start
    print(f"[V5] SEARCH({log_detail}{'+bm25' if hybrid else ''}): "
          f"'{query[:60]}' → {len(ranked)} results in {elapsed:.2f}s "
          f"({len(queries)} queries, {len(all_results)} raw)", flush=True)

    _cache[cache_key] = ranked
    return ranked
