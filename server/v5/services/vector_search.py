"""
PerovskiteGPT v1.5 — 向量检索核心

基于 numpy 的余弦相似度检索，配合 Ollama mxbai-embed-large 做 query 向量化。
支持 journal_rank 加权提升 Nature 系列期刊排名。

用法:
  作为模块: from .vector_search import search
  CLI 测试:  python -m server.v5.services.vector_search "查询文本" --top_k 5
"""

import json
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import requests

# ── 配置（运行时由 retrieval / config 注入） ──

OLLAMA_EMBED_URL = "http://127.0.0.1:11435/api/embed"
EMBED_MODEL = "mxbai-embed-large"

JOURNAL_WEIGHTS = {1: 1.5, 2: 1.4, 3: 1.3, 4: 1.2, 5: 1.15, 6: 1.05, 7: 1.0}
JOURNAL_NAMES = {1: "Nature", 2: "NatEnergy", 3: "NatMater", 4: "NatPhoton",
                 5: "NatNanotech", 6: "NatComm", 7: "Other"}

# 数据路径（由 init 时设置）
_data_dir: Path = None
_vectors = None
_texts = None

# S2 向量库
_s2_data_dir: Path = None
_s2_vectors = None
_s2_texts = None
_s2_loaded = False


def init(data_dir: str | Path):
    """初始化向量库路径。在服务启动时调用一次。"""
    global _data_dir
    _data_dir = Path(data_dir)


def _ensure_loaded():
    """惰性加载向量和文本数据。"""
    global _vectors, _texts

    if _vectors is not None and _texts is not None:
        return _vectors, _texts

    if _data_dir is None:
        raise RuntimeError("vector_search not initialized — call init(data_dir) first")

    vec_path = _data_dir / "vectors.npy"
    txt_path = _data_dir / "texts.jsonl"

    if not vec_path.exists():
        raise FileNotFoundError(f"vectors.npy not found at {vec_path}")
    if not txt_path.exists():
        raise FileNotFoundError(f"texts.jsonl not found at {txt_path}")

    print(f"[vector_search] Loading index from {_data_dir}...", file=sys.stderr, flush=True)
    t0 = time.time()
    _vectors = np.load(vec_path)
    _texts = []
    with open(txt_path, "r") as f:
        for line in f:
            _texts.append(json.loads(line))

    print(f"[vector_search] Loaded {_vectors.shape[0]} vectors, {len(_texts)} texts "
          f"({time.time() - t0:.1f}s)", file=sys.stderr, flush=True)
    return _vectors, _texts


def search(query: str, top_k: int = 10, journal_boost: bool = True) -> list[dict]:
    """语义搜索。

    Args:
        query: 英文搜索查询
        top_k: 返回结果数
        journal_boost: 是否启用期刊加权

    Returns:
        [{rank, similarity, journal_rank, journal_name, source, path, content, idx}, ...]
    """
    vectors, texts = _ensure_loaded()

    # 获取 query 向量
    resp = requests.post(OLLAMA_EMBED_URL,
                         json={"model": EMBED_MODEL, "input": [query]},
                         timeout=30)
    resp.raise_for_status()
    vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    # 余弦相似度
    sims = vectors @ vec

    # 候选池: top_k * 3
    cand = min(top_k * 3, len(sims))
    top_idx = np.argpartition(sims, -cand)[-cand:]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    if journal_boost:
        scored = [
            (idx, float(sims[idx]) * JOURNAL_WEIGHTS.get(texts[idx].get("journal_rank", 7), 1.0))
            for idx in top_idx
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_idx = [s[0] for s in scored[:top_k]]
    else:
        top_idx = top_idx[:top_k]

    results = []
    for idx in top_idx:
        rec = texts[idx]
        jr = rec.get("journal_rank", 7)
        content = rec.get("content", "") or ""
        results.append({
            "rank": len(results) + 1,
            "similarity": round(float(sims[idx]), 4),
            "journal_rank": jr,
            "journal_name": JOURNAL_NAMES.get(jr, rec.get("journal", "Other")),
            "source": rec.get("source", ""),
            "path": rec.get("path", ""),
            "content": content[:6000],
            "idx": int(idx),
        })

    return results


# ── S2 向量库 ──

def init_s2(data_dir: str | Path):
    """初始化 S2 向量库路径。"""
    global _s2_data_dir
    _s2_data_dir = Path(data_dir)


def _ensure_s2_loaded():
    """惰性加载 S2 向量和文本数据。"""
    global _s2_vectors, _s2_texts, _s2_loaded

    if _s2_loaded:
        return _s2_vectors, _s2_texts

    if _s2_data_dir is None:
        _s2_loaded = True
        return None, None

    vec_path = _s2_data_dir / "vectors.npy"
    txt_path = _s2_data_dir / "texts.jsonl"

    if not vec_path.exists() or not txt_path.exists():
        print(f"[vector_search] S2 index not found at {_s2_data_dir}, skipping S2.",
              file=sys.stderr, flush=True)
        _s2_loaded = True
        return None, None

    print(f"[vector_search] Loading S2 index from {_s2_data_dir}...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    _s2_vectors = np.load(vec_path)
    _s2_texts = []
    with open(txt_path, "r") as f:
        for line in f:
            _s2_texts.append(json.loads(line))

    _s2_loaded = True
    print(f"[vector_search] S2: {_s2_vectors.shape[0]} vectors, {len(_s2_texts)} texts "
          f"({time.time() - t0:.1f}s)", file=sys.stderr, flush=True)
    return _s2_vectors, _s2_texts


def _s2_citation_boost(citation_count: int) -> float:
    """引用数 boost: log10(citations) * factor, 上限 30%。"""
    import math
    if not citation_count or citation_count <= 0:
        return 1.0
    boost = 1.0 + math.log10(citation_count + 1) * S2_CITATION_BOOST_FACTOR
    return min(boost, 1.3)


S2_CITATION_BOOST_FACTOR = 0.1


def search_all(
    query: str,
    top_k: int = 10,
    journal_boost: bool = True,
    include_s2: bool = True,
) -> list[dict]:
    """双集合搜索: Nature + S2。

    Nature 全文结果权重 1.0, S2 Tier 1 (全文) ×0.9, S2 Tier 2 (摘要) ×0.8。
    S2 结果额外附加引用数 boost。

    Args:
        query: 英文搜索查询
        top_k: 返回结果数
        journal_boost: 是否启用期刊加权
        include_s2: 是否包含 S2 集合

    Returns:
        [{rank, similarity, journal_name, source, content, _s2_citation_count?, ...}, ...]
    """
    # 1. Nature 集合搜索
    nature_results = search(query, top_k=top_k * 2, journal_boost=journal_boost)

    if not include_s2:
        return nature_results[:top_k]

    # 2. S2 集合搜索
    s2_vecs, s2_texts = _ensure_s2_loaded()
    if s2_vecs is None or s2_texts is None:
        return nature_results[:top_k]

    # 获取 query 向量 (复用 search 中已计算的, 这里重新算一次)
    resp = requests.post(OLLAMA_EMBED_URL,
                         json={"model": EMBED_MODEL, "input": [query]},
                         timeout=30)
    resp.raise_for_status()
    vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    # S2 余弦相似度
    s2_sims = s2_vecs @ vec
    cand = min(top_k * 3, len(s2_sims))
    s2_top_idx = np.argpartition(s2_sims, -cand)[-cand:]
    s2_top_idx = s2_top_idx[np.argsort(-s2_sims[s2_top_idx])]

    s2_results = []
    for idx in s2_top_idx[:top_k * 2]:
        rec = s2_texts[idx]
        tier = rec.get("_s2_tier", 2)
        citations = rec.get("_s2_citation_count", 0) or 0

        # S2 权重: Tier 1 = 0.9, Tier 2 = 0.8
        tier_weight = 0.9 if tier == 1 else 0.8
        citation_boost = _s2_citation_boost(citations)
        score = float(s2_sims[idx]) * tier_weight * citation_boost

        s2_results.append({
            "idx": -1,  # S2 论文没有 Nature 索引
            "similarity": round(float(s2_sims[idx]), 4),
            "score": round(score, 4),
            "journal_rank": rec.get("journal_rank", 8),
            "journal_name": rec.get("journal", "Unknown"),
            "source": rec.get("source", ""),
            "content": (rec.get("content", "") or "")[:6000],
            "_s2_citation_count": citations,
            "_s2_year": rec.get("_s2_year"),
            "_s2_tier": tier,
            "_s2_paper_id": rec.get("_s2_paper_id", ""),
            "_s2_doi": rec.get("_s2_doi", ""),
            "is_s2": True,
        })

    # 3. 合并排序
    # S2 分数归一化到 Nature 相似度范围
    if s2_results:
        max_s2_score = max(r["score"] for r in s2_results)
        if max_s2_score > 0:
            for r in s2_results:
                r["similarity"] = round(r["score"] / max_s2_score * 0.85, 4)

    merged = nature_results + s2_results
    merged.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    merged = merged[:top_k]

    # 重新编号 rank
    for i, r in enumerate(merged):
        r["rank"] = i + 1

    return merged


def search_s2(
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """S2-only 语义搜索（已替代 Nature + S2 双集合）。

    S2 已覆盖 Nature 论文，无需单独维护 Nature 向量库。
    Tier 1 (全文) ×0.9, Tier 2 (摘要) ×0.8, 外加引用数 boost。
    """
    s2_vecs, s2_texts = _ensure_s2_loaded()
    if s2_vecs is None or s2_texts is None:
        return []

    # 获取 query 向量
    resp = requests.post(OLLAMA_EMBED_URL,
                         json={"model": EMBED_MODEL, "input": [query]},
                         timeout=30)
    resp.raise_for_status()
    vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    # 余弦相似度
    s2_sims = s2_vecs @ vec
    cand = min(top_k * 3, len(s2_sims))
    s2_top_idx = np.argpartition(s2_sims, -cand)[-cand:]
    s2_top_idx = s2_top_idx[np.argsort(-s2_sims[s2_top_idx])]

    results = []
    for idx in s2_top_idx[:top_k]:
        rec = s2_texts[idx]
        tier = rec.get("_s2_tier", 2)
        citations = rec.get("_s2_citation_count", 0) or 0

        tier_weight = 0.9 if tier == 1 else 0.8
        citation_boost = _s2_citation_boost(citations)
        score = float(s2_sims[idx]) * tier_weight * citation_boost

        results.append({
            "rank": 0,
            "similarity": round(score, 4),
            "journal_rank": rec.get("journal_rank", 8),
            "journal_name": rec.get("journal", "Unknown"),
            "source": rec.get("source", ""),
            "content": (rec.get("content", "") or "")[:6000],
            "_s2_citation_count": citations,
            "_s2_year": rec.get("_s2_year"),
            "_s2_tier": tier,
            "_s2_paper_id": rec.get("_s2_paper_id", ""),
            "_s2_doi": rec.get("_s2_doi", ""),
            "is_s2": True,
        })

    # 归一化 + 重排
    if results:
        max_score = max(r["similarity"] for r in results)
        if max_score > 0:
            for r in results:
                r["similarity"] = round(r["similarity"] / max_score, 4)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


# ── CLI ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PerovskiteGPT v1.5 向量检索")
    parser.add_argument("query", type=str, help="英文搜索查询")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--no-journal-boost", action="store_true")
    parser.add_argument("--data-dir", type=str, default="/data1/perovskite-rag/data/vector_db",
                        help="向量库目录 (默认 data/vector_db)")
    args = parser.parse_args()

    init(args.data_dir)
    start = time.time()
    results = search(args.query, top_k=args.top_k,
                     journal_boost=not args.no_journal_boost)
    elapsed = time.time() - start

    print(f"[vector_search] {len(results)} results ({elapsed:.2f}s)",
          file=sys.stderr, flush=True)
    json.dump(results, sys.stdout, ensure_ascii=False)
    print()
