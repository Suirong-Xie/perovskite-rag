#!/usr/bin/env python3
"""
Phase 1 — 相似度搜索工具。
给定查询文本，用 Ollama 嵌入后做余弦相似度搜索。

用法：
  python3 02_search.py "你的查询" [--top_k 10] [--journal_boost]

依赖：
  pip install numpy requests
"""

import json
import sys
import time
import argparse
import requests
from pathlib import Path

import numpy as np

DATA_DIR = Path("/data1/perovskite-rag/sunny-rag/data")
OLLAMA_EMBED_URL = "http://127.0.0.1:11435/api/embed"
EMBED_MODEL = "mxbai-embed-large"

# 期刊权重
JOURNAL_WEIGHTS = {
    1: 1.5,   # Nature
    2: 1.4,   # Nature Energy
    3: 1.3,   # Nature Materials
    4: 1.2,   # Nature Photonics
    5: 1.15,  # Nature Nanotechnology
    6: 1.05,  # Nature Communications
    7: 1.0,   # Other
}

def load_index():
    """加载预计算的向量索引和文本数据。"""
    print("[INDEX] Loading vectors...", end=" ", flush=True)
    t0 = time.time()
    vectors = np.load(DATA_DIR / "vectors.npy")
    print(f"{vectors.shape} ({vectors.nbytes/1024/1024:.0f} MB) in {time.time()-t0:.1f}s")

    print("[INDEX] Loading texts...", end=" ", flush=True)
    t0 = time.time()
    texts = []
    with open(DATA_DIR / "texts.jsonl", "r") as f:
        for line in f:
            texts.append(json.loads(line))
    print(f"{len(texts)} entries in {time.time()-t0:.1f}s")

    return vectors, texts


def embed_query(query: str) -> np.ndarray:
    """调 Ollama 嵌入接口，返回归一化的 1024 维向量。"""
    resp = requests.post(OLLAMA_EMBED_URL, json={
        "model": EMBED_MODEL,
        "input": [query],
    })
    resp.raise_for_status()
    vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
    # L2 归一化（余弦相似度 = 内积）
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def search(query: str, top_k: int = 10, journal_boost: bool = True,
            vectors: np.ndarray = None, texts: list = None):
    """执行相似度搜索。"""
    if vectors is None or texts is None:
        vectors, texts = load_index()

    print(f"\n[SEARCH] Query: {query}")
    print(f"[SEARCH] top_k={top_k}, journal_boost={journal_boost}")

    # 1. 嵌入查询
    t0 = time.time()
    qvec = embed_query(query)
    print(f"[SEARCH] Embedding: {time.time()-t0:.2f}s")

    # 2. 计算余弦相似度 (矩阵乘法，已归一化)
    t0 = time.time()
    similarities = vectors @ qvec  # (N,) dot product
    print(f"[SEARCH] Similarity calc: {time.time()-t0:.2f}s")

    # 3. 最优 top_k (多取一些用于重排)
    candidate_k = min(top_k * 3, len(similarities))
    top_indices = np.argpartition(similarities, -candidate_k)[-candidate_k:]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]

    # 4. 期刊加权重排
    if journal_boost:
        scored = []
        for idx in top_indices:
            rank = texts[idx]["journal_rank"]
            weight = JOURNAL_WEIGHTS.get(rank, 1.0)
            weighted = float(similarities[idx]) * weight
            scored.append((idx, weighted))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = [s[0] for s in scored[:top_k]]
    else:
        top_indices = top_indices[:top_k]

    print(f"[SEARCH] Results ({len(top_indices)}):\n")

    # 5. 输出
    results = []
    for i, idx in enumerate(top_indices):
        sim = float(similarities[idx])
        rec = texts[idx]
        content_preview = rec["content"][:150].replace("\n", " ")
        
        journal_name = {1: "Nature", 2: "NatEnergy", 3: "NatMater",
                       4: "NatPhoton", 5: "NatNanotech", 6: "NatComm",
                       7: "Other"}.get(rec["journal_rank"], "Other")
        
        print(f"  [{i+1}] score={sim:.4f} | rank={rec['journal_rank']} ({journal_name})")
        print(f"       source: {rec['source']}")
        print(f"       {content_preview}")
        print()

        results.append({
            "rank": i + 1,
            "similarity": round(sim, 4),
            "journal_rank": rec["journal_rank"],
            "journal_name": journal_name,
            "source": rec["source"],
            "path": rec["path"],
            "content": rec["content"],
            "idx": int(idx),
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perovskite RAG 相似度搜索")
    parser.add_argument("query", type=str, help="查询文本")
    parser.add_argument("--top_k", type=int, default=10, help="返回结果数")
    parser.add_argument("--no-journal-boost", action="store_true", help="禁用期刊加权重排")
    parser.add_argument("--profile", action="store_true", help="打印加载时间")

    args = parser.parse_args()

    v, t = load_index()
    results = search(args.query, top_k=args.top_k,
                     journal_boost=not args.no_journal_boost,
                     vectors=v, texts=t)

    # JSON 输出
    print("=" * 60)
    print(json.dumps(results, ensure_ascii=False, indent=2)[:2000])
