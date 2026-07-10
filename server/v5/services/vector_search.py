"""
PerovskiteGPT V5 — 向量检索核心

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


# ── CLI ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PerovskiteGPT V5 向量检索")
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
