#!/usr/bin/env python3
"""
Sunny-RAG 搜索工具 v3 — 接入 11,779 篇 Nature 期刊语义分块。
用法：
  python3 search_tool_v3.py "查询文本" [--top_k 10] [--no-journal-boost] [--data-version v3]
输出：一行 JSON（stdout），日志走 stderr

--data-version 参数：
  v1 (默认) — 旧库，188k chunks，从 sunny-rag/data/ 读取（旧 arxiv 批量）
  v2       — 新库，4.6k journals 语义 chunk，从 sunny-rag/data_v2/ 读取
  v3       — 最新库，11,779 篇 Nature 期刊语义分块，从 sunny-rag/data_v3/ 读取
"""

import json, sys, time, argparse, requests
from pathlib import Path
import numpy as np

BASE_DIR = Path("/data1/perovskite-rag/sunny-rag")

OLLAMA_EMBED_URL = "http://127.0.0.1:11435/api/embed"
EMBED_MODEL = "mxbai-embed-large"

# journal_rank: 1=Nature,2=NatEnergy,3=NatMater,4=NatPhoton,5=NatNanotech,6=NatComm,7=Other
JOURNAL_WEIGHTS = {1: 1.5, 2: 1.4, 3: 1.3, 4: 1.2, 5: 1.15, 6: 1.05, 7: 1.0}
JOURNAL_NAMES  = {1: "Nature", 2: "NatEnergy", 3: "NatMater", 4: "NatPhoton",
                  5: "NatNanotech", 6: "NatComm", 7: "Other"}

_vectors = None
_texts  = None
_current_data_version = None


def get_data_dir(version="v1"):
    if version == "v2":
        return BASE_DIR / "data_v2"
    elif version == "v3":
        return BASE_DIR / "data_v3"
    return BASE_DIR / "data"


def ensure_loaded(version="v1"):
    global _vectors, _texts, _current_data_version
    if _vectors is not None and _texts is not None and _current_data_version == version:
        return _vectors, _texts

    data_dir = get_data_dir(version)
    vec_path = data_dir / "vectors.npy"
    txt_path = data_dir / "texts.jsonl"

    if not vec_path.exists():
        print(f"[tool] ERROR: vectors.npy not found at {vec_path}", file=sys.stderr, flush=True)
        sys.exit(1)
    if not txt_path.exists():
        print(f"[tool] ERROR: texts.jsonl not found at {txt_path}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[tool] Loading {version} index from {data_dir}...", file=sys.stderr, flush=True)

    _vectors = np.load(vec_path)
    _texts = []
    with open(txt_path, "r") as f:
        for line in f:
            _texts.append(json.loads(line))

    _current_data_version = version
    print(f"[tool] Loaded {_vectors.shape[0]} vectors, {len(_texts)} texts", file=sys.stderr, flush=True)
    return _vectors, _texts


def search(query, top_k=10, journal_boost=True, data_version="v1"):
    vectors, texts = ensure_loaded(data_version)

    # 获取 query 向量
    resp = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": [query]}, timeout=30)
    resp.raise_for_status()
    vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    # 余弦相似度（向量已归一化）
    sims = vectors @ vec

    # 候选池：取 top_k * 3 个高分候选
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
            "content": content[:6000],  # 截断过长的 content
            "idx": int(idx),
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--no-journal-boost", action="store_true")
    parser.add_argument("--data-version", choices=["v1", "v2", "v3"], default="v1",
                        help="向量库版本: v1=旧库(188k), v2=新journals库(4.6k), v3=最新Nature期刊(11.8k)")
    args = parser.parse_args()

    start = time.time()
    results = search(args.query, top_k=args.top_k,
                     journal_boost=not args.no_journal_boost,
                     data_version=args.data_version)
    elapsed = time.time() - start

    print(f"[tool] Found {len(results)} results from {args.data_version} ({elapsed:.2f}s)",
          file=sys.stderr, flush=True)

    json.dump(results, sys.stdout, ensure_ascii=False)
    print()
