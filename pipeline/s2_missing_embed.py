#!/usr/bin/env python3
"""
补嵌 s2_vector_db 中缺失的 chunk。
从 s2_chunks.jsonl 中找出尚未嵌入的 chunks，嵌入后追加到 texts.jsonl + vectors.npy。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests

# ── 加载 .env ──
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                if _key not in os.environ:
                    os.environ[_key] = _val.strip().strip('"').strip("'")

BASE_DIR = "/data1/perovskite-rag"
CORPUS_DIR = os.path.join(BASE_DIR, "data", "s2_corpus")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "s2_vector_db")

CHUNKED_FILE = os.path.join(CORPUS_DIR, "s2_chunks.jsonl")
TEXTS_FILE = os.path.join(VECTOR_DB_DIR, "texts.jsonl")
VECTORS_FILE = os.path.join(VECTOR_DB_DIR, "vectors.npy")

OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://127.0.0.1:11435/api/embed")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
EMBED_BATCH_SIZE = 20


def _clean_text(text: str) -> str:
    """轻量清洗，去掉控制字符和多余空白。"""
    import re
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {3,}', '  ', text)
    return text.strip()


def embed_batch(texts: list[str]) -> np.ndarray:
    """批量嵌入，返回 L2 归一化向量。"""
    clean_texts = [_clean_text(t) for t in texts]
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": OLLAMA_EMBED_MODEL, "input": clean_texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    vectors = np.array(data["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vectors /= norms
    return vectors


def main():
    # 1. 加载已有文本 ID
    print("[1/4] Loading existing text IDs...", flush=True)
    existing_ids = set()
    if os.path.exists(TEXTS_FILE):
        with open(TEXTS_FILE) as f:
            for line in f:
                obj = json.loads(line.strip())
                existing_ids.add(obj["id"])
    print(f"  {len(existing_ids)} existing texts", flush=True)

    # 2. 加载已有向量
    print("[2/4] Loading existing vectors...", flush=True)
    existing_vectors = None
    if os.path.exists(VECTORS_FILE):
        existing_vectors = np.load(VECTORS_FILE)
        print(f"  {existing_vectors.shape[0]} existing vectors, shape={existing_vectors.shape}", flush=True)
    else:
        print("  No existing vectors", flush=True)
        sys.exit(1)

    # 3. 找出缺失的 chunks
    print("[3/4] Finding missing chunks...", flush=True)
    missing_chunks = []
    with open(CHUNKED_FILE) as f:
        for line in f:
            chunk = json.loads(line.strip())
            if chunk["id"] not in existing_ids:
                missing_chunks.append(chunk)

    print(f"  {len(missing_chunks)} chunks to embed", flush=True)
    if not missing_chunks:
        print("  Nothing to do!", flush=True)
        return

    # Show tier distribution
    from collections import Counter
    tiers = Counter(c.get("_s2_tier", "?") for c in missing_chunks)
    print(f"  Tiers: {dict(tiers)}", flush=True)

    # 4. 嵌入缺失 chunks
    print("[4/4] Embedding...", flush=True)
    new_vectors_list = []

    with open(TEXTS_FILE, "a") as fout:
        for batch_start in range(0, len(missing_chunks), EMBED_BATCH_SIZE):
            batch = missing_chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
            texts = [c["content"] for c in batch]

            try:
                vecs = embed_batch(texts)
            except Exception as e:
                print(f"  Batch {batch_start} error: {e}, retrying one-by-one...", flush=True)
                vecs_list = []
                for j, t in enumerate(texts):
                    try:
                        v = embed_batch([t])
                        vecs_list.append(v[0])
                    except Exception as e2:
                        print(f"    Skipping chunk {batch_start+j} (id={batch[j]['id']}): {e2}", flush=True)
                if vecs_list:
                    vecs = np.array(vecs_list, dtype=np.float32)
                else:
                    vecs = np.zeros((0, 1024), dtype=np.float32)

            if vecs.shape[0] > 0:
                new_vectors_list.append(vecs)

            # 只写成功嵌入的文本
            for j, c in enumerate(batch):
                if j < vecs.shape[0]:
                    fout.write(json.dumps(c, ensure_ascii=False) + "\n")

            progress = min(batch_start + EMBED_BATCH_SIZE, len(missing_chunks))
            print(f"  {progress}/{len(missing_chunks)} chunks embedded", flush=True)

    # 5. 合并向量
    print("[5/5] Merging vectors...", flush=True)
    if new_vectors_list:
        new_vectors = np.concatenate(new_vectors_list, axis=0)
        print(f"  New: {new_vectors.shape}", flush=True)
        merged = np.concatenate([existing_vectors, new_vectors], axis=0)
        print(f"  Merged: {merged.shape}", flush=True)

        # Backup old vectors
        backup = VECTORS_FILE + ".bak"
        np.save(backup, existing_vectors)
        print(f"  Backup saved to {backup}", flush=True)

        np.save(VECTORS_FILE, merged)
        print(f"  Saved: {merged.shape} → {VECTORS_FILE}", flush=True)

        norms = np.linalg.norm(merged, axis=1)
        print(f"  Norm check: min={norms.min():.6f} max={norms.max():.6f} mean={norms.mean():.6f}", flush=True)
    else:
        print("  No new vectors to merge", flush=True)

    # 6. 验证
    print("\n[Verify]")
    final_texts = 0
    with open(TEXTS_FILE) as f:
        for _ in f:
            final_texts += 1
    final_vecs = np.load(VECTORS_FILE)
    print(f"  texts: {final_texts}, vectors: {final_vecs.shape[0]}", flush=True)
    print(f"  missing after fix: {len(missing_chunks) - (final_texts - len(existing_ids))}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
