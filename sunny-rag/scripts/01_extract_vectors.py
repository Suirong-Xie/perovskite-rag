#!/usr/bin/env python3
"""
Phase 1 — 从 Qdrant local (SQLite) 导出向量 + 文本到轻量索引。
一次性任务，后续直接读取导出的索引文件做相似度搜索。

用法：python3 01_extract_vectors.py
输出：../data/vectors.npy (向量) + ../data/texts.jsonl (文本+元数据)
"""

import sqlite3
import pickle
import json
import sys
import time
from pathlib import Path

SQLITE_PATH = "/data1/perovskite-rag/data/qdrant_data/collection/perovskite_papers/storage.sqlite"
OUTPUT_DIR = Path("/data1/perovskite-rag/sunny-rag/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5000

def extract():
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM points")
    total = cursor.fetchone()[0]
    print(f"Total points in SQLite: {total}")

    cursor = conn.execute("SELECT id, point FROM points ORDER BY id")

    vectors = []
    count = 0
    t0 = time.time()

    # texts 先写 jsonl，每条是 JSON 行（避免一次性吃太多内存）
    texts_f = open(OUTPUT_DIR / "texts.jsonl", "w", encoding="utf-8")
    id_map = {}  # id_str → index in vectors array

    for row in cursor:
        point = pickle.loads(row[1])

        # 提取向量
        vec_raw = point.vector
        if isinstance(vec_raw, dict):
            vec_list = vec_raw.get("", [])
        else:
            vec_list = list(vec_raw)

        if len(vec_list) != 1024:
            print(f"  [SKIP] unexpected vector size {len(vec_list)} for id={row[0]}")
            continue

        vectors.append(vec_list)
        idx = len(vectors) - 1
        id_map[str(point.id)] = idx

        # 提取 payload
        payload = point.payload or {}
        content = payload.get("page_content", "")
        metadata = payload.get("metadata", {})

        rec = {
            "id": point.id,
            "idx": idx,
            "content": content,
            "source": metadata.get("source", ""),
            "path": metadata.get("path", ""),
            "journal_rank": metadata.get("journal_rank", 7),
        }
        texts_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        count += 1
        if count % BATCH_SIZE == 0:
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            print(f"  Processed {count}/{total} ({count/total*100:.1f}%) — {rate:.0f} pts/s")

    texts_f.close()
    conn.close()

    elapsed = time.time() - t0
    print(f"\nDone! Processed {count}/{total} points in {elapsed:.1f}s ({count/elapsed:.0f} pts/s)")

    # 保存向量
    import numpy as np
    vec_array = np.array(vectors, dtype=np.float32)
    print(f"Vectors shape: {vec_array.shape}")

    np.save(OUTPUT_DIR / "vectors.npy", vec_array)

    with open(OUTPUT_DIR / "id_to_index.json", "w") as f:
        json.dump(id_map, f)

    print(f"Saved: {OUTPUT_DIR / 'vectors.npy'}  ({vec_array.nbytes / 1024 / 1024:.1f} MB)")
    print(f"Saved: {OUTPUT_DIR / 'texts.jsonl'}")
    print(f"Saved: {OUTPUT_DIR / 'id_to_index.json'}")

if __name__ == "__main__":
    extract()
