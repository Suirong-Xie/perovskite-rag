#!/usr/bin/env python3
"""
给 chunks.jsonl 的每一条 metadata 加上 journal_rank 字段。
journal_rank 从文件名前缀解析，用于检索时的期刊权重排序。
"""
import json, os, re
from tqdm import tqdm

CHUNKS_PATH = "/data1/perovskite-rag/data/chunked_data/chunks.jsonl"
BACKUP_PATH = CHUNKS_PATH + ".bak"

# 期刊缩写 → rank (1=最高优先级)
JOURNAL_RANK_MAP = {
    "Nature":  1,
    "NatEnergy": 2,
    "NatMater": 3,
    "NatPhoton": 4,
    "NatNanotech": 5,
    "NatComm": 6,
    # 非 Nature 系列的默认 7
}

def get_journal_rank(source: str) -> int:
    """从文件名解析期刊 rank"""
    for prefix, rank in JOURNAL_RANK_MAP.items():
        if source.startswith(prefix + "_"):
            return rank
    return 7

print("=" * 60)
print("给 chunks.jsonl 添加 journal_rank")
print("=" * 60)

# 统计
total = 0
updated = 0
rank_counts = {}
missing = 0

# 先备份
print(f"\n备份原文件到 {BACKUP_PATH} ...")
os.system(f"cp {CHUNKS_PATH} {BACKUP_PATH}")

# 读出所有行，更新后写出
print(f"处理 {CHUNKS_PATH} ...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
    for line in tqdm(lines):
        total += 1
        d = json.loads(line)
        source = d.get("metadata", {}).get("source", "")
        rank = get_journal_rank(source)
        d["metadata"]["journal_rank"] = rank
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        updated += 1

print(f"\n✅ 完成!")
print(f"   总条数: {total}")
print(f"   已更新: {updated}")
print(f"\n   期刊 rank 分布:")
rank_names = {1: "Nature", 2: "NatEnergy", 3: "NatMater", 4: "NatPhoton", 5: "NatNanotech", 6: "NatComm", 7: "Other"}
for r in sorted(rank_counts):
    name = rank_names.get(r, f"rank_{r}")
    print(f"     rank {r} ({name}): {rank_counts[r]:>8,} 条")

print(f"\n   备份: {BACKUP_PATH}")
print(f"   源文件: {CHUNKS_PATH}")
