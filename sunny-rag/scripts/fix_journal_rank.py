#!/usr/bin/env python3
"""
修复 texts.jsonl 中的 journal_rank 字段。
根据 source 文件名前缀推断期刊排名。
"""
import json
from pathlib import Path

DATA_DIR = Path("/data1/perovskite-rag/sunny-rag/data")

PREFIX_RANK = {
    "Nature_": 1,
    "NatEnergy_": 2,
    "NatMater_": 3,
    "NatPhoton_": 4,
    "NatNanotech_": 5,
    "NatComm_": 6,
}
DEFAULT_RANK = 7

def source_to_rank(source: str) -> int:
    for prefix, rank in PREFIX_RANK.items():
        if source.startswith(prefix):
            return rank
    return DEFAULT_RANK

def main():
    path = DATA_DIR / "texts.jsonl"
    with open(path, "r") as f:
        lines = f.readlines()

    stats = {}
    fixed = 0
    total = len(lines)

    for i, line in enumerate(lines):
        rec = json.loads(line)
        old_rank = rec["journal_rank"]
        new_rank = source_to_rank(rec["source"])
        if old_rank != new_rank:
            rec["journal_rank"] = new_rank
            lines[i] = json.dumps(rec, ensure_ascii=False) + "\n"
            fixed += 1
        stats[new_rank] = stats.get(new_rank, 0) + 1

    names = {1:"Nature",2:"NatEnergy",3:"NatMater",4:"NatPhoton",5:"NatNanotech",6:"NatComm",7:"Other"}
    print(f"Total: {total}")
    print(f"Fixed (was default 7, now correct): {fixed}")
    print()
    for k in sorted(stats):
        print(f"  Rank {k} ({names[k]}): {stats[k]} ({stats[k]/total*100:.1f}%)")

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"\nWritten back to {path} ✅")

if __name__ == "__main__":
    main()
