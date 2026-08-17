#!/usr/bin/env python3
"""
s2_bulk_harvest.py — 从 Semantic Scholar 批量收割钙钛矿论文元数据

策略: 按年份分片查询，每年每个 query 独立翻页，避免 offset 超限 (9999)。
Checkpoint/resume 支持中断后继续。

用法:
  python3 s2_bulk_harvest.py                 # 全量收割
  python3 s2_bulk_harvest.py --dry-run       # 只统计每个 query+year 的结果数
  python3 s2_bulk_harvest.py --year 2024     # 只收单年
  python3 s2_bulk_harvest.py --query-idx 0   # 只跑第 0 个 query (调试用)
"""

import json
import os
import sys
import time
import argparse
from typing import Optional

# 确保能找到 server/app 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from app.services.semantic_scholar_service import search_semantic_scholar_bulk, search_semantic_scholar

# ── 配置 ──

BASE_DIR = "/data1/perovskite-rag"
CORPUS_DIR = os.path.join(BASE_DIR, "data", "s2_corpus")
CHECKPOINT_FILE = os.path.join(CORPUS_DIR, "harvest_checkpoint.json")
OUTPUT_FILE = os.path.join(CORPUS_DIR, "s2_papers_raw.jsonl")

os.makedirs(CORPUS_DIR, exist_ok=True)

# 搜索查询（覆盖钙钛矿太阳能电池的主要方向）
QUERIES = [
    # 主查询
    "perovskite solar cell",
    "perovskite solar cells",
    # 材料变体
    "lead halide perovskite solar",
    "tin perovskite solar cell",
    "inorganic perovskite solar cell",
    "mixed cation perovskite solar",
    "2D perovskite solar cell",
    "perovskite tandem solar cell",
    # 器件工程
    "perovskite solar cell interface passivation",
    "perovskite solar cell hole transport layer",
    "perovskite solar cell electron transport layer",
    "perovskite solar cell stability",
    # 特色方向
    "FAPbI3 perovskite solar",
    "wide bandgap perovskite solar",
    "all-perovskite tandem",
    "perovskite solar module",
    "flexible perovskite solar",
    "perovskite quantum dot solar",
]

YEAR_START = 2009
YEAR_END = 2026

FLUSH_INTERVAL = 500  # 每 500 条 flush 一次


# ── Checkpoint ──

def load_checkpoint() -> dict:
    """加载进度: {(query, year): last_offset}"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            raw = json.load(f)
        # key → offset mapping
        return {tuple(k.split("||")): v for k, v in raw.items()}
    return {}


def save_checkpoint(ckpt: dict):
    """保存进度"""
    raw = {"||".join(k): v for k, v in ckpt.items()}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(raw, f, indent=2)


# ── 核心收割逻辑 ──

def harvest_query_year(
    query: str,
    year: int,
    max_per_year: int = 5000,
    checkpoint: Optional[dict] = None,
    dry_run: bool = False,
) -> tuple[int, list[dict]]:
    """收割某个 query + year 的所有论文。

    Returns:
        (total_harvested, papers_list)
    """
    key = f"{query}||{year}"

    # Checkpoint: 如果已完成则跳过
    if checkpoint and key in checkpoint and checkpoint[key] == "done":
        print(f"  [SKIP] '{query[:50]}' year={year} — already done", flush=True)
        return 0, []

    if dry_run:
        # 只查总数
        result = search_semantic_scholar(
            query=query, max_results=1, offset=0,
            year_min=year, year_max=year,
        )
        # 需要单独探总数
        print(f"  [DRY-RUN] '{query[:50]}' year={year}: probing...", flush=True)
        return 0, []

    print(f"  [HARVEST] '{query[:50]}' year={year}: starting...", flush=True)

    papers = search_semantic_scholar_bulk(
        query=query,
        max_total=max_per_year,
        year_min=year,
        year_max=year,
    )

    # 标注 query source
    for p in papers:
        p["_query_source"] = query
        p["_harvested_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # 标记完成
    if checkpoint is not None:
        checkpoint[key] = "done"
        save_checkpoint(checkpoint)

    return len(papers), papers


def flush_papers(papers: list[dict], mode: str = "a"):
    """将论文列表追加写入 JSONL。"""
    with open(OUTPUT_FILE, mode) as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  [FLUSH] {len(papers)} papers → {OUTPUT_FILE}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="S2 批量元数据收割")
    parser.add_argument("--dry-run", action="store_true", help="只统计不下载")
    parser.add_argument("--year", type=int, help="只收指定年份")
    parser.add_argument("--query-idx", type=int, help="只跑指定 query 索引")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复")
    parser.add_argument("--reset", action="store_true", help="清除 checkpoint 重新开始")
    args = parser.parse_args()

    # 确定要跑的 query 列表
    if args.query_idx is not None:
        queries = [QUERIES[args.query_idx]]
        print(f"[HARVEST] Single query: '{queries[0]}'", flush=True)
    else:
        queries = QUERIES
        print(f"[HARVEST] {len(queries)} queries total", flush=True)

    # 确定年份范围
    if args.year:
        years = [args.year]
    else:
        years = list(range(YEAR_START, YEAR_END + 1))
    print(f"[HARVEST] Years: {min(years)}–{max(years)} ({len(years)} years)", flush=True)

    # Checkpoint
    if args.reset and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("[HARVEST] Checkpoint reset.", flush=True)

    checkpoint = load_checkpoint() if not args.dry_run else None
    if checkpoint:
        done_count = len([v for v in checkpoint.values() if v == "done"])
        print(f"[HARVEST] Checkpoint: {done_count} query-year pairs done, "
              f"{len(checkpoint) - done_count} partial.", flush=True)

    # 重置输出文件（除非 resume）
    if not args.resume and not args.dry_run:
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        flush_papers([], "w")  # touch

    total_papers = 0
    start_time = time.time()
    batch: list[dict] = []

    for qi, query in enumerate(queries):
        for yi, year in enumerate(years):
            n, papers = harvest_query_year(
                query=query,
                year=year,
                max_per_year=5000,
                checkpoint=checkpoint,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                continue

            total_papers += n
            batch.extend(papers)

            # 每 FLUSH_INTERVAL 条 flush 一次
            if len(batch) >= FLUSH_INTERVAL:
                flush_papers(batch)
                batch = []

            # ETA
            elapsed = time.time() - start_time
            tasks_done = yi + 1 + qi * len(years)
            tasks_total = len(queries) * len(years)
            eta = (elapsed / tasks_done) * (tasks_total - tasks_done) if tasks_done > 0 else 0
            print(f"[HARVEST] Overall: {total_papers} papers so far, "
                  f"ETA {eta/60:.0f}min remaining", flush=True)

    # 最后 flush
    if batch:
        flush_papers(batch)

    elapsed = time.time() - start_time
    print(f"\n[HARVEST] COMPLETE: {total_papers} papers in {elapsed/60:.1f} min", flush=True)
    print(f"[HARVEST] Output: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
