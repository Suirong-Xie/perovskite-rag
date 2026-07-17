#!/usr/bin/env python3
"""
rebuild_checkpoint.py — 扫描 journals_pdf/ 已有文件, 更新 batch_checkpoint.json

场景:
  - 手动下载了一批 PDF 或者用其他工具下载了
  - batch_checkpoint.json 丢了 / 损坏了
  - 想重建 checkpoint 以反映磁盘上真实的下载状态

工作流:
  1. 扫描 journals_pdf/ 下所有 PDF
  2. 从文件名还原 DOI
  3. 在 missing_non_wiley.jsonl 中验证 DOI 存在
  4. 写入 batch_checkpoint.json (标记为 "done")

用法:
  python pipeline\\windows_PC\\rebuild_checkpoint.py
  python pipeline\\windows_PC\\rebuild_checkpoint.py --dry-run    # 仅查看
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path

# ── 配置 ──

PDF_DIR = "./journals_pdf"
MISSING_FILE = "./missing_non_wiley.jsonl"
CHECKPOINT_FILE = "./batch_checkpoint.json"


def doi_from_filename(filename: str, paper_index: dict[str, dict]) -> str | None:
    """从 safe_filename 还原 DOI。

    "10.1038_s41586-021-03446-x.pdf" → "10.1038/s41586-021-03446-x"
    "10.1088_1402-4896_ae3701.pdf" → "10.1088/1402-4896/ae3701"

    尝试多种分割方案, 取 paper_index 中存在的。
    """
    name = filename.removesuffix(".pdf")
    if "_" not in name:
        # 简单 DOI, 没有多级路径
        if name.lower() in paper_index:
            return name
        return None

    parts = name.split("_")
    prefix = parts[0]  # e.g. "10.1088"

    # 尝试不同数量的 segments 作为 DOI 第二部分
    candidates = []
    for split_at in range(1, min(len(parts), 6)):
        suffix = "/".join(parts[1:split_at + 1])
        remain = "_".join(parts[split_at + 1:])
        if remain:
            doi = f"{prefix}/{suffix}/{remain}"
        else:
            doi = f"{prefix}/{suffix}"
        candidates.append(doi)

    # 返回第一个在 paper_index 中存在的
    for doi in candidates:
        if doi.lower() in paper_index:
            return doi

    # 没有匹配 → 返回最合理的猜测
    return candidates[0] if candidates else name


def load_paper_index() -> dict[str, dict]:
    """加载全量论文索引。"""
    index = {}
    if not os.path.exists(MISSING_FILE):
        print(f"⚠️  {MISSING_FILE} not found — will not validate DOIs.")
        return index
    with open(MISSING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            doi = p.get("doi", "")
            if doi:
                index[doi.lower()] = p
    return index


def main():
    parser = argparse.ArgumentParser(description="从 journals_pdf/ 已有文件重建 checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="仅查看, 不写入")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_FILE,
                        help=f"checkpoint 文件路径 (默认: {CHECKPOINT_FILE})")
    args = parser.parse_args()

    if not os.path.isdir(PDF_DIR):
        print(f"❌ {PDF_DIR} not found!")
        sys.exit(1)

    paper_index = load_paper_index()
    print(f"📋 Paper index: {len(paper_index)} entries\n")

    # ── 扫描 PDF ──
    found: dict[str, str] = {}   # doi_lower → filename
    unknown: list[str] = []       # filenames that can't be matched

    for root, dirs, files in os.walk(PDF_DIR):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            doi = doi_from_filename(fname, paper_index)
            if doi and doi.lower() in paper_index:
                found[doi.lower()] = fname
            else:
                unknown.append(os.path.join(os.path.basename(root), fname))

    # ── 加载已有 checkpoint (保留旧的 "done" 记录) ──
    old_checkpoint: dict[str, str] = {}
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint, "r", encoding="utf-8") as f:
            old_checkpoint = json.load(f)
    old_count = sum(1 for v in old_checkpoint.values() if v == "done")

    # ── 合并: 磁盘上有的全部标 done ──
    new_checkpoint = dict(old_checkpoint)
    new_found = 0
    for doi_lower in found:
        if new_checkpoint.get(doi_lower) != "done":
            new_checkpoint[doi_lower] = "done"
            new_found += 1

    # ── 报告 ──
    print(f"📂 Scanned {PDF_DIR}/")
    print(f"   PDFs matched to paper index:  {len(found):>6}")
    print(f"   PDFs with unknown DOI:         {len(unknown):>6}")
    print(f"   Old checkpoint entries (done): {old_count:>6}")
    print(f"   New entries added:             {new_found:>6}")
    print(f"   Total checkpoint entries:      {sum(1 for v in new_checkpoint.values() if v == 'done'):>6}")

    if unknown:
        print(f"\n⚠️  {len(unknown)} file(s) could not be matched to any paper:")
        for f in unknown[:20]:
            print(f"     {f}")
        if len(unknown) > 20:
            print(f"     ... and {len(unknown) - 20} more")

    if args.dry_run:
        print(f"\n💡 Dry run — no changes written. Remove --dry-run to update {args.checkpoint}.")
        return

    # ── 写入 ──
    with open(args.checkpoint, "w", encoding="utf-8") as f:
        json.dump(new_checkpoint, f, indent=2)

    total_done = sum(1 for v in new_checkpoint.values() if v == "done")
    print(f"\n✅ Wrote {total_done} entries to {args.checkpoint}")


if __name__ == "__main__":
    main()
