#!/usr/bin/env python3
"""
audit_classification.py — 检查已下载 PDF 的归类是否正确，可选自动整理

工作流:
  1. 扫描 journals_pdf/ 下所有 PDF 文件
  2. 从文件名提取 DOI, 在 missing_non_wiley.jsonl 中查找论文元数据
  3. 用 journal_classifier 计算正确的目录名
  4. 统计归类错误, 打印报告
  5. (可选) 将错放的 PDF 移动到正确目录

用法:
  # 仅检查 (dry-run)
  python audit_classification.py

  # 检查 + 整理 (移动错放文件)
  python audit_classification.py --fix

  # 检查指定目录
  python audit_classification.py --pdf-dir /path/to/journals_pdf

  # 检查 + 也清理空目录
  python audit_classification.py --fix --cleanup-empty-dirs
"""

import json
import os
import re
import sys
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

# 共享的期刊分类器
from journal_classifier import classify_venue


# ── DOI 提取 ──

def doi_from_filename(filename: str, paper_index: dict[str, dict] | None = None) -> str:
    """从 PDF 文件名还原 DOI。

    DOI 中可能有多个 / (如 10.1088/1402-4896/ae3701),
    safe_filename 把所有 / 替换为 _, 所以逆向时需要尝试多个分割点。

    "10.1038_s41586-021-03446-x.pdf" → "10.1038/s41586-021-03446-x"
    "10.1088_1402-4896_ae3701.pdf" → "10.1088/1402-4896/ae3701"

    如果提供了 paper_index, 则验证候选 DOI 是否存在。
    """
    name = filename.removesuffix(".pdf")
    if "_" not in name:
        return name

    # 生成所有可能的分割方案 (第一个 _ 之后, 每遇到一个 _ 都可能是一次 / → _ 替换)
    parts = name.split("_")
    if len(parts) < 2:
        return name

    prefix = parts[0]  # e.g. "10.1088"

    # 尝试不同数量的 segments 作为 DOI 的第二部分
    candidates = []
    for split_at in range(1, min(len(parts), 5)):
        suffix = "/".join(parts[1:split_at+1])
        remain = "_".join(parts[split_at+1:])
        if remain:
            doi = f"{prefix}/{suffix}/{remain}"
        else:
            doi = f"{prefix}/{suffix}"
        candidates.append(doi)

    # 如果提供了 paper_index, 返回第一个匹配的
    if paper_index is not None:
        for doi in candidates:
            if doi.lower() in paper_index:
                return doi

    # 否则返回最合理的猜测: 取第一部分作为 suffix, 其余作为额外层级
    return candidates[0] if candidates else name


# ── 加载论文元数据 ──

def load_paper_index(missing_file: str) -> dict[str, dict]:
    """加载 missing_non_wiley.jsonl, 按小写 DOI 建索引。"""
    index = {}
    if not os.path.exists(missing_file):
        print(f"⚠️  Not found: {missing_file}")
        return index

    with open(missing_file, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            doi = p.get("doi", "")
            if doi:
                index[doi.lower()] = p
    return index


# ── 扫描 PDF ──

def scan_pdfs(pdf_base_dir: str, paper_index: dict[str, dict]) -> list[dict]:
    """扫描目录树, 返回每个 PDF 的信息。

    Returns:
        [{filename, current_dir, doi, path}, ...]
    """
    pdfs = []
    for root, dirs, files in os.walk(pdf_base_dir):
        current_dir = os.path.basename(root)
        for fname in files:
            if fname.lower().endswith(".pdf"):
                doi = doi_from_filename(fname, paper_index)
                pdfs.append({
                    "filename": fname,
                    "current_dir": current_dir,
                    "doi": doi,
                    "path": os.path.join(root, fname),
                })
    return pdfs


# ── 主逻辑 ──

def audit(pdf_base_dir: str, paper_index: dict[str, dict],
          fix: bool = False, cleanup_empty: bool = False):
    """审计 PDF 归类, 可选进行修复。"""

    pdfs = scan_pdfs(pdf_base_dir, paper_index)
    print(f"📂 Scanned {len(pdfs)} PDFs in {pdf_base_dir}")
    print(f"📄 Paper index: {len(paper_index)} entries\n")

    # ── 统计 ──
    correct = 0
    mismatched = 0
    no_venue = 0          # 在 index 中没找到
    empty_venue = 0       # 在 index 中找到但 venue 为空
    mismatches = defaultdict(list)  # {current_dir: [(doi, correct_dir, venue), ...]}
    dir_correct_counts = defaultdict(int)     # {dir: count}
    dir_should_counts = defaultdict(int)      # {dir: count}
    dir_current_counts = defaultdict(int)     # {dir: count}

    for pdf in pdfs:
        doi_lower = pdf["doi"].lower()
        paper = paper_index.get(doi_lower)

        dir_current_counts[pdf["current_dir"]] += 1

        if not paper:
            no_venue += 1
            continue

        venue = paper.get("venue", "") or ""
        if not venue:
            empty_venue += 1
            continue

        correct_dir = classify_venue(venue)
        dir_should_counts[correct_dir] += 1

        if pdf["current_dir"] == correct_dir:
            correct += 1
            dir_correct_counts[correct_dir] += 1
        else:
            mismatched += 1
            mismatches[pdf["current_dir"]].append({
                "doi": pdf["doi"],
                "venue": venue,
                "correct_dir": correct_dir,
                "path": pdf["path"],
                "filename": pdf["filename"],
            })

    # ── 打印报告 ──

    total_classified = correct + mismatched
    print("=" * 70)
    print("📊 CLASSIFICATION AUDIT REPORT")
    print("=" * 70)
    print(f"  Total PDFs scanned:     {len(pdfs):5d}")
    print(f"  Correctly classified:   {correct:5d}  ({correct/max(total_classified,1)*100:.1f}%)")
    print(f"  MISCLASSIFIED:          {mismatched:5d}  ({mismatched/max(total_classified,1)*100:.1f}%)")
    print(f"  Not in paper index:     {no_venue:5d}")
    print(f"  Empty venue in index:   {empty_venue:5d}")
    print()

    # Top 错误目录
    if mismatches:
        print(f"{'='*70}")
        print("🔴 TOP MISCLASSIFICATION PATTERNS (current_dir → correct_dir)")
        print(f"{'='*70}")

        # 按错误数排
        pattern_counts = defaultdict(int)
        pattern_examples = {}
        for cur_dir, items in mismatches.items():
            for item in items:
                key = (cur_dir, item["correct_dir"])
                pattern_counts[key] += 1
                if key not in pattern_examples:
                    pattern_examples[key] = item

        for (cur_dir, correct_dir), count in sorted(pattern_counts.items(),
                                                      key=lambda x: -x[1])[:30]:
            ex = pattern_examples[(cur_dir, correct_dir)]
            print(f"  {count:4d}  {cur_dir:<35s} → {correct_dir}")
            print(f"         venue=\"{ex['venue'][:70]}\"")
            print(f"         doi={ex['doi']}")

    if not fix:
        print()
        print("💡 Run with --fix to reorganize misclassified PDFs.")
        return

    # ── 修复: 移动文件 ──

    if not mismatched:
        print("\n✅ No misclassified files to fix.")
        return

    print(f"\n{'='*70}")
    print("🔧 REORGANIZING — moving {mismatched} files...")
    print(f"{'='*70}")

    moved = 0
    errors = 0
    moved_dirs = set()  # 记录哪些目录被写入过

    for cur_dir, items in mismatches.items():
        for item in items:
            src = item["path"]
            correct_dir = item["correct_dir"]
            dst_dir = os.path.join(pdf_base_dir, correct_dir)
            dst = os.path.join(dst_dir, item["filename"])

            # 检查目标是否已存在
            if os.path.exists(dst):
                # 比较文件大小
                src_size = os.path.getsize(src)
                dst_size = os.path.getsize(dst)
                if src_size == dst_size:
                    print(f"  ⏭️  SKIP (target exists, same size): {item['filename']}")
                    os.remove(src)  # 删除源文件 (重复)
                    moved += 1
                    continue
                else:
                    # 不同大小, 加后缀避免覆盖
                    base, ext = os.path.splitext(item["filename"])
                    dst = os.path.join(dst_dir, f"{base}_DUP{ext}")
                    print(f"  ⚠️  Target exists (diff size), saving as: {os.path.basename(dst)}")

            os.makedirs(dst_dir, exist_ok=True)
            try:
                shutil.move(src, dst)
                moved += 1
                moved_dirs.add(correct_dir)
                if moved <= 10 or moved % 50 == 0:
                    print(f"  ✅ [{moved}] {cur_dir}/{item['filename']} → {correct_dir}/")
            except Exception as e:
                errors += 1
                print(f"  ❌ FAILED: {src} → {dst}: {e}")

    print(f"\n📦 Moved: {moved}, Errors: {errors}")

    # ── 清理空目录 ──
    if cleanup_empty:
        print(f"\n🧹 Cleaning empty directories...")
        for root, dirs, files in os.walk(pdf_base_dir, topdown=False):
            for d in dirs:
                dpath = os.path.join(root, d)
                try:
                    if not os.listdir(dpath):
                        os.rmdir(dpath)
                        print(f"  🗑️  Removed empty: {dpath}")
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description="审计已下载 PDF 的期刊归类, 可选自动整理"
    )
    parser.add_argument("--pdf-dir", type=str, default="./journals_pdf",
                        help="PDF 存储根目录 (默认: ./journals_pdf)")
    parser.add_argument("--missing-file", type=str,
                        default="./missing_non_wiley.jsonl",
                        help="论文元数据文件路径")
    parser.add_argument("--fix", action="store_true",
                        help="将错放的 PDF 移动到正确目录")
    parser.add_argument("--cleanup-empty-dirs", action="store_true",
                        help="整理后删除空目录 (需配合 --fix)")
    args = parser.parse_args()

    if not os.path.isdir(args.pdf_dir):
        print(f"❌ PDF directory not found: {args.pdf_dir}")
        sys.exit(1)

    paper_index = load_paper_index(args.missing_file)
    if not paper_index:
        print("❌ Paper index is empty — cannot audit.")
        sys.exit(1)

    audit(args.pdf_dir, paper_index,
          fix=args.fix,
          cleanup_empty=args.cleanup_empty_dirs)


if __name__ == "__main__":
    main()
