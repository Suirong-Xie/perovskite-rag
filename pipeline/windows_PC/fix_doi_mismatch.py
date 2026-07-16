#!/usr/bin/env python3
"""
fix_doi_mismatch.py — 从 PDF 内容提取 DOI, 与文件名 DOI 比对, 不匹配的移到 unmatched/ 重新分类

工作流:
  1. 扫描 journals_pdf/ 下所有 PDF
  2. 从 PDF 第一页提取真实 DOI
  3. 与文件名中的 DOI 比对
  4. 匹配 → 保留原位
  5. 不匹配 → 用真实 DOI + 正确期刊名 → 移动到 unmatched/ 下
  6. 提取不到 DOI → 移到 unmatched/_no_doi_found/

用法:
  python fix_doi_mismatch.py                   # 检查 + 整理
  python fix_doi_mismatch.py --dry-run         # 仅检查, 不动文件
  python fix_doi_mismatch.py --pdf-dir ./journals_pdf
"""

import json
import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# 复用分类器
from journal_classifier import classify_venue

PDF_BASE_DIR = "./journals_pdf"
UNMATCHED_DIR = "./unmatched"
MISSING_FILE = "./missing_non_wiley.jsonl"

# DOI 正则: 匹配 10.xxxx/xxxxx... 格式
DOI_PATTERN = re.compile(r'\b(10\.\d{4,}/[^\s()\[\]<>"]+)', re.IGNORECASE)
# 去掉末尾的标点 (DOI 不应该以 . , ; 结尾)
DOI_CLEAN = re.compile(r'[.,;:]+$')


def normalize_doi(doi: str) -> str:
    """规范化 DOI 用于比较: 小写, 统一分隔符。

    safe_filename 把 / \\ : 全替换成 _, 逆向无法区分。
    所以两边都转成同一个形式再比较: 全部 -> 小写, / \\ : -> _
    """
    return doi.strip().lower().replace("/", "_").replace("\\", "_").replace(":", "_")


def doi_from_filename(filename: str) -> str:
    """从文件名提取 DOI key (去掉 .pdf, 保留 _ 不做还原)。"""
    return filename.removesuffix(".pdf")


def extract_all_text_from_pdf(pdf_path: str) -> str:
    """提取 PDF 全部页面文本 (兜底, 用于前2页+末页找不到 DOI 的情况)。"""
    # pypdf 全页 (限 50 页, 正常论文不会超)
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[:50]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if len(text.strip()) >= 50:
            return text.strip()
    except Exception:
        pass

    # pdftotext 全页
    try:
        import subprocess
        proc = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace",
        )
        if len(proc.stdout.strip()) >= 50:
            return proc.stdout.strip()
    except Exception:
        pass

    # 二进制暴力: 读整个文件
    try:
        with open(pdf_path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        lines = [l for l in text.split("\n") if 5 < len(l.strip()) < 500]
        text = "\n".join(lines[:500])
        if len(text.strip()) >= 50:
            return text.strip()
    except Exception:
        pass

    return ""


def extract_text_from_pdf(pdf_path: str, include_last: bool = False) -> str:
    """从 PDF 提取文本。默认取前 2 页；include_last=True 时也取末页。

    原因: Science / AAAS 的 DOI 印在末页, 不在首页。
    Windows 上可能没有 pdftotext, 优先用 pypdf (纯 Python)。
    """
    # 方法 1: pypdf (纯 Python, 跨平台, 支持末页)
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        total = len(reader.pages)
        pages = list(reader.pages[:2])
        if include_last and total > 2:
            pages.append(reader.pages[-1])
        text = ""
        for page in pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if len(text.strip()) >= 50:
            return text.strip()
    except Exception:
        pass

    # 方法 2: pdftotext 命令行
    try:
        import subprocess
        text_parts = []

        proc = subprocess.run(
            ["pdftotext", "-l", "2", pdf_path, "-"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        if proc.stdout.strip():
            text_parts.append(proc.stdout)

        if include_last:
            try:
                info = subprocess.run(
                    ["pdfinfo", pdf_path],
                    capture_output=True, text=True, timeout=10,
                    encoding="utf-8", errors="replace",
                )
                pages_match = re.search(r'Pages:\s+(\d+)', info.stdout)
                total_pages = int(pages_match.group(1)) if pages_match else 0
                if total_pages > 2:
                    proc_last = subprocess.run(
                        ["pdftotext", "-f", str(total_pages), "-l", str(total_pages), pdf_path, "-"],
                        capture_output=True, text=True, timeout=30,
                        encoding="utf-8", errors="replace",
                    )
                    if proc_last.stdout.strip():
                        text_parts.append(proc_last.stdout)
            except Exception:
                pass

        text = "\n".join(text_parts)
        if len(text.strip()) >= 50:
            return text.strip()
    except Exception:
        pass

    # 方法 4: 二进制暴力解析 (兜底)
    # 读前 50KB (含首页) + 末尾 20KB (含末页, 如果 include_last)
    try:
        with open(pdf_path, "rb") as f:
            head = f.read(50000)
        text = head.decode("utf-8", errors="replace")
        if include_last:
            with open(pdf_path, "rb") as f:
                f.seek(max(0, os.path.getsize(pdf_path) - 20000))
                tail = f.read(20000)
            text += "\n" + tail.decode("utf-8", errors="replace")
        lines = [l for l in text.split("\n") if 5 < len(l.strip()) < 500]
        text = "\n".join(lines[:200])
        if len(text.strip()) >= 50:
            return text.strip()
    except Exception:
        pass

    return ""


def extract_doi_from_metadata(pdf_path: str) -> str | None:
    """从 PDF 元数据中提取 DOI (dc:identifier, prism:doi 等字段)。"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        meta = reader.metadata
        if not meta:
            return None
        for key in ["/dc:identifier", "/prism:doi", "/doi", "/DOI",
                    "/WPS-ARTICLEDOI", "/crossmark:doi"]:
            val = meta.get(key) or meta.get(key.lower(), "")
            if val and "10." in str(val):
                m = DOI_PATTERN.search(str(val))
                if m:
                    doi = DOI_CLEAN.sub('', m.group(1))
                    if len(doi) >= 15:
                        return doi
        for key in ["/Subject", "/Description"]:
            val = meta.get(key, "")
            if val and "10." in str(val):
                m = DOI_PATTERN.search(str(val))
                if m:
                    doi = DOI_CLEAN.sub('', m.group(1))
                    if len(doi) >= 15:
                        return doi
    except Exception:
        pass
    return None


def extract_doi_from_text(text: str, search_last_page: bool = False) -> str | None:
    """从文本中提取文章自身的 DOI (非参考文献中的 DOI)。

    策略:
      1. "DOI:" 标签: 只在前 1/3 搜 (标题区) 或只在后 1/5 搜 (末页区, Science 等)
         避免中间页的 reference 区
      2. 前 1/3 范围的裸 DOI (无标签)
      3. 兜底: 全文取第一个
    """
    if not text:
        return None

    doi_label = re.compile(r'DOI\s*[：:]\s*(10\.\d{4,}/[^\s]+)', re.IGNORECASE)

    if search_last_page:
        # 只在末尾 20% 搜 (末页区, Science 的 DOI 在这)
        last_start = max(len(text) * 4 // 5, len(text) - 3000)
        m = doi_label.search(text, last_start)
        if m:
            doi = DOI_CLEAN.sub('', m.group(1))
            if len(doi) >= 15:
                return doi

    # ── 策略 1: "DOI:" 标签 — 只在前 1/4 (标题/页眉区) ──
    first_quarter = len(text) // 4
    m = doi_label.search(text[:first_quarter])
    if m:
        doi = DOI_CLEAN.sub('', m.group(1))
        if len(doi) >= 15:
            return doi

    # ── 策略 2: 裸 DOI — 只在前 1/3 (参考文献都在后半截) ──
    split_point = max(len(text) // 3, 2000)
    matches = DOI_PATTERN.findall(text[:split_point])
    for m in matches:
        m = DOI_CLEAN.sub('', m)
        if len(m) >= 15:
            return m

    # ── 策略 3: 全文兜底, 取第一个 ──
    matches = DOI_PATTERN.findall(text)
    for m in matches:
        m = DOI_CLEAN.sub('', m)
        if len(m) >= 15:
            return m

    return None


def load_paper_index() -> dict[str, dict]:
    """加载全量论文索引。"""
    index = {}
    if not os.path.exists(MISSING_FILE):
        print(f"⚠️  {MISSING_FILE} not found")
        return index
    with open(MISSING_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = json.loads(line.strip())
            doi = p.get("doi", "")
            if doi:
                index[doi.lower()] = p
    print(f"📋 Paper index: {len(index)} entries")
    return index


def main():
    parser = argparse.ArgumentParser(description="PDF 内容 DOI 校验 & 重归类")
    parser.add_argument("--pdf-dir", type=str, default=PDF_BASE_DIR)
    parser.add_argument("--dry-run", action="store_true", help="仅检查不动文件")
    args = parser.parse_args()

    if not os.path.isdir(args.pdf_dir):
        print(f"❌ {args.pdf_dir} not found")
        sys.exit(1)

    paper_index = load_paper_index()

    # 统计
    stats = {
        "scanned": 0,
        "doi_found": 0,
        "doi_match": 0,
        "doi_mismatch": 0,
        "doi_not_found": 0,
        "no_text": 0,
        "moved": 0,
    }
    mismatches = []  # [(path, filename_doi, real_doi, venue, title), ...]
    no_doi = []      # [(path, filename_doi), ...]

    print(f"\n🔍 Scanning {args.pdf_dir}...")
    for root, dirs, files in os.walk(args.pdf_dir):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue

            stats["scanned"] += 1
            src_path = os.path.join(root, fname)
            filename_doi = doi_from_filename(fname)

            # 提取 DOI: 元数据 → 正文首页 → 正文首页+末页
            real_doi = extract_doi_from_metadata(src_path)

            # Science (10.1126): DOI 严格只在末页, 不搜首页避免 reference 里的 Science DOI
            is_science = filename_doi.lower().startswith("10.1126") or "10.1126" in filename_doi.lower()

            if not real_doi and not is_science:
                text = extract_text_from_pdf(src_path)
                real_doi = extract_doi_from_text(text)

            if not real_doi:
                text = extract_text_from_pdf(src_path, include_last=True)
                real_doi = extract_doi_from_text(text, search_last_page=True)

            # 末页也没找到 → 全页兜底
            if not real_doi:
                text = extract_all_text_from_pdf(src_path)
                real_doi = extract_doi_from_text(text, search_last_page=True)

            if not real_doi:
                stats["no_text"] += 1
                no_doi.append((src_path, filename_doi))
                if stats["no_text"] <= 20:
                    print(f"  ❓ No DOI found: {fname[:60]}")
                continue

            if not real_doi:
                stats["doi_not_found"] += 1
                no_doi.append((src_path, filename_doi))
                print(f"  ❓ No DOI in text: {fname[:60]}")
                continue

            stats["doi_found"] += 1

            if normalize_doi(real_doi) == normalize_doi(filename_doi):
                stats["doi_match"] += 1
                continue

            # — 不匹配 —
            stats["doi_mismatch"] += 1

            # 去全量 index 找 venue 和 title
            paper = paper_index.get(real_doi.lower(), {})
            venue = paper.get("venue", "") or ""
            title = paper.get("title", "")[:80]

            mismatches.append((src_path, filename_doi, real_doi, venue, title))
            print(f"  ❌ MISMATCH: {fname[:50]}")
            print(f"     filename DOI: {filename_doi}")
            print(f"     content  DOI: {real_doi}")
            if title:
                print(f"     title: {title}")

    # ── 报告 ──
    print(f"\n{'='*60}")
    print(f"📊 REPORT")
    print(f"{'='*60}")
    print(f"  Scanned:          {stats['scanned']:5d}")
    print(f"  DOI found in PDF: {stats['doi_found']:5d}")
    print(f"  DOI match:        {stats['doi_match']:5d}  ✅")
    print(f"  DOI MISMATCH:     {stats['doi_mismatch']:5d}  ❌")
    print(f"  DOI not found:    {stats['doi_not_found']:5d}  ❓")
    print(f"  No extractable text: {stats['no_text']:5d}")

    if stats["doi_mismatch"] == 0 and stats["doi_not_found"] == 0:
        print("\n✅ All clean!")
        return

    if args.dry_run:
        print("\n💡 Dry run — no files moved. Use without --dry-run to fix.")
        return

    # ── 修复: 移动不匹配的文件 ──
    print(f"\n🔧 Moving {len(mismatches)} mismatched files to {UNMATCHED_DIR}/ ...")

    for src_path, filename_doi, real_doi, venue, title in mismatches:
        # 确定目标目录
        if venue:
            journal = classify_venue(venue)
        else:
            journal = "_unknown_journal"

        dst_dir = os.path.join(UNMATCHED_DIR, journal)
        os.makedirs(dst_dir, exist_ok=True)

        # 用真实 DOI 做文件名
        safe_name = real_doi.replace("/", "_").replace("\\", "_").replace(":", "_") + ".pdf"
        dst_path = os.path.join(dst_dir, safe_name)

        # 避免覆盖
        if os.path.exists(dst_path):
            base = safe_name.removesuffix(".pdf")
            dst_path = os.path.join(dst_dir, f"{base}_DUP.pdf")

        shutil.move(src_path, dst_path)
        stats["moved"] += 1

    # ── 移动无法提取 DOI 的文件 ──
    if no_doi:
        nd_dir = os.path.join(UNMATCHED_DIR, "_no_doi_found")
        os.makedirs(nd_dir, exist_ok=True)
        for src_path, filename_doi in no_doi:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(nd_dir, fname)
            if os.path.exists(dst_path):
                base = fname.removesuffix(".pdf")
                dst_path = os.path.join(nd_dir, f"{base}_DUP.pdf")
            shutil.move(src_path, dst_path)
            stats["moved"] += 1

    print(f"\n✅ Done. {stats['moved']} files moved to {UNMATCHED_DIR}/")
    print(f"   Run audit_classification.py --pdf-dir {UNMATCHED_DIR} to verify.")


if __name__ == "__main__":
    main()
