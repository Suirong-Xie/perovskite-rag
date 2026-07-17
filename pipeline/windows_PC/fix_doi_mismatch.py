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
DOI_PATTERN = re.compile(r'\b(10\.\d{4,}/[a-zA-Z0-9.\-/_]+)', re.IGNORECASE)
# 去掉末尾的标点 (DOI 不应该以 . , ; 结尾)
DOI_CLEAN = re.compile(r'[.,;:]+$')


def normalize_doi(doi: str) -> str:
    """规范化 DOI 用于比较: 小写, 统一分隔符, 去除连字符。

    safe_filename 把 / \\ : 全替换成 _, 逆向无法区分。
    pdftotext/pypdf 可能丢失 ISSN 中间的连字符 (1361-651X → 1361651X)。
    连字符不影响 DOI 唯一性, 比较时去除。
    """
    return (doi.strip().lower()
            .replace("/", "_").replace("\\", "_").replace(":", "_")
            .replace("-", ""))


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
      1. "DOI:" 标签: 前 1/4 (标题/页眉区) → 安全优先
      1b. DOI 标注: 全文本 (匹配 "DOI: 10.xxx/..." 或 "https://doi.org/10.xxx/...")
      2. 前 1/3 范围的裸 DOI (无标签/URL)
      3. 兜底: 全文取第一个
    """
    if not text:
        return None

    # ── 预处理: 拼接被换行截断的 DOI ──
    # PDF 页眉/页脚中 DOI 常跨行: "10.1088/1674-4926/\n43/5/052201"
    text = re.sub(r'/\s*\n\s*(\w)', r'/\1', text)
    text = re.sub(r'(10\.\d{4,}/\S{2,})/\s*\n\s*(\w)', r'\1/\2', text)
    # pypdf 在连字符后截断: "10.1088/1361-\n651X/ae8043"
    text = re.sub(r'(10\.\d{4,}/\S*?)-(\s*\n\s*)(\d)', r'\1-\3', text)

    # ── 预处理: 消除 pypdf 在 DOI 中插入的空格 ──
    # pypdf 因 PDF kerning 常在 DOI 数字间插空格: "1361 -6633" "1088/ 1361"
    text = re.sub(r'(\d)\s+(-)', r'\1\2', text)       # "1361 -6633" → "1361-6633"
    text = re.sub(r'(/)\s+(\d)', r'\1\2', text)        # "1088/ 1361" → "1088/1361"
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)        # "10.137 1" → "10.1371"
    # PLOS ONE 等期刊中 "https://doi.or g" → "https://doi.org"
    text = re.sub(r'\bdoi\s*\.\s*o\s*r\s*g\s*/\s*', 'doi.org/', text)

    # ── 预处理: DOI 字符串中残留空格一次性清除 ──
    # pypdf 可能把 DOI 拆得粉碎: "journal.po ne.03132 66" → "journal.pone.0313266"
    # 只在 DOI 上下文 (10.XXXX/ 之后) 合并, 遇到大写字母自动停止
    text = re.sub(
        r'\b(10\.\d{4,}/[a-z0-9.\-/]*(?:\s+[a-z0-9.\-/]+)*)',
        lambda m: m.group(0).replace(' ', ''),
        text,
    )

    doi_label = re.compile(r'DOI\s*[：:]\s*(10\.\d{4,}/[a-zA-Z0-9.\-/_]+)', re.IGNORECASE)

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

    # ── 策略 1b: DOI 标注 — 全文本搜 (IOP 页眉用 https://doi.org/10.xxx/...) ──
    # 匹配: "DOI: 10.xxx/..." 或 "https://doi.org/10.xxx/..."
    # 参考文献中极少出现完整 doi.org URL, 全量搜安全
    doi_annotated = re.compile(
        r'(?:DOI\s*[：:]\s*|https?://doi\.org/|Digital\s+Object\s+Identifier\s+)\s*(10\.\d{4,}/[a-zA-Z0-9.\-/_]+)',
        re.IGNORECASE,
    )
    m = doi_annotated.search(text)
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

            # DOI 可能粘了 ISSN/版权文字, 从尾部逐字符裁剪重试
            if real_doi.lower() not in paper_index:
                for trim_len in range(len(real_doi) - 1, 14, -1):
                    candidate = real_doi[:trim_len].rstrip('./-')
                    if candidate.lower() in paper_index:
                        real_doi = candidate
                        break

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
