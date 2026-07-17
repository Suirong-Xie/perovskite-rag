#!/usr/bin/env python3
"""
batch_open.py — 分批打开文章网页, 手动点 PDF, 自动归类到期刊文件夹

用法:
  python batch_open.py                    # 每次 20 篇, 按出版者分组
  python batch_open.py --batch 30         # 每次 30 篇
  python batch_open.py --publisher acs    # 只处理 ACS
  python batch_open.py --resume           # 断点续传

工作流:
  1. 脚本按出版者分组, 逐批处理
  2. 每批: 清空 batch_tmp → 打开 20 个 DOI → 你下载到 batch_tmp
  3. 按 Enter → 从文件名提取 DOI 匹配 → 归入 journals_pdf/<期刊>/
  4. 下一批

匹配策略:
  - 从 PDF 文件名中查找 DOI 片段 (suffix / full / article-ID)
  - 不依赖下载顺序 — 随便先点哪个 PDF 下载都可以
  - 未匹配的会明确报告, 让你手动处理 (如 Elsevier 的 1-s2.0-Sxxx-main.pdf)
"""

import json
import os
import re
import sys
import time
import glob
import shutil
import webbrowser
import argparse

# ── 配置 ──

MISSING_FILE = "missing_non_wiley.jsonl"
CHECKPOINT_FILE = "batch_checkpoint.json"
PDF_DIR = "journals_pdf"
DOWNLOADS_DIR = r"D:\Edge浏览器下载"
BATCH_TMP = os.path.join(DOWNLOADS_DIR, "batch_tmp")
BATCH_SIZE = 20
UNMATCHED_DIR = "unmatched"

# DOI 正则 (用于 PDF 内容提取)
DOI_PATTERN = re.compile(r'\b(10\.\d{4,}/[^\s()\[\]<>"]+)', re.IGNORECASE)
DOI_CLEAN_RE = re.compile(r'[.,;:]+$')

DOI_PREFIX_MAP = {
    "10.1016": "elsevier", "10.1021": "acs", "10.1039": "rsc",
    "10.1038": "nature", "10.1126": "science", "10.1007": "springer",
    "10.3390": "mdpi", "10.1088": "iop", "10.1063": "aip",
    "10.1109": "ieee", "10.1103": "aps", "10.1093": "oup",
    "10.2139": "ssrn", "10.1073": "pnas", "10.1371": "plos",
}

PUBLISHER_INFO = {
    "elsevier":  {"name": "Elsevier / ScienceDirect"},
    "acs":       {"name": "ACS Publications"},
    "rsc":       {"name": "RSC Publishing"},
    "nature":    {"name": "Nature.com"},
    "science":   {"name": "Science / AAAS"},
    "springer":  {"name": "Springer Link"},
    "mdpi":      {"name": "MDPI (OA)"},
    "iop":       {"name": "IOPscience"},
    "aip":       {"name": "AIP Publishing"},
    "ieee":      {"name": "IEEE Xplore"},
    "aps":       {"name": "APS Journals"},
    "pnas":      {"name": "PNAS"},
}

JOURNAL_MAP = {
    "nature": "Nature", "nature energy": "NatEnergy",
    "nature materials": "NatMater", "nature photonics": "NatPhoton",
    "nature nanotechnology": "NatNanotech", "nature communications": "NatComm",
    "science": "Science", "science advances": "Science_Advances",
    "joule": "Joule", "matter": "Matter", "chem": "Chem",
    "acs energy letters": "ACS_Energy_Letters",
    "journal of the american chemical society": "JACS",
    "chemistry of materials": "Chemistry_of_Materials",
    "acs nano": "ACS_Nano", "nano letters": "Nano_Letters",
    "acs applied materials": "ACS_Applied_Materials_Interfaces",
    "advanced materials": "Advanced_Materials",
    "advanced energy materials": "Advanced_Energy_Materials",
    "advanced functional materials": "Advanced_Functional_Materials",
    "angewandte chemie": "Angewandte_Chemie",
    "advanced science": "Advanced_Science",
    "small": "Small", "small methods": "Small_Methods",
    "solar rrl": "Solar_RRL",
    "energy & environmental science": "Energy_Environmental_Science",
    "journal of materials chemistry": "J_Materials_Chemistry_A",
    "nano energy": "Nano_Energy",
    "chemical engineering journal": "Chemical_Engineering_Journal",
    "nanoscale": "Nanoscale",
}


def publisher_from_doi(doi: str) -> str:
    return DOI_PREFIX_MAP.get(doi.split("/")[0], "unknown")


def safe_filename(doi: str) -> str:
    return doi.replace("/", "_").replace("\\", "_").replace(":", "_") + ".pdf"


def journal_dir(venue: str) -> str:
    v = (venue or "").lower().strip()
    for key, d in JOURNAL_MAP.items():
        if key in v:
            return d
    safe = re.sub(r'[^\w\s-]', '', venue or "Unknown").strip()[:40].replace(" ", "_")
    return safe if safe else "Unknown_Journal"


def clear_tmp():
    """清空 batch_tmp 文件夹。"""
    if os.path.isdir(BATCH_TMP):
        for f in os.listdir(BATCH_TMP):
            if f.endswith(".pdf"):
                os.remove(os.path.join(BATCH_TMP, f))
    else:
        os.makedirs(BATCH_TMP, exist_ok=True)


def list_tmp_pdfs() -> list[str]:
    """列出 batch_tmp 中的 PDF, 按文件名排序 (顺序不再影响匹配)。"""
    pdfs = glob.glob(os.path.join(BATCH_TMP, "*.pdf"))
    return sorted(pdfs, key=lambda p: os.path.basename(p))


def pdf_matches_doi(pdf_path: str, doi: str) -> bool:
    """检查 PDF 文件名中是否包含该 DOI (或其关键片段)。

    浏览器下载 PDF 时, 文件名通常包含 DOI 后缀 (如 s41560-020-00735-z.pdf)。
    此函数尝试多种 DOI 表示形式来匹配文件名。
    """
    doi = doi.strip().lower()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    if "/" not in doi:
        return False

    prefix, suffix = doi.split("/", 1)

    fname = os.path.basename(pdf_path).lower()
    # 去掉浏览器重名标记 "(1)", "(2)" 等, 以及 .pdf 扩展名
    fname_clean = re.sub(r'\s*\(\d+\)', '', fname).replace(".pdf", "")

    # 候选匹配串: 从最特异到最泛
    # 1. DOI 后缀完整匹配, 如 "s41560-020-00735-z" 或 "acsnano.5b00001"
    if suffix in fname_clean:
        return True
    # 2. 完整 DOI (斜杠替换为下划线), 如 "10.1038_s41560-020-00735-z"
    if doi.replace("/", "_") in fname_clean:
        return True
    # 3. 完整 DOI (斜杠替换为点), 如 "10.1038.s41560-020-00735-z"
    if doi.replace("/", ".") in fname_clean:
        return True
    # 4. DOI 后缀的最后一段 (文章 ID), 长度 > 4 以防止太短误匹配
    #    如 "jac.5b00001" → "5b00001"
    parts = suffix.rsplit(".", 1)
    if len(parts) > 1 and len(parts[1]) > 4 and parts[1] in fname_clean:
        return True

    return False


# ── 全量论文索引 (全局匹配用, 惰性加载) ──

_paper_index_cache: dict[str, dict] | None = None


def load_paper_index() -> dict[str, dict]:
    """加载全量论文索引 (DOI → paper dict), 惰性缓存。"""
    global _paper_index_cache
    if _paper_index_cache is not None:
        return _paper_index_cache
    _paper_index_cache = {}
    if not os.path.exists(MISSING_FILE):
        print(f"  ⚠️  {MISSING_FILE} not found, global matching disabled.")
        return _paper_index_cache
    with open(MISSING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            doi = p.get("doi", "")
            if doi:
                _paper_index_cache[doi.lower()] = p
    print(f"  📋 Paper index loaded: {len(_paper_index_cache)} entries")
    return _paper_index_cache


def normalize_doi(doi: str) -> str:
    """规范化 DOI 为统一 key (小写, 分隔符统一为 _)。"""
    return doi.strip().lower().replace("/", "_").replace("\\", "_").replace(":", "_")


# ── PDF 内容 DOI 提取 (移植自 fix_doi_mismatch.py) ──

def _extract_doi_from_metadata(pdf_path: str) -> str | None:
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
                    doi = DOI_CLEAN_RE.sub('', m.group(1))
                    if len(doi) >= 15:
                        return doi
        for key in ["/Subject", "/Description"]:
            val = meta.get(key, "")
            if val and "10." in str(val):
                m = DOI_PATTERN.search(str(val))
                if m:
                    doi = DOI_CLEAN_RE.sub('', m.group(1))
                    if len(doi) >= 15:
                        return doi
    except Exception:
        pass
    return None


def _extract_text_from_pdf(pdf_path: str, include_last: bool = False) -> str:
    """从 PDF 提取文本 (前 2 页 + 可选末页)。

    多级兜底: pypdf → pdftotext → 二进制暴力解析。
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
        proc = subprocess.run(
            ["pdftotext", "-l", "2", pdf_path, "-"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        text = proc.stdout.strip()
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
                        text += "\n" + proc_last.stdout.strip()
            except Exception:
                pass
        if len(text.strip()) >= 50:
            return text.strip()
    except Exception:
        pass

    # 方法 3: 二进制暴力解析 (兜底)
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


def _extract_all_text_from_pdf(pdf_path: str) -> str:
    """提取 PDF 大量页面文本 (全页兜底, 限 50 页)。"""
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


def _extract_doi_from_text(text: str, search_last_page: bool = False) -> str | None:
    """从文本中提取文章自身 DOI (非参考文献中的 DOI)。

    策略:
      1. "DOI:" 标签: 前 1/4 搜 (标题区) 或后 1/5 搜 (末页区, Science 等)
      2. 前 1/3 范围的裸 DOI
      3. 兜底: 全文取第一个
    """
    if not text:
        return None

    doi_label = re.compile(r'DOI\s*[：:]\s*(10\.\d{4,}/[^\s]+)', re.IGNORECASE)

    if search_last_page:
        last_start = max(len(text) * 4 // 5, len(text) - 3000)
        m = doi_label.search(text, last_start)
        if m:
            doi = DOI_CLEAN_RE.sub('', m.group(1))
            if len(doi) >= 15:
                return doi

    # 策略 1: "DOI:" 标签 — 只在前 1/4 (标题/页眉区)
    first_quarter = len(text) // 4
    m = doi_label.search(text[:first_quarter])
    if m:
        doi = DOI_CLEAN_RE.sub('', m.group(1))
        if len(doi) >= 15:
            return doi

    # 策略 2: 裸 DOI — 只在前 1/3 (参考文献都在后半截)
    split_point = max(len(text) // 3, 2000)
    matches = DOI_PATTERN.findall(text[:split_point])
    for m in matches:
        m = DOI_CLEAN_RE.sub('', m)
        if len(m) >= 15:
            return m

    # 策略 3: 全文兜底, 取第一个
    matches = DOI_PATTERN.findall(text)
    for m in matches:
        m = DOI_CLEAN_RE.sub('', m)
        if len(m) >= 15:
            return m

    return None


def extract_doi_from_pdf_content(pdf_path: str) -> str | None:
    """从 PDF 内容提取真实 DOI (元数据 → 前2页 → +末页 → 全页)。"""
    # 1. 元数据
    real_doi = _extract_doi_from_metadata(pdf_path)
    if real_doi:
        return real_doi

    # 2. 前 2 页
    text = _extract_text_from_pdf(pdf_path)
    real_doi = _extract_doi_from_text(text)
    if real_doi:
        return real_doi

    # 3. 前 2 页 + 末页 (Science 的 DOI 在末页)
    text = _extract_text_from_pdf(pdf_path, include_last=True)
    real_doi = _extract_doi_from_text(text, search_last_page=True)
    if real_doi:
        return real_doi

    # 4. 全页兜底
    text = _extract_all_text_from_pdf(pdf_path)
    real_doi = _extract_doi_from_text(text, search_last_page=True)
    return real_doi


# ── 全局匹配 ──

def global_match_pdf(pdf_path: str, paper_index: dict[str, dict]) -> tuple[str | None, dict | None]:
    """尝试将 PDF 与全量论文索引匹配。

    两层:
      1. 文件名含 DOI → 查 paper_index
      2. 从 PDF 内容提取 DOI → 查 paper_index

    Returns (matched_doi, paper_dict) or (None, None).
    """
    fname = os.path.basename(pdf_path)

    # 第一层: 全局文件名匹配 (遍历所有论文 DOI, 检查是否出现在文件名中)
    for doi, paper in paper_index.items():
        if pdf_matches_doi(pdf_path, doi):
            return doi, paper

    # 第二层: PDF 内容提取 DOI
    print(f"    🔬 Extracting DOI from PDF content: {fname[:60]}")
    real_doi = extract_doi_from_pdf_content(pdf_path)
    if real_doi:
        doi_key = real_doi.lower()
        if doi_key in paper_index:
            return doi_key, paper_index[doi_key]
        else:
            print(f"    ⚠️  Content DOI {real_doi} not found in paper index")

    return None, None


def process_batch(batch_papers: list[dict], checkpoint: dict) -> int:
    """处理一批论文: 打开页面 → 等下载 → 按文件名匹配 DOI → 归位。"""
    clear_tmp()

    print(f"\n  📂 Download folder: {BATCH_TMP}")
    for i, p in enumerate(batch_papers):
        title = p.get("title", "")[:80]
        venue = p.get("venue", "")[:30]
        print(f"  [{i+1:2d}] {venue} — {title}")

    print(f"\n  👉 Opening {len(batch_papers)} tabs in your browser...")
    print(f"  👉 Set Edge to download PDFs to: {BATCH_TMP}")
    print(f"  👉 Click PDF download on each page, then press Enter\n")
    input("  Press Enter to open browser...")

    for p in batch_papers:
        webbrowser.open(f"https://doi.org/{p['doi']}")
        time.sleep(0.3)

    print(f"\n  ✅ {len(batch_papers)} tabs opened!")
    input("  Press Enter when ALL downloads are complete...")

    # ── 按文件名匹配 DOI ──
    pdfs = list_tmp_pdfs()
    print(f"\n  🔍 Found {len(pdfs)} PDFs in {BATCH_TMP}")

    # 建立匹配: pdf_path → paper_index (每篇论文只配一个 PDF)
    pdf_used = set()
    paper_to_pdf = {}   # paper_index → pdf_path

    for pi, paper in enumerate(batch_papers):
        for pdf_path in pdfs:
            if pdf_path in pdf_used:
                continue
            if pdf_matches_doi(pdf_path, paper["doi"]):
                paper_to_pdf[pi] = pdf_path
                pdf_used.add(pdf_path)
                break

    # ── 归位已匹配的 PDF ──
    moved = 0
    for pi, pdf_path in paper_to_pdf.items():
        paper = batch_papers[pi]
        doi = paper["doi"]
        venue = paper.get("venue", "")

        jdir = os.path.join(PDF_DIR, journal_dir(venue))
        dst = os.path.join(jdir, safe_filename(doi))
        os.makedirs(jdir, exist_ok=True)

        if os.path.exists(dst):
            print(f"    ⏭️  Already exists: {safe_filename(doi)}")
            checkpoint[doi] = "done"
            moved += 1
            continue

        shutil.move(pdf_path, dst)
        checkpoint[doi] = "done"
        moved += 1
        print(f"    📁 {os.path.basename(pdf_path)} → {jdir}/{safe_filename(doi)}")

    # ── 未匹配 PDF: 3-tier 逐级匹配 ──
    unmatched_papers = [i for i in range(len(batch_papers)) if i not in paper_to_pdf]
    unmatched_pdfs   = [p for p in pdfs if p not in pdf_used]

    if unmatched_pdfs:
        print(f"\n  🔄 {len(unmatched_pdfs)} PDF(s) not matched in batch → trying global match...")
        paper_index = load_paper_index()

        still_unmatched = []
        for pdf_path in unmatched_pdfs:
            matched_doi, matched_paper = global_match_pdf(pdf_path, paper_index)
            if matched_paper:
                venue = matched_paper.get("venue", "")
                jdir = os.path.join(PDF_DIR, journal_dir(venue))
                dst = os.path.join(jdir, safe_filename(matched_doi))
                os.makedirs(jdir, exist_ok=True)

                matched_label = "[filename]" if pdf_matches_doi(pdf_path, matched_doi) else "[content]"
                if os.path.exists(dst):
                    print(f"    ⏭️  {matched_label} Already exists: {safe_filename(matched_doi)}")
                else:
                    shutil.move(pdf_path, dst)
                    print(f"    🌐 {matched_label} {os.path.basename(pdf_path)} → {jdir}/{safe_filename(matched_doi)}")
                checkpoint[matched_doi] = "done"
                moved += 1
            else:
                still_unmatched.append(pdf_path)

        # ── 仍无法匹配 → 移入 unmatched/ ──
        if still_unmatched:
            print(f"\n  🗑️  {len(still_unmatched)} PDF(s) still unmatched → {UNMATCHED_DIR}/_unknown_batch/")
            unmatched_dst = os.path.join(UNMATCHED_DIR, "_unknown_batch")
            os.makedirs(unmatched_dst, exist_ok=True)
            for pdf_path in still_unmatched:
                fname = os.path.basename(pdf_path)
                dst_path = os.path.join(unmatched_dst, fname)
                if os.path.exists(dst_path):
                    base = fname.removesuffix(".pdf")
                    dst_path = os.path.join(unmatched_dst, f"{base}_DUP.pdf")
                shutil.move(pdf_path, dst_path)
                print(f"     {fname}")
            print(f"  💡 Run fix_doi_mismatch.py --pdf-dir {UNMATCHED_DIR} to re-check content DOIs.")

    if unmatched_papers:
        print(f"\n  ⚠️  {len(unmatched_papers)} paper(s) have no matching PDF:")
        for i in unmatched_papers:
            p = batch_papers[i]
            print(f"     [{i+1:2d}] https://doi.org/{p['doi']} — {p.get('title','')[:80]}")
        print(f"  💡 These may need manual download or the PDF filenames don't contain their DOIs.")

    # 清除残留 (所有 PDF 都已移走)
    clear_tmp()
    return moved


def main():
    parser = argparse.ArgumentParser(description="分批下载论文 PDF, 自动归类")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--publisher", type=str)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(MISSING_FILE):
        print(f"❌ {MISSING_FILE} not found!")
        sys.exit(1)

    papers = []
    with open(MISSING_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            if args.publisher and publisher_from_doi(p["doi"]) != args.publisher:
                continue
            papers.append(p)

    # 按出版者分组
    by_pub = {}
    for p in papers:
        pub = publisher_from_doi(p["doi"])
        by_pub.setdefault(pub, []).append(p)
    pub_order = sorted(by_pub.keys(), key=lambda p: -len(by_pub[p]))

    total = sum(len(ps) for ps in by_pub.values())
    print(f"📊 {total} papers from {len(by_pub)} publishers:")
    for pub in pub_order:
        info = PUBLISHER_INFO.get(pub, {"name": pub})
        print(f"   {info['name']}: {len(by_pub[pub])}")

    # Checkpoint
    checkpoint = {}
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            checkpoint = json.load(f)

    grand_done = sum(1 for v in checkpoint.values() if v == "done")

    for pub in pub_order:
        pub_papers = by_pub[pub]
        info = PUBLISHER_INFO.get(pub, {"name": pub})

        # 过滤已完成的
        pending = []
        for p in pub_papers:
            doi = p["doi"]
            if checkpoint.get(doi) == "done":
                continue
            jdir = os.path.join(PDF_DIR, journal_dir(p.get("venue", "")))
            if os.path.isfile(os.path.join(jdir, safe_filename(doi))):
                checkpoint[doi] = "done"
                continue
            pending.append(p)

        if not pending:
            continue

        pub_done = len(pub_papers) - len(pending)
        print(f"\n{'='*60}")
        print(f"  📚 {info['name']}: {len(pub_papers)} total, {pub_done} done, {len(pending)} remaining")
        print(f"{'='*60}")

        batch_num = 0
        for start in range(0, len(pending), args.batch):
            batch = pending[start:start + args.batch]
            batch_num += 1

            print(f"\n  --- Batch {batch_num} ({len(batch)} papers) ---")
            moved = process_batch(batch, checkpoint)

            # 保存进度
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f)

            now_done = sum(1 for v in checkpoint.values() if v == "done")
            print(f"  📋 {info['name']}: {now_done - grand_done} new this batch, {now_done}/{total} overall")
            grand_done = now_done

    print(f"\n🎉 All done! {grand_done}/{total} papers")


if __name__ == "__main__":
    main()
