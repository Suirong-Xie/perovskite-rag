#!/usr/bin/env python3
"""
s2_download_pdfs.py — 通过 Sci-Hub + arXiv 批量下载钙钛矿论文 PDF

下载优先级:
  1. Sci-Hub (doi → https://sci-hub.se/{doi})  — 覆盖面 ~90%
  2. arXiv (arxivId → arxiv.org/pdf/{id}.pdf)   — 预印本
  3. openAccessUrl (S2 自带的 OA 链接)            — 合法 OA
  4. 失败 → 降级为 abstract-only

PDF 存储: /data/data/pkb/01_raw_data/journals_pdf/{Journal}/{doi_suffix}.pdf
文本存储: data/s2_corpus/s2_texts_tier1.jsonl (全文) / tier2.jsonl (摘要)

用法:
  python3 s2_download_pdfs.py                    # 全量下载
  python3 s2_download_pdfs.py --limit 100        # 只下前 100 篇
  python3 s2_download_pdfs.py --no-scihub        # 跳过 Sci-Hub (仅用 arXiv + OA)
  python3 s2_download_pdfs.py --resume           # 从 checkpoint 恢复
"""

import json
import os
import re
import sys
import time
import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(__file__))
from pdf_text_extractor import extract_text_column_aware
from journal_classifier import classify_venue, journal_to_dirname

# ── 配置 ──

BASE_DIR = "/data1/perovskite-rag"
CORPUS_DIR = os.path.join(BASE_DIR, "data", "s2_corpus")
PDF_BASE_DIR = "/data/data/pkb/01_raw_data/journals_pdf"
DEDUPED_FILE = os.path.join(CORPUS_DIR, "s2_papers_deduped.jsonl")
TIER1_FILE = os.path.join(CORPUS_DIR, "s2_texts_tier1.jsonl")
TIER2_FILE = os.path.join(CORPUS_DIR, "s2_texts_tier2.jsonl")
CHECKPOINT_FILE = os.path.join(CORPUS_DIR, "download_checkpoint.json")

os.makedirs(CORPUS_DIR, exist_ok=True)

# Sci-Hub 域名池 (只用已验证能用的)
SCIHUB_DOMAINS = [
    "sci-hub.ru",
    "sci-hub.st",
]

# 下载超时 & 重试
DOWNLOAD_TIMEOUT = 10
MAX_RETRIES = 1
REQUEST_DELAY = 1.0  # Sci-Hub 间隔 (秒)

# User-Agent
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def doi_to_filename(doi: str) -> str:
    """DOI → 安全文件名。"""
    # 例如 10.1038/s41586-021-03446-x → 10.1038_s41586-021-03446-x
    safe = doi.replace("/", "_").replace("\\", "_")
    safe = re.sub(r'[^\w\-_.]', '', safe)
    return f"{safe}.pdf"


# ── Checkpoint ──

def load_checkpoint() -> dict:
    """加载下载进度: {paperId: status}"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)


# ── Sci-Hub PDF 下载 ──

def scihub_download(doi: str, save_path: str) -> bool:
    """通过 Sci-Hub 下载 PDF。

    解析 Sci-Hub 页面中的 PDF 链接（支持新旧两种页面格式）。

    Returns:
        True if PDF downloaded and saved successfully.
    """
    for domain in SCIHUB_DOMAINS:
        try:
            scihub_url = f"https://{domain}/{doi}"
            resp = requests.get(
                scihub_url,
                headers={"User-Agent": UA},
                timeout=DOWNLOAD_TIMEOUT,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                continue

            # 如果直接返回 PDF
            if "application/pdf" in resp.headers.get("Content-Type", ""):
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                return True

            html = resp.text

            # 方法 1: JS 中嵌入的 PDF 路径
            # 新格式: { url: '/storage/twin/.../paper.pdf', doi: '...' }
            js_match = re.search(
                r"url\s*:\s*['\"](/[^'\"]+\.pdf)['\"]",
                html,
            )
            if js_match:
                pdf_path = js_match.group(1)
                pdf_url = urljoin(scihub_url, pdf_path)
                if _download_pdf_url(pdf_url, save_path):
                    return True

            # 方法 2: <iframe src="..."> 或 <embed src="...">
            soup = BeautifulSoup(html, "html.parser")
            for tag_name in ["iframe", "embed", "object"]:
                for tag in soup.find_all(tag_name):
                    src = tag.get("src", "") or tag.get("data", "")
                    if src and (".pdf" in src.lower() or "storage" in src.lower()):
                        pdf_url = urljoin(scihub_url, src)
                        if _download_pdf_url(pdf_url, save_path):
                            return True

            # 方法 3: <a href="..."> 中包含 pdf
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if ".pdf" in href.lower() and "sci-hub" not in href.lower():
                    pdf_url = urljoin(scihub_url, href)
                    if _download_pdf_url(pdf_url, save_path):
                        return True

            # 方法 4: 页面中所有可能的 PDF URL
            matches = re.findall(r'(https?://[^"\'\s]+\.pdf)', html)
            for pdf_url in matches[:3]:
                if _download_pdf_url(pdf_url, save_path):
                    return True

            # 方法 5: 旧格式 onclick
            onclick_matches = re.findall(r"location\.href\s*=\s*['\"]([^'\"]+\.pdf[^'\"]*)['\"]", html)
            for pdf_url in onclick_matches:
                full_url = urljoin(scihub_url, pdf_url)
                if _download_pdf_url(full_url, save_path):
                    return True

        except requests.Timeout:
            continue
        except Exception:
            continue

    return False


def _download_pdf_url(pdf_url: str, save_path: str, referer: str = "") -> bool:
    """下载已知的 PDF URL 到本地。"""
    try:
        headers = {"User-Agent": UA}
        if referer:
            headers["Referer"] = referer
        else:
            # 从 PDF URL 推导 Referer
            from urllib.parse import urlparse as _up
            parsed = _up(pdf_url)
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"

        resp = requests.get(pdf_url, headers=headers, timeout=DOWNLOAD_TIMEOUT)
        if resp.status_code == 200 and (
            "application/pdf" in resp.headers.get("Content-Type", "")
            or resp.content[:5] == b"%PDF-"
        ):
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


# ── arXiv PDF 下载 ──

def arxiv_download(arxiv_id: str, save_path: str) -> bool:
    """从 arXiv 下载 PDF。"""
    try:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        resp = requests.get(
            url,
            headers={"User-Agent": UA},
            timeout=DOWNLOAD_TIMEOUT,
        )
        if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


# ── 直接 OA 下载 ──

def direct_oa_download(url: str, save_path: str) -> bool:
    """直接下载 openAccessUrl 指向的 PDF。"""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": UA},
            timeout=DOWNLOAD_TIMEOUT,
            allow_redirects=True,
        )
        if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


# ── PDF → 文本提取 ──

def pdf_to_text(pdf_path: str, max_chars: int = 12000) -> Optional[str]:
    """将 PDF 转为纯文本 (复用 pdf_text_extractor)。

    如果 extract_text_column_aware 不可用, 回退到 pdftotext 命令行。
    """
    try:
        text = extract_text_column_aware(pdf_path, max_chars=max_chars)
        if text and len(text.strip()) >= 100:
            return text.strip()
    except Exception:
        pass

    # Fallback: pdftotext
    try:
        proc = subprocess.run(
            ["pdftotext", "-l", "10", pdf_path, "-"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0 and len(proc.stdout.strip()) >= 100:
            return proc.stdout.strip()
    except Exception:
        pass

    return None


# ── 主流程 ──

def filename_to_source(filename: str, venue: str) -> str:
    """构建与现有格式兼容的 source 字段。"""
    return f"{journal_to_dirname(venue)}_{filename}"


def process_paper(
    paper: dict,
    checkpoint: dict,
    tier1_fp,   # file handle
    tier2_fp,   # file handle
    stats: dict,
    no_scihub: bool = False,
) -> bool:
    """处理单篇论文: 下载 PDF → 提取文本 → 写入 JSONL。

    Returns:
        True if full-text obtained, False if abstract-only.
    """
    paper_id = paper.get("paperId", "")
    doi = paper.get("doi", "")
    arxiv_id = paper.get("arxivId", "")
    oa_url = paper.get("openAccessUrl", "")
    venue = paper.get("_venue_normalized", "") or paper.get("venue", "")
    title = paper.get("title", "")

    # Checkpoint: 已处理过则跳过
    if paper_id and paper_id in checkpoint:
        return checkpoint[paper_id] == "fulltext"

    journal_dir = os.path.join(PDF_BASE_DIR, journal_to_dirname(venue))
    os.makedirs(journal_dir, exist_ok=True)

    pdf_path = None
    source = ""

    # 确定文件名
    if doi:
        filename = doi_to_filename(doi)
    else:
        filename = f"{paper_id}.pdf" if paper_id else hashlib.md5(title.encode()).hexdigest()[:16] + ".pdf"

    save_path = os.path.join(journal_dir, filename)

    # 跳过已存在且有效的 PDF
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        pdf_path = save_path
        stats["cached"] += 1

    # 下载流程
    if not pdf_path:
        downloaded = False

        # 1. Sci-Hub
        if doi and not no_scihub:
            stats["scihub_tried"] += 1
            if scihub_download(doi, save_path):
                downloaded = True
                stats["scihub_ok"] += 1
                time.sleep(REQUEST_DELAY)

        # 2. arXiv
        if not downloaded and arxiv_id:
            stats["arxiv_tried"] += 1
            if arxiv_download(arxiv_id, save_path):
                downloaded = True
                stats["arxiv_ok"] += 1
                time.sleep(REQUEST_DELAY)

        # 3. Direct OA
        if not downloaded and oa_url:
            stats["oa_tried"] += 1
            if direct_oa_download(oa_url, save_path):
                downloaded = True
                stats["oa_ok"] += 1

        if downloaded:
            pdf_path = save_path

    # 提取文本
    if pdf_path and os.path.exists(pdf_path):
        text = pdf_to_text(pdf_path)
        if text and len(text) >= 100:
            # Tier 1: 全文
            record = {
                "paper_id": paper_id,
                "doi": doi,
                "title": title,
                "year": paper.get("year"),
                "venue": venue,
                "citation_count": paper.get("citationCount", 0),
                "tier": 1,
                "source": f"{journal_to_dirname(venue)}_{filename}",
                "content_raw": text,
            }
            tier1_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            tier1_fp.flush()
            stats["fulltext"] += 1
            if paper_id:
                checkpoint[paper_id] = "fulltext"
            return True

    # Tier 2: 摘要
    abstract = paper.get("abstract", "") or ""
    record = {
        "paper_id": paper_id,
        "doi": doi,
        "title": title,
        "year": paper.get("year"),
        "venue": venue,
        "citation_count": paper.get("citationCount", 0),
        "tier": 2,
        "source": f"s2:{paper_id}",
        "content_raw": abstract,
    }
    tier2_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    tier2_fp.flush()
    stats["abstract_only"] += 1
    if paper_id:
        checkpoint[paper_id] = "abstract"
    return False


def main():
    parser = argparse.ArgumentParser(description="S2 论文 PDF 批量下载")
    parser.add_argument("--limit", type=int, help="只处理前 N 篇")
    parser.add_argument("--no-scihub", action="store_true", help="跳过 Sci-Hub")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="跳过已下载的 PDF (默认)")
    args = parser.parse_args()

    # 加载论文
    if not os.path.exists(DEDUPED_FILE):
        print(f"[DOWNLOAD] ERROR: Not found: {DEDUPED_FILE}", flush=True)
        print("[DOWNLOAD] Run s2_quality_filter.py first.", flush=True)
        sys.exit(1)

    print(f"[DOWNLOAD] Loading papers from {DEDUPED_FILE}...", flush=True)
    papers = []
    with open(DEDUPED_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    print(f"[DOWNLOAD] Loaded {len(papers)} papers.", flush=True)
    if args.limit:
        papers = papers[:args.limit]
        print(f"[DOWNLOAD] Limited to {args.limit}.", flush=True)

    # 统计分级
    tier1_count = sum(1 for p in papers if p.get("_tier") == 1)
    tier2_count = sum(1 for p in papers if p.get("_tier") == 2)
    print(f"[DOWNLOAD] Tier 1 (OA/arXiv): {tier1_count}, Tier 2 (abstract): {tier2_count}", flush=True)

    # Checkpoint
    checkpoint = load_checkpoint() if args.resume else {}

    # 打开输出文件
    mode_tier1 = "a" if args.resume else "w"
    mode_tier2 = "a" if args.resume else "w"
    tier1_fp = open(TIER1_FILE, mode_tier1)
    tier2_fp = open(TIER2_FILE, mode_tier2)

    stats = {
        "total": 0,
        "fulltext": 0,
        "abstract_only": 0,
        "scihub_tried": 0,
        "scihub_ok": 0,
        "arxiv_tried": 0,
        "arxiv_ok": 0,
        "oa_tried": 0,
        "oa_ok": 0,
        "cached": 0,
    }

    start_time = time.time()
    last_ckpt_flush = 0

    try:
        for i, paper in enumerate(papers):
            stats["total"] += 1
            is_fulltext = process_paper(
                paper, checkpoint,
                tier1_fp, tier2_fp,
                stats, no_scihub=args.no_scihub,
            )

            # 进度报告
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(papers) - i - 1) / rate if rate > 0 else 0
                print(f"[DOWNLOAD] {i+1}/{len(papers)} "
                      f"(fulltext: {stats['fulltext']}, abstract: {stats['abstract_only']}, "
                      f"SH: {stats['scihub_ok']}/{stats['scihub_tried']}, "
                      f"arXiv: {stats['arxiv_ok']}/{stats['arxiv_tried']}) "
                      f"| {rate:.1f} p/min | ETA {eta/60:.0f}min", flush=True)

            # 每 100 篇 flush checkpoint
            if (i + 1) % 100 == 0:
                save_checkpoint(checkpoint)

    finally:
        tier1_fp.close()
        tier2_fp.close()
        save_checkpoint(checkpoint)

    elapsed = time.time() - start_time
    print(f"\n[DOWNLOAD] COMPLETE in {elapsed/60:.1f}min", flush=True)
    print(f"[DOWNLOAD] Fulltext: {stats['fulltext']}, Abstract: {stats['abstract_only']}", flush=True)
    print(f"[DOWNLOAD] Sci-Hub: {stats['scihub_ok']}/{stats['scihub_tried']} "
          f"({100*stats['scihub_ok']/max(stats['scihub_tried'],1):.0f}%)", flush=True)
    print(f"[DOWNLOAD] arXiv: {stats['arxiv_ok']}/{stats['arxiv_tried']}", flush=True)
    print(f"[DOWNLOAD] OA direct: {stats['oa_ok']}/{stats['oa_tried']}", flush=True)
    print(f"[DOWNLOAD] Cached: {stats['cached']}", flush=True)


if __name__ == "__main__":
    main()
