#!/usr/bin/env python3
"""
campus_download.py — 校内机器专用: 真浏览器批量下载论文 PDF

使用 Playwright (真 Chrome) 而非 requests, 避免反爬检测。
工作流:
  1. 启动 Chrome 窗口 (你能看到)
  2. 每个出版者: 打开样本 DOI → 你在浏览器中手动登录
  3. 登录完成后按 Enter → 脚本接管, 逐个打开 DOI 页面, 点 PDF 按钮, 下载
  4. 下一个出版者, 重复

安装:
  pip install playwright
  playwright install chromium

用法:
  python campus_download.py
  python campus_download.py --publisher elsevier
  python campus_download.py --limit 10  (每个出版者最多 N 篇)
"""

import json
import os
import re
import sys
import time
import argparse
# 共享的期刊分类器 (与 s2_download_pdfs.py / audit_classification.py 共用)
from journal_classifier import classify_venue

# ── 配置 ──

PDF_BASE_DIR = "./journals_pdf"
MISSING_FILE = "./missing_non_wiley.jsonl"
CHECKPOINT_FILE = "./campus_checkpoint.json"
LOG_FILE = "./campus_download.log"

MIN_DELAY = 3.0   # 页面间最小间隔
MAX_DELAY = 8.0   # 页面间最大间隔
TIMEOUT = 30_000  # Playwright 超时 (毫秒)

DOI_PREFIX_MAP = {
    "10.1016": "elsevier",
    "10.1021": "acs",
    "10.1039": "rsc",
    "10.1038": "nature",
    "10.1126": "science",
    "10.1007": "springer",
    "10.3390": "mdpi",
    "10.1088": "iop",
    "10.1063": "aip",
    "10.1109": "ieee",
    "10.1103": "aps",
    "10.1093": "oup",
    "10.2139": "ssrn",
    "10.1073": "pnas",
    "10.1371": "plos",
    "10.1002": "wiley",
}

PUBLISHER_INFO = {
    "elsevier":  {"name": "Elsevier / ScienceDirect", "sample": "10.1016/j.joule.2024.02.019"},
    "acs":       {"name": "ACS Publications", "sample": "10.1021/acs.chemrev.3c00396"},
    "rsc":       {"name": "RSC Publishing", "sample": "10.1039/d2ra05903g"},
    "nature":    {"name": "Nature.com", "sample": "10.1038/s41586-023-05825-y"},
    "science":   {"name": "Science / AAAS", "sample": "10.1126/science.abp8873"},
    "springer":  {"name": "Springer Link", "sample": "10.1007/s10904-023-02777-8"},
    "mdpi":      {"name": "MDPI (OA)", "sample": "10.3390/nano11051218"},
    "iop":       {"name": "IOPscience", "sample": "10.1088/1361-6528/acb123"},
    "aip":       {"name": "AIP Publishing", "sample": "10.1063/5.0123456"},
    "ieee":      {"name": "IEEE Xplore", "sample": "10.1109/JPHOTOV.2023.1234567"},
    "aps":       {"name": "APS Journals", "sample": "10.1103/PhysRevB.107.125203"},
    "pnas":      {"name": "PNAS", "sample": "10.1073/pnas.2301234567"},
}

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def publisher_from_doi(doi: str) -> str:
    return DOI_PREFIX_MAP.get(doi.split("/")[0], "unknown")


def safe_filename(doi: str) -> str:
    return doi.replace("/", "_").replace("\\", "_").replace(":", "_") + ".pdf"


# ── 全量论文索引 & DOI 匹配 ──

_paper_index: dict[str, dict] | None = None  # 懒加载


def load_paper_index() -> dict[str, dict]:
    """加载全量 missing_non_wiley.jsonl → {doi_lower: paper}."""
    global _paper_index
    if _paper_index is not None:
        return _paper_index
    _paper_index = {}
    if not os.path.exists(MISSING_FILE):
        log(f"⚠️  {MISSING_FILE} not found — DOI verification disabled")
        return _paper_index
    with open(MISSING_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            doi = p.get("doi", "")
            if doi:
                _paper_index[doi.lower()] = p
    log(f"📋 Loaded {len(_paper_index)} papers into index")
    return _paper_index


# ── Playwright 下载 ──

def authenticate_publisher_ui(page, publisher: str, papers: list[dict]):
    """打开样本 DOI 让用户手动登录。"""
    info = PUBLISHER_INFO.get(publisher, {"name": publisher, "sample": papers[0]["doi"]})
    sample_doi = info["sample"]
    name = info["name"]

    print(f"\n{'='*60}")
    print(f"  📚 {name.upper()} — {len(papers)} papers")
    print(f"  🌐 Opening: https://doi.org/{sample_doi}")
    print(f"  👉 Please log in via your institution (CARSI / SSO) in the browser window")
    print(f"  👉 After you see the article page, come back here and press Enter")
    print(f"{'='*60}")

    page.goto(f"https://doi.org/{sample_doi}", wait_until="domcontentloaded")
    input("\n  Press Enter when logged in...")

    # 验证登录状态
    current_url = page.url.lower()
    if any(kw in current_url for kw in ["login", "signin", "authenticate", "idp.", "wayf."]):
        log(f"  ⚠️  Still on login page! Continue anyway? (y/n)")
        if input("  > ").lower() != 'y':
            return False
    else:
        log(f"  ✅ Auth seems OK ({page.url[:100]})")

    return True


# ── 主流程 ──

def main():
    parser = argparse.ArgumentParser(description="校内批量下载论文 PDF (Playwright 真浏览器)")
    parser.add_argument("--publisher", type=str, help="只处理指定出版者")
    parser.add_argument("--limit", type=int, help="每个出版者最多下载 N 篇")
    parser.add_argument("--dry-run", action="store_true", help="仅统计")
    args = parser.parse_args()

    # 加载论文
    if not os.path.exists(MISSING_FILE):
        log(f"❌ {MISSING_FILE} not found!")
        sys.exit(1)

    papers = []
    with open(MISSING_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            pub = publisher_from_doi(p["doi"])
            if args.publisher and pub != args.publisher:
                continue
            papers.append(p)

    by_pub = {}
    for p in papers:
        pub = publisher_from_doi(p["doi"])
        by_pub.setdefault(pub, []).append(p)

    pub_order = sorted(by_pub.keys(), key=lambda p: -len(by_pub[p]))
    total = sum(len(ps) for ps in by_pub.values())
    log(f"📊 {total} papers from {len(by_pub)} publishers:")
    for pub in pub_order:
        info = PUBLISHER_INFO.get(pub, {"name": pub})
        log(f"   {info['name']}: {len(by_pub[pub])}")

    if args.dry_run:
        return

    # 加载全量论文索引 (用于下载后 DOI 校验)
    paper_index = load_paper_index()

    checkpoint = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            checkpoint = json.load(f)
        log(f"📋 Checkpoint: {sum(1 for v in checkpoint.values() if v == 'ok')} done")

    # 启动 Playwright → Edge (反检测配置)
    from playwright.sync_api import sync_playwright
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        channel="msedge",
        headless=False,
        args=[
            "--start-maximized",
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context(
        accept_downloads=True,
        viewport={"width": 1280, "height": 800},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    )
    # 隐藏 webdriver 标记
    page = context.new_page()
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => false });
        Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
    """)
    page = context.new_page()

    grand_ok, grand_fail, grand_skip = 0, 0, 0

    try:
        for pub in pub_order:
            pub_papers = by_pub[pub]
            if args.limit:
                pub_papers = pub_papers[:args.limit]

            info = PUBLISHER_INFO.get(pub, {"name": pub})
            log(f"\n{'='*60}")
            log(f"📚 {info['name']}: {len(pub_papers)} papers")

            # 认证
            if not authenticate_publisher_ui(page, pub, pub_papers):
                log(f"⏭️  Skipping {info['name']}")
                continue

            ok, fail, skip = 0, 0, 0
            failed_dois = []
            start_time = time.time()

            # 过滤出待下载的论文
            pending = []
            for p in pub_papers:
                doi = p["doi"]
                if doi in checkpoint and checkpoint[doi] == "ok":
                    skip += 1
                    continue
                venue = p.get("venue", "")
                jdir = os.path.join(PDF_BASE_DIR, classify_venue(venue))
                save_path = os.path.join(jdir, safe_filename(doi))
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                    ok += 1
                    grand_ok += 1
                    checkpoint[doi] = "ok"
                    continue
                pending.append(p)

            if not pending:
                log(f"  ✅ All {len(pub_papers)} papers already done!")
                continue

            log(f"  📋 {len(pending)} papers to download")

            # ── 批量下载: 每批开 N 个 tab ──
            BATCH_SIZE = 20
            paper_index = load_paper_index()

            for batch_start in range(0, len(pending), BATCH_SIZE):
                batch = pending[batch_start:batch_start + BATCH_SIZE]
                batch_num = batch_start // BATCH_SIZE + 1
                total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

                log(f"\n  📑 Batch {batch_num}/{total_batches}: opening {len(batch)} tabs...")

                # 收集本轮下载
                captured_downloads: list[tuple[any, str, str]] = []  # [(download, doi, title), ...]

                def make_download_handler():
                    """闭包: 捕获下载事件并记录。"""
                    def handler(download):
                        suggested = download.suggested_filename or ""
                        log(f"    📥 Download: {suggested[:60]}")
                        captured_downloads.append((download, "", suggested))
                    return handler

                # 注册上下文级别的下载监听
                context.on("download", make_download_handler())

                # 开 tab, 导航到 DOI
                tabs = []
                for paper in batch:
                    doi = paper["doi"]
                    title = paper.get("title", "")[:80]
                    venue = paper.get("venue", "")
                    tab = context.new_page()
                    tab.goto(f"https://doi.org/{doi}", wait_until="domcontentloaded", timeout=TIMEOUT)
                    tabs.append((tab, paper))
                    log(f"    🔗 {title[:70]}")
                    time.sleep(0.5)  # 避免太快被反爬

                log(f"  🖐️  {len(tabs)} tabs ready — click PDF download on each tab")
                log(f"  👉 Press Enter in terminal when all downloads finish (or type 'q' to skip this batch)")
                user_input = input("  > ").strip().lower()
                if user_input == 'q':
                    for tab, _ in tabs:
                        try:
                            tab.close()
                        except Exception:
                            pass
                    continue

                # 等所有下载完成
                time.sleep(3)  # 再等 3 秒确保最后的下载也被抓到

                # ── 处理捕获的下载 ──
                processed = set()
                for download, _, suggested_filename in captured_downloads:
                    # 用 suggested_filename 去全量 index 匹配真实 DOI
                    resolved_doi = None
                    matched_paper = None

                    for doi_lower, p in paper_index.items():
                        # 从 suggested filename 提取关键标识符匹配
                        name_key = suggested_filename.lower().removesuffix(".pdf")
                        if len(name_key) < 5:
                            continue
                        # 直接子串匹配
                        if name_key in doi_lower or doi_lower.replace("/", "_").replace(":", "_") in name_key.replace("-", "_"):
                            resolved_doi = p["doi"]
                            matched_paper = p
                            break

                    if not resolved_doi:
                        # 兜底: 遍历 batch 看看哪个 DOI 最匹配
                        for paper in batch:
                            expected_doi = paper["doi"].lower()
                            doi_suffix = expected_doi.split("/")[-1].replace("-", "_")
                            if doi_suffix in suggested_filename.lower():
                                resolved_doi = paper["doi"]
                                matched_paper = paper
                                break

                    if not resolved_doi:
                        log(f"    ⚠️  Cannot match download: {suggested_filename[:60]}")
                        fail += 1
                        continue

                    if resolved_doi in processed:
                        continue
                    processed.add(resolved_doi)

                    # 存盘
                    venue = matched_paper.get("venue", "") or ""
                    jdir = os.path.join(PDF_BASE_DIR, classify_venue(venue))
                    os.makedirs(jdir, exist_ok=True)
                    save_path = os.path.join(jdir, safe_filename(resolved_doi))

                    try:
                        download.save_as(save_path)
                        if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                            ok += 1
                            grand_ok += 1
                            checkpoint[resolved_doi] = "ok"
                            title = matched_paper.get("title", "")[:60]
                            log(f"    ✅ [{ok}] {classify_venue(venue)}/{safe_filename(resolved_doi)}")
                            log(f"       {title}")
                        else:
                            fail += 1
                            grand_fail += 1
                            checkpoint[resolved_doi] = "fail"
                            log(f"    ❌ Save failed: too small")
                    except Exception as e:
                        fail += 1
                        grand_fail += 1
                        log(f"    ❌ Save error: {e}")

                # 清理: 关 tab
                for tab, _ in tabs:
                    try:
                        tab.close()
                    except Exception:
                        pass

                # 保存 checkpoint
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f)

                # 未匹配到的下载标记为失败
                for paper in batch:
                    if paper["doi"] not in processed and paper["doi"] not in checkpoint:
                        fail += 1
                        grand_fail += 1
                        checkpoint[paper["doi"]] = "fail"
                        log(f"    🔴 No download: {paper['doi']}")

                elapsed = time.time() - start_time
                done = ok + fail
                rate = done / elapsed * 60 if elapsed > 0 else 0
                remaining = len(pending) - batch_start - len(batch)
                eta = remaining / rate if rate > 0 else 0
                log(f"  [{info['name']}] batch {batch_num}/{total_batches} done | ok={ok} fail={fail} | ETA {eta:.0f}min")

            # 本 publisher 结束
            grand_skip += skip
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f)

    finally:
        browser.close()
        playwright.stop()

    log(f"\n🎉 ALL DONE: ok={grand_ok} fail={grand_fail} skip={grand_skip}")


if __name__ == "__main__":
    main()
