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
import random
import argparse
from pathlib import Path
from urllib.parse import urljoin

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


def human_click(page, element):
    """模拟真人点击: 先慢移鼠标, 再点击。"""
    box = element.bounding_box()
    if box:
        x = box["x"] + box["width"] * random.uniform(0.3, 0.7)
        y = box["y"] + box["height"] * random.uniform(0.3, 0.7)
        # 分多步移动 (模拟真人轨迹)
        steps = random.randint(3, 6)
        for step in range(steps):
            mx = x + random.uniform(-5, 5)
            my = y + random.uniform(-3, 3)
            page.mouse.move(mx, my)
            time.sleep(random.uniform(0.02, 0.08))
        time.sleep(random.uniform(0.1, 0.3))
        page.mouse.click(x, y)
    else:
        element.click()
    time.sleep(random.uniform(0.5, 1.5))


def human_scroll(page):
    """模拟真人滚动页面。"""
    page.evaluate(f"window.scrollBy(0, {random.randint(100, 400)})")
    time.sleep(random.uniform(0.3, 0.8))


def find_and_click_pdf(page) -> bool:
    """在当前页面找 PDF 下载按钮并点击, 等待下载。"""
    # 不同出版者的 PDF 按钮/链接模式
    selectors = [
        # 通用: 包含 "PDF" 的链接或按钮
        "a:has-text('PDF')",
        "a:has-text('pdf')",
        "button:has-text('PDF')",
        "a[href*='pdf']",
        # Elsevier: "View PDF" 按钮
        "a.pdf-download-btn",
        "a[href*='/pdfft']",
        # ACS: PDF 链接
        "a[href*='/doi/pdf/']",
        # Nature: PDF 链接
        "a[href$='.pdf']",
        # Science: PDF 标签
        "a[href*='/doi/pdf/']",
        # Springer: "Download PDF"
        "a:has-text('Download PDF')",
        "a:has-text('Download book PDF')",
        # RSC: PDF 链接
        "a[href*='articlepdf']",
    ]

    for sel in selectors:
        try:
            elem = page.locator(sel).first
            if elem.is_visible(timeout=1000):
                href = elem.get_attribute("href") or ""
                # 跳过补充材料
                if "suppl" in href.lower() or "supplementary" in href.lower():
                    continue
                # 真人式点击
                human_click(page, elem)
                time.sleep(1)
                # 检查是否打开了 PDF (内嵌预览或新标签)
                if ".pdf" in page.url.lower():
                    return True, None  # PDF 已在当前页面
                # 尝试捕获下载事件
                try:
                    with page.expect_download(timeout=5000) as dl:
                        pass  # 下载可能已触发
                    return True, dl.value
                except Exception:
                    pass
                # 如果页面变化了, 可能是 PDF 内嵌
                if "application/pdf" in page.content()[:500]:
                    return True, None
        except Exception:
            continue

    return False, None


def download_paper_ui(page, doi: str, save_path: str) -> bool:
    """用 Playwright 下载单篇论文。"""
    # 打开 DOI 页面
    try:
        page.goto(f"https://doi.org/{doi}", wait_until="domcontentloaded", timeout=TIMEOUT)
    except Exception as e:
        log(f"    ❌ Page load: {e}")
        return False

    # 模拟真人浏览: 随机滚动几下
    for _ in range(random.randint(1, 3)):
        human_scroll(page)

    # 检查是否是错误页
    url_low = page.url.lower()
    if any(kw in url_low for kw in ["login", "signin", "authenticate", "idp."]):
        log(f"    ⚠️  Redirected to login — cookies may be expired")
        return False

    # 检查标题是否包含 "Not Found" 等
    title = page.title().lower()
    if any(kw in title for kw in ["not found", "404", "error"]):
        log(f"    ❌ Page not found: {title[:80]}")
        return False

    # 检查是否直接打开 PDF (浏览器内嵌预览)
    if ".pdf" in page.url.lower() or "application/pdf" in page.content()[:500].lower():
        try:
            pdf_bytes = page.evaluate("""
                async () => {
                    const resp = await fetch(window.location.href);
                    const buf = await resp.arrayBuffer();
                    return Array.from(new Uint8Array(buf));
                }
            """)
            if pdf_bytes and bytes(pdf_bytes[:5]) == b"%PDF-":
                with open(save_path, "wb") as f:
                    f.write(bytes(pdf_bytes))
                return True
        except Exception:
            pass

    # 尝试点击 PDF 按钮下载
    found, download = find_and_click_pdf(page)

    if found and download:
        try:
            download.save_as(save_path)
            return True
        except Exception as e:
            log(f"    ❌ Save failed: {e}")

    # Fallback: 直接构造 PDF URL 并尝试保存
    pub = publisher_from_doi(doi)
    pdf_url = guess_pdf_url(doi, pub)
    if pdf_url:
        try:
            # 方法 A: 导航到 PDF URL, 触发下载
            with page.expect_download(timeout=15000) as dl:
                page.goto(pdf_url, wait_until="domcontentloaded", timeout=TIMEOUT)
            download = dl.value
            download.save_as(save_path)
            if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                return True
        except Exception:
            pass

        try:
            # 方法 B: 导航到 PDF URL, 直接保存响应内容
            resp = page.goto(pdf_url, wait_until="domcontentloaded", timeout=TIMEOUT)
            if resp and resp.body()[:5] == b"%PDF-":
                with open(save_path, "wb") as f:
                    f.write(resp.body())
                return True
        except Exception:
            pass

        try:
            # 方法 C: 用 page.evaluate 发起 fetch 拿 PDF
            pdf_bytes = page.evaluate("""
                async () => {
                    const resp = await fetch(arguments[0]);
                    const buf = await resp.arrayBuffer();
                    return Array.from(new Uint8Array(buf));
                }
            """, pdf_url)
            if pdf_bytes and bytes(pdf_bytes[:5]) == b"%PDF-":
                with open(save_path, "wb") as f:
                    f.write(bytes(pdf_bytes))
                return True
        except Exception:
            pass

    return False


def guess_pdf_url(doi: str, publisher: str) -> str | None:
    if publisher == "nature":
        return f"https://www.nature.com/articles/{doi.split('/',1)[1]}.pdf"
    elif publisher == "science":
        return f"https://www.science.org/doi/pdf/{doi}"
    elif publisher == "acs":
        return f"https://pubs.acs.org/doi/pdf/{doi}"
    elif publisher == "rsc":
        return f"https://pubs.rsc.org/en/content/articlepdf/{doi.split('/',1)[1]}"
    elif publisher == "elsevier":
        return f"https://www.sciencedirect.com/science/article/pii/{doi.split('/',1)[1]}/pdfft"
    elif publisher == "springer":
        return f"https://link.springer.com/content/pdf/{doi}.pdf"
    elif publisher == "mdpi":
        return f"https://www.mdpi.com/{doi.split('/',1)[1]}/pdf"
    return None


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

            for i, paper in enumerate(pub_papers):
                doi = paper["doi"]
                title = paper.get("title", "")[:80]
                venue = paper.get("venue", "")

                log(f"\n  [{i+1}/{len(pub_papers)}] https://doi.org/{doi}")
                log(f"    {title}")

                if doi in checkpoint and checkpoint[doi] == "ok":
                    skip += 1
                    log(f"    ⏭️  Already done (skip)")
                    continue

                jdir = os.path.join(PDF_BASE_DIR, classify_venue(venue))
                os.makedirs(jdir, exist_ok=True)
                save_path = os.path.join(jdir, safe_filename(doi))

                if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                    ok += 1
                    checkpoint[doi] = "ok"
                    log(f"    ✅ Cached")
                    continue

                if download_paper_ui(page, doi, save_path):
                    ok += 1
                    checkpoint[doi] = "ok"
                    size = os.path.getsize(save_path) if os.path.exists(save_path) else 0
                    log(f"    ✅ {size} bytes")
                else:
                    fail += 1
                    checkpoint[doi] = "fail"
                    failed_dois.append(doi)
                    log(f"    🔴 FAILED — manual: https://doi.org/{doi}")

                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                    eta = (len(pub_papers) - i - 1) / rate if rate > 0 else 0
                    log(f"  [{info['name']}] {i+1}/{len(pub_papers)} ok={ok} fail={fail} | {rate:.1f}p/min ETA {eta:.0f}min")

                if (i + 1) % 100 == 0:
                    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                        json.dump(checkpoint, f)

                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            elapsed = time.time() - start_time
            log(f"  ✅ {info['name']} DONE: ok={ok} fail={fail} skip={skip} in {elapsed/60:.1f}min")
            if failed_dois:
                log(f"  🔴 CHECK ({len(failed_dois)}):")
                for d in failed_dois:
                    log(f"     https://doi.org/{d}")

            grand_ok += ok; grand_fail += fail; grand_skip += skip
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f)

    finally:
        browser.close()
        playwright.stop()

    log(f"\n🎉 ALL DONE: ok={grand_ok} fail={grand_fail} skip={grand_skip}")


if __name__ == "__main__":
    main()
