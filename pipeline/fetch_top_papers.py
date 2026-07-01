#!/usr/bin/env python3
"""
fetch_top_papers.py — 从 Nature 系列期刊扒钙钛矿论文并下载 PDF

用法:
  python3 fetch_top_papers.py                  # 全量运行（搜索 + 下载）
  python3 fetch_top_papers.py --dry-run         # 只搜元数据不下载
  python3 fetch_top_papers.py --search-only     # 搜索所有期刊但不下载
  python3 fetch_top_papers.py --daily           # 每日增量搜索（凌晨 cron 用）
"""

import json, os, time, sys
from datetime import datetime
from habanero import Crossref
import requests
from collections import Counter

# ============================================================
# 配置
# ============================================================
CROSSREF_DELAY = 1.1

# 当前可下载的期刊（Nature 系列 —— 无需认证）
JOURNALS = {
    "0028-0836": "Nature",
    "2058-7546": "Nature Energy",
    "1476-1122": "Nature Materials",
    "1749-4893": "Nature Photonics",
    "1748-3395": "Nature Nanotechnology",
    "2041-1723": "Nature Communications",
}

QUERY = "perovskite solar cell"
QUERY_ALT = "perovskite photovoltaic"
YEAR_FROM = 2020  # 首次全量搜索起始年份
YEAR_TO = 2026
# 每日增量搜索：只搜当天发表的新文章
DAILY_DAYS_BACK = 0  # 0 = 只搜当天
CITATION_THRESHOLD = 50
CITATION_THRESHOLD_RECENT = 0  # 新文章不设引用下限
MAX_PER_JOURNAL = 200
# 每日模式每本期刊最多取多少篇
DAILY_MAX_PER_JOURNAL = 10

# PDF 保存位置：每本期刊一个独立子目录
PDF_BASE_DIR = "/data/data/pkb/01_raw_data/journals_pdf"

# 元数据和下载记录
DATA_DIR = "/data1/perovskite-rag/data/pipeline"
METADATA_FILE = os.path.join(DATA_DIR, "fetched_papers_metadata.json")
TRACKER_FILE = os.path.join(DATA_DIR, "download_tracker.json")

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 出版商识别
# ============================================================
def get_publisher(issn):
    nature = {"0028-0836","2058-7546","1476-1122","1749-4893","1748-3395","2041-1723"}
    if issn in nature: return "nature"
    return "unknown"


# ============================================================
# PDF 下载（Nature 系列）
# ============================================================
def download_nature_pdf(doi_suffix, save_path):
    """Nature 系列 PDF 下载
    1. 先走主链接 https://www.nature.com/articles/{doi}.pdf
    2. 如果被反爬拦截（返回 HTML 而非 PDF），fallback 到 _reference.pdf（预印本校样版）
    3. 都失败则 return None
    """
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/pdf,*/*"}

    # 方案 A：主链接
    url = f"https://www.nature.com/articles/{doi_suffix}.pdf"
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", ""):
        with open(save_path, "wb") as f:
            f.write(r.content)
        return len(r.content)

    # 方案 B：_reference.pdf（预印本校样版，不受登录墙拦截）
    url_ref = f"https://www.nature.com/articles/{doi_suffix}_reference.pdf"
    r = requests.get(url_ref, headers=headers, timeout=30)
    if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", ""):
        with open(save_path, "wb") as f:
            f.write(r.content)
        return len(r.content)

    return None


# ============================================================
# CrossRef API 搜索
# ============================================================
def search_journal(issn, name, yf, yt, query, date_from=None, date_to=None):
    cr = Crossref()
    papers = []
    offset = 0
    rows = 50
    while True:
        try:
            r = cr.works(
                query=query, limit=rows, offset=offset,
                filter={
                    "issn": issn, "type": "journal-article",
                    "from-pub-date": date_from or f"{yf}-01-01",
                    "until-pub-date": date_to or f"{yt}-12-31",
                },
                sort="relevance", order="desc",
            )
        except Exception as e:
            print(f"  ⚠ API err: {e}")
            time.sleep(5)
            continue
        items = r["message"].get("items", [])
        total = r["message"].get("total-results", 0)
        if not items:
            break
        for item in items:
            doi = item.get("DOI", "")
            if not doi:
                continue
            title = item.get("title", [""])[0]
            cited = item.get("is-referenced-by-count", 0)
            year = (
                item.get("published-print", {}).get("date-parts", [[0]])[0][0]
                or item.get("published-online", {}).get("date-parts", [[0]])[0][0]
                or 0
            )
            threshold = CITATION_THRESHOLD_RECENT if year >= 2025 else CITATION_THRESHOLD
            if cited < threshold:
                continue
            authors = []
            for a in item.get("author", []):
                if a.get("family"):
                    authors.append(f"{a.get('given', '')} {a['family']}".strip())
            # 关键词过滤：标题和摘要必须至少匹配一个太阳能关键词
            text_to_check = title + " " + (item.get("abstract", "") or "")
            if not is_perovskite_solar(text_to_check):
                continue

            papers.append({
                "doi": doi,
                "title": title,
                "journal": name,
                "year": year,
                "cited": cited,
                "authors": authors,
                "abstract": (item.get("abstract", "") or "")[:500],
                "url": item.get("URL", ""),
                "publisher": get_publisher(issn),
            })
        offset += rows
        if offset >= MAX_PER_JOURNAL or offset >= total:
            break
        time.sleep(CROSSREF_DELAY)
    return papers


# ============================================================
# 关键词过滤
# ============================================================

# 论文标题/摘要必须包含以下至少一个关键词才算真正的钙钛矿太阳能论文
SOLAR_KEYWORDS = [
    "perovskite solar",
    "perovskite photovoltaic",
    "perovskite/silicon tandem",
    "perovskite-organic tandem",
    "all-perovskite tandem",
    "perovskite cell",
    "perovskite module",
    "perovskite absorber",
    "perovskite film",
    "perovskite layer",
    "perovskite device",
    "hole transport",
    "electron transport",
    "passivation",
    "FAPbI3",
    "formamidinium",
    "methylammonium",
    "CsPbI3",
    "CsPbBr3",
    "tin perovskite",
    "lead perovskite",
    "halide perovskite",
    "perovskite crystal",
    "perovskite precursor",
    "SnO2",
    "spiro-OMeTAD",
    "NiOx",
]


def is_perovskite_solar(text):
    """判断一段文本是否与钙钛矿太阳能相关"""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in SOLAR_KEYWORDS)


# ============================================================
# 工具函数
# ============================================================
def get_doi_suffix(doi):
    return doi.split("/", 1)[1] if "/" in doi else doi


def journal_abbr(name):
    m = {
        "Nature": "Nature",
        "Nature Energy": "NatEnergy",
        "Nature Materials": "NatMater",
        "Nature Photonics": "NatPhoton",
        "Nature Nanotechnology": "NatNanotech",
        "Nature Communications": "NatComm",
    }
    return m.get(name, name[:10])


# ============================================================
# 主流程
# ============================================================
def get_date_filter(daily=False):
    """获取日期过滤参数"""
    if not daily:
        return {
            "from-pub-date": f"{YEAR_FROM}-01-01",
            "until-pub-date": f"{YEAR_TO}-12-31",
        }
    # 每日模式：只搜当天发表的文章
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    return {
        "from-pub-date": today_str,
        "until-pub-date": today_str,
    }


def search_all(daily=False):
    """搜索所有配置的期刊"""
    all_p = []
    date_filter = get_date_filter(daily)
    max_per = DAILY_MAX_PER_JOURNAL if daily else MAX_PER_JOURNAL
    label = "📅 每日增量" if daily else f"🔍 {YEAR_FROM}-{YEAR_TO}"
    print("=" * 60)
    print(f"{label} 钙钛矿论文 | {len(JOURNALS)} 本 Nature 期刊")
    print(f"   日期范围: {date_filter['from-pub-date']} ~ {date_filter['until-pub-date']}")
    print("=" * 60)
    for issn, name in JOURNALS.items():
        pub = get_publisher(issn)
        print(f"\n📖 {name} ({pub})")
        papers = search_journal(
            issn, name,
            int(date_filter["from-pub-date"][:4]),
            int(date_filter["until-pub-date"][:4]),
            QUERY,
            date_from=date_filter["from-pub-date"] if daily else None,
            date_to=date_filter["until-pub-date"] if daily else None,
        )
        if len(papers) < 10:
            p2 = search_journal(
                issn, name,
                int(date_filter["from-pub-date"][:4]),
                int(date_filter["until-pub-date"][:4]),
                QUERY_ALT,
                date_from=date_filter["from-pub-date"] if daily else None,
                date_to=date_filter["until-pub-date"] if daily else None,
            )
            seen = {p["doi"] for p in papers}
            for p in p2:
                if p["doi"] not in seen:
                    papers.append(p)
                    seen.add(p["doi"])
        papers.sort(key=lambda x: x["cited"], reverse=True)
        papers = papers[:max_per]
        print(f"  ✅ {len(papers)} 篇")
        if papers:
            print(f"  🏆 最高引: [{papers[0]['cited']}] {papers[0]['title'][:60]}")
        all_p.extend(papers)
        time.sleep(CROSSREF_DELAY)
    all_p.sort(key=lambda x: x["cited"], reverse=True)
    return all_p


def download_all(papers):
    """下载论文 PDF"""
    tracker = {}
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE) as f:
            tracker = json.load(f)

    stats = {"total": len(papers), "success": 0, "skipped": 0, "failed": 0}
    fails = []

    print(f"\n{'=' * 60}")
    print(f"⬇️  开始下载 {len(papers)} 篇论文 PDF")
    print(f"{'=' * 60}")

    for i, paper in enumerate(papers):
        doi = paper["doi"]
        journal = paper["journal"]
        year = paper["year"]
        pub = paper["publisher"]
        suf = get_doi_suffix(doi)
        fname = f"{journal_abbr(journal)}_{year}_{suf.replace('/', '_')}.pdf"

        # 按期刊名建子目录
        journal_dir = os.path.join(PDF_BASE_DIR, journal_abbr(journal))
        os.makedirs(journal_dir, exist_ok=True)
        path = os.path.join(journal_dir, fname)

        # 跳过已下载
        if os.path.exists(path):
            tracker[doi] = {"status": "skipped", "path": path}
            stats["skipped"] += 1
            continue
        if doi in tracker and tracker[doi].get("status") == "success" and os.path.exists(tracker[doi].get("path", "")):
            stats["skipped"] += 1
            continue

        # 下载
        try:
            if pub == "nature":
                sz = download_nature_pdf(suf, path)
                if sz and sz > 100000:
                    tracker[doi] = {"status": "success", "path": path, "size": sz}
                    stats["success"] += 1
                else:
                    raise Exception(f"PDF 太小或为空: {sz} bytes")
            else:
                tracker[doi] = {"status": "pending", "doi": doi, "publisher": pub}
                stats["skipped"] += 1
        except Exception as e:
            tracker[doi] = {"status": "failed", "error": str(e)}
            fails.append((doi, str(e)))
            stats["failed"] += 1

        # 保存进度
        with open(TRACKER_FILE, "w") as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)
        time.sleep(0.5)

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(papers)}")

    return stats, fails


def print_report(papers, stats, fails, start_time):
    """打印摘要报告"""
    jc = Counter(p["journal"] for p in papers)
    print(f"\n{'=' * 60}")
    print("📊 报告")
    print(f"{'=' * 60}")

    print(f"\n📚 论文分布:")
    for j, c in jc.most_common():
        top = max((p for p in papers if p["journal"] == j), key=lambda x: x["cited"])
        print(f"  {j}: {c} 篇 (最高引: {top['cited']} — {top['title'][:50]})")

    print(f"\n📥 下载统计:")
    print(f"  总计: {stats['total']} 篇")
    print(f"  ✅ 成功: {stats['success']}")
    print(f"  ⏭️  跳过: {stats['skipped']}")
    print(f"  ❌ 失败: {stats['failed']}")

    if fails:
        print(f"\n失败列表:")
        for d, e in fails[:10]:
            print(f"  ✗ {d}: {e}")

    print(f"\n⏱️  总耗时: {datetime.now() - start_time}")
    print(f"📄 元数据: {METADATA_FILE}")
    print(f"📄 进度: {TRACKER_FILE}")
    print(f"📁 PDF: {PDF_BASE_DIR}/")


# ============================================================
# 入口
# ============================================================
def main():
    dry_run = "--dry-run" in sys.argv
    search_only = "--search-only" in sys.argv
    daily = "--daily" in sys.argv
    start = datetime.now()

    print("搜索中...")
    papers = search_all(daily=daily)
    print(f"\n共找到 {len(papers)} 篇论文")

    if dry_run:
        for p in papers[:20]:
            print(f"  [{p['cited']:4d}] [{p['journal']}] {p['title'][:70]}")
        return

    if search_only or daily:
        # daily 模式下只搜不下，因为新文章可能还没卷到 RAG 里
        with open(METADATA_FILE, "w") as f:
            json.dump(
                {"generated_at": datetime.now().isoformat(), "total": len(papers), "papers": papers},
                f, indent=2, ensure_ascii=False,
            )
        print(f"\n{'='*60}")
        print(f"{'='*60}")
        print(f"✅ 每日扫描完成! 找到 {len(papers)} 篇新论文")
        for p in papers:
            print(f"  [{p['cited']:4d}] [{p['journal']}] {p['title'][:80]}")
        print(f"元数据: {METADATA_FILE}")
        print(f"耗时: {datetime.now() - start}")
        return

    stats, fails = download_all(papers)
    print_report(papers, stats, fails, start)


if __name__ == "__main__":
    main()
