#!/usr/bin/env python3
"""
s2_quality_filter.py — S2 论文元数据质量过滤 & 去重

策略:
  1. 质量门槛: citationCount>=5 OR year>=2026 OR venue in 顶刊白名单
  2. 去重: DOI 精确匹配 → 标题 n-gram Jaccard (>0.85)
  3. 分级: Tier 1 (有 OA PDF) → Tier 2 (仅摘要)

用法:
  python3 s2_quality_filter.py                          # 全量过滤
  python3 s2_quality_filter.py --input s2_papers_raw.jsonl --output s2_papers_filtered.jsonl
"""

import json
import os
import sys
import re
import argparse
from collections import Counter
from typing import Optional

# ── 配置 ──

BASE_DIR = "/data1/perovskite-rag"
CORPUS_DIR = os.path.join(BASE_DIR, "data", "s2_corpus")
INPUT_FILE = os.path.join(CORPUS_DIR, "s2_papers_raw.jsonl")
OUTPUT_FILE = os.path.join(CORPUS_DIR, "s2_papers_deduped.jsonl")
REPORT_FILE = os.path.join(CORPUS_DIR, "filter_report.json")

# ── 顶刊白名单 (钙钛矿太阳能电池领域已发表) ──

VENUE_WHITELIST = {
    # Nature 系列
    "Nature", "Nature Energy", "Nature Materials", "Nature Photonics",
    "Nature Nanotechnology", "Nature Communications", "Nature Reviews Materials",
    "Nature Reviews Chemistry", "Communications Materials",
    # Science 系
    "Science", "Science Advances",
    # Cell 系
    "Joule", "Matter", "Chem", "Cell Reports Physical Science", "iScience",
    # ACS
    "ACS Energy Letters", "Journal of the American Chemical Society",
    "Chemistry of Materials", "ACS Applied Materials & Interfaces",
    "ACS Nano", "Nano Letters", "ACS Central Science", "JACS Au",
    "The Journal of Physical Chemistry Letters",
    "The Journal of Physical Chemistry C",
    "ACS Applied Energy Materials", "ACS Materials Letters",
    # Wiley
    "Advanced Materials", "Advanced Energy Materials",
    "Advanced Functional Materials", "Angewandte Chemie",
    "Advanced Science", "Small", "Small Methods", "Solar RRL",
    "Advanced Optical Materials", "InfoMat",
    # RSC
    "Energy & Environmental Science", "Journal of Materials Chemistry A",
    "Materials Horizons", "Nanoscale", "Nanoscale Horizons",
    "Sustainable Energy & Fuels", "Chemical Science",
    "Physical Chemistry Chemical Physics",
    # Elsevier
    "Nano Energy", "Chemical Engineering Journal",
    "Solar Energy Materials and Solar Cells", "Materials Today Energy",
    "Applied Surface Science", "Electrochimica Acta",
    "Nano Today", "Materials Science and Engineering: R: Reports",
    "Progress in Materials Science",
    # AIP / APS
    "Applied Physics Letters", "Journal of Applied Physics",
    "APL Materials", "Physical Review Applied", "Physical Review Materials",
    # Nature 子刊 (补充)
    "Light: Science & Applications", "NPG Asia Materials",
    # 综合顶刊
    "Proceedings of the National Academy of Sciences",
    "Accounts of Chemical Research", "Chemical Society Reviews",
    "Reports on Progress in Physics",
    # 能源/材料专门
    "Progress in Photovoltaics", "Journal of Energy Chemistry",
    "Materials Today", "Advanced Photonics",
}

# 标准化 venue 名称映射 (S2 有时缩写不一致)
VENUE_NORMALIZE = {
    "energy & environmental science": "Energy & Environmental Science",
    "energy environ. sci.": "Energy & Environmental Science",
    "j. am. chem. soc.": "Journal of the American Chemical Society",
    "angew. chem. int. ed.": "Angewandte Chemie",
    "angew. chem. int. ed. engl.": "Angewandte Chemie",
    "adv. mater.": "Advanced Materials",
    "adv. energy mater.": "Advanced Energy Materials",
    "adv. funct. mater.": "Advanced Functional Materials",
    "acs energy lett.": "ACS Energy Letters",
    "nano lett.": "Nano Letters",
    "acs nano": "ACS Nano",
    "chem. mater.": "Chemistry of Materials",
    "j. phys. chem. lett.": "The Journal of Physical Chemistry Letters",
    "j. phys. chem. c": "The Journal of Physical Chemistry C",
    "acs appl. mater. interfaces": "ACS Applied Materials & Interfaces",
    "adv. sci.": "Advanced Science",
    "nat. commun.": "Nature Communications",
    "nat. energy": "Nature Energy",
    "nat. mater.": "Nature Materials",
    "nat. photon.": "Nature Photonics",
    "nat. nanotechnol.": "Nature Nanotechnology",
    "sci. adv.": "Science Advances",
    "j. mater. chem. a": "Journal of Materials Chemistry A",
    "sustain. energy fuels": "Sustainable Energy & Fuels",
    "nanoscale horiz.": "Nanoscale Horizons",
    "sol. energy mater. sol. cells": "Solar Energy Materials and Solar Cells",
    "chem. eng. j.": "Chemical Engineering Journal",
    "appl. surf. sci.": "Applied Surface Science",
    "phys. chem. chem. phys.": "Physical Chemistry Chemical Physics",
    "j. energy chem.": "Journal of Energy Chemistry",
    "appl. phys. lett.": "Applied Physics Letters",
    "j. appl. phys.": "Journal of Applied Physics",
    "acs appl. energy mater.": "ACS Applied Energy Materials",
    "mater. today energy": "Materials Today Energy",
    "mater. horiz.": "Materials Horizons",
}


def normalize_venue(name: str) -> str:
    """标准化期刊名。"""
    if not name:
        return ""
    name = name.strip()
    key = name.lower().rstrip(".")
    if key in VENUE_NORMALIZE:
        return VENUE_NORMALIZE[key]
    return name


def is_high_quality(paper: dict) -> bool:
    """判断论文是否高质量 (三选一)。

    1. citationCount >= 5
    2. year >= 2026 (新论文, 引用还来不及积累)
    3. venue 在顶刊白名单中
    """
    citations = paper.get("citationCount", 0) or 0
    if citations >= 5:
        return True

    year = paper.get("year")
    if year and year >= 2026:
        return True

    venue = normalize_venue(paper.get("venue", ""))
    if venue and venue in VENUE_WHITELIST:
        return True

    return False


def load_papers(path: str) -> list[dict]:
    """加载 JSONL 文件。"""
    papers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


# ── 去重 ──

def deduplicate(papers: list[dict]) -> list[dict]:
    """去重: DOI 精确匹配 + 标题 n-gram Jaccard 模糊匹配。"""

    # Pass 1: DOI 去重
    seen_doi: dict[str, dict] = {}
    no_doi: list[dict] = []

    for p in papers:
        doi = (p.get("doi") or "").lower().strip()
        if doi:
            existing = seen_doi.get(doi)
            if not existing or (p.get("citationCount", 0) or 0) > (existing.get("citationCount", 0) or 0):
                seen_doi[doi] = p
        else:
            no_doi.append(p)

    deduped = list(seen_doi.values())
    print(f"  [DEDUP] DOI pass: {len(papers)} → {len(deduped)} (unique DOIs), "
          f"{len(no_doi)} papers without DOI", flush=True)

    # Pass 2: 标题 n-gram Jaccard 去重 (仅对无 DOI 的论文)
    if len(no_doi) > 1:
        kept = title_dedup(no_doi, threshold=0.85)
        deduped.extend(kept)
        print(f"  [DEDUP] Title pass: {len(no_doi)} → {len(kept)} (unique titles)", flush=True)

    return deduped


def title_dedup(papers: list[dict], threshold: float = 0.85) -> list[dict]:
    """标题 n-gram Jaccard 去重 (O(n²), n≤5000 可接受)。"""
    kept: list[dict] = []

    for p in papers:
        title = normalize_title(p.get("title", ""))
        if not title:
            continue

        is_dup = False
        for i, existing in enumerate(kept):
            existing_title = normalize_title(existing.get("title", ""))
            sim = jaccard_ngram_similarity(title, existing_title, n=3)
            if sim >= threshold:
                # 保留引用数更高的
                if (p.get("citationCount", 0) or 0) > (existing.get("citationCount", 0) or 0):
                    kept[i] = p
                is_dup = True
                break

        if not is_dup:
            kept.append(p)

    return kept


def normalize_title(title: str) -> str:
    """归一化标题用于比较: 小写, 去标点, 合并空格。"""
    title = title.lower()
    title = re.sub(r'[^\w\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def jaccard_ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """计算两个字符串的 n-gram Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0

    ngrams_a = {a[i:i+n] for i in range(len(a) - n + 1)}
    ngrams_b = {b[i:i+n] for i in range(len(b) - n + 1)}

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union) if union else 0.0


# ── 分级 ──

TIER_MAP = {
    1: "fulltext_oa",      # 有 openAccessUrl, 可以尝试下全文
    2: "abstract_only",    # 只有摘要
}


def classify(paper: dict) -> int:
    """分类: 1=有 OA PDF, 2=摘要。"""
    if paper.get("openAccessUrl"):
        return 1
    # 有 arXiv ID 也算 Tier 1 (可以用 arXiv 预印本)
    if paper.get("arxivId"):
        return 1
    return 2


# ── 主流程 ──

def filter_and_dedup(
    input_path: str,
    output_path: str,
    report_path: Optional[str] = None,
) -> tuple[int, int, dict]:
    """过滤 + 去重主函数。

    Returns:
        (raw_count, filtered_count, report_dict)
    """
    # 1. 加载
    print(f"[FILTER] Loading papers from {input_path}...", flush=True)
    raw_papers = load_papers(input_path)
    print(f"[FILTER] Loaded {len(raw_papers)} raw papers.", flush=True)

    # 2. 质量过滤
    before_filter = len(raw_papers)
    papers = [p for p in raw_papers if is_high_quality(p)]
    filtered_out = before_filter - len(papers)
    print(f"[FILTER] Quality filter: {before_filter} → {len(papers)} "
          f"(removed {filtered_out} low-quality)", flush=True)

    # 3. 去重
    before_dedup = len(papers)
    papers = deduplicate(papers)
    dups_removed = before_dedup - len(papers)
    print(f"[FILTER] Dedup: {before_dedup} → {len(papers)} "
          f"(removed {dups_removed} duplicates)", flush=True)

    # 4. 分级
    tier_counts = Counter()
    venue_counts = Counter()
    year_counts = Counter()

    for p in papers:
        # 标准化 venue
        venue = normalize_venue(p.get("venue", ""))
        p["_venue_normalized"] = venue
        p["_tier"] = classify(p)

        tier_counts[p["_tier"]] += 1
        venue_counts[venue or "(unknown)"] += 1
        year = p.get("year")
        if year:
            year_counts[year] += 1

    # 5. 按引用数排序
    papers.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)

    # 6. 写入输出
    with open(output_path, "w") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n[FILTER] OUTPUT: {len(papers)} papers → {output_path}", flush=True)
    print(f"  Tier 1 (fulltext): {tier_counts[1]} ({100*tier_counts[1]/len(papers):.0f}%)", flush=True)
    print(f"  Tier 2 (abstract): {tier_counts[2]} ({100*tier_counts[2]/len(papers):.0f}%)", flush=True)
    print(f"\n[FILTER] Top venues:", flush=True)
    for v, c in venue_counts.most_common(20):
        print(f"  {v}: {c}", flush=True)
    print(f"\n[FILTER] Year distribution:", flush=True)
    for y in sorted(year_counts):
        print(f"  {y}: {year_counts[y]}", flush=True)

    # 7. 报告
    report = {
        "total_raw": before_filter,
        "total_filtered": len(papers),
        "filtered_out_low_quality": filtered_out,
        "duplicates_removed": dups_removed,
        "tier_counts": dict(tier_counts),
        "top_venues": venue_counts.most_common(30),
        "year_distribution": dict(sorted(year_counts.items())),
    }
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[FILTER] Report: {report_path}", flush=True)

    return before_filter, len(papers), report


def main():
    parser = argparse.ArgumentParser(description="S2 论文质量过滤 & 去重")
    parser.add_argument("--input", default=INPUT_FILE, help="原始 JSONL 路径")
    parser.add_argument("--output", default=OUTPUT_FILE, help="过滤后 JSONL 路径")
    parser.add_argument("--report", default=REPORT_FILE, help="报告 JSON 路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[FILTER] ERROR: Input file not found: {args.input}", flush=True)
        sys.exit(1)

    filter_and_dedup(args.input, args.output, args.report)


if __name__ == "__main__":
    main()
