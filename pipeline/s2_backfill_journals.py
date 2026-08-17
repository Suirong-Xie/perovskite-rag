"""
S2 向量库元数据补全 — 通过 Crossref API 用 DOI 反查期刊名

补全 texts_review.jsonl 和 texts_experimental.jsonl 中 journal=Unknown/"": 的条目
99.8% 有 DOI (来自 _s2_doi 字段)，380 条无 DOI 无法补全

用法: python pipeline/s2_backfill_journals.py
"""
import json
import time
import re
import os
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent.parent / "data/s2_vector_db"
REVIEW_PATH = BASE / "texts_review.jsonl"
EXPT_PATH = BASE / "texts_experimental.jsonl"
MAPPING_PATH = BASE / "doi_journal_map.json"

CROSSREF_URL = "https://api.crossref.org/works/"
REQUEST_DELAY = 0.15  # ~6 req/s, polite to Crossref public API
BATCH_SAVE_INTERVAL = 500  # 每500个DOI保存一次进度


def load_existing_map():
    if MAPPING_PATH.exists():
        with open(MAPPING_PATH) as f:
            return json.load(f)
    return {}


def query_crossref(doi: str) -> str | None:
    """返回 container-title 或 None"""
    url = CROSSREF_URL + doi
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PerovskiteGPT/5.0 (mailto:xiesuirong@westlake.edu.cn)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        msg = data.get("message", {})
        titles = msg.get("container-title", [])
        if titles and titles[0]:
            return titles[0]
        # fallback: publisher
        publisher = msg.get("publisher", "")
        if publisher:
            return publisher
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # DOI not in Crossref
        print(f"  HTTP {e.code} for {doi}")
    except Exception as e:
        print(f"  Error for {doi}: {e}")
    return None


def collect_missing_dois():
    """收集所有缺期刊名且有 DOI 的记录，去重 DOI"""
    dois = set()
    missing_stats = Counter()

    for path, label in [(REVIEW_PATH, "review"), (EXPT_PATH, "experimental")]:
        with open(path) as f:
            for line in f:
                r = json.loads(line.strip())
                j = r.get("journal", "")
                if j in ("Unknown", "", None):
                    doi = r.get("_s2_doi", "")
                    if doi:
                        dois.add(doi)
                        missing_stats[label] += 1
                    else:
                        missing_stats[f"{label}_no_doi"] += 1

    print(f"收集到 {len(dois):,} 唯一 DOI 需要查询")
    print(f"  review 缺期刊: {missing_stats['review']:,} (有DOI)")
    print(f"  experimental 缺期刊: {missing_stats['experimental']:,} (有DOI)")
    print(f"  review 无DOI: {missing_stats.get('review_no_doi', 0):,}")
    print(f"  experimental 无DOI: {missing_stats.get('experimental_no_doi', 0):,}")
    return sorted(dois)


def main():
    existing = load_existing_map()
    print(f"已有映射: {len(existing):,} DOI")

    all_dois = collect_missing_dois()
    remaining = [d for d in all_dois if d not in existing]
    print(f"待查询: {len(remaining):,} DOI")

    if not remaining:
        print("无需查询，直接进入更新阶段")
    else:
        print(f"\n开始查询 Crossref API ({len(remaining):,} DOI, "
              f"预计 {len(remaining)*REQUEST_DELAY/60:.0f} 分钟)...\n")

        new_count = 0
        fail_count = 0
        t0 = time.time()

        for i, doi in enumerate(remaining):
            journal = query_crossref(doi)
            if journal:
                existing[doi] = journal
                new_count += 1
            else:
                fail_count += 1

            time.sleep(REQUEST_DELAY)

            # 进度显示 + 定期保存
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(remaining) - i - 1) / rate
                print(f"  [{i+1}/{len(remaining)}] {new_count} found, {fail_count} failed "
                      f"| {rate:.1f} req/s | ETA {eta/60:.1f}min", flush=True)

            if (i + 1) % BATCH_SAVE_INTERVAL == 0:
                with open(MAPPING_PATH, "w") as f:
                    json.dump(existing, f, ensure_ascii=False)
                print(f"  💾 已保存 {len(existing):,} 条映射", flush=True)

        # 最终保存
        with open(MAPPING_PATH, "w") as f:
            json.dump(existing, f, ensure_ascii=False)
        print(f"\n查询完成: {new_count} 新映射, {fail_count} 失败, "
              f"总计 {len(existing):,} DOI", flush=True)

    # 阶段2: 更新 JSONL 文件
    print("\n--- 更新 JSONL 文件 ---")
    for path, label in [(REVIEW_PATH, "review"), (EXPT_PATH, "experimental")]:
        backup = Path(str(path) + ".bak")
        if not backup.exists():
            os.rename(path, backup)
            print(f"  备份: {backup}")
        else:
            print(f"  备份已存在，覆盖读取: {backup}")

        updated = 0
        unchanged = 0
        total = 0

        with open(backup) as fin, open(path, "w") as fout:
            for line in fin:
                r = json.loads(line.strip())
                total += 1
                j = r.get("journal", "")
                if j in ("Unknown", "", None):
                    doi = r.get("_s2_doi", "")
                    new_j = existing.get(doi)
                    if new_j:
                        r["journal"] = new_j
                        updated += 1
                    else:
                        unchanged += 1
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"  {label}: {total:,} 行, 更新 {updated:,}, "
              f"仍缺 {unchanged:,} ({100*unchanged/total:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
