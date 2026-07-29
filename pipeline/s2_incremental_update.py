#!/usr/bin/env python3
"""
s2_incremental_update.py — S2 向量库增量更新 + review/experimental 拆分

步骤:
  1. 构建 S2 DB 现有 DOI 索引
  2. 扫描新 PDF，匹配 DOI → REPLACE / NEW / SKIP
  3. 分类现有条目 (review / experimental)
  4. 对 REPLACE + NEW PDF 做 PyMuPDF4LLM 提取 + DeepSeek LLM 切分
  5. 替换旧条目或追加新条目
  6. 拆分为 texts_review.jsonl + texts_experimental.jsonl
  7. 分别 Ollama 向量化
  8. 输出到 s2_vector_db/

用法:
  python3 s2_incremental_update.py               # 全量
  python3 s2_incremental_update.py --dry-run     # 预览不执行
  python3 s2_incremental_update.py --chunk-only  # 只切分，不嵌入
  python3 s2_incremental_update.py --embed-only  # 只嵌入
  python3 s2_incremental_update.py --resume      # 从断点续传
"""

import json, os, re, sys, time, argparse, hashlib
from pathlib import Path
from typing import Optional

# ── 环境 ──
_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ENV.exists():
    for _line in open(_ENV):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            if k not in os.environ:
                os.environ[k] = v.strip().strip('"').strip("'")

BASE = Path("/data1/perovskite-rag")
S2_DB_DIR = BASE / "data/s2_vector_db"
TEXTS_FILE = S2_DB_DIR / "texts.jsonl"
TEXTS_REVIEW = S2_DB_DIR / "texts_review.jsonl"
TEXTS_EXPT = S2_DB_DIR / "texts_experimental.jsonl"
VECTORS_REVIEW = S2_DB_DIR / "vectors_review.npy"
VECTORS_EXPT = S2_DB_DIR / "vectors_experimental.npy"
PROGRESS_FILE = S2_DB_DIR / "incremental_progress.json"
NEW_PDF_DIR = Path("/tmp/literature_new/Literature_download/journals_pdf")
EXISTING_PDF_DIR = BASE / "journals_pdf"

# LLM
LLM_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/chat/completions"
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_EMBED_URL", "http://127.0.0.1:11435/api/embed")
OLLAMA_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# ── 综述关键词 ──
REVIEW_KW = [
    "review", "progress", "advances", "survey", "overview",
    "comprehensive", "perspective", "roadmap", "retrospect",
    "tutorial", "state of the art", "state-of-the-art",
    "critical review", "mini review", "recent progress",
    "recent advances", "current status", "综述", "进展", "回顾", "概述",
]


def log(msg):
    print(f"[S2-Up] {msg}", flush=True)


def doi_from_filename(fname: str) -> Optional[str]:
    """从文件名提取 DOI: 10.1007_s40820-025-02022-6.pdf → 10.1007/s40820-025-02022-6"""
    fname = os.path.basename(fname).replace(".pdf", "")
    # 匹配 DOI pattern: 10.XXXX/YYYY
    m = re.match(r'(10\.\d{4,})_(.+)', fname)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # arXiv pattern
    m = re.match(r'(arXiv.*)', fname, re.IGNORECASE)
    if m:
        return fname
    return fname  # fallback: use filename as key


def classify_paper(source: str, content: str = "") -> str:
    """启发式分类：review 或 experimental。"""
    text = (source + " " + (content or "")[:500]).lower()
    for kw in REVIEW_KW:
        if kw.lower() in text:
            return "review"
    return "experimental"


# ═══════════════════════════════════════════════════════════
# Step 1: Build DOI index
# ═══════════════════════════════════════════════════════════

def build_doi_index():
    """扫描 S2 DB，构建 DOI → 行号列表 的索引。"""
    log("Building DOI index from S2 DB...")
    doi_map: dict[str, list[int]] = {}
    if not TEXTS_FILE.exists():
        log("WARN: texts.jsonl not found, starting fresh")
        return doi_map, []

    lines = []
    with open(TEXTS_FILE) as f:
        for i, line in enumerate(f):
            lines.append(line)
            try:
                d = json.loads(line)
                doi = d.get("_s2_doi", "")
                if doi:
                    doi_map.setdefault(doi, []).append(i)
            except Exception:
                pass
    log(f"  Indexed {len(lines)} entries, {len(doi_map)} unique DOIs")
    return doi_map, lines


# ═══════════════════════════════════════════════════════════
# Step 2: Scan new PDFs
# ═══════════════════════════════════════════════════════════

def scan_new_pdfs(doi_index: dict):
    """扫描新 PDF，分类为 REPLACE / NEW。"""
    log(f"Scanning new PDFs in {NEW_PDF_DIR}...")
    replace_list = []
    new_list = []

    if not NEW_PDF_DIR.exists():
        log(f"WARN: {NEW_PDF_DIR} does not exist!")
        return replace_list, new_list

    for root, dirs, files in os.walk(NEW_PDF_DIR):
        for fname in files:
            if not fname.endswith(".pdf"):
                continue
            fpath = os.path.join(root, fname)
            doi = doi_from_filename(fname)
            fsize = os.path.getsize(fpath)

            if doi and doi in doi_index:
                replace_list.append((fpath, doi, fsize))
            else:
                new_list.append((fpath, doi or fname, fsize))

    replace_list.sort(key=lambda x: x[2], reverse=True)
    new_list.sort(key=lambda x: x[2], reverse=True)
    log(f"  REPLACE: {len(replace_list)}, NEW: {len(new_list)}")
    return replace_list, new_list


# ═══════════════════════════════════════════════════════════
# Step 3: Extract text from PDF
# ═══════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: str) -> str:
    """PyMuPDF4LLM 提取文本，无 OCR。"""
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(pdf_path, use_ocr=False)
        # 清理
        from s2_chunk_and_embed import clean_markdown_text
        return clean_markdown_text(md)[:12000]
    except Exception as e:
        log(f"  PyMuPDF4LLM failed for {pdf_path}: {e}")
        # fallback: fitz
        try:
            import fitz
            doc = fitz.open(pdf_path)
            pages = [page.get_text("text") for page in doc if page.get_text("text").strip()]
            doc.close()
            return "\n\n".join(pages)[:12000]
        except Exception:
            return ""


# ═══════════════════════════════════════════════════════════
# Step 4: LLM chunking
# ═══════════════════════════════════════════════════════════

CHUNK_PROMPT = """Split this perovskite solar cell paper into semantic chunks for RAG.
Each chunk: 500-2000 chars, self-contained, follow the paper's structure.
Skip: references, acknowledgments, supporting info.

Return ONLY JSON: {"chunks": ["text1...", "text2..."], "n": int}"""


def chunk_text(text: str, max_retries: int = 3) -> list[str]:
    """DeepSeek LLM 语义切分。"""
    import requests
    for attempt in range(max_retries):
        try:
            r = requests.post(
                LLM_URL,
                headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": CHUNK_PROMPT},
                        {"role": "user", "content": text[:12000]},
                    ],
                    "max_tokens": 4096,
                    "temperature": 0,
                },
                timeout=120,
            )
            if r.status_code != 200:
                log(f"  LLM error {r.status_code}: {r.text[:200]}")
                time.sleep(2 ** attempt)
                continue
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            # Extract JSON
            json_match = re.search(r'\{.*"chunks".*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("chunks", [text[:1500]])
            return [content[:1500]]
        except Exception as e:
            log(f"  LLM retry {attempt}: {e}")
            time.sleep(2 ** attempt)
    return [text[:1500]]


# ═══════════════════════════════════════════════════════════
# Step 5: Process new PDFs (chunk + classify)
# ═══════════════════════════════════════════════════════════

def process_pdfs(pdf_list: list, paper_type_hint: str = ""):
    """对 PDF 列表做提取+切分，返回 chunk 记录列表。"""
    results = []
    total = len(pdf_list)
    for i, (fpath, doi, fsize) in enumerate(pdf_list):
        log(f"  [{i+1}/{total}] {os.path.basename(fpath)[:70]} ({fsize/1024:.0f}KB)")
        text = extract_pdf_text(fpath)
        if not text or len(text) < 200:
            log(f"    SKIP: insufficient text ({len(text)} chars)")
            continue

        chunks = chunk_text(text)
        ptype = paper_type_hint or classify_paper(os.path.basename(fpath), text)
        journal = guess_journal(os.path.basename(fpath))

        for chunk_text_content in chunks:
            if len(chunk_text_content) < 200:
                continue
            results.append({
                "source": os.path.basename(fpath),
                "content": chunk_text_content,
                "journal": journal,
                "journal_rank": journal_rank(journal),
                "_s2_doi": doi,
                "_s2_paper_id": doi_to_paper_id(doi),
                "_s2_year": extract_year(os.path.basename(fpath)),
                "_s2_tier": 1,
                "_s2_citation_count": 0,
                "_extract_method": "pymupdf4llm",
                "_paper_type": ptype,
            })
        log(f"    → {len(chunks)} chunks, type={ptype}")
    return results


def guess_journal(fname: str) -> str:
    """从文件名推断期刊名。"""
    parts = fname.replace(".pdf", "").split("_")
    # 跳过 DOI 部分
    journal_parts = []
    for p in parts:
        if re.match(r'10\.\d+', p):
            break
        journal_parts.append(p)
    return "_".join(journal_parts).replace("_", " ") if journal_parts else "Unknown"


def journal_rank(journal: str) -> int:
    """期刊排名 0-10。"""
    ranks = {
        "Nature": 10, "Science": 10,
        "Nature Energy": 9, "Nature Materials": 9,
        "Nature Photonics": 9, "Nature Nanotechnology": 9,
        "Nature Communications": 8, "Science Advances": 8,
        "Joule": 8, "Energy & Environmental Science": 8,
        "Advanced Materials": 7, "Advanced Energy Materials": 7,
        "ACS Energy Letters": 7, "ACS Nano": 7,
        "Angewandte Chemie": 7, "JACS": 7,
        "Nano Energy": 6, "Chemistry of Materials": 6,
    }
    for k, v in ranks.items():
        if k.lower() in journal.lower():
            return v
    return 5


def doi_to_paper_id(doi: str) -> str:
    return hashlib.sha1(doi.encode()).hexdigest()[:40]


def extract_year(fname: str) -> str:
    m = re.search(r'(20\d{2})', fname)
    return m.group(1) if m else ""


# ═══════════════════════════════════════════════════════════
# Step 6: Merge + Replace
# ═══════════════════════════════════════════════════════════

def merge_and_replace(existing_lines: list, doi_index: dict,
                      replace_doi_set: set, new_chunks: list):
    """替换旧条目 + 追加新条目 + 分类全部。"""
    log("Merging: replacing old entries + adding new chunks...")

    # 找出要删除的行号
    lines_to_delete: set[int] = set()
    for doi in replace_doi_set:
        if doi in doi_index:
            lines_to_delete.update(doi_index[doi])
    log(f"  Deleting {len(lines_to_delete)} old lines (replaced by new chunks)")

    # 保留未删除的行 + 分类
    kept = []
    for i, line in enumerate(existing_lines):
        if i in lines_to_delete:
            continue
        try:
            d = json.loads(line)
            if "_paper_type" not in d:
                d["_paper_type"] = classify_paper(d.get("source", ""), d.get("content", ""))
            kept.append(d)
        except Exception:
            kept.append(json.loads(line) if line.strip() else {"_paper_type": "experimental"})

    # 追加新 chunk
    kept.extend(new_chunks)

    # 重新分配 idx
    for i, d in enumerate(kept):
        d["idx"] = i
        d["id"] = i

    log(f"  Total after merge: {len(kept)} entries")
    return kept


# ═══════════════════════════════════════════════════════════
# Step 7: Split & Save
# ═══════════════════════════════════════════════════════════

def split_and_save(all_entries: list):
    """拆分为 review/experimental，写入文件。"""
    reviews = [d for d in all_entries if d.get("_paper_type") == "review"]
    expts = [d for d in all_entries if d.get("_paper_type") != "review"]

    log(f"Splitting: {len(reviews)} review chunks, {len(expts)} experimental chunks")

    # Re-idx within each category
    for i, d in enumerate(reviews):
        d["idx"] = i
        d["id"] = i
    for i, d in enumerate(expts):
        d["idx"] = i
        d["id"] = i

    with open(TEXTS_REVIEW, "w") as f:
        for d in reviews:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(TEXTS_EXPT, "w") as f:
        for d in expts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    log(f"  Saved: {TEXTS_REVIEW} ({len(reviews)} lines)")
    log(f"  Saved: {TEXTS_EXPT} ({len(expts)} lines)")
    return reviews, expts


# ═══════════════════════════════════════════════════════════
# Step 8: Embed
# ═══════════════════════════════════════════════════════════

def embed_and_save(entries: list, out_file: Path, label: str):
    """Ollama 批量嵌入 + 写入 .npy。"""
    import numpy as np
    import requests

    texts = [d["content"] for d in entries]
    batch_size = 20
    all_vectors = []

    log(f"Embedding {label}: {len(texts)} texts...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                r = requests.post(OLLAMA_URL, json={
                    "model": OLLAMA_MODEL,
                    "input": batch,
                }, timeout=60)
                if r.status_code == 200:
                    vecs = r.json()["embeddings"]
                    all_vectors.extend(vecs)
                    break
                else:
                    log(f"  Embed error {r.status_code}, retry {attempt}")
                    time.sleep(2)
            except Exception as e:
                log(f"  Embed exception: {e}, retry {attempt}")
                time.sleep(2)
        if (i // batch_size) % 50 == 0:
            log(f"  {label}: {i}/{len(texts)}")

    vecs_array = np.array(all_vectors, dtype=np.float32)
    np.save(str(out_file), vecs_array)
    log(f"  Saved {vecs_array.shape} to {out_file}")


# ═══════════════════════════════════════════════════════════
# Step 9: Copy PDFs
# ═══════════════════════════════════════════════════════════

def copy_new_pdfs(pdf_list: list):
    """复制新 PDF 到 journals_pdf/。"""
    import shutil
    copied = 0
    for fpath, doi, fsize in pdf_list:
        fname = os.path.basename(fpath)
        journal = guess_journal(fname)
        dest_dir = EXISTING_PDF_DIR / journal.replace(" ", "_")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname
        if not dest.exists():
            shutil.copy2(fpath, dest)
            copied += 1
    log(f"Copied {copied} new PDFs to {EXISTING_PDF_DIR}")


# ═══════════════════════════════════════════════════════════
# Progress
# ═══════════════════════════════════════════════════════════

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"phase": 0, "processed_dois": [], "replace_done": False,
            "new_done": False, "chunked_count": 0}


def save_progress(state: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(state, f)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-only", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-copy", action="store_true", help="Skip copying PDFs")
    args = parser.parse_args()

    if args.dry_run:
        log("=== DRY RUN ===")
        doi_index, lines = build_doi_index()
        replace_list, new_list = scan_new_pdfs(doi_index)
        log(f"Would REPLACE: {len(replace_list)}")
        log(f"Would NEW:     {len(new_list)}")
        total_mb = sum(x[2] for x in replace_list + new_list) / 1024 / 1024
        log(f"Total PDF size: {total_mb:.0f} MB")
        return

    progress = load_progress()
    start_phase = progress.get("phase", 0) if args.resume else 0

    # ── Phase 1: Index + Scan ──
    if start_phase <= 1 and not args.embed_only:
        doi_index, existing_lines = build_doi_index()
        replace_list, new_list = scan_new_pdfs(doi_index)

        # Copy PDFs
        if not args.skip_copy:
            copy_new_pdfs(replace_list + new_list)

        # Classify existing
        log("Classifying existing entries...")
        review_count = 0
        for line in existing_lines:
            d = json.loads(line)
            if classify_paper(d.get("source", ""), d.get("content", "")) == "review":
                review_count += 1
        log(f"  Existing: {review_count} review, {len(existing_lines) - review_count} experimental")

        progress["phase"] = 2
        progress["replace_count"] = len(replace_list)
        progress["new_count"] = len(new_list)
        progress["review_count"] = review_count
        save_progress(progress)

    # ── Phase 2: Chunk new PDFs ──
    if start_phase <= 2 and not args.embed_only:
        replace_doi_set = {doi for _, doi, _ in replace_list}

        # Process REPLACE
        new_chunks = []
        if replace_list:
            log(f"=== Chunking {len(replace_list)} REPLACE PDFs ===")
            new_chunks = process_pdfs(replace_list)

        # Process NEW
        if new_list:
            log(f"=== Chunking {len(new_list)} NEW PDFs ===")
            new_chunks += process_pdfs(new_list)

        log(f"Total new chunks: {len(new_chunks)}")
        progress["chunked_count"] = len(new_chunks)
        progress["phase"] = 3
        save_progress(progress)

        # Merge
        all_entries = merge_and_replace(
            existing_lines, doi_index, replace_doi_set, new_chunks
        )
        reviews, expts = split_and_save(all_entries)
        progress["phase"] = 4
        progress["total_review"] = len(reviews)
        progress["total_expt"] = len(expts)
        save_progress(progress)

    if args.chunk_only:
        log("--chunk-only: done after chunking")
        return

    # ── Phase 3: Embed ──
    if start_phase <= 4 or args.embed_only:
        # Load texts if embed-only
        if args.embed_only:
            reviews = [json.loads(l) for l in open(TEXTS_REVIEW) if l.strip()]
            expts = [json.loads(l) for l in open(TEXTS_EXPT) if l.strip()]

        embed_and_save(reviews, VECTORS_REVIEW, "review")
        embed_and_save(expts, VECTORS_EXPT, "experimental")

        # Update original texts.jsonl as combined (backward compat)
        all_entries = reviews + expts
        all_entries.sort(key=lambda d: (d.get("_paper_type", "x"), d.get("idx", 0)))
        for i, d in enumerate(all_entries):
            d["idx"] = i
            d["id"] = i
        with open(TEXTS_FILE, "w") as f:
            for d in all_entries:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        progress["phase"] = 5
        save_progress(progress)

    log("=== DONE ===")
    log(f"Review chunks:     {TEXTS_REVIEW}")
    log(f"Experimental chunks: {TEXTS_EXPT}")
    log(f"Review vectors:    {VECTORS_REVIEW}")
    log(f"Expt vectors:      {VECTORS_EXPT}")


if __name__ == "__main__":
    main()
