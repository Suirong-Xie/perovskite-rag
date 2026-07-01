#!/usr/bin/env python3
"""
llm_semantic_chunker.py — v3: DeepSeek API 语义分块
1. 检查 relevance → 不相关直接跳过
2. 分段语义 chunk → 写入 JSONL
"""

import os, sys, json, glob, argparse, time, re
import fitz
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
MAX_CHARS = 6000
MAX_WORKERS = 6
PDF_ROOT = "/data/data/pkb/01_raw_data/papers_pdf"
OUTPUT_DIR = "/data1/perovskite-rag/data/chunked_data"
OUT_NATURE = os.path.join(OUTPUT_DIR, "chunks_v2_nature.jsonl")
OUT_ARXIV  = os.path.join(OUTPUT_DIR, "chunks_v2_arxiv.jsonl")

RELEVANCE_PROMPT = """Assess relevance to perovskite solar cells (PSCs). 
RELEVANT topics: perovskite composition, SAM/HTL/ETL, device stability, defect passivation, interface modification, charge transport, band alignment, ion migration.
IRRELEVANT topics: oxide perovskites (non-PSC), ferroelectric, spintronics, catalysis, LED, memristor, non-halide.
Reply ONLY "RELEVANT" or "IRRELEVANT"."""

CHUNK_PROMPT = """Split this paper section into self-contained semantic chunks for RAG retrieval on perovskite solar cells.
Each chunk: 500-2000 characters, complete semantic unit, follows paper structure. Exclude references/acknowledgments.
Return ONLY a JSON array of strings."""

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except:
        return ""
    pages = []
    for page in doc:
        t = page.get_text("text")
        if len(t.strip()) < 50: continue
        lines = [l.strip() for l in t.split("\n") if len(l.strip()) >= 5]
        pages.append("\n".join(lines))
    doc.close()
    return "\n\n".join(pages)

def split_text(text, max_c=MAX_CHARS):
    paras = text.split("\n\n")
    segs, cur, cur_len = [], [], 0
    for p in paras:
        pl = len(p)
        if cur_len + pl > max_c and cur:
            segs.append("\n\n".join(cur))
            cur, cur_len = [p], pl
        else:
            cur.append(p)
            cur_len += pl
    if cur: segs.append("\n\n".join(cur))
    return segs

def llm_call(messages, max_tok=4096):
    for _ in range(3):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "temperature": 0.05, "max_tokens": max_tok},
                timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except:
            time.sleep(3)
    return ""

def check_relevance(text):
    ans = llm_call([
        {"role": "system", "content": RELEVANCE_PROMPT},
        {"role": "user", "content": text[:3000]}
    ], max_tok=20)
    return "RELEVANT" in ans.upper()

def call_chunk(text):
    c = llm_call([
        {"role": "system", "content": CHUNK_PROMPT},
        {"role": "user", "content": text}
    ])
    if not c: return []
    c = c.strip()
    if c.startswith("```"):
        lines = c.split("\n")
        c = "\n".join(lines[1:] if lines[0].startswith("```") else lines[:-1])
        if c.endswith("```"): c = c[:-3].strip()
    try:
        js = json.loads(c)
        if isinstance(js, list):
            return [x for x in js if isinstance(x, str) and len(x.strip()) > 50]
    except:
        pass
    return []

def get_journal(fname):
    if "arXiv" in fname: return "arXiv", 7
    m = {"Nature_":("Nature",1), "NatEnergy_":("Nature Energy",2), "NatMater_":("Nature Materials",3),
         "NatPhoton_":("Nature Photonics",4), "NatNanotech_":("Nature Nanotechnology",5), "NatComm_":("Nature Communications",6)}
    for k, v in m.items():
        if fname.startswith(k): return v
    return "Other", 7

def process_one(pdf_path):
    fname = os.path.basename(pdf_path)
    journal, rank = get_journal(fname)
    txt = extract_text(pdf_path)
    if not txt.strip():
        return [], None, False

    # Step 1: relevance check
    if not check_relevance(txt):
        return [], None, True  # skipped

    # Step 2: chunk
    segs = split_text(txt)
    all_c = []
    for seg in segs:
        c = call_chunk(seg)
        all_c.extend(c)
        time.sleep(0.3)

    records = [{"content": c, "metadata": {"source": fname, "journal": journal, "journal_rank": rank}}
               for c in all_c]
    cat = "arXiv" if "arXiv" in fname else "Nature"
    return records, cat, False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="全部 PDF")
    parser.add_argument("--year", help="单年")
    parser.add_argument("--dir", help="目录")
    parser.add_argument("--test", action="store_true", help="测 3 篇")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    if not API_KEY:
        print("[ERROR] Set LLM_API_KEY")
        sys.exit(1)

    pdfs = []
    if args.test:
        pdfs = sorted(glob.glob(os.path.join(PDF_ROOT, "2021", "05", "*.pdf")))[:3]
    elif args.dir:
        pdfs = sorted(glob.glob(os.path.join(args.dir, "*.pdf")))
    elif args.year:
        pdfs = sorted(glob.glob(os.path.join(PDF_ROOT, args.year, "**", "*.pdf"), recursive=True))
    elif args.all:
        pdfs = sorted(glob.glob(os.path.join(PDF_ROOT, "**", "*.pdf"), recursive=True))

    for p in [OUT_NATURE, OUT_ARXIV]:
        if os.path.exists(p): os.remove(p)

    nbuf, abuf, skip_cnt, ok_cnt = [], [], 0, 0
    start = time.time()

    def flush():
        if nbuf:
            with open(OUT_NATURE, "a", encoding="utf-8") as f:
                for r in nbuf: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            nbuf.clear()
        if abuf:
            with open(OUT_ARXIV, "a", encoding="utf-8") as f:
                for r in abuf: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            abuf.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(process_one, p): p for p in pdfs}
        for i, f in enumerate(as_completed(fut)):
            rec, cat, skipped = f.result()
            if skipped:
                skip_cnt += 1
            elif cat == "Nature":
                nbuf.extend(rec)
                ok_cnt += 1
            else:
                abuf.extend(rec)
                ok_cnt += 1
            if (i + 1) % 20 == 0:
                flush()
                elapsed = time.time() - start
                rate = (i+1) / elapsed * 60
                remaining = (len(pdfs) - i - 1) / rate if rate else 0
                print(f"  [{i+1}/{len(pdfs)}] {rate:.0f}/min ETA {remaining:.0f}min | kept={ok_cnt} skip={skip_cnt}")

    flush()
    elapsed = time.time() - start
    n_kb = os.path.getsize(OUT_NATURE)//1024 if os.path.exists(OUT_NATURE) else 0
    a_kb = os.path.getsize(OUT_ARXIV)//1024 if os.path.exists(OUT_ARXIV) else 0
    print(f"\nDone! {len(pdfs)} PDFs in {elapsed/60:.1f}min")
    print(f"  Kept: {ok_cnt} | Skipped: {skip_cnt}")
    print(f"  Nature: {n_kb} KB | arXiv: {a_kb} KB")

if __name__ == "__main__":
    main()
