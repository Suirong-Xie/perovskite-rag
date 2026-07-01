#!/usr/bin/env python3
import os, sys, json, glob, time, fitz, requests
from concurrent.futures import ProcessPoolExecutor, as_completed

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = "deepseek-chat"
MAX_CHARS = 12000
PDF_ROOT = "/data/data/pkb/01_raw_data/papers_pdf"
OUT_DIR = "/data1/perovskite-rag/data/chunked_data"
OUT_NATURE = os.path.join(OUT_DIR, "chunks_v2_nature.jsonl")
OUT_ARXIV = os.path.join(OUT_DIR, "chunks_v2_arxiv.jsonl")

PROMPT = """You are a paper processing assistant for a perovskite solar cell (PSC) RAG system.

Step 1 - Relevance: Is this paper about perovskite solar cells?
  RELEVANT: perovskite composition, SAMs, HTL/ETL, device stability, passivation, interface engineering
  IRRELEVANT: oxide perovskites (non-PSC), ferroelectric, spintronics, catalysis

Step 2 - If RELEVANT: Split into semantic chunks for RAG.
  Each: 500-2000 chars, complete unit, follow paper structure.
  EXCLUDE: references, acknowledgments.
  MERGE figure captions with describing paragraph.

OUTPUT (JSON only):
  IRRELEVANT: {"skip": true}
  RELEVANT: {"chunks": ["text1...", "text2..."], "n": int}
"""

def extract_text(p):
    try:
        d = fitz.open(p)
    except:
        return ""
    ps = []
    for page in d:
        t = page.get_text("text")
        if len(t.strip()) < 50: continue
        ps.append("\n".join([l.strip() for l in t.split("\n") if len(l.strip()) >= 5]))
    d.close()
    return "\n\n".join(ps)[:MAX_CHARS]

def journal_info(f):
    if "arXiv" in f: return "arXiv", 7
    m = {"Nature_":("Nature",1), "NatEnergy_":("Nature Energy",2), "NatMater_":("Nature Materials",3),
         "NatPhoton_":("Nature Photonics",4), "NatNanotech_":("Nature Nanotechnology",5), "NatComm_":("Nature Communications",6)}
    for k,v in m.items():
        if f.startswith(k): return v
    return "Other", 7

def process(text):
    for _ in range(3):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": text}
                ], "temperature": 0.05, "max_tokens": 4096}, timeout=120)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n",1)[1]
                if "```" in content: content = content.rsplit("```",1)[0]
                content = content.strip()
            js = json.loads(content)
            if isinstance(js, dict):
                if js.get("skip"): return []
                ch = js.get("chunks", [])
                return [c for c in ch if isinstance(c,str) and len(c.strip())>50]
        except:
            time.sleep(5)
    return []

def process_one(pdf_path):
    fname = os.path.basename(pdf_path)
    journal, rank = journal_info(fname)
    text = extract_text(pdf_path)
    if not text.strip(): return [],None,False
    chunks = process(text)
    if not chunks: return [],None,True
    recs = [{"content":c,"metadata":{"source":fname,"journal":journal,"journal_rank":rank}} for c in chunks]
    cat = "arXiv" if "arXiv" in fname else "Nature"
    return recs, cat, False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not API_KEY: print("ERROR: LLM_API_KEY"); sys.exit(1)

    pdfs = sorted(glob.glob(os.path.join(PDF_ROOT, args.year, "**", "*.pdf"), recursive=True))
    print(f"[{args.year}] {len(pdfs)} PDFs")

    for p in [OUT_NATURE, OUT_ARXIV]:
        if os.path.exists(p): os.remove(p)

    nbuf, abuf, skip, ok = [], [], 0, 0
    start = time.time()

    def flush():
        if nbuf:
            with open(OUT_NATURE,"a") as f:
                for r in nbuf: f.write(json.dumps(r,ensure_ascii=False)+"\n")
            nbuf.clear()
        if abuf:
            with open(OUT_ARXIV,"a") as f:
                for r in abuf: f.write(json.dumps(r,ensure_ascii=False)+"\n")
            abuf.clear()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(process_one, p): p for p in pdfs}
        for i, f in enumerate(as_completed(fut)):
            rec, cat, skipped = f.result()
            if skipped: skip += 1
            elif cat == "Nature": nbuf.extend(rec); ok += 1
            else: abuf.extend(rec); ok += 1
            if (i+1) % 10 == 0:
                flush()
                el = time.time()-start
                rate = (i+1)/el*60 if el else 0
                print(f"[{args.year}] {i+1}/{len(pdfs)} {rate:.0f}/min ETA={(len(pdfs)-i-1)/rate if rate else 0:.0f}min kept={ok} skip={skip}")

    flush()
    el = time.time()-start
    print(f"[{args.year}] {len(pdfs)} PDFs in {el/60:.1f}min | kept={ok} skip={skip}")
    for p in [OUT_NATURE, OUT_ARXIV]:
        if os.path.exists(p): print(f"  {p}: {os.path.getsize(p)//1024}KB")

if __name__ == "__main__":
    main()
