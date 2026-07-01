#!/usr/bin/env python3
"""
ingest_with_rank.py — 带期刊排名的 PDF 分块脚本

期刊优先级排名（1=最高）：
  1. Nature
  2. Nature Energy
  3. Nature Materials
  4. Nature Photonics
  5. Nature Nanotechnology
  6. Nature Communications
  7. 其他期刊/arXiv（最低优先级）
"""

import os, json, glob, re
from datetime import datetime
from tqdm import tqdm

PDF_ROOT = "/data/data/pkb/01_raw_data/papers_pdf"
OUTPUT_DIR = "/data1/perovskite-rag/data/chunked_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 期刊优先级映射 ──
JOURNAL_RANK = {
    "Nature":      1,
    "NatEnergy":   2,
    "NatMater":    3,
    "NatPhoton":   4,
    "NatNanotech": 5,
    "NatComm":     6,
}

JOURNAL_FULL_NAME = {
    "Nature": "Nature",
    "NatEnergy": "Nature Energy",
    "NatMater": "Nature Materials",
    "NatPhoton": "Nature Photonics",
    "NatNanotech": "Nature Nanotechnology",
    "NatComm": "Nature Communications",
}

def get_journal_info(fname):
    """从文件名提取期刊信息（如 Nature_2021_xxx.pdf -> rank=1）"""
    for abbr in sorted(JOURNAL_RANK.keys(), key=len, reverse=True):
        if fname.startswith(abbr + "_"):
            return JOURNAL_RANK[abbr], JOURNAL_FULL_NAME[abbr], abbr
    return 7, "other", "Other"

# ── PDF 文本提取 ──
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)

# ── 遍历所有 PDF ──
pdf_files = glob.glob(os.path.join(PDF_ROOT, "**/*.pdf"), recursive=True)
print(f"找到 {len(pdf_files)} 个 PDF 文件")

all_chunks = []
rank_stats = {r: 0 for r in range(1, 8)}

for pdf_path in tqdm(pdf_files):
    try:
        text = extract_text_from_pdf(pdf_path)
    except:
        continue
    if not text.strip():
        continue

    file_name = os.path.basename(pdf_path)
    rank, full_name, abbr = get_journal_info(file_name)
    rank_stats[rank] = rank_stats.get(rank, 0) + 1

    chunks = splitter.create_documents(
        [text],
        metadatas=[{
            "source": file_name,
            "path": pdf_path,
            "journal": full_name,
            "journal_abbr": abbr,
            "journal_rank": rank,
        }]
    )
    all_chunks.extend(chunks)

# ── 保存 JSONL ──
output_path = os.path.join(OUTPUT_DIR, "chunks.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        record = {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n完成，共 {len(all_chunks)} 个块")
print(f"保存至 {output_path}")
print("\n期刊分布：")
for rank in sorted(rank_stats.keys()):
    if rank_stats[rank] > 0:
        label = {1: "Nature", 2: "Nat Energy", 3: "Nat Mater",
                 4: "Nat Photon", 5: "Nat Nanotech", 6: "Nat Comm",
                 7: "Other/arXiv"}.get(rank, f"Rank {rank}")
        print(f"  {label}: {rank_stats[rank]} 篇")
