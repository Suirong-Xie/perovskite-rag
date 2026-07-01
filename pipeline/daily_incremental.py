#!/usr/bin/env python3
"""
daily_incremental.py — Nature 每日增量爬取 + 增量向量更新

用法:
  python3 daily_incremental.py

流程:
  1. 用--daily模式搜索当天新文章（只搜不下）
  2. 如果有新文章 → 下载PDF
  3. 新PDF同步到papers_pdf
  4. 只对新PDF做文本提取+分块
  5. 追加chunks到chunks.jsonl
  6. 只追加新向量到Qdrant（不动旧的）
  7. [removed] 不再重启RAG server
"""
import os, sys, json, glob, shutil, re, subprocess, time
from datetime import datetime
from tqdm import tqdm

PIPELINE_DIR = "/data1/perovskite-rag/pipeline"
RAGENV = "/data1/perovskite-rag/.RAGenv/bin/python"
JOURNALS_PDF = "/data/data/pkb/01_raw_data/journals_pdf"
PAPERS_PDF = "/data/data/pkb/01_raw_data/papers_pdf"
CHUNKED_OUT = "/data1/perovskite-rag/data/chunked_data"
TRACKER_FILE = "/data1/perovskite-rag/data/pipeline/download_tracker.json"
METADATA_FILE = "/data1/perovskite-rag/data/pipeline/fetched_papers_metadata.json"
os.makedirs(CHUNKED_OUT, exist_ok=True)

extract_year = re.compile(r"_(20\d{2})_")
SUDO_PREFIX = ["sudo", "-S"]

def sudo_run(cmd, **kwargs):
    """Run command with sudo"""
    full_cmd = ["echo", "P@ssw0rd"] + ["|"] + SUDO_PREFIX + cmd
    del kwargs

def run_pipeline(cmd_list, timeout=300):
    full = ' '.join(cmd_list)
    print(f"  $ {full[:120]}...")
    r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"  ⚠ stderr: {r.stderr[-200:]}")
    return r.stdout + r.stderr

def find_year_from_name(fname):
    m = extract_year.search(fname)
    return m.group(1) if m else str(datetime.now().year)

# ============================================================
# Main
# ============================================================
start = datetime.now()
print(f"\n{'='*60}")
print(f"📅 Nature 每日增量流程  |  {start.strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}")

# Step 1: 搜索当天新文章
print(f"\n🔍 Step 1: 搜索当天新文章...")
os.chdir("/data1/perovskite-rag")
result = run_pipeline([RAGENV, "pipeline/fetch_top_papers.py", "--daily"], timeout=600)
print(result[-500:] if len(result) > 500 else result)

# 解析结果
new_count = 0
for line in result.split("\n"):
    if "找到" in line and "篇新论文" in line:
        import re as re2
        m = re2.search(r"(\d+)", line)
        if m:
            new_count = int(m.group(1))
    elif "✅ 每日扫描完成!" in line:
        pass

if new_count == 0:
    print("\n📭 今天没有新文章，跳过下载和向量更新。")
    print(f"\n⏱️  总耗时: {datetime.now() - start}")
    sys.exit(0)

# 额外检查：搜索到的论文是否已经在 tracker 里标记为 success
# 如果是，说明已入库，跳过下载和后续流程
tracker_data = {}
if os.path.exists(TRACKER_FILE):
    with open(TRACKER_FILE) as f:
        tracker_data = json.load(f)

with open(METADATA_FILE) as f:
    meta = json.load(f)

already_done = True
for paper in meta.get("papers", []):
    doi = paper.get("doi", "")
    # tracker 中有记录（不论 success/skipped）就算已处理过
    if doi not in tracker_data:
        already_done = False
        break

if already_done and new_count > 0:
    print(f"\n✅ 这 {new_count} 篇论文均已成功下载并入库，跳过。")
    print(f"\n⏱️  总耗时: {datetime.now() - start}")
    sys.exit(0)

print(f"\n📥 Step 2: 下载 {new_count} 篇新文章...")

# Step 2: 搜索也会下载（daily模式只搜不下），需要再跑全量（实际只下载新文章）
# 但fetch_top_papers.py的full模式会重新搜索+下载
# 用现有metadata和tracker来增量下载
run_pipeline([RAGENV, "pipeline/fetch_top_papers.py"], timeout=1200)

print(f"\n📋 Step 3: 识别新增PDF并同步到papers_pdf...")

# 收集journals_pdf中所有文件名
journals_pdfs = glob.glob(os.path.join(JOURNALS_PDF, "**/*.pdf"), recursive=True)

# papers_pdf已有文件名集合
existing_in_papers = set()
for root, dirs, files in os.walk(PAPERS_PDF):
    for f in files:
        if f.endswith(".pdf"):
            existing_in_papers.add(f)

new_files = []
for src_path in journals_pdfs:
    fname = os.path.basename(src_path)
    if fname not in existing_in_papers:
        new_files.append(src_path)

print(f"  新增PDF: {len(new_files)} 篇")

if not new_files:
    print("  没有新PDF需要处理，结束。")
    print(f"\n⏱️  总耗时: {datetime.now() - start}")
    sys.exit(0)

# Sync to papers_pdf
copied = 0
for src_path in tqdm(new_files):
    fname = os.path.basename(src_path)
    year = find_year_from_name(fname)
    month = datetime.now().strftime("%m")
    dest_dir = os.path.join(PAPERS_PDF, year, month)
    dest_path = os.path.join(dest_dir, fname)
    if os.path.exists(dest_path):
        continue
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    copied += 1
print(f"  synced: {copied}")

# Step 4: 提取新PDF文本 + 分块
print(f"\n📝 Step 4: 提取文本 + 分块...")
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

all_chunks = []
failed = 0
for pdf_path in tqdm(new_files):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
    except Exception:
        failed += 1
        continue
    if not text.strip():
        continue
    file_name = os.path.basename(pdf_path)
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": file_name, "path": pdf_path}]
    )
    all_chunks.extend(chunks)

print(f"  新chunks: {len(all_chunks)} (失败: {failed})")

# Step 5: 追加到chunks.jsonl
print(f"\n💾 Step 5: 追加到chunks.jsonl...")
output_path = os.path.join(CHUNKED_OUT, "chunks.jsonl")
existing_count = 0
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        existing_count = sum(1 for _ in f)

with open(output_path, "a", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(json.dumps({
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }, ensure_ascii=False) + "\n")

print(f"  之前: {existing_count} → 现在: {existing_count + len(all_chunks)}")

# Step 6: 追加向量到Qdrant
print(f"\n🧠 Step 6: 增量追加向量到Qdrant...")
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

embed_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11435"
)

texts = [c.page_content for c in all_chunks]
metadatas = [c.metadata for c in all_chunks]

if len(texts) == 0:
    print("  没有新chunks需要嵌入。")
else:
    client = QdrantClient(path="/data1/perovskite-rag/data/qdrant_data")
    cn = "perovskite_papers"
    vectorstore = QdrantVectorStore(client=client, collection_name=cn, embedding=embed_model)

    batch_size = 500
    for i in range(0, len(texts), batch_size):
        bt = texts[i:i+batch_size]
        bm = metadatas[i:i+batch_size]
        vectorstore.add_texts(texts=bt, metadatas=bm)
        pct = min(i+batch_size, len(texts))
        print(f"  {pct}/{len(texts)} ({100*pct//len(texts)}%)")

    print(f"  ✅ 新增 {len(all_chunks)} 个向量")

print(f"\n{'='*60}")
print(f"✅ 每日增量完成!")
print(f"   新论文: {new_count} 篇")
print(f"   新PDF: {len(new_files)} 篇")
print(f"   新chunks: {len(all_chunks)}")
print(f"   总耗时: {datetime.now() - start}")
print(f"{'='*60}")
