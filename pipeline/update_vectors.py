#!/usr/bin/env python3
import os, sys, json, glob, shutil, re
from datetime import datetime
from tqdm import tqdm

JOURNALS_PDF = "/data/data/pkb/01_raw_data/journals_pdf"
PAPERS_PDF = "/data/data/pkb/01_raw_data/papers_pdf"
CHUNKED_OUT = "/data1/perovskite-rag/data/chunked_data"
os.makedirs(CHUNKED_OUT, exist_ok=True)

extract_year = re.compile(r"_(20\d{2})_")

def find_year_from_name(fname):
    m = extract_year.search(fname)
    return m.group(1) if m else str(datetime.now().year)

print("Step 1: Sync journals_pdf -> papers_pdf")
copied = 0
existing = 0
journals_pdfs = glob.glob(os.path.join(JOURNALS_PDF, "**/*.pdf"), recursive=True)
for src_path in tqdm(journals_pdfs):
    fname = os.path.basename(src_path)
    year = find_year_from_name(fname)
    month = datetime.now().strftime("%m")
    dest_dir = os.path.join(PAPERS_PDF, year, month)
    dest_path = os.path.join(dest_dir, fname)
    if os.path.exists(dest_path):
        existing += 1
        continue
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    copied += 1
print(f"  copied={copied}, existing={existing}, total={len(journals_pdfs)}")

print()
print("Step 2: PDF text extraction + chunking")
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""])
all_pdf_files = glob.glob(os.path.join(PAPERS_PDF, "**/*.pdf"), recursive=True)
print(f"Found {len(all_pdf_files)} PDF files")

all_chunks = []
for pdf_path in tqdm(all_pdf_files):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        continue
    if not text.strip():
        continue
    file_name = os.path.basename(pdf_path)
    chunks = splitter.create_documents([text],
        metadatas=[{"source": file_name, "path": pdf_path}])
    all_chunks.extend(chunks)

output_path = os.path.join(CHUNKED_OUT, "chunks.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(json.dumps({"content": chunk.page_content,
            "metadata": chunk.metadata}, ensure_ascii=False) + "\n")
print(f"Done: {len(all_chunks)} chunks -> {output_path}")

print()
print("Step 3: Rebuild Qdrant vector store")
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

embed_model = OllamaEmbeddings(model="mxbai-embed-large",
    base_url="http://127.0.0.1:11435")

with open(output_path, "r", encoding="utf-8") as f:
    chunks_data = [json.loads(line) for line in f]
texts = [c["content"] for c in chunks_data]
metadatas = [c["metadata"] for c in chunks_data]
print(f"Loaded {len(texts)} items")

client = QdrantClient(path="/data1/perovskite-rag/data/qdrant_data")
cn = "perovskite_papers"
if client.collection_exists(cn):
    client.delete_collection(cn)
    print("Deleted old collection")
client.create_collection(collection_name=cn,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
print("Created new collection")

vectorstore = QdrantVectorStore(client=client,
    collection_name=cn, embedding=embed_model)
batch_size = 2000
for i in range(0, len(texts), batch_size):
    bt = texts[i:i+batch_size]
    bm = metadatas[i:i+batch_size]
    vectorstore.add_texts(texts=bt, metadatas=bm)
    print(f"  {min(i+batch_size, len(texts))}/{len(texts)}")

print(f"\nDone! {len(all_pdf_files)} PDFs, {len(all_chunks)} chunks")
