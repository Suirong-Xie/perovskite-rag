import os
import json
import glob
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_ROOT = "/data/data/pkb/01_raw_data/papers_pdf/"
OUTPUT_DIR = "chunked_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 分块器：每块约 500 字符，重叠 100
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

def extract_text_from_pdf(pdf_path):
    """使用 PyMuPDF 提取文本，尝试按阅读顺序输出"""
    doc = fitz.open(pdf_path)
    full_text = []
    for page in doc:
        text = page.get_text("text")
        if text:
            full_text.append(text)
    return "\n".join(full_text)

# 遍历所有 PDF
pdf_files = glob.glob(os.path.join(PDF_ROOT, "**/*.pdf"), recursive=True)
print(f"找到 {len(pdf_files)} 个 PDF 文件")

all_chunks = []
for pdf_path in tqdm(pdf_files):
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"跳过错误文件 {pdf_path}: {e}")
        continue

    if not text.strip():
        continue

    # 文件名作为元数据
    file_name = os.path.basename(pdf_path)
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": file_name, "path": pdf_path}]
    )
    all_chunks.extend(chunks)

# 存储为 JSONL
output_path = os.path.join(OUTPUT_DIR, "chunks.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        record = {
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"完成，总共 {len(all_chunks)} 个块，保存至 {output_path}")
