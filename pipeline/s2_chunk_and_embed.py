#!/usr/bin/env python3
"""
s2_chunk_and_embed.py — S2 论文 切分 + 向量化入库 (v2: PyMuPDF4LLM + 断点续传)

步骤:
  1. 构建 PDF 索引 (Sci-Hub + campus downloads)
  2. 读取 s2_texts_tier1.jsonl (全文) + s2_texts_tier2.jsonl (摘要)
  3. Tier 1 (有 PDF): PyMuPDF4LLM 提取 → 清洗 → DeepSeek LLM 语义切分
  4. Tier 1 (无 PDF): 使用现有 content_raw → normalize → clean → LLM 切分
  5. Tier 2 (有 PDF): 同上 PyMuPDF4LLM 提取 → LLM 切分 (校园下载补全)
  6. Tier 2 (摘要): 无需切分, 摘要本身就是 1 个 chunk
  7. Ollama mxbai-embed-large 批量嵌入
  8. 输出 → data/s2_vector_db/texts.jsonl + vectors.npy
  9. 断点续传: s2_chunks_progress.json 追踪已处理 paper_id

与现有向量库格式兼容, 额外字段: _s2_citation_count, _s2_year, _s2_tier

用法:
  python3 s2_chunk_and_embed.py                    # 全量处理 (支持断点续传)
  python3 s2_chunk_and_embed.py --limit 100        # 只处理前 100 篇
  python3 s2_chunk_and_embed.py --chunk-only       # 只切分, 不嵌入
  python3 s2_chunk_and_embed.py --embed-only       # 从已有的 chunk 文件嵌入
  python3 s2_chunk_and_embed.py --fresh            # 忽略断点, 重新开始
"""

import json, os, re, sys, time, argparse, glob as globmod
from typing import Optional
from pathlib import Path

import requests as httpx
import numpy as np

# ── 加载 .env ──

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                if _key not in os.environ:
                    os.environ[_key] = _val.strip().strip('"').strip("'")

# ── 配置 ──

BASE_DIR = "/data1/perovskite-rag"
CORPUS_DIR = os.path.join(BASE_DIR, "data", "s2_corpus")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "s2_vector_db")

TIER1_FILE = os.path.join(CORPUS_DIR, "s2_texts_tier1.jsonl")
TIER2_FILE = os.path.join(CORPUS_DIR, "s2_texts_tier2.jsonl")
CHUNKED_FILE = os.path.join(CORPUS_DIR, "s2_chunks.jsonl")
PROGRESS_FILE = os.path.join(CORPUS_DIR, "s2_chunks_progress.json")
TEXTS_FILE = os.path.join(VECTOR_DB_DIR, "texts.jsonl")
VECTORS_FILE = os.path.join(VECTOR_DB_DIR, "vectors.npy")

os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# PDF 搜索路径
PDF_SEARCH_DIRS = [
    "/data/data/pkb/01_raw_data/journals_pdf",
    "/data1/perovskite-rag/journals_pdf",
]

# DeepSeek API
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Ollama
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://127.0.0.1:11435/api/embed")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
EMBED_BATCH_SIZE = 20

# Chunking
CHUNK_MIN_CHARS = 300
CHUNK_MAX_CHARS = 2000
LLM_MAX_INPUT = 12000  # 最多送 LLM 的字符数


# ═══════════════════════════════════════════════════════════════════
# PDF 索引
# ═══════════════════════════════════════════════════════════════════

_pdf_index: dict[str, str] | None = None  # filename → full_path


def build_pdf_index(search_dirs: list[str] | None = None) -> dict[str, str]:
    """构建 PDF 文件名 → 完整路径的索引。"""
    global _pdf_index
    if _pdf_index is not None:
        return _pdf_index

    dirs = search_dirs or PDF_SEARCH_DIRS
    _pdf_index = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, dirs_, files in os.walk(d):
            for f in files:
                if f.lower().endswith('.pdf') and f not in _pdf_index:
                    _pdf_index[f] = os.path.join(root, f)

    print(f"[PDF] Indexed {len(_pdf_index)} PDFs from {len(dirs)} search dirs", flush=True)
    return _pdf_index


def find_pdf(record: dict) -> str | None:
    """为 paper record 查找 PDF 文件路径。"""
    idx = build_pdf_index()

    # 1. 从 source 字段推断
    source = record.get('source', '') or ''
    if source.endswith('.pdf'):
        if source in idx:
            return idx[source]
        # 去掉 journal 前缀: "JACS_10.1021_ja809598r.pdf" → "10.1021_ja809598r.pdf"
        parts = source.rsplit('_', 1)
        if len(parts) == 2 and parts[0].isalpha():
            fname = parts[1]
            if fname in idx:
                return idx[fname]

    # 2. 从 DOI 推断
    doi = (record.get('doi', '') or '').strip()
    if doi:
        doi_file = doi.replace('/', '_') + '.pdf'
        if doi_file in idx:
            return idx[doi_file]
        # Try lowercase
        doi_file_lower = doi_file.lower()
        for fname, path in idx.items():
            if fname.lower() == doi_file_lower:
                return path

    return None


# ═══════════════════════════════════════════════════════════════════
# PyMuPDF4LLM 文本提取 + 清洗
# ═══════════════════════════════════════════════════════════════════

def extract_text_pymupdf4llm(pdf_path: str) -> tuple[str, str]:
    """用 PyMuPDF4LLM 提取 PDF 文本 (Markdown 格式), 并清洗噪声。

    Returns:
        (cleaned_text, method)
        method: 'pymupdf4llm' | 'pymupdf4llm+ocr' | 'failed'
    """
    import pymupdf4llm

    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
    except Exception as e:
        print(f"  [PDF] PyMuPDF4LLM failed: {e}, falling back to fitz", flush=True)
        return "", "failed"

    cleaned = clean_markdown_text(md_text)
    # 判断是否走了 OCR (扫描版 PDF)
    is_ocr = "<!-- Start of picture text -->" in md_text and len(cleaned) < 2000
    method = 'pymupdf4llm+ocr' if is_ocr else 'pymupdf4llm'
    return cleaned, method


def clean_markdown_text(md_text: str) -> str:
    """清洗 PyMuPDF4LLM 输出的 markdown 文本。"""
    if not md_text:
        return ""

    # 1. 移除图片文字标记块
    md_text = re.sub(
        r'<!-- Start of picture text -->.*?<!-- End of picture text -->',
        '', md_text, flags=re.DOTALL
    )

    # 2. 移除仅有 <br> 的行
    md_text = re.sub(r'^\s*(?:\w+\s*)?(?:<br>\s*)+$', '', md_text, flags=re.MULTILINE)

    # 3. 移除校正声明类噪声行
    md_text = re.sub(
        r'^\s*(?:Corrected|Erratum|Corrigendum)\s+\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\.?\s*(?:See full text\.?)?\s*$',
        '', md_text, flags=re.MULTILINE | re.IGNORECASE
    )

    # 4. 移除空白的 markdown header (# 后面没有内容)
    md_text = re.sub(r'^#{1,4}\s*$', '', md_text, flags=re.MULTILINE)

    # 5. 合并连续空行
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)

    # 6. 截断 References / Acknowledgments 之后的内容
    cutoff_pattern = re.compile(
        r'^(?:#{1,4}\s*)?'
        r'(?:R[eE][fF][eE][rR][eE][nN][cC][eE][sS]?(?:\s+[aA][nN][dD]\s+[Nn][oO][tT][eE][sS]?)?|'
        r'Acknowledge?ments?|'
        r'Author\s+[Cc]ontributions?|'
        r'Competing\s+[Ii]nterests?|'
        r'Conflict\s+of\s+[Ii]nterest|'
        r'Supplementary\s+(?:Information|Materials?|Data|Notes?)|'
        r'Supporting\s+Information|'
        r'Data\s+[Aa]vailability|'
        r'Code\s+[Aa]vailability|'
        r'Funding\s+[Ss]ources?|'
        r'Notes?\s+and\s+[Rr]eferences?)'
        r'\s*$'
    )

    lines = md_text.split('\n')
    cutoff_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if cutoff_pattern.match(stripped):
            # Confirm: next line should be different content or end
            # Avoid false positive on section mentions in body text
            if len(stripped) < 80 and ('references' in stripped.lower() or 'acknowledg' in stripped.lower()):
                cutoff_idx = i
                break

    if cutoff_idx is not None:
        md_text = '\n'.join(lines[:cutoff_idx])
        print(f"  [CLEAN] Truncated at line {cutoff_idx} (marker: '{lines[cutoff_idx].strip()[:60]}')", flush=True)

    # 7. 移除纯 figure caption 行 (markdown 格式)
    md_text = re.sub(r'^\s*\*?\*?(?:Fig(?:ure)?\.?\s*\d+|Table\s+\d+|Scheme\s+\d+)\b.*$',
                     '', md_text, flags=re.MULTILINE | re.IGNORECASE)

    return md_text.strip()


# ═══════════════════════════════════════════════════════════════════
# 当前方法的 text normalization (fallback)
# ═══════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """将 PDF 提取的断行文本重组为段落结构 (fitz fallback)。"""
    if not text:
        return ""
    lines = text.split('\n')
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith('-') and i + 1 < len(lines):
            next_line = lines[i + 1].lstrip()
            if next_line and next_line[0].islower():
                line = line.rstrip()[:-1] + next_line
                i += 2
                merged.append(line.rstrip())
                continue
        if not line.strip():
            merged.append('')
            i += 1
            continue
        stripped = line.strip()
        if len(stripped) < 40 and (stripped.endswith('.') or stripped.isupper()
                                    or stripped[0].isdigit()):
            merged.append(stripped)
            i += 1
            continue
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            next_stripped = lines[i].strip()
            if not next_stripped:
                break
            if len(next_stripped) < 40 and (next_stripped.endswith('.') or next_stripped[0].isdigit()):
                break
            if (next_stripped[0].isdigit() and '.' in next_stripped[:4]) or \
               (next_stripped.isupper() and len(next_stripped) < 60):
                break
            para_lines.append(next_stripped)
            i += 1
        para = ' '.join(para_lines)
        para = re.sub(r'\s+', ' ', para).strip()
        merged.append(para)
    result = '\n\n'.join(l for l in merged if l or (merged[0] if merged else True))
    return result


CUTOFF_MARKERS = [
    r'^references?\s*$', r'^references?\s+\d', r'^bibliography\s*$',
    r'^acknowledgments?\s*$', r'^acknowledgements?\s*$',
    r'^author\s+contributions?\s*$', r'^competing\s+interests?\s*$',
    r'^conflict\s+of\s+interest\s*$', r'^supplementary\s+information\s*$',
    r'^supporting\s+information\s*$', r'^supplementary\s+materials?\s*$',
    r'^electronic\s+supplementary\s+material',
    r'^additional\s+information\s*$', r'^notes?\s+and\s+references?\s*$',
    r'^footnotes?\s*$', r'^appendix\s',
]

SKIP_LINE_PATTERNS = [
    r'^\s*fig(?:ure)?\.?\s+\d+\.', r'^\s*table\s+\d+\.',
    r'^\s*scheme\s+\d+\.', r'^\s*chart\s+\d+\.', r'^\s*plate\s+\d+\.',
    r'^\s*box\s+\d+\.', r'^\s*\d+\s*$', r'^\s*[a-z]?\d{2,4}\s*$',
    r'^\s*doi:\s', r'^\s*https?://', r'^\s*received:?\s', r'^\s*published:?\s',
    r'^\s*accepted:?\s', r'^\s*©\s', r'^\s*copyright\s',
    r'^\s*all rights reserved', r'^\s*this journal is\s', r'^\s*\|\s*\d+\s*$',
]


def clean_paper_text(text: str) -> str:
    """清洗论文文本：截断参考文献/致谢, 移除图表标题/页码/版权声明 (fitz fallback)。"""
    if not text:
        return ""
    lines = text.split('\n')
    cleaned_lines = []
    cutoff_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern in CUTOFF_MARKERS:
            if re.match(pattern, stripped, re.IGNORECASE):
                cutoff_idx = i
                break
        if cutoff_idx is not None:
            break
        skip = False
        for pattern in SKIP_LINE_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                skip = True
                break
        if not skip and stripped:
            cleaned_lines.append(line)
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip()
    return result


# ═══════════════════════════════════════════════════════════════════
# LLM Chunking (DeepSeek API)
# ═══════════════════════════════════════════════════════════════════

CHUNK_PROMPT = """You are a paper processing assistant for a perovskite solar cell (PSC) RAG system.

Step 1 - Relevance: Is this text from a paper about perovskite solar cells?
  RELEVANT: perovskite composition/devices, SAMs, HTL/ETL, device stability,
  passivation, interface engineering, tandem cells, lead-free perovskites,
  perovskite photophysics (PL, TRPL, TA), perovskite film processing
  IRRELEVANT: oxide perovskites (non-PSC), ferroelectric, spintronics, catalysis,
  pure organic photovoltaics (OPV, BHJ without perovskite), quantum optics,
  superconductors, batteries, fuel cells, water splitting (unless PSC-related),
  non-solar-cell perovskite (LEDs, lasers, detectors — unless clearly about PSC)

Step 2 - If RELEVANT: Split into semantic chunks for RAG.
  Each chunk MUST be 500-2000 characters. Hard limit: NEVER exceed 2000 chars.
  Rules:
  - Each chunk is a complete, self-contained semantic unit
  - EXCLUDE: references, acknowledgments, figure captions (lines starting with "Fig." or "Figure"), author contributions, competing interests statements
  - Preserve original English wording — do NOT summarize or translate
  - Follow the paper's natural section structure (Introduction → Methods → Results → Discussion)

OUTPUT (JSON only, no other text):
  IRRELEVANT: {"skip": true}
  RELEVANT: {"chunks": ["chunk1...", "chunk2..."], "n": <int>}

TEXT:
{text}

JSON OUTPUT:"""


def chunk_text_deepseek(text: str, max_retries: int = 3) -> list[str]:
    """用 DeepSeek API 做语义切分。"""
    if len(text) <= CHUNK_MIN_CHARS:
        return []

    prompt = CHUNK_PROMPT.replace("{text}", text[:LLM_MAX_INPUT])

    for attempt in range(max_retries):
        try:
            api_url = LLM_BASE_URL.rstrip("/")
            if not api_url.endswith("/v1"):
                api_url += "/v1"
            api_url += "/chat/completions"

            resp = httpx.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.05,
                    "max_tokens": 4096,
                },
                timeout=120,
            )

            if resp.status_code != 200:
                print(f"  [CHUNK] API error {resp.status_code}: {resp.text[:200]}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return _simple_split(text)

            content = resp.json()["choices"][0]["message"]["content"]
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                return _simple_split(text)

            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                return _simple_split(text)

            if result.get("skip"):
                return []

            chunks = result.get("chunks", [])
            if not chunks:
                return []

            valid = []
            for c in chunks:
                c = c.strip()
                if len(c) >= CHUNK_MIN_CHARS:
                    if len(c) > CHUNK_MAX_CHARS:
                        c = c[:CHUNK_MAX_CHARS]
                    valid.append(c)
            if valid:
                return valid

            return _simple_split(text)

        except Exception as e:
            print(f"  [CHUNK] Error: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return []

    return []


def _clean_text(text: str) -> str:
    """清洗文本: 移除不可打印字符, 保留可读内容。"""
    cleaned = []
    for ch in text:
        if ch.isprintable() or ch in '\n\r\t':
            cleaned.append(ch)
        else:
            cleaned.append(' ')
    return ''.join(cleaned)


def _simple_split(text: str) -> list[str]:
    """简单切分: 按段落/句子边界, 严格保证每块 <= CHUNK_MAX_CHARS。"""
    text = _clean_text(text)
    if len(text) <= CHUNK_MAX_CHARS:
        return [text] if len(text) >= CHUNK_MIN_CHARS else []

    paragraphs = text.split("\n\n")
    segments = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= CHUNK_MAX_CHARS:
            segments.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) > CHUNK_MAX_CHARS:
                    for i in range(0, len(sent), CHUNK_MAX_CHARS):
                        chunk = sent[i:i + CHUNK_MAX_CHARS].strip()
                        if len(chunk) >= CHUNK_MIN_CHARS:
                            segments.append(chunk)
                else:
                    segments.append(sent)

    chunks = []
    current = ""
    for seg in segments:
        if len(current) + len(seg) + 2 <= CHUNK_MAX_CHARS:
            current = (current + "\n\n" + seg) if current else seg
        else:
            if current and len(current) >= CHUNK_MIN_CHARS:
                chunks.append(current)
            elif current:
                current = current + "\n\n" + seg
                if len(current) > CHUNK_MAX_CHARS:
                    current = current[:CHUNK_MAX_CHARS]
                if len(current) >= CHUNK_MIN_CHARS:
                    chunks.append(current)
                current = ""
                continue
            current = seg

    if current and len(current) >= CHUNK_MIN_CHARS:
        chunks.append(current)

    if not chunks and text:
        chunks = [text[:CHUNK_MAX_CHARS]]

    return chunks


# ═══════════════════════════════════════════════════════════════════
# Embedding (Ollama)
# ═══════════════════════════════════════════════════════════════════

def embed_batch(texts: list[str]) -> np.ndarray:
    """批量嵌入, 返回 L2 归一化的向量数组。"""
    clean_texts = [_clean_text(t) for t in texts]
    resp = httpx.post(
        OLLAMA_EMBED_URL,
        json={"model": OLLAMA_EMBED_MODEL, "input": clean_texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    vectors = np.array(data["embeddings"], dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vectors /= norms

    return vectors


# ═══════════════════════════════════════════════════════════════════
# 断点续传管理
# ═══════════════════════════════════════════════════════════════════

def load_progress() -> dict[str, int]:
    """加载断点进度: {paper_id: num_chunks}。"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict[str, int]) -> None:
    """保存断点进度。"""
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    os.replace(tmp, PROGRESS_FILE)


def _venue_rank(venue: str) -> int:
    """期刊分级 (用于排序权重)。"""
    top = {
        "Nature": 1, "Science": 1,
        "Nature Energy": 2, "Nature Materials": 2, "Joule": 2,
        "Nature Photonics": 3, "Nature Nanotechnology": 3, "Matter": 3,
        "Nature Communications": 4, "Science Advances": 4, "Chem": 4,
        "ACS Energy Letters": 5, "Advanced Materials": 5,
        "Advanced Energy Materials": 5, "Energy & Environmental Science": 5,
        "Angewandte Chemie": 5, "JACS": 5,
        "Advanced Functional Materials": 6, "Nano Energy": 6,
        "ACS Nano": 6, "Nano Letters": 6, "Chemistry of Materials": 6,
        "Solar RRL": 7, "Small": 7, "Advanced Science": 7,
        "Journal of Materials Chemistry A": 7, "ACS Applied Materials & Interfaces": 7,
    }
    for key, rank in top.items():
        if key.lower() in venue.lower():
            return rank
    return 8


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def load_texts(path: str) -> list[dict]:
    """加载 JSONL 论文数据。"""
    records = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ═══════════════════════════════════════════════════════════════════
# 并行 Chunking
# ═══════════════════════════════════════════════════════════════════

from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import threading

LLM_WORKERS = 4  # 并行 LLM API 调用数
COLLECT_BATCH = LLM_WORKERS * 3  # 攒够 N 个 future 后开始收集结果


def _extract_text_for_paper(rec: dict) -> tuple[str, int, str]:
    """为单篇论文提取文本。

    Returns:
        (content, tier, status)
        status: 'pymupdf4llm' | 'pymupdf4llm+ocr' | 'fallback' | 'fallback_no_pdf' | 'abstract' | 'failed'
    """
    content = rec.get("content_raw", "") or ""
    tier = rec.get("tier", 2)

    if tier == 1:
        pdf_path = find_pdf(rec)
        if pdf_path:
            extracted, method = extract_text_pymupdf4llm(pdf_path)
            if extracted and len(extracted) > CHUNK_MIN_CHARS:
                return extracted, 1, method
            else:
                # Fallback: PyMuPDF4LLM 失败 (OCR 质量差 / 提取为空)
                if content:
                    content = clean_paper_text(normalize_text(content))
                    return content, 1, 'fallback'
                return "", 1, 'failed'
        else:
            content = clean_paper_text(normalize_text(content))
            return content, 1, 'fallback_no_pdf'
    else:
        # Tier 2: 检查是否有 PDF (校园下载补全)
        pdf_path = find_pdf(rec)
        if pdf_path:
            extracted, method = extract_text_pymupdf4llm(pdf_path)
            if extracted and len(extracted) > CHUNK_MIN_CHARS:
                return extracted, 1, method  # 升级为全文
        return content[:CHUNK_MAX_CHARS], 2, 'abstract'


def _run_llm_chunk(content: str) -> list[str]:
    """LLM 切分 wrapper (供线程池调用)。"""
    if len(content) > CHUNK_MAX_CHARS:
        return chunk_text_deepseek(content)
    else:
        content = content[:CHUNK_MAX_CHARS].strip()
        return [content] if len(content) >= CHUNK_MIN_CHARS else []


def _make_chunk_records(chunks: list[str], rec: dict, tier: int, start_id: int,
                        extract_method: str = "") -> list[dict]:
    """生成 chunk records。"""
    records = []
    for j, chunk in enumerate(chunks):
        record = {
            "id": start_id + j,
            "idx": start_id + j,
            "content": chunk,
            "source": rec.get("source", f"s2:{rec.get('paper_id', '')}"),
            "journal": rec.get("venue", "Unknown"),
            "journal_rank": _venue_rank(rec.get("venue", "")),
            "_s2_citation_count": rec.get("citation_count", 0) or 0,
            "_s2_year": rec.get("year"),
            "_s2_paper_id": rec.get("paper_id", ""),
            "_s2_tier": tier,
            "_s2_doi": rec.get("doi", ""),
        }
        if extract_method:
            record["_extract_method"] = extract_method
        records.append(record)
    return records


def chunk_all(
    tier1_path: str,
    tier2_path: str,
    chunked_path: str,
    limit: Optional[int] = None,
    fresh: bool = False,
) -> None:
    """切分所有论文 (并行版), 输出 chunked JSONL。支持断点续传。"""
    tier1 = load_texts(tier1_path)
    tier2 = load_texts(tier2_path)

    print(f"[CHUNK] Tier 1 (fulltext): {len(tier1)} papers", flush=True)
    print(f"[CHUNK] Tier 2 (abstract): {len(tier2)} papers", flush=True)
    print(f"[CHUNK] LLM workers: {LLM_WORKERS} (parallel)", flush=True)

    # 构建 PDF 索引
    build_pdf_index()

    # 断点续传
    progress = {} if fresh else load_progress()
    if progress:
        print(f"[RESUME] {len(progress)} papers already processed, resuming...", flush=True)

    # 最新论文优先 — PyMuPDF4LLM 对 2020+ 论文效果最好,
    # 旧扫描版 PDF 走 OCR 质量差, 放后面 fallback
    tier1.sort(key=lambda r: r.get('year', 0) or 0, reverse=True)
    all_records = tier1 + tier2
    if limit:
        all_records = all_records[:limit]

    # 统计
    total_chunks = sum(progress.values())
    deepseek_calls = 0
    pymupdf4llm_used = 0
    pymupdf4llm_ocr = 0
    fallback_used = 0
    fallback_no_pdf = 0
    pymupdf4llm_failed = 0
    skipped_count = 0

    write_lock = threading.Lock()
    file_mode = "a" if progress else "w"

    with open(chunked_path, file_mode) as fout, \
         ThreadPoolExecutor(max_workers=LLM_WORKERS) as executor:

        pending: dict[Future, tuple[dict, int, str, str]] = {}
        # {future: (rec, tier, paper_id, extract_method)}

        def _collect_one():
            """收集一个完成的 future, 写入结果。非阻塞: 如果没有完成的就返回 False。"""
            nonlocal total_chunks, deepseek_calls

            done_futures = [f for f in pending if f.done()]
            if not done_futures:
                return False

            f = done_futures[0]
            rec, tier, paper_id, extract_method = pending.pop(f)

            try:
                chunks = f.result(timeout=10)
            except Exception as e:
                print(f"  [WORKER] Error for {paper_id[:16]}: {e}", flush=True)
                chunks = []

            with write_lock:
                if chunks:
                    chunk_records = _make_chunk_records(chunks, rec, tier, total_chunks, extract_method)
                    for cr in chunk_records:
                        fout.write(json.dumps(cr, ensure_ascii=False) + "\n")
                    total_chunks += len(chunks)

                progress[paper_id] = len(chunks)
                save_progress(progress)
                fout.flush()

            return True

        t_start = time.time()
        extracted_count = 0

        for i, rec in enumerate(all_records):
            paper_id = rec.get("paper_id", f"unknown_{i}")

            # 断点续传: 跳过已处理的
            if paper_id in progress:
                skipped_count += 1
                continue

            # ── 提取文本 (主线程, 顺序执行) ──
            content, tier, status = _extract_text_for_paper(rec)
            extracted_count += 1

            if status == 'pymupdf4llm':
                pymupdf4llm_used += 1
            elif status == 'pymupdf4llm+ocr':
                pymupdf4llm_ocr += 1
            elif status == 'fallback':
                fallback_used += 1
            elif status == 'fallback_no_pdf':
                fallback_no_pdf += 1
            elif status == 'failed':
                pymupdf4llm_failed += 1

            # ── 提交 LLM 任务 (线程池并行) ──
            if tier == 1 and content:
                future = executor.submit(_run_llm_chunk, content)
                deepseek_calls += 1
            else:
                # 摘要或空文本: 直接包装结果
                future = executor.submit(lambda c=content: [c[:CHUNK_MAX_CHARS]] if len(c.strip()) >= CHUNK_MIN_CHARS else [])

            pending[future] = (rec, tier, paper_id, status)

            # ── 收集已完成的结果 ──
            if len(pending) >= COLLECT_BATCH:
                while len(pending) >= COLLECT_BATCH:
                    _collect_one()

            # ── 打印进度 ──
            current = len(progress) + len(pending)
            if current % 100 == 0 or current <= 10:
                elapsed = time.time() - t_start
                rate = extracted_count / (elapsed / 60) if elapsed > 0 else 0
                in_flight = len(pending)
                print(f"[CHUNK] {current} papers → {total_chunks} chunks "
                      f"(rate: {rate:.1f}/min, in-flight: {in_flight}, "
                      f"P4LLM: {pymupdf4llm_used}, OCR: {pymupdf4llm_ocr}, "
                      f"FB: {fallback_used}, noPDF: {fallback_no_pdf})", flush=True)

        # ── 收集所有剩余结果 ──
        print(f"[CHUNK] Extraction done. Collecting {len(pending)} remaining LLM results...", flush=True)
        while pending:
            _collect_one()
            if pending:
                time.sleep(0.5)  # 避免忙等待

    # ── 最终统计 ──
    elapsed_total = time.time() - t_start
    print(f"\n[CHUNK] Done: {len(progress)} papers → {total_chunks} chunks "
          f"in {elapsed_total/60:.1f}min "
          f"({len(progress)/(elapsed_total/60):.1f} papers/min)", flush=True)
    print(f"[CHUNK]   PyMuPDF4LLM (native): {pymupdf4llm_used} papers", flush=True)
    print(f"[CHUNK]   PyMuPDF4LLM (via OCR): {pymupdf4llm_ocr} papers", flush=True)
    print(f"[CHUNK]   Fallback (content_raw): {fallback_used} papers", flush=True)
    print(f"[CHUNK]   Fallback (no PDF found): {fallback_no_pdf} papers", flush=True)
    print(f"[CHUNK]   PyMuPDF4LLM failed: {pymupdf4llm_failed} papers", flush=True)
    print(f"[CHUNK]   DeepSeek API calls: {deepseek_calls}", flush=True)
    print(f"[CHUNK] Output: {chunked_path}", flush=True)
    print(f"[CHUNK] Progress: {PROGRESS_FILE} ({len(progress)} entries)", flush=True)


def embed_all(chunked_path: str, vector_dir: str):
    """嵌入所有 chunk, 输出 texts.jsonl + vectors.npy。"""
    chunks = load_texts(chunked_path)
    if not chunks:
        print("[EMBED] No chunks to embed!", flush=True)
        return

    print(f"[EMBED] {len(chunks)} chunks to embed", flush=True)

    all_vectors = []
    texts_path = os.path.join(vector_dir, "texts.jsonl")

    with open(texts_path, "w") as fout:
        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
            texts = [c["content"] for c in batch]

            try:
                vecs = embed_batch(texts)
            except Exception as e:
                print(f"[EMBED] Batch {batch_start} error: {e}, retrying one-by-one...", flush=True)
                vecs = []
                skip_indices = set()
                for j, t in enumerate(texts):
                    try:
                        v = embed_batch([t])
                        vecs.append(v[0])
                    except Exception as e2:
                        print(f"  [EMBED] Skipping chunk {batch_start+j}: {e2}", flush=True)
                        skip_indices.add(j)
                vecs = np.array(vecs, dtype=np.float32) if vecs else np.zeros((0, 1024), dtype=np.float32)
                if skip_indices:
                    for j in sorted(skip_indices, reverse=True):
                        del batch[j]

            all_vectors.append(vecs)

            for j, c in enumerate(batch):
                c["id"] = batch_start + j
                c["idx"] = batch_start + j
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")

            progress_val = min(batch_start + EMBED_BATCH_SIZE, len(chunks))
            print(f"[EMBED] {progress_val}/{len(chunks)} chunks embedded", flush=True)

    vectors = np.concatenate(all_vectors, axis=0)
    vectors_path = os.path.join(vector_dir, "vectors.npy")
    np.save(vectors_path, vectors)

    print(f"\n[EMBED] Done: {vectors.shape} vectors → {vectors_path}", flush=True)
    print(f"[EMBED] Texts: {texts_path}", flush=True)
    norms = np.linalg.norm(vectors, axis=1)
    print(f"[EMBED] Norm check: min={norms.min():.6f} max={norms.max():.6f} mean={norms.mean():.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="S2 论文切分 & 嵌入 (v2: PyMuPDF4LLM + 断点续传)")
    parser.add_argument("--limit", type=int, help="只处理前 N 篇论文")
    parser.add_argument("--chunk-only", action="store_true", help="只切分, 不嵌入")
    parser.add_argument("--embed-only", action="store_true", help="只嵌入 (从已有 chunk 文件)")
    parser.add_argument("--fresh", action="store_true", help="忽略断点, 从头开始 (删除进度文件)")
    args = parser.parse_args()

    print("[PIPELINE] === S2 Chunk & Embed v2 (PyMuPDF4LLM + 断点续传) ===", flush=True)
    print(f"[PIPELINE] LLM: {LLM_MODEL} @ {LLM_BASE_URL}", flush=True)
    print(f"[PIPELINE] Embed: {OLLAMA_EMBED_MODEL} @ {OLLAMA_EMBED_URL}", flush=True)

    if args.fresh:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print(f"[PIPELINE] --fresh: 已删除断点文件, 从头开始", flush=True)
        if os.path.exists(CHUNKED_FILE) and not args.embed_only:
            os.remove(CHUNKED_FILE)
            print(f"[PIPELINE] --fresh: 已删除旧 chunk 文件", flush=True)

    if not args.embed_only:
        print("\n[PIPELINE] === Phase 4.1: Chunking ===", flush=True)
        t0 = time.time()
        chunk_all(TIER1_FILE, TIER2_FILE, CHUNKED_FILE,
                  limit=args.limit, fresh=args.fresh)
        print(f"[PIPELINE] Chunking done in {(time.time()-t0)/60:.1f}min", flush=True)

    if not args.chunk_only:
        print("\n[PIPELINE] === Phase 4.2: Embedding ===", flush=True)
        t0 = time.time()
        embed_all(CHUNKED_FILE, VECTOR_DB_DIR)
        print(f"[PIPELINE] Embedding done in {(time.time()-t0)/60:.1f}min", flush=True)

    print("\n[PIPELINE] Phase 4 complete!", flush=True)


if __name__ == "__main__":
    main()
