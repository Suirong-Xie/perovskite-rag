#!/usr/bin/env python3
"""
PyMuPDF4LLM vs 当前方法 — Chunking 质量对比测试
用法: python3 test_pymupdf4llm_chunking.py [pdf_path]
"""

import json, os, re, sys, time, argparse
from pathlib import Path

# 加载 .env
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                if _key not in os.environ:
                    os.environ[_key] = _val.strip().strip('"').strip("'")

import requests as httpx
import fitz
import pymupdf4llm

# ── 配置 ──
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

CHUNK_MIN_CHARS = 300
CHUNK_MAX_CHARS = 2000

# ── 当前方法的 normalize/clean (从 s2_chunk_and_embed.py 复制) ──

def normalize_text(text: str) -> str:
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


# ── PyMuPDF4LLM 专用清洗 ──

def clean_markdown_text(md_text: str) -> str:
    """清洗 PyMuPDF4LLM 输出，保留 markdown 结构但去除噪声。"""
    if not md_text:
        return ""

    # 1. 移除图片文字标记
    md_text = re.sub(r'<!-- Start of picture text -->.*?<!-- End of picture text -->',
                     '', md_text, flags=re.DOTALL)

    # 2. 移除单独的行内图片文字 (如 "and<br>to<br>")
    md_text = re.sub(r'^\s*(?:\w+)?<br>\s*$', '', md_text, flags=re.MULTILINE)

    # 3. 合并连续空行
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)

    # 4. 移除空白的 markdown header
    md_text = re.sub(r'^#{1,4}\s*$', '', md_text, flags=re.MULTILINE)

    # 5. 尝试截断 References/Acknowledgments 之后的内容
    for marker in [
        r'^#{1,4}\s*R[eE][fF][eE][rR][eE][nN][cC][eE][sS]?\s*$',
        r'^#{1,4}\s*References?\s+[aA]nd\s+[Nn]otes?\s*$',
        r'^#{1,4}\s*Acknowledge?ments?\s*$',
        r'^#{1,4}\s*Author\s+[Cc]ontributions?\s*$',
        r'^#{1,4}\s*Competing\s+[Ii]nterests?\s*$',
        r'^#{1,4}\s*Supplementary\s+(?:Information|Materials?)\s*$',
    ]:
        lines = md_text.split('\n')
        for i, line in enumerate(lines):
            if re.match(marker, line.strip()):
                md_text = '\n'.join(lines[:i])
                break

    return md_text.strip()


# ── LLM Chunking (DeepSeek, 与 s2_chunk_and_embed.py 一致) ──

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


def chunk_with_llm(text: str, label: str = "") -> list[str]:
    """用 DeepSeek API 做语义切分，返回 chunk 列表。"""
    if not text or len(text) < CHUNK_MIN_CHARS:
        return []

    prompt = CHUNK_PROMPT.replace("{text}", text[:12000])
    api_url = LLM_BASE_URL.rstrip("/")
    if not api_url.endswith("/v1"):
        api_url += "/v1"
    api_url += "/chat/completions"

    start = time.time()
    for attempt in range(3):
        try:
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
                print(f"  [{label}] API error {resp.status_code}: {resp.text[:200]}", flush=True)
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                return []

            content = resp.json()["choices"][0]["message"]["content"]
            elapsed = time.time() - start

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                print(f"  [{label}] No JSON in response ({elapsed:.1f}s)", flush=True)
                return []

            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                print(f"  [{label}] JSON parse error ({elapsed:.1f}s)", flush=True)
                return []

            if result.get("skip"):
                print(f"  [{label}] LLM marked as IRRELEVANT (skip=true)", flush=True)
                return []

            chunks = result.get("chunks", [])
            valid = []
            for c in chunks:
                c = c.strip()
                if len(c) >= CHUNK_MIN_CHARS:
                    if len(c) > CHUNK_MAX_CHARS:
                        c = c[:CHUNK_MAX_CHARS]
                    valid.append(c)

            if valid:
                print(f"  [{label}] {len(valid)} chunks in {elapsed:.1f}s "
                      f"(avg {sum(len(c) for c in valid)//len(valid)} chars/chunk)", flush=True)
                return valid

            return []

        except Exception as e:
            print(f"  [{label}] Error: {e}", flush=True)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return []

    return []


# ── 评估函数 ──

def evaluate_chunks(chunks: list[str], label: str) -> dict:
    """评估 chunk 质量。"""
    if not chunks:
        return {"n": 0, "label": label, "avg_len": 0, "min_len": 0, "max_len": 0,
                "sections_detected": [], "issues": ["No chunks produced"]}

    lengths = [len(c) for c in chunks]
    # 检测 section 关键词
    section_keywords = [
        "introduction", "method", "experiment", "result", "discussion",
        "conclusion", "device fabrication", "characterization", "performance",
        "stability", "efficiency", "structure", "morphology", "optical",
        "photovoltaic", "solar cell"
    ]
    detected = []
    for kw in section_keywords:
        for c in chunks:
            if kw.lower() in c.lower()[:100]:  # 出现在前100字符
                if kw not in detected:
                    detected.append(kw)

    issues = []
    # 过长
    too_long = [l for l in lengths if l > CHUNK_MAX_CHARS]
    if too_long:
        issues.append(f"{len(too_long)} chunks exceed {CHUNK_MAX_CHARS} chars")
    # 过短
    too_short = [l for l in lengths if l < CHUNK_MIN_CHARS]
    if too_short:
        issues.append(f"{len(too_short)} chunks below {CHUNK_MIN_CHARS} chars")
    # 重复内容检测
    seen_starts = set()
    duplicates = 0
    for c in chunks:
        start = c[:60].strip().lower()
        if start in seen_starts:
            duplicates += 1
        seen_starts.add(start)
    if duplicates:
        issues.append(f"{duplicates} potential duplicate chunks")

    return {
        "n": len(chunks),
        "label": label,
        "avg_len": sum(lengths) // len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "sections_detected": detected,
        "issues": issues if issues else ["None"],
    }


# ── 主流程 ──

def main():
    parser = argparse.ArgumentParser(description="对比 PyMuPDF4LLM vs 当前方法的 chunking 效果")
    parser.add_argument("pdf_path", nargs="?", help="PDF 文件路径")
    parser.add_argument("--no-llm", action="store_true", help="只提取文本，不调用 LLM chunking")
    args = parser.parse_args()

    pdf_path = args.pdf_path or "/data1/perovskite-rag/journals_pdf/Science/10.1126_science.abh1885.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    print("=" * 70)
    print("PyMuPDF4LLM vs 当前方法 — Chunking 对比测试")
    print(f"PDF: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path)/1024:.0f} KB")
    print(f"LLM: {LLM_MODEL} @ {LLM_BASE_URL}")
    print("=" * 70)

    # ── Step 1: 文本提取 ──
    print("\n📝 Step 1: 文本提取")

    # 方法 1: fitz + normalize + clean
    t0 = time.time()
    doc = fitz.open(pdf_path)
    raw_pages = [page.get_text() for page in doc]
    n_pages = len(raw_pages)
    raw_text = '\n'.join(raw_pages)
    doc.close()
    current_text = clean_paper_text(normalize_text(raw_text))
    t1 = time.time()
    print(f"  当前方法 (fitz→normalize→clean): {len(current_text)} chars, {n_pages} pages, {t1-t0:.1f}s")

    # 方法 2: PyMuPDF4LLM
    t0b = time.time()
    md_text_raw = pymupdf4llm.to_markdown(pdf_path)
    md_text = clean_markdown_text(md_text_raw)
    t1b = time.time()
    print(f"  PyMuPDF4LLM (to_markdown→clean): {len(md_text)} chars, {t1b-t0b:.1f}s")

    # ── Step 2: 文本质量对比 ──
    print(f"\n📊 Step 2: 文本质量对比")
    print(f"  {'指标':<35} {'当前方法':>15} {'PyMuPDF4LLM':>15}")
    print(f"  {'-'*35} {'-'*15} {'-'*15}")

    # 段落数
    curr_paras = len([p for p in current_text.split('\n\n') if p.strip()])
    md_paras = len([p for p in md_text.split('\n\n') if p.strip()])
    print(f"  {'段落数':<35} {curr_paras:>15} {md_paras:>15}")

    # 平均段落长度
    curr_avg_para = sum(len(p) for p in current_text.split('\n\n')) // max(curr_paras, 1)
    md_avg_para = sum(len(p) for p in md_text.split('\n\n')) // max(md_paras, 1)
    print(f"  {'平均段落长度 (chars)':<35} {curr_avg_para:>15} {md_avg_para:>15}")

    # 结构标记
    print(f"  {'Markdown headers':<35} {'-':>15} {len(re.findall(r'^#{1,4}\s', md_text, re.MULTILINE)):>15}")
    print(f"  {'Bold/Italic 标记':<35} {'-':>15} {len(re.findall(r'\*\*[^*]+\*\*|\*[^*]+\*', md_text)):>15}")
    pic_count = md_text_raw.count('<!-- Start of picture text -->')
    print(f"  {'图片文字块 (已移除)':<35} {'-':>15} {pic_count:>15}")

    # Section 关键词覆盖
    sections = ['introduction', 'method', 'result', 'discussion', 'conclusion',
                'device fabrication', 'stability', 'efficiency']
    curr_sections = [s for s in sections if s.lower() in current_text.lower()]
    md_sections = [s for s in sections if s.lower() in md_text.lower()]
    print(f"  {'Section 关键词覆盖':<35} {len(curr_sections):>15} {len(md_sections):>15}")

    # 显示文本样本
    print(f"\n📄 文本样本对比:")
    print(f"\n  ── 当前方法 (前 600 chars) ──")
    print(f"  {current_text[:600]}")
    print(f"\n  ── PyMuPDF4LLM (前 600 chars) ──")
    print(f"  {md_text[:600]}")

    if args.no_llm:
        print("\n⏩ --no-llm: 跳过 LLM chunking")
        return

    # ── Step 3: LLM Chunking ──
    print(f"\n🤖 Step 3: LLM Chunking ({LLM_MODEL})")

    print("\n  [Method A] 当前方法 → LLM chunking...")
    chunks_a = chunk_with_llm(current_text, "当前方法")

    print("\n  [Method B] PyMuPDF4LLM → LLM chunking...")
    chunks_b = chunk_with_llm(md_text, "PyMuPDF4LLM")

    # ── Step 4: Chunk 质量评估 ──
    print(f"\n📊 Step 4: Chunk 质量评估")
    eval_a = evaluate_chunks(chunks_a, "当前方法")
    eval_b = evaluate_chunks(chunks_b, "PyMuPDF4LLM")

    for ev in [eval_a, eval_b]:
        print(f"\n  [{ev['label']}]")
        print(f"    Chunks: {ev['n']}")
        print(f"    Avg length: {ev['avg_len']} chars")
        print(f"    Range: {ev['min_len']}–{ev['max_len']} chars")
        print(f"    Sections detected: {ev['sections_detected']}")
        print(f"    Issues: {', '.join(ev['issues'])}")

    # ── Step 5: Chunk 内容对比 ──
    print(f"\n📝 Step 5: Chunk 内容样本")
    if chunks_a:
        print(f"\n  [当前方法] Chunk #1 ({len(chunks_a[0])} chars):")
        print(f"  {chunks_a[0][:500]}")
        if len(chunks_a) > 1:
            print(f"  ... Chunk #{len(chunks_a)} ({len(chunks_a[-1])} chars):")
            print(f"  {chunks_a[-1][:500]}")

    if chunks_b:
        print(f"\n  [PyMuPDF4LLM] Chunk #1 ({len(chunks_b[0])} chars):")
        print(f"  {chunks_b[0][:500]}")
        if len(chunks_b) > 1:
            mid = len(chunks_b) // 2
            print(f"  ... Chunk #{mid+1} ({len(chunks_b[mid])} chars):")
            print(f"  {chunks_b[mid][:500]}")
            print(f"  ... Chunk #{len(chunks_b)} ({len(chunks_b[-1])} chars):")
            print(f"  {chunks_b[-1][:500]}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("🏆 总结")
    print(f"{'='*70}")
    print(f"  PyMuPDF4LLM 提取了 {len(md_text) - len(current_text):+d} chars "
          f"({100*(len(md_text)-len(current_text))/max(len(current_text),1):+.0f}%) 更多内容")
    print(f"  Chunks: 当前方法={eval_a['n']}, PyMuPDF4LLM={eval_b['n']}")
    print(f"  结构: PyMuPDF4LLM 保留了 Markdown headers, bold/italic 等语义标记")
    print(f"  耗时: 当前方法 ~0.1s (extract) + LLM, PyMuPDF4LLM ~{t1b-t0b:.0f}s (extract) + LLM")

    # Write results to JSON for record-keeping
    result = {
        "pdf": pdf_path,
        "comparison": {
            "current": {
                "text_len": len(current_text),
                "paragraphs": curr_paras,
                "chunks": eval_a,
                "extract_time_s": round(t1 - t0, 2),
            },
            "pymupdf4llm": {
                "text_len": len(md_text),
                "paragraphs": md_paras,
                "chunks": eval_b,
                "extract_time_s": round(t1b - t0b, 1),
            },
        }
    }
    result_path = "/data1/perovskite-rag/data/s2_corpus/pymupdf4llm_test_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 结果已保存: {result_path}")


if __name__ == "__main__":
    main()
