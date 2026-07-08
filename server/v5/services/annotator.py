"""
PerovskiteGPT V5 — PDF 高亮元数据提取

只做文本分析和匹配，不修改 PDF 文件。
所有视觉高亮由前端 pdf-reader.html 的 CSS overlay 完成。
"""
import json
import os
import re
import unicodedata
from pathlib import Path
from collections import defaultdict
from ..core.config import V5_DIR


def log(msg: str):
    print(f"[V5] {msg}", flush=True)


# ── IDF 词频（惰性构建） ──
_idf_cache = None


def _get_idf():
    global _idf_cache
    if _idf_cache is not None:
        return _idf_cache
    from ..core.config import SUNNY_RAG_DIR, SEARCH_DATA_VERSION
    data_dir = SUNNY_RAG_DIR / f"data_{SEARCH_DATA_VERSION}"
    txt_path = data_dir / "texts.jsonl"
    if not txt_path.exists():
        _idf_cache = {}
        return _idf_cache
    doc_count = 0
    df = {}
    with open(txt_path) as f:
        for line in f:
            rec = json.loads(line)
            words = set(_normalize(rec.get("content", "")).split())
            for w in words:
                df[w] = df.get(w, 0) + 1
            doc_count += 1
    _idf_cache = {w: __import__('math').log(doc_count / (c + 1)) for w, c in df.items()}
    return _idf_cache


def _normalize(text: str) -> str:
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    return " ".join(text.lower().split())


def _word_overlap(chunk_words: set, line_words: set) -> float:
    if not line_words:
        return 0.0
    idf = _get_idf()
    inter = chunk_words & line_words
    if not inter:
        return 0.0
    weighted_inter = sum(idf.get(w, 1.0) for w in inter)
    weighted_union = sum(idf.get(w, 1.0) for w in (chunk_words | line_words))
    coverage = len(inter) / len(line_words) if line_words else 0
    return (weighted_inter / max(weighted_union, 0.001)) * 0.5 + coverage * 0.5


def extract_highlight_meta(pdf_path: str, chunk_texts: list) -> dict:
    """提取高亮元数据：匹配 chunk 文本到 PDF 段落/行。

    不修改 PDF，只返回前端需要的结构化数据：
    {
      "pages": [1, 2, ...],
      "chunks": [
        {
          "idx": 0,
          "pages": [1, 2],
          "text_preview": "...",
          "text": "...",           # 完整归一化文本，供 pdf.js 搜索
        }
      ]
    }

    注意：不再生成 annotated PDF 文件。
    """
    import fitz

    stem = Path(pdf_path).stem
    meta_path = V5_DIR / "annotated_pdfs" / f"{stem}_meta.json"

    # 缓存命中
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f) if os.path.getsize(meta_path) > 0 else {}
            log(f"[ANNOTATE] Cache hit for {stem} ({meta.get('pages', [])})")
            return meta
        except Exception:
            pass

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return {"pages": [], "chunks": []}

    # ── 提取段落和行 ──
    page_paragraphs = {}
    page_lines = {}

    for pg in range(len(doc)):
        page = doc[pg]
        page_w = page.rect.width

        text_blocks = []
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            text = b[4].strip()
            if len(text) < 5:
                continue
            text_blocks.append((b[1], b[3], b[0], b[2], text))  # y0, y1, x0, x1, text

        text_blocks.sort(key=lambda t: t[0])

        # 合并相邻块为段落
        merged = []
        cur = []
        for tb in text_blocks:
            y0, y1, x0, x1, text = tb
            if not cur:
                cur.append(tb)
            else:
                prev = cur[-1]
                line_h = y1 - y0
                if abs(y0 - prev[1]) < max(line_h * 2, 15):
                    prev_cx = (prev[2] + prev[3]) / 2
                    cur_cx = (x0 + x1) / 2
                    if abs(cur_cx - prev_cx) < page_w * 0.4:
                        cur.append(tb)
                    else:
                        merged.append(cur)
                        cur = [tb]
                else:
                    merged.append(cur)
                    cur = [tb]
        if cur:
            merged.append(cur)

        page_paragraphs[pg] = []
        page_lines[pg] = []

        for blocks in merged:
            para_text = " ".join(b[4] for b in blocks)
            if len(para_text) < 50:
                continue
            norm_words = set(_normalize(para_text).split())
            if len(norm_words) < 5:
                continue
            px0 = min(b[2] for b in blocks)
            py0 = min(b[0] for b in blocks)
            px1 = max(b[3] for b in blocks)
            py1 = max(b[1] for b in blocks)
            page_paragraphs[pg].append(([px0, py0, px1, py1], norm_words, para_text))

            for b in blocks:
                y0, y1, x0, x1, text = b
                if len(text.strip()) >= 10:
                    page_lines[pg].append(([x0, y0, x1, y1], text.strip()))

    doc.close()

    # ── 匹配 chunks ──
    all_pages: set[int] = set()
    chunk_meta: list[dict] = []

    for ci, chunk in enumerate(chunk_texts):
        chunk_words = set(_normalize(chunk).split())
        if len(chunk_words) < 5:
            continue

        # 段落级匹配
        para_scores = []
        for pg in range(len(page_paragraphs)):
            for bbox, para_words, text in page_paragraphs[pg]:
                score = _word_overlap(chunk_words, para_words)
                if score > 0.05:
                    para_scores.append((score, pg, bbox))

        if not para_scores:
            continue

        para_scores.sort(key=lambda x: -x[0])
        top_paras = [(pg, bbox) for _, pg, bbox in para_scores[:3] if _ > 0.06]

        chunk_pages: set[int] = set()
        for pg, para_bbox in top_paras:
            px0, py0, px1, py1 = para_bbox
            py0 -= 2
            py1 += 2

            for line_arr, line_text in page_lines.get(pg, []):
                lx0, ly0, lx1, ly1 = line_arr
                if ly0 < py0 or ly1 > py1:
                    continue
                line_words = set(_normalize(line_text).split())
                if not line_words:
                    continue
                if _word_overlap(chunk_words, line_words) < 0.03:
                    continue
                chunk_pages.add(pg + 1)
                all_pages.add(pg + 1)

        if chunk_pages:
            chunk_meta.append({
                "idx": ci,
                "pages": sorted(chunk_pages),
                "text_preview": _normalize(chunk)[:120],
                "text": _normalize(chunk),
            })

    meta = {"pages": sorted(all_pages), "chunks": chunk_meta}

    # 缓存
    (V5_DIR / "annotated_pdfs").mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    hit_count = len([c for c in chunk_meta if c["pages"]])
    log(f"[ANNOTATE] {stem}: {hit_count}/{len(chunk_texts)} chunks matched, "
        f"pages {sorted(all_pages)}")
    return meta
