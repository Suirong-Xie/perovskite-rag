"""
PerovskiteGPT V5 — PDF 高亮标注服务

策略（v3）：段落定位 + 行级精确高亮
1. 提取 PDF 所有段落及 bbox
2. chunk 文本与每个段落计算词重叠度 — 找到相关段落
3. 在匹配段落内，逐行检查词重叠 — 只高亮匹配行（不跨行、不整段）
"""
import json
import os
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from ..core.config import V5_DIR


def log(msg: str):
    print(f"[V5] {msg}", flush=True)


CHUNK_COLORS = [
    (1.0, 0.85, 0.3), (0.3, 0.85, 1.0), (0.6, 1.0, 0.4),
    (1.0, 0.5, 0.7), (0.7, 0.6, 1.0), (1.0, 0.75, 0.5),
]


def _normalize(text: str) -> str:
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = unicodedata.normalize('NFKC', text)
    # 去标点，只保留字母数字和连字符
    text = re.sub(r'[^\w\s-]', ' ', text)
    return " ".join(text.lower().split())


# ── IDF 词频（惰性构建） ──
_idf_cache = None


def _get_idf():
    """构建全局词频表用于 IDF 加权。"""
    global _idf_cache
    if _idf_cache is not None:
        return _idf_cache
    from ..core.config import SUNNY_RAG_DIR, SEARCH_DATA_VERSION
    data_dir = SUNNY_RAG_DIR / f"data_{SEARCH_DATA_VERSION}"
    txt_path = data_dir / "texts.jsonl"
    if not txt_path.exists():
        _idf_cache = {}
        return _idf_cache
    import json as _json
    doc_count = 0
    df = {}
    with open(txt_path) as f:
        for line in f:
            rec = _json.loads(line)
            words = set(_normalize(rec.get("content", "")).split())
            for w in words:
                df[w] = df.get(w, 0) + 1
            doc_count += 1
    _idf_cache = {w: __import__('math').log(doc_count / (c + 1)) for w, c in df.items()}
    return _idf_cache


def _word_overlap(chunk_words: set, line_words: set) -> float:
    """IDF 加权的词重叠度。稀有词（技术术语）权重高，常见词权重低。"""
    if not line_words:
        return 0.0
    idf = _get_idf()
    inter = chunk_words & line_words
    if not inter:
        return 0.0
    # IDF 加权交集的得分
    weighted_inter = sum(idf.get(w, 1.0) for w in inter)
    weighted_union = sum(idf.get(w, 1.0) for w in (chunk_words | line_words))
    # 覆盖率（段落中有多少行匹配）
    coverage = len(inter) / len(line_words) if line_words else 0
    return (weighted_inter / max(weighted_union, 0.001)) * 0.5 + coverage * 0.5


def annotate_pdf(pdf_path: str, chunk_texts: list) -> tuple[str, dict]:
    """段落定位 + 行级精确高亮。

    1. 提取所有段落（blocks），计算 chunk 与每个段落的词重叠
    2. 对匹配段落，提取逐行文本（lines），逐行检查词重叠
    3. 只高亮匹配行（每行独立 rect，不跨行合并）

    Returns:
        (annotated_pdf_path, highlight_meta)
    """
    import fitz

    annotated_dir = V5_DIR / "annotated_pdfs"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(pdf_path).stem
    out_path = str(annotated_dir / f"{stem}_annotated.pdf")

    if os.path.exists(out_path):
        meta_path = out_path.replace("_annotated.pdf", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f) if os.path.getsize(meta_path) > 0 else {}
            log(f"[ANNOTATE] Cache hit for {stem}")
            return out_path, meta
        return out_path, {}

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    shutil.copy2(pdf_path, tmp_path)

    doc = fitz.open(tmp_path)

    # ── 步骤 1: 提取每页的段落和行级文本 ──
    # 先合并相邻文本块为段落（Nature 等期刊的 PDF 常把一行拆成一个 block）
    page_paragraphs = {}
    page_lines = {}

    for pg in range(len(doc)):
        page = doc[pg]
        page_w = page.rect.width

        # 收集所有文本块
        text_blocks = []
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            text = b[4].strip()
            if len(text) < 5:
                continue
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
            text_blocks.append((y0, y1, x0, x1, text))

        # 按 y 排序
        text_blocks.sort(key=lambda t: t[0])

        # 合并相邻块为段落：同列且 y 间距 < line_height*2
        merged_paras = []
        cur_blocks = []
        for tb in text_blocks:
            y0, y1, x0, x1, text = tb
            if not cur_blocks:
                cur_blocks.append(tb)
            else:
                prev = cur_blocks[-1]
                py1 = prev[1]
                # 同行或很近的行才合并
                line_h = y1 - y0
                if abs(y0 - py1) < max(line_h * 2, 15):
                    # 检查是否同列（x 中心接近）
                    prev_cx = (prev[2] + prev[3]) / 2
                    cur_cx = (x0 + x1) / 2
                    if abs(cur_cx - prev_cx) < page_w * 0.4:
                        cur_blocks.append(tb)
                    else:
                        merged_paras.append(cur_blocks)
                        cur_blocks = [tb]
                else:
                    merged_paras.append(cur_blocks)
                    cur_blocks = [tb]
        if cur_blocks:
            merged_paras.append(cur_blocks)

        # 从合并后的段落提取 paragraph + lines
        page_paragraphs[pg] = []
        page_lines[pg] = []

        for blocks in merged_paras:
            para_text = " ".join(b[4] for b in blocks)
            if len(para_text) < 50:
                continue
            norm_words = set(_normalize(para_text).split())
            if len(norm_words) < 5:
                continue
            # 合并 bbox
            px0 = min(b[2] for b in blocks)
            py0 = min(b[0] for b in blocks)
            px1 = max(b[3] for b in blocks)
            py1 = max(b[1] for b in blocks)
            page_paragraphs[pg].append(([px0, py0, px1, py1], norm_words, para_text))

            # 每个原始 block 作为一行
            for b in blocks:
                y0, y1, x0, x1, text = b
                if len(text.strip()) >= 10:
                    page_lines[pg].append(([x0, y0, x1, y1], text.strip()))

    # ── 步骤 2: 对每个 chunk 找匹配段落 → 逐行高亮 ──
    total_annotations = 0
    all_pages: set[int] = set()
    chunk_meta: list[dict] = []

    for ci, chunk in enumerate(chunk_texts):
        chunk_words = set(_normalize(chunk).split())
        if len(chunk_words) < 5:
            continue

        # 找匹配段落
        para_scores = []
        for pg in range(len(doc)):
            for bbox, para_words, text in page_paragraphs[pg]:
                score = _word_overlap(chunk_words, para_words)
                if score > 0.05:
                    para_scores.append((score, pg, bbox))

        if not para_scores:
            continue

        para_scores.sort(key=lambda x: -x[0])
        # 取 top 3 段落
        top_paras = [(pg, bbox) for _, pg, bbox in para_scores[:3] if _ > 0.06]

        chunk_pages: set[int] = set()
        chunk_count = 0

        # 在匹配段落内，逐行检查匹配
        for pg, para_bbox in top_paras:
            px0, py0, px1, py1 = para_bbox
            # 稍微扩展段落边界以包含边缘行
            py0 -= 2
            py1 += 2

            for line_arr, line_text in page_lines.get(pg, []):
                lx0, ly0, lx1, ly1 = line_arr
                # 检查行是否在段落区域内
                if ly0 < py0 or ly1 > py1:
                    continue
                line_words = set(_normalize(line_text).split())
                if not line_words:
                    continue
                line_score = _word_overlap(chunk_words, line_words)
                if line_score < 0.03:
                    continue

                # 高亮这一行：x 用行自己的 bbox（精准），y 用行的 bbox
                x0 = max(px0, lx0)  # 取段落和行的交集 x
                x1 = min(px1, lx1)
                # 确保矩形有效
                w = x1 - x0
                h = ly1 - ly0
                if w < 20 or h < 5:
                    continue
                rect = fitz.Rect(x0, ly0, x1, ly1)
                try:
                    doc[pg].add_highlight_annot(rect)
                except Exception:
                    continue
                chunk_count += 1
                total_annotations += 1
                chunk_pages.add(pg + 1)
                all_pages.add(pg + 1)

        if chunk_count > 0:
            chunk_meta.append({
                "idx": ci,
                "pages": sorted(chunk_pages),
                "count": chunk_count,
                "text_preview": _normalize(chunk)[:120],
                "text": _normalize(chunk),
            })

    if total_annotations == 0:
        doc.close()
        os.remove(tmp_path)
        log(f"[ANNOTATE] No matches for {stem}")
        return "", {}

    doc.save(out_path, incremental=False, garbage=4, deflate=True)
    doc.close()
    os.remove(tmp_path)

    meta = {"pages": sorted(all_pages), "chunks": chunk_meta}
    meta_path = out_path.replace("_annotated.pdf", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    log(f"[ANNOTATE] {stem}: {total_annotations} line highlights "
        f"on pages {sorted(all_pages)} from {len(chunk_meta)}/{len(chunk_texts)} chunks")
    return out_path, meta
