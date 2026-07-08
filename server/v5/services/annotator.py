"""
PerovskiteGPT V5 — PDF 高亮标注服务

策略（v2）：段落级语义匹配
1. 提取 PDF 所有段落及 bbox
2. chunk 文本与每个段落计算词重叠度
3. 高亮最佳匹配段落（整段高亮，不用碎片匹配）
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
    """归一化文本：去 markdown、Unicode 归一化、小写。"""
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = unicodedata.normalize('NFKC', text)
    return " ".join(text.lower().split())


def _word_overlap(chunk_words: set, para_words: set) -> float:
    """计算词重叠度：Jaccard + 覆盖率加权。"""
    if not para_words:
        return 0.0
    intersection = chunk_words & para_words
    # Jaccard: 交集 / 并集
    union = chunk_words | para_words
    jaccard = len(intersection) / len(union) if union else 0
    # 覆盖率: 段落中有多少词被 chunk 命中
    coverage = len(intersection) / len(para_words) if para_words else 0
    return jaccard * 0.4 + coverage * 0.6


def annotate_pdf(pdf_path: str, chunk_texts: list) -> tuple[str, dict]:
    """段落级语义高亮。

    1. 提取 PDF 所有段落及 bbox
    2. 每个 chunk 与各段落计算词重叠度
    3. 高亮 top-N 匹配段落

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
            log(f"[ANNOTATE] Cache hit for {stem} ({meta.get('pages', [])})")
            return out_path, meta
        return out_path, {}

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    shutil.copy2(pdf_path, tmp_path)

    doc = fitz.open(tmp_path)

    # ── 步骤 1: 提取 PDF 所有段落及 bbox ──
    paragraphs = []  # [(page_num, bbox, words_set, text), ...]
    for pg in range(len(doc)):
        page = doc[pg]
        blocks = page.get_text("blocks")
        for b in blocks:
            if b[6] != 0:  # 非文本块
                continue
            text = b[4].strip()
            if len(text) < 80:  # 太短（标题、图表标签等），跳过
                continue
            norm_words = set(_normalize(text).split())
            if len(norm_words) < 5:
                continue
            bbox = [b[0], b[1], b[2], b[3]]
            paragraphs.append((pg, bbox, norm_words, text))

    if not paragraphs:
        log(f"[ANNOTATE] No paragraphs found in {stem}")
        doc.close()
        os.remove(tmp_path)
        return "", {}

    # ── 步骤 2: 对每个 chunk 找最佳匹配段落 ──
    total_annotations = 0
    all_pages: set[int] = set()
    chunk_meta: list[dict] = []

    for ci, chunk in enumerate(chunk_texts):
        chunk_words = set(_normalize(chunk).split())
        if len(chunk_words) < 5:
            continue

        # 计算每个段落与 chunk 的词重叠度
        scored = []
        for pg, bbox, para_words, text in paragraphs:
            score = _word_overlap(chunk_words, para_words)
            if score > 0.05:  # 最低阈值
                scored.append((score, pg, bbox, text))

        if not scored:
            continue

        scored.sort(key=lambda x: -x[0])

        # 取 top 5 段落（只要 score > 0.06 即有一半以上的词重叠）
        top = [s for s in scored[:5] if s[0] > 0.06]
        chunk_pages: set[int] = set()
        chunk_count = 0

        for score, pg, bbox, text in top:
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            doc[pg].add_highlight_annot(rect)
            chunk_pages.add(pg + 1)
            chunk_count += 1
            total_annotations += 1
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

    log(f"[ANNOTATE] {stem}: {total_annotations} paragraph highlights "
        f"on pages {sorted(all_pages)} from {len(chunk_meta)}/{len(chunk_texts)} chunks")
    return out_path, meta
