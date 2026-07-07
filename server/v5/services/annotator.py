"""
PerovskiteGPT V5 — PDF 高亮标注服务

基于 PyMuPDF search_for() 的精确文本定位，不做自定义词级匹配和手动合并。
search_for() 返回的 Rect 天然是逐行精确的，自动处理换行、分栏、连字符。
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


# ── 多 chunk 配色（HSL 色相均匀分布，半透明） ──
CHUNK_COLORS = [
    (1.0, 0.85, 0.3),   # gold
    (0.3, 0.85, 1.0),   # sky blue
    (0.6, 1.0, 0.4),    # green
    (1.0, 0.5, 0.7),    # pink
    (0.7, 0.6, 1.0),    # lavender
    (1.0, 0.75, 0.5),   # orange
]


def _normalize_for_search(text: str) -> str:
    """将 chunk 文本归一化为可在 PDF 中搜索的形式。

    处理 LLM 语义分块时引入的格式化差异：
      - markdown 斜体/粗体: _n–i–p_ → n–i–p
      - 标题标记: ## Title → Title
      - 括号内多余空格: ( x ) → (x)
      - Unicode 归一化: 全角→半角, 连字→分开
    """
    # 去掉 markdown 标记
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # 括号空格
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    # Unicode 归一化
    text = unicodedata.normalize('NFKC', text)
    # 合并空白
    text = " ".join(text.split())
    return text


def _generate_queries(chunk_text: str) -> list[str]:
    """从 chunk 文本生成一组 PDF 搜索查询。

    策略：
      1. 整体文本（归一化后，≤300 chars）
      2. 句子级（智能分句，跳过小数点如 26.7%）
      3. 短语级（逗号、分号分割）

    查询按长度降序排列——长查询匹配更精确（更少但更准确的 rects）。
    """
    text = _normalize_for_search(chunk_text)
    queries = []

    # 1. 整体文本（限制 300 chars 避免因换行/特殊字符差异全量失败）
    if len(text) >= 20:
        queries.append(text[:300])

    # 2. 句子级分割：跳过小数点（如 26.7%, 3.8 eV）
    for sent in re.split(r'(?<!\d)[.!?](?!\d)', text):
        s = sent.strip()
        if 30 <= len(s) <= 200:
            queries.append(s)

    # 3. 短语级：逗号、分号分割
    for phrase in re.split(r'[,;]', text):
        p = phrase.strip()
        if 40 <= len(p) <= 150:
            queries.append(p)

    # 去重保序，长查询优先
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    unique.sort(key=len, reverse=True)
    return unique


def annotate_pdf(pdf_path: str, chunk_texts: list) -> tuple[str, dict]:
    """在 PDF 中搜索 chunk 文本并添加高亮标注。

    使用 PyMuPDF search_for() 做精确文本定位，rects 直接用作高亮矩形。
    不做词级匹配、不做手动合并——search_for 已经处理了换行和分栏。

    标注后的副本保存到 annotated_pdfs/ 目录。

    Returns:
        (annotated_pdf_path, highlight_meta)
        - annotated_pdf_path: 标注后的 PDF 路径，无匹配则为空字符串
        - highlight_meta: {"pages": [1,2,3], "chunks": [{"idx": 0, "pages": [1], "count": 5, "text_preview": "..."}, ...]}
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
        log(f"[ANNOTATE] Cache hit for {stem} (no meta)")
        return out_path, {}

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    shutil.copy2(pdf_path, tmp_path)

    doc = fitz.open(tmp_path)
    total_annotations = 0
    all_pages: set[int] = set()
    chunk_meta: list[dict] = []

    for ci, chunk in enumerate(chunk_texts):
        queries = _generate_queries(chunk)
        chunk_pages: set[int] = set()
        chunk_count = 0

        for q in queries:
            if len(q) < 20:
                continue
            for pg in range(len(doc)):
                results = doc[pg].search_for(q)
                if results:
                    chunk_pages.add(pg + 1)  # 转为 1-based 页码
                    all_pages.add(pg + 1)
                    for rect in results:
                        if rect.y0 > 80 and rect.y1 < doc[pg].rect.height - 60 and rect.width > 30:
                            doc[pg].add_highlight_annot(rect)
                            chunk_count += 1
                            total_annotations += 1

        if chunk_count > 0:
            chunk_meta.append({
                "idx": ci,
                "pages": sorted(chunk_pages),
                "count": chunk_count,
                "text_preview": _normalize_for_search(chunk)[:120],
                "text": _normalize_for_search(chunk),  # 完整文本供前端 pdf.js 搜索
            })

    if total_annotations == 0:
        doc.close()
        os.remove(tmp_path)
        log(f"[ANNOTATE] No text found for {stem}")
        return "", {}

    doc.save(out_path, incremental=False, garbage=4, deflate=True)
    doc.close()
    os.remove(tmp_path)

    meta = {"pages": sorted(all_pages), "chunks": chunk_meta}
    # 保存 metadata 供缓存命中时使用
    meta_path = out_path.replace("_annotated.pdf", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    log(f"[ANNOTATE] {stem}: {total_annotations} highlights on pages {sorted(all_pages)} "
        f"from {len(chunk_meta)}/{len(chunk_texts)} chunks")
    return out_path, meta
