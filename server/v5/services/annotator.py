"""
PerovskiteGPT V5 — PDF 高亮标注服务
从 v4 server.py 抽离，在引用的 PDF 中搜索 chunk 文本并添加高亮注释
"""
import os
import re
import shutil
import tempfile
from pathlib import Path
from ..core.config import V5_DIR


def log(msg: str):
    print(f"[V5] {msg}", flush=True)


def annotate_pdf(pdf_path: str, chunk_texts: list) -> str:
    """在 PDF 文件中搜索 chunk 文本并写入原生高亮标注。
    策略：对一个 chunk，用其文本中所有可能的子句和片段搜索 PDF。
    不修改原始 PDF，标注后的副本保存到 annotated_pdfs/ 目录。

    Returns:
        标注后的 PDF 路径，如果未找到任何文本则返回空字符串
    """
    import fitz

    annotated_dir = V5_DIR / "annotated_pdfs"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(pdf_path).stem
    out_path = str(annotated_dir / f"{stem}_annotated.pdf")

    if os.path.exists(out_path):
        log(f"[ANNOTATE] Cache hit for {stem}")
        return out_path

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    shutil.copy2(pdf_path, tmp_path)

    doc = fitz.open(tmp_path)
    total_annotations = 0

    def normalize(s):
        return " ".join(s.replace("\\n", " ").replace("\\r", " ").split())

    for chunk in chunk_texts:
        text = chunk.strip()
        if not text:
            continue

        flat = normalize(text)
        words = flat.split()
        page_rects = {}

        candidates = set()
        # 滑动窗口
        for i in range(0, len(words), max(1, len(words) // 12)):
            frag = " ".join(words[i:i + min(14, len(words) - i)])
            if len(frag) > 20:
                candidates.add(frag[:180])
        # 每个完整句
        for sent in re.split(r'[.!?]', flat):
            s = sent.strip()
            if len(s) > 30:
                candidates.add(s[:200])
        # 完整文本
        candidates.add(flat[:250])

        for q in candidates:
            qq = " ".join(q.split())
            if len(qq) < 20:
                continue
            for pg in range(len(doc)):
                for r in doc[pg].search_for(qq):
                    page_rects.setdefault(pg, []).append((r.y0, r.y1, r.x0, r.x1))

        if not page_rects:
            continue

        # 每页合并重叠/相邻矩形
        for pg, rects in page_rects.items():
            page_h = doc[pg].rect.height
            rects.sort()

            merged = []
            for y0, y1, x0, x1 in rects:
                if y0 < 100 or y1 > page_h - 80:
                    continue
                found = False
                for i, (my0, my1, mx0, mx1) in enumerate(merged):
                    if not (y1 + 15 < my0 or y0 - 15 > my1):
                        merged[i] = (min(my0, y0), max(my1, y1),
                                     min(mx0, x0), max(mx1, x1))
                        found = True
                        break
                if not found:
                    merged.append((y0, y1, x0, x1))

            for y0, y1, x0, x1 in merged:
                w = x1 - x0
                if w < 60:
                    continue
                rect = fitz.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1)
                doc[pg].add_highlight_annot(rect)
                total_annotations += 1

    if total_annotations == 0:
        doc.close()
        os.remove(tmp_path)
        log(f"[ANNOTATE] No text found for {stem}")
        return ""

    doc.save(out_path, incremental=False, garbage=4, deflate=True)
    doc.close()
    os.remove(tmp_path)
    log(f"[ANNOTATE] Saved annotated PDF: {out_path} ({total_annotations} merged highlights)")
    return out_path
