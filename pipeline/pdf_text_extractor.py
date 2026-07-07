#!/usr/bin/env python3
"""
双栏 PDF 结构化提取器
- 文本提取：按 x 坐标分栏，每栏内按 y 排序，保证阅读顺序正确
- bbox 输出：每个文本块带精确坐标，供前端 pdf.js CSS overlay 高亮使用

输出格式（extract_structured）：
{
  "pages": [
    {
      "page": 1,
      "width": 595.0, "height": 842.0,
      "blocks": [
        {"bbox": [x0,y0,x1,y1], "text": "...", "column": 0},
        ...
      ]
    }
  ]
}
"""

import fitz
from typing import Optional


def _detect_columns(page: fitz.Page) -> tuple[Optional[float], float]:
    """检测页面分栏。返回 (分界线_x, 页宽)。单栏返回 (None, 页宽)。"""
    page_w = page.rect.width
    blocks = page.get_text("blocks")
    if len(blocks) < 3:
        return None, page_w

    x_centers = sorted([
        (b[0] + b[2]) / 2
        for b in blocks
        if b[6] == 0 and len(b[4].strip()) > 15
    ])
    if len(x_centers) < 5:
        return None, page_w

    # 方法 1: 检查中位数的分布 —— 如果多数块分属左右两个簇，则是双栏
    mid = page_w / 2
    left = [x for x in x_centers if x < mid]
    right = [x for x in x_centers if x >= mid]
    if not left or not right:
        return None, page_w

    # 如果左侧簇的右边界和右侧簇的左边界之间有明显间隙（>10% 页宽），则是双栏
    gap = min(right) - max(left)
    if gap > page_w * 0.08:
        boundary = (max(left) + min(right)) / 2
        return boundary, page_w

    # 方法 2: 直方图检测 —— 在 x 轴上以 5% 页宽为单位建直方图，找空 bin
    bin_w = page_w * 0.05
    bins = {}
    for x in x_centers:
        bi = int(x / bin_w)
        bins[bi] = bins.get(bi, 0) + 1
    # 在 30%-70% 范围内找连续 2 个以上的空 bin（= 栏间隙）
    empty_streak = 0
    gap_bin = None
    for bi in range(int(0.3 * page_w / bin_w), int(0.7 * page_w / bin_w)):
        if bins.get(bi, 0) == 0:
            empty_streak += 1
            if empty_streak >= 2 and gap_bin is None:
                gap_bin = bi
        else:
            empty_streak = 0
            gap_bin = None
    if gap_bin is not None:
        boundary = gap_bin * bin_w + bin_w  # 间隙中心
        return boundary, page_w

    return None, page_w


def extract_structured(pdf_path: str, max_pages: int = 20) -> dict:
    """提取 PDF 的结构化文本块（带 bbox）。"""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return {"pages": []}

    pages_out = []
    for pi in range(min(len(doc), max_pages)):
        page = doc[pi]
        blocks = page.get_text("blocks")
        if not blocks:
            continue

        col_boundary, page_w = _detect_columns(page)

        blocks_out = []
        for b in blocks:
            if b[6] != 0:  # 非文本块（图片等）
                continue
            text = b[4].strip()
            if len(text) < 5:
                continue

            bbox = [b[0], b[1], b[2], b[3]]  # x0, y0, x1, y1
            x_center = (b[0] + b[2]) / 2

            # 确定所属栏
            col = 0
            if col_boundary is not None:
                col = 0 if x_center < col_boundary else 1

            blocks_out.append({
                "bbox": bbox,
                "text": text,
                "column": col,
            })

        # 按栏和 y 排序
        if col_boundary is not None:
            blocks_out.sort(key=lambda b: (b["column"], b["bbox"][1]))
        else:
            blocks_out.sort(key=lambda b: b["bbox"][1])

        pages_out.append({
            "page": pi + 1,
            "width": page.rect.width,
            "height": page.rect.height,
            "blocks": blocks_out,
        })

    doc.close()
    return {"pages": pages_out}


def extract_text_column_aware(pdf_path: str, max_chars: int = 12000) -> str:
    """从 PDF 提取文本，自动检测并处理双栏排版（向后兼容）。"""
    data = extract_structured(pdf_path)
    all_pages = []
    total = 0
    for p in data["pages"]:
        page_text = "\n".join(b["text"] for b in p["blocks"])
        if len(page_text.strip()) >= 50:
            all_pages.append(page_text)
        total += len(page_text)
        if total >= max_chars:
            break
    return "\n\n".join(all_pages)[:max_chars]


def extract_text(pdf_path: str, max_chars: int = 12000) -> str:
    """兼容旧接口。"""
    return extract_text_column_aware(pdf_path, max_chars)
