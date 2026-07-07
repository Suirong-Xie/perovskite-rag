#!/usr/bin/env python3
"""
双栏 PDF 文本提取器 — 替代 PyMuPDF 默认 get_text("text")。

问题：默认 get_text("text") 对双栏 PDF 按行扫描，左右栏文本交替混合，
导致后续 LLM 分块收到乱序输入，chunk 质量差，高亮跨栏。

方案：用 get_text("dict") 获取带位置信息的文本块，按 x 坐标分栏，
每栏内按 y 排序，保证阅读顺序正确。
"""

import fitz
from typing import Optional


def _column_x_center(page: fitz.Page) -> float:
    """返回页面的栏分界 x 坐标（两栏时的中间位置）。"""
    # 收集所有文本块的 x 中心
    blocks = page.get_text("blocks")
    if len(blocks) < 3:
        return page.rect.width / 2  # 单栏或空页

    # 取前 10 个非空块的 x 中心，用中位数估计栏分界
    x_centers = sorted([
        (b[0] + b[2]) / 2
        for b in blocks[:20]
        if b[6] == 0 and len(b[4].strip()) > 20  # type 0 = text block
    ])
    if len(x_centers) < 3:
        return page.rect.width / 2

    mid = len(x_centers) // 2
    return x_centers[mid]


def extract_text_column_aware(pdf_path: str, max_chars: int = 12000) -> str:
    """
    从 PDF 提取文本，自动检测并处理双栏排版。

    Args:
        pdf_path: PDF 文件路径
        max_chars: 最大返回字符数（截断）

    Returns:
        按阅读顺序组织的文本
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""

    all_pages = []

    for page in doc:
        blocks = page.get_text("blocks")
        if not blocks:
            continue

        # 检测是否为双栏页面
        col_boundary = _column_x_center(page)
        page_w = page.rect.width

        # 如果分界线距离边缘 < 20% 页宽，视为单栏
        if col_boundary < page_w * 0.2 or col_boundary > page_w * 0.8:
            # 单栏：按 y 排序即可
            text_blocks = []
            for b in blocks:
                if b[6] == 0:  # type 0 = text
                    t = b[4].strip()
                    if len(t) > 5:
                        text_blocks.append((b[1], t))  # (y0, text)
            text_blocks.sort(key=lambda x: (x[0], 0))
            page_text = "\n".join(t for _, t in text_blocks)

        else:
            # 双栏：分左右两栏
            left_col = []
            right_col = []
            for b in blocks:
                if b[6] != 0:
                    continue
                t = b[4].strip()
                if len(t) < 5:
                    continue
                x_center = (b[0] + b[2]) / 2
                if x_center < col_boundary:
                    left_col.append((b[1], t))  # (y0, text)
                else:
                    right_col.append((b[1], t))

            # 每栏内按 y 排序
            left_col.sort(key=lambda x: x[0])
            right_col.sort(key=lambda x: x[0])

            # 按 y 交错合并两栏（同高度时左栏优先）
            left_text = "\n".join(t for _, t in left_col)
            right_text = "\n".join(t for _, t in right_col)

            # 简单策略：先左栏后右栏（大多数期刊的阅读顺序）
            page_text = left_text + "\n" + right_text

        if len(page_text.strip()) >= 50:
            all_pages.append(page_text)

        if sum(len(p) for p in all_pages) >= max_chars:
            break

    doc.close()
    result = "\n\n".join(all_pages)
    return result[:max_chars]


# ── 向后兼容接口 ──

def extract_text(pdf_path: str, max_chars: int = 12000) -> str:
    """兼容旧接口，内部使用分栏感知提取。"""
    return extract_text_column_aware(pdf_path, max_chars)
