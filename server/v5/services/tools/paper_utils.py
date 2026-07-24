"""PDF 查找工具 — 供 read_paper、extract_data、compare 等工具共享使用。"""

import re
from typing import Optional
from pathlib import Path
from ...core.config import PAPERS_DIR, JOURNALS_PDF_DIR

JOURNAL_DIR_MAP = {
    "Nature": "Nature",
    "NatEnergy": "NatEnergy",
    "NatMater": "NatMater",
    "NatPhoton": "NatPhoton",
    "NatNanotech": "NatNanotech",
    "NatComm": "NatComm",
    "Science": "Science",
}

# 缓存 journals_pdf 目录下的所有 PDF 文件名 → 相对路径
_pdf_index: Optional[dict[str, str]] = None


def _build_pdf_index() -> dict[str, str]:
    """构建 PDF 文件名 → 完整路径的索引（惰性初始化，只做一次）。"""
    global _pdf_index
    if _pdf_index is not None:
        return _pdf_index
    _pdf_index = {}
    if JOURNALS_PDF_DIR.exists():
        for journal_dir in JOURNALS_PDF_DIR.iterdir():
            if not journal_dir.is_dir():
                continue
            for pdf_file in journal_dir.iterdir():
                if pdf_file.suffix.lower() == ".pdf":
                    _pdf_index[pdf_file.name] = str(pdf_file)
    return _pdf_index


def _extract_doi_from_source(source: str) -> Optional[str]:
    """从 chunk source 中提取 DOI 文件名。

    chunk source 格式:
      - Nature: "Nature_2021_s41467-021-26121-1.pdf" (无 DOI，保留原名)
      - S2:     "{Journal}_10.XXXX_rest.pdf" → DOI = "10.XXXX_rest.pdf"
      - S2:     "s2:paperId" → 无 DOI
      - S2:     "Unknown_Journal_hash.pdf" → 无 DOI
    """
    # s2:paperId 格式 — 没有 PDF
    if source.startswith("s2:"):
        return None
    # 按 _10. 分割提取 DOI 部分
    # (Unknown_Journal_ 前缀也可能有有效 DOI, 不跳过)
    m = re.search(r'_10\.', source)
    if m:
        return source[m.start() + 1:]  # "10.XXXX_rest.pdf"
    # 无 DOI 格式 (如 Unknown_Journal_hash.pdf, Nature_2021_xxx.pdf)
    # 返回原名，靠精确匹配
    return source


def find_pdf_fast(source: str, journal_name: str = "") -> Optional[str]:
    """按 source 文件名 + journal_name 查找 PDF。

    匹配策略（按优先级）:
    1. journals_pdf/{journal}/{source} — O(1) 精确匹配
    2. journals_pdf/*/ — 扫描所有期刊目录做精确匹配
    3. DOI 提取匹配 — S2 chunk source 提取 DOI 后在 PDF 库中查找
    4. papers_pdf/{year}/{month}/{source} — 旧 arXiv 数据
    """
    # 1. journals_pdf/{journal}/{source} — O(1) with journal_name
    journal_dir_name = JOURNAL_DIR_MAP.get(journal_name, "")
    if journal_dir_name:
        pdf_file = JOURNALS_PDF_DIR / journal_dir_name / source
        if pdf_file.exists():
            return str(pdf_file)
    # 2. journals_pdf/*/ — scan all journal dirs for exact match (fallback)
    if JOURNALS_PDF_DIR.exists():
        for journal_dir in JOURNALS_PDF_DIR.iterdir():
            if not journal_dir.is_dir():
                continue
            if journal_dir.name == journal_dir_name:
                continue  # already checked above
            pdf_file = journal_dir / source
            if pdf_file.exists():
                return str(pdf_file)
    # 3. DOI 提取匹配 — S2 chunk source  →  磁盘 DOI 文件名
    doi_filename = _extract_doi_from_source(source)
    if doi_filename and doi_filename != source:
        pdf_index = _build_pdf_index()
        if doi_filename in pdf_index:
            return pdf_index[doi_filename]
    # 4. papers_pdf/{year}/{month}/{source} — 旧 arXiv 数据
    for year_dir in sorted(PAPERS_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            pdf_file = month_dir / source
            if pdf_file.exists():
                return str(pdf_file)
    return None


def find_pdf_path(source: str) -> Optional[str]:
    """按 source 文件名查找 PDF（兼容旧接口，委托给 find_pdf_fast）"""
    return find_pdf_fast(source)
