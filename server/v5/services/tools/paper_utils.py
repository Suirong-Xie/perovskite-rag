"""PDF 查找工具 — 供 read_paper、extract_data、compare 等工具共享使用。"""

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


def find_pdf_fast(source: str, journal_name: str = "") -> Optional[str]:
    """按 source 文件名 + journal_name 查找 PDF。
    先用 journal_name 做 O(1) 查找 journals_pdf/{journal}/，
    fallback 到扫描 papers_pdf/{year}/{month}/。
    """
    # 1. journals_pdf/{journal}/{source} — O(1) with journal_name
    journal_dir_name = JOURNAL_DIR_MAP.get(journal_name, "")
    if journal_dir_name:
        pdf_file = JOURNALS_PDF_DIR / journal_dir_name / source
        if pdf_file.exists():
            return str(pdf_file)
    # 2. journals_pdf/*/ — scan all journal dirs (fallback)
    if JOURNALS_PDF_DIR.exists():
        for journal_dir in JOURNALS_PDF_DIR.iterdir():
            if not journal_dir.is_dir():
                continue
            if journal_dir.name == journal_dir_name:
                continue  # already checked above
            pdf_file = journal_dir / source
            if pdf_file.exists():
                return str(pdf_file)
    # 3. papers_pdf/{year}/{month}/{source} — 旧 arXiv 数据
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
