"""
PerovskiteGPT V5 — 论文 + PDF API Router
"""
import os
import re
import sqlite3
import pickle
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse
from ..core.config import BASE_DIR, ANNOTATED_DIR
from ..services.agent import find_pdf_fast

router = APIRouter()

# 文章库缓存
_papers_cache = None

PREFIX_JOURNAL = {
    "Nature_": "Nature",
    "NatEnergy_": "Nature Energy",
    "NatMater_": "Nature Materials",
    "NatPhoton_": "Nature Photonics",
    "NatNanotech_": "Nature Nanotechnology",
    "NatComm_": "Nature Communications",
    "arXiv_": "arXiv / Preprint",
}


def file_id_from_source(source: str) -> str:
    return source.replace(".pdf", "")


def find_pdf_path(source: str) -> Optional[str]:
    """委托给 agent 模块的 find_pdf_fast（统一 PDF 查找逻辑）"""
    return find_pdf_fast(source)


def _load_papers_cache():
    global _papers_cache
    if _papers_cache is not None:
        return
    sqlite_path = BASE_DIR / "data/qdrant_data/collection/perovskite_papers/storage.sqlite"
    if not sqlite_path.exists():
        _papers_cache = []
        return
    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.execute("SELECT point FROM points")
    papers = {}
    for row in cursor:
        p = pickle.loads(row[0])
        meta = p.payload.get("metadata", {})
        source = meta.get("source", "")
        path = meta.get("path", "")
        content = p.payload.get("page_content", "")
        if not source:
            continue
        journal = "Other"
        for prefix, name in PREFIX_JOURNAL.items():
            if source.startswith(prefix):
                journal = name
                break
        if source not in papers:
            papers[source] = {
                "id": file_id_from_source(source),
                "source": source,
                "path": path,
                "journal": journal,
                "title_preview": "",
                "chunk_count": 0,
                "year": None,
            }
            if path:
                m = re.search(r'/(\d{4})/', path)
                if m:
                    papers[source]["year"] = int(m.group(1))
        papers[source]["chunk_count"] += 1
        if not papers[source]["title_preview"] and len(content) > 20:
            papers[source]["title_preview"] = content[:150].replace("\n", " ").strip()
    conn.close()
    _papers_cache = sorted(papers.values(), key=lambda x: -(x["year"] or 0))


@router.get("/api/papers")
def list_papers(
    category: Optional[str] = Query(None, description="期刊分类"),
    year: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    _load_papers_cache()
    sorted_papers = _papers_cache
    if year:
        sorted_papers = [p for p in sorted_papers if p["year"] == year]
    total = len(sorted_papers)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "papers": sorted_papers[start:end],
        "categories": [
            "Nature", "Nature Energy", "Nature Materials",
            "Nature Photonics", "Nature Nanotechnology",
            "Nature Communications", "arXiv / Preprint", "Other",
        ],
    }


@router.get("/api/pdf/{file_id}")
def get_pdf(file_id: str, page: Optional[int] = Query(None)):
    # 优先返回 annotated 版本（带高亮标注）
    annotated_path = os.path.join(str(ANNOTATED_DIR), f"{file_id}_annotated.pdf")
    if os.path.exists(annotated_path):
        return FileResponse(
            annotated_path, media_type="application/pdf",
            filename=f"{file_id}.pdf",
            headers={"Content-Disposition": f'inline; filename="{file_id}.pdf"'},
        )
    pdf_path = find_pdf_path(f"{file_id}.pdf")
    if not pdf_path:
        raise HTTPException(404, "PDF not found")
    return FileResponse(
        pdf_path, media_type="application/pdf",
        filename=f"{file_id}.pdf",
        headers={"Content-Disposition": f'inline; filename="{file_id}.pdf"'},
    )


@router.get("/api/sessions/{session_id}/refs")
async def get_session_refs(session_id: str):
    """refs.json 已废弃（v5 PDF 高亮直接嵌入 PDF 文件）"""
    return {}
