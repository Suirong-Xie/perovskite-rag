"""search_papers — 本地向量库搜索 (Nature + S2)。"""

from ...core.schemas import ToolCall, ToolResult
from ..retrieval import search_papers


SCHEMA = {
    "name": "search_papers",
    "description": (
        "Search the perovskite solar cell research paper database for papers "
        "matching a scientific query. Returns ranked results with journal name, "
        "source filename, and content snippets. Use this to find relevant papers "
        "before answering."
    ),
    "parameters": {
        "query": "English search query string (e.g., 'inverted perovskite solar cell stability')",
        "top_k": "Number of results to return (default 5, max 10)",
    },
}


def execute(arguments: dict) -> tuple:
    query = arguments.get("query", "")
    top_k = min(int(arguments.get("top_k", 5)), 10)
    if not query:
        return (ToolResult(ToolCall("search_papers", arguments), "", error="query is required"), [])

    results = search_papers(query, top_k=top_k)
    if not results:
        return (ToolResult(ToolCall("search_papers", arguments), "No results found."), [])

    # 延迟导入避免循环: retrieval → tools.__init__ → search_local → retrieval
    from .paper_utils import find_pdf_path as _find_pdf
    for r in results:
        src = r.get("source", "")
        r["has_pdf"] = bool(src and _find_pdf(src))

    output_lines = [f"Found {len(results)} results for '{query}':\n"]
    for i, r in enumerate(results):
        file_id = r.get("source", "").replace(".pdf", "")
        has_pdf = r.get("has_pdf", True)
        pdf_tag = "📄 PDF" if has_pdf else "🔗 DOI only"
        doi = r.get("_s2_doi", "")
        output_lines.append(
            f"[{i+1}] {r.get('journal_name', 'Unknown')} | "
            f"Sim: {r.get('similarity', 0):.3f} | "
            f"{pdf_tag} | "
            f"Source: {r.get('source', 'N/A')}\n"
            f"    Content: {r.get('content', '')[:600]}"
        )
        if not has_pdf and doi:
            output_lines[-1] += f"\n    DOI: https://doi.org/{doi}"
    return (ToolResult(ToolCall("search_papers", arguments), "\n".join(output_lines)), results)
