"""search_arxiv — arXiv 预印本搜索。"""

from ...core.schemas import ToolCall, ToolResult
from ..arxiv_service import search_arxiv as _search_arxiv


SCHEMA = {
    "name": "search_arxiv",
    "description": (
        "Search arXiv for perovskite solar cell preprints. Returns titles, "
        "abstracts, authors, publication dates, and PDF download links. "
        "arXiv has 160,000+ perovskite-related preprints, often covering "
        "the latest research (2024-2026) before journal publication. "
        "Use this alongside search_papers to find recent work not yet in "
        "the local database."
    ),
    "parameters": {
        "query": "English search query (e.g., 'inverted perovskite stability 2024')",
        "max_results": "Number of results to return (default 5, max 10)",
    },
}


def execute(arguments: dict) -> tuple:
    query = arguments.get("query", "")
    max_results = min(int(arguments.get("max_results", 5)), 10)
    if not query:
        return (ToolResult(ToolCall("search_arxiv", arguments), "", error="query is required"), [])

    results = _search_arxiv(query, max_results=max_results)
    if not results:
        return (ToolResult(ToolCall("search_arxiv", arguments), "No results on arXiv."), [])

    output_lines = [f"Found {len(results)} arXiv papers for '{query}':\n"]
    for i, r in enumerate(results):
        authors_str = ", ".join(r.get("authors", [])[:3])
        if len(r.get("authors", [])) > 3:
            authors_str += " et al."
        output_lines.append(
            f"[{i+1}] {r.get('title', 'N/A')}\n"
            f"    Authors: {authors_str}\n"
            f"    Published: {r.get('published', 'N/A')}\n"
            f"    arXiv ID: {r.get('arxiv_id', 'N/A')}\n"
            f"    PDF: {r.get('pdf_url', 'N/A')}\n"
            f"    Category: {r.get('category', 'N/A')}\n"
            f"    Abstract: {r.get('summary', '')[:500]}"
        )
    return (ToolResult(ToolCall("search_arxiv", arguments), "\n".join(output_lines)), results)
