"""search_semantic_scholar — Semantic Scholar API 搜索。"""

from ...core.schemas import ToolCall, ToolResult
from ..semantic_scholar_service import search_semantic_scholar as _search_s2


SCHEMA = {
    "name": "search_semantic_scholar",
    "description": (
        "Search Semantic Scholar, a massive academic database covering 200M+ "
        "published papers across all scientific disciplines. Returns titles, "
        "abstracts, authors, journal/venue, publication year, and citation "
        "counts. Citation count is a strong quality signal — highly-cited "
        "papers are usually landmark works. "
        "Use this for: discovering papers beyond our local Nature collection, "
        "finding the most influential papers on a topic (sort by citations), "
        "or searching for papers from non-Nature journals (Science, ACS, "
        "Wiley, RSC, etc.). "
        "Complements search_papers (local full-text) and search_arxiv (preprints)."
    ),
    "parameters": {
        "query": "English search query (e.g., 'perovskite stability under humidity')",
        "max_results": "Number of results (default 5, max 10)",
        "year_min": "Optional: earliest publication year (e.g., 2022)",
        "year_max": "Optional: latest publication year (e.g., 2026)",
    },
}


def execute(arguments: dict) -> tuple:
    query = arguments.get("query", "")
    max_results = min(int(arguments.get("max_results", 5)), 10)
    year_min = arguments.get("year_min")
    year_max = arguments.get("year_max")

    if year_min:
        try: year_min = int(year_min)
        except (ValueError, TypeError): year_min = None
    if year_max:
        try: year_max = int(year_max)
        except (ValueError, TypeError): year_max = None

    if not query:
        return (ToolResult(ToolCall("search_semantic_scholar", arguments), "", error="query is required"), [])

    results = _search_s2(query, max_results=max_results, year_min=year_min, year_max=year_max)
    if not results:
        return (ToolResult(ToolCall("search_semantic_scholar", arguments), "No results on Semantic Scholar."), [])

    output_lines = [f"Found {len(results)} papers on Semantic Scholar for '{query}':\n"]
    for i, r in enumerate(results):
        authors_str = ", ".join(r.get("authors", [])[:3])
        if len(r.get("authors", [])) > 3:
            authors_str += " et al."
        year_citations = f"{r.get('year', 'N/A')} · {r.get('citationCount', 0)} citations"
        output_lines.append(
            f"[{i+1}] {r.get('title', 'N/A')}\n"
            f"    Authors: {authors_str}\n"
            f"    Venue: {r.get('venue', 'N/A')} · {year_citations}\n"
            f"    DOI: {r.get('doi', 'N/A')}\n"
            f"    Abstract: {r.get('abstract', '(not available)')[:500]}"
        )
    return (ToolResult(ToolCall("search_semantic_scholar", arguments), "\n".join(output_lines)), results)
