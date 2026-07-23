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

    output_lines = [f"Found {len(results)} results for '{query}':\n"]
    for i, r in enumerate(results):
        file_id = r.get("source", "").replace(".pdf", "")
        output_lines.append(
            f"[{i+1}] {r.get('journal_name', 'Unknown')} | "
            f"Similarity: {r.get('similarity', 0):.3f} | "
            f"Source: {r.get('source', 'N/A')} | "
            f"File ID: {file_id}\n"
            f"    Content: {r.get('content', '')[:600]}"
        )
    return (ToolResult(ToolCall("search_papers", arguments), "\n".join(output_lines)), results)
