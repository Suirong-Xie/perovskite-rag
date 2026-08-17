"""get_citations + get_references — 引用追踪 (Semantic Scholar API)。"""

import urllib.request
import json
from ...core.config import S2_API_KEY
from ...core.schemas import ToolCall, ToolResult

S2_API_URL = "https://api.semanticscholar.org/graph/v1"


def _s2_get(endpoint: str, params: dict) -> dict:
    url = f"{S2_API_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-api-key": S2_API_KEY or ""})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _resolve_paper_id(identifier: str) -> str | None:
    """Resolve DOI or title to S2 paper ID."""
    # Already a paper ID (40-char hex)
    if len(identifier) == 40 and all(c in '0123456789abcdef' for c in identifier):
        return identifier
    # DOI lookup
    try:
        data = _s2_get(f"paper/DOI:{identifier}", {"fields": "paperId"})
        return data.get("paperId")
    except Exception:
        pass
    # Title search
    try:
        data = _s2_get("paper/search", {"query": identifier, "limit": 1, "fields": "paperId"})
        papers = data.get("data", [])
        if papers:
            return papers[0].get("paperId")
    except Exception:
        pass
    return None


def _format_papers(data: list[dict], max_n: int = 10) -> list[str]:
    lines = []
    for i, p in enumerate(data[:max_n]):
        citing = p.get("citingPaper") or p  # citations endpoint wraps in citingPaper
        title = citing.get("title", "N/A")
        year = citing.get("year", "?")
        venue = citing.get("venue", "") or (citing.get("journal", {}) or {}).get("name", "")
        authors = [a.get("name", "") for a in citing.get("authors", [])[:3]]
        authors_str = ", ".join(authors) if authors else "N/A"
        citation_count = citing.get("citationCount", 0)
        paper_id = citing.get("paperId", "")
        lines.append(
            f"[{i+1}] {title}\n"
            f"    Authors: {authors_str}\n"
            f"    Venue: {venue} · {year} · {citation_count} citations\n"
            f"    S2 ID: {paper_id}"
        )
    return lines


# ── get_citations ──

CITATIONS_SCHEMA = {
    "name": "get_citations",
    "description": (
        "Find papers that cite a given paper. Given a Semantic Scholar paper ID, "
        "DOI, or paper title, returns a list of citing papers with titles, authors, "
        "venues, and citation counts. Use this to find follow-up work, understand "
        "a paper's impact, or discover related research. Highly-cited citing papers "
        "are often important subsequent developments."
    ),
    "parameters": {
        "identifier": "Semantic Scholar paper ID (40-char hex), DOI, or paper title",
    },
}


def execute_citations(arguments: dict) -> tuple:
    identifier = arguments.get("identifier", "")
    if not identifier:
        return (ToolResult(ToolCall("get_citations", arguments), "", error="identifier is required"), [])

    try:
        paper_id = _resolve_paper_id(identifier)
        if not paper_id:
            return (ToolResult(ToolCall("get_citations", arguments),
                               f"Could not resolve '{identifier}' to a paper ID."), [])

        data = _s2_get(f"paper/{paper_id}/citations", {
            "fields": "citingPaper.title,citingPaper.authors,citingPaper.venue,citingPaper.year,citingPaper.citationCount,citingPaper.paperId",
            "limit": "20",
        })
        papers = data.get("data", [])
        if not papers:
            return (ToolResult(ToolCall("get_citations", arguments), "No citations found."), [])

        lines = [f"Papers citing '{identifier}' ({len(papers)} total, showing top {min(len(papers), 10)}):"]
        lines.extend(_format_papers(papers))
        return (ToolResult(ToolCall("get_citations", arguments), "\n".join(lines)), papers)
    except Exception as e:
        return (ToolResult(ToolCall("get_citations", arguments), "", error=str(e)), [])


# ── get_references ──

REFERENCES_SCHEMA = {
    "name": "get_references",
    "description": (
        "Find papers that are cited by a given paper (its bibliography). "
        "Given a Semantic Scholar paper ID, DOI, or title, returns its reference list "
        "with titles, authors, venues, and citation counts. "
        "Use this to discover foundational works in a research area."
    ),
    "parameters": {
        "identifier": "Semantic Scholar paper ID (40-char hex), DOI, or paper title",
    },
}


def execute_references(arguments: dict) -> tuple:
    identifier = arguments.get("identifier", "")
    if not identifier:
        return (ToolResult(ToolCall("get_references", arguments), "", error="identifier is required"), [])

    try:
        paper_id = _resolve_paper_id(identifier)
        if not paper_id:
            return (ToolResult(ToolCall("get_references", arguments),
                               f"Could not resolve '{identifier}' to a paper ID."), [])

        data = _s2_get(f"paper/{paper_id}/references", {
            "fields": "citedPaper.title,citedPaper.authors,citedPaper.venue,citedPaper.year,citedPaper.citationCount,citedPaper.paperId",
            "limit": "20",
        })
        papers = data.get("data", [])
        if not papers:
            return (ToolResult(ToolCall("get_references", arguments), "No references found."), [])

        lines = [f"Papers cited by '{identifier}' ({len(papers)} total, showing top {min(len(papers), 10)}):"]
        lines.extend(_format_papers(papers))
        return (ToolResult(ToolCall("get_references", arguments), "\n".join(lines)), papers)
    except Exception as e:
        return (ToolResult(ToolCall("get_references", arguments), "", error=str(e)), [])


SCHEMAS = [CITATIONS_SCHEMA, REFERENCES_SCHEMA]
EXECUTOR_MAP = {
    "get_citations": execute_citations,
    "get_references": execute_references,
}
