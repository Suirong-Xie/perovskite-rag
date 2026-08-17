"""compare_papers — 多论文对比表格生成。"""

import subprocess
from ...core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from ...core.schemas import ToolCall, ToolResult
from .paper_utils import find_pdf_path
from .read_paper import _extract_pdf_text


SCHEMA = {
    "name": "compare_papers",
    "description": (
        "Compare multiple papers side-by-side on key performance metrics. "
        "Given a list of paper source filenames and optional metric names, "
        "extracts structured data from each paper and generates a comparison "
        "table in markdown format. "
        "Use this to answer questions like 'compare the PCE and stability of "
        "these 3 papers' or 'which of these papers achieved the highest Voc'. "
        "Metrics can include: PCE, Voc, Jsc, FF, stability, device_structure, "
        "perovskite_composition, key_innovation."
    ),
    "parameters": {
        "sources": "Comma-separated list of paper source filenames (e.g., 'Nature_2021_xxx.pdf,NatEnergy_2023_yyy.pdf')",
        "metrics": "Comma-separated list of metrics to compare (default: 'PCE,Voc,Jsc,FF,stability')",
    },
}


_COMPARE_PROMPT = (
    "You are analyzing multiple perovskite solar cell papers. Given the extracted "
    "content from each paper, create a comparison table in markdown format.\n\n"
    "For each paper, extract these metrics: {metrics}\n\n"
    "Output:\n"
    "1. A markdown table with papers as rows and metrics as columns\n"
    "2. A brief summary (2-3 sentences) of the key differences\n"
    "3. A recommendation on which paper has the best overall performance\n\n"
    "Use 'N/A' for any metric that cannot be found in the provided content."
)


def execute(arguments: dict) -> tuple:
    sources_raw = arguments.get("sources", "")
    metrics_raw = arguments.get("metrics", "PCE,Voc,Jsc,FF,stability")

    if not sources_raw:
        return (ToolResult(ToolCall("compare_papers", arguments), "", error="sources is required"), None)

    sources = [s.strip() for s in sources_raw.split(",") if s.strip()]
    metrics = [m.strip() for m in metrics_raw.split(",") if m.strip()]

    if len(sources) < 2:
        return (ToolResult(ToolCall("compare_papers", arguments),
                           "Need at least 2 papers to compare. Use extract_data for a single paper."), None)
    if len(sources) > 5:
        sources = sources[:5]

    # 从每篇论文提取文本
    paper_texts = []
    for src in sources:
        pdf_path = find_pdf_path(src)
        if not pdf_path:
            paper_texts.append((src, f"[无全文: {src}]"))
            continue
        try:
            text = _extract_pdf_text(pdf_path)
            paper_texts.append((src, text[:3000]))
        except Exception:
            paper_texts.append((src, f"[提取失败: {src}]"))

    # 构造 LLM prompt
    papers_block = "\n\n---\n\n".join(
        f"### Paper {i+1}: {src}\n{text}"
        for i, (src, text) in enumerate(paper_texts)
    )
    metrics_str = ", ".join(metrics)

    try:
        import requests
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": _COMPARE_PROMPT.format(metrics=metrics_str)},
                    {"role": "user", "content": papers_block},
                ],
                "temperature": 0.1, "max_tokens": 2048,
            },
            timeout=60,
        )
        content = resp.json()["choices"][0]["message"]["content"]
        return (ToolResult(ToolCall("compare_papers", arguments), content), None)
    except Exception as e:
        return (ToolResult(ToolCall("compare_papers", arguments), "", error=str(e)), None)
