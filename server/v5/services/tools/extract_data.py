"""extract_data — LLM 驱动的论文结构化数据提取。"""

import subprocess
import requests
from ...core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from ...core.schemas import ToolCall, ToolResult
from .paper_utils import find_pdf_path


SCHEMA = {
    "name": "extract_data",
    "description": (
        "Extract structured performance data from a perovskite solar cell paper. "
        "Returns key metrics: PCE (power conversion efficiency), Voc, Jsc, FF, "
        "device architecture (n-i-p or p-i-n), perovskite composition, "
        "and stability test results if available."
    ),
    "parameters": {
        "source": "Paper source filename (e.g., 'Nature_2021_xxx.pdf')",
    },
}


_EXTRACT_PROMPT = (
    "Extract perovskite solar cell performance data from this paper excerpt. "
    "Return ONLY a JSON object with these fields (use null if not found): "
    "pce (max power conversion efficiency %), voc (V), jsc (mA/cm²), ff (%), "
    "device_structure (n-i-p or p-i-n), perovskite_composition, "
    "stability (hours tested, conditions), key_innovation (one sentence summary)."
)


def execute(arguments: dict) -> tuple:
    source = arguments.get("source", "")
    if not source:
        return (ToolResult(ToolCall("extract_data", arguments), "", error="source is required"), None)

    pdf_path = find_pdf_path(source)
    if not pdf_path:
        return (ToolResult(ToolCall("extract_data", arguments), f"PDF not found: {source}", error="PDF not found"), None)

    try:
        proc = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return (ToolResult(ToolCall("extract_data", arguments), "", error=f"pdftotext error: {proc.stderr[:200]}"), None)
        text = proc.stdout[:10000]

        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": _EXTRACT_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.1, "max_tokens": 512,
            },
            timeout=30,
        )
        content = resp.json()["choices"][0]["message"]["content"]
        return (ToolResult(ToolCall("extract_data", arguments), content), {"source": source, "content": content[:600]})
    except Exception as e:
        return (ToolResult(ToolCall("extract_data", arguments), "", error=str(e)), {})
