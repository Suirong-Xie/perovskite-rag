"""search_perovskite_database — Perovskite Database Project 结构化查询。"""

import urllib.request
import json
from ...core.schemas import ToolCall, ToolResult

PDP_API = "https://api.perovskitedatabase.com"


SCHEMA = {
    "name": "search_perovskite_database",
    "description": (
        "Search the Perovskite Database Project — a curated database of 40,000+ "
        "perovskite solar cell devices with structured performance data. "
        "Returns device-level metrics: PCE, Voc, Jsc, FF, perovskite composition, "
        "device stack, deposition methods, and stability data. "
        "Use this to query known device performance for specific compositions "
        "or processing conditions. Much more structured than paper search — each "
        "entry is a specific fabricated device with exact values."
    ),
    "parameters": {
        "composition": "Perovskite composition to search (e.g., 'FAPbI3', 'Cs0.1FA0.9PbI3', 'MAPbBr3')",
        "property": "Optional: specific property to focus on (e.g., 'PCE', 'stability', 'Voc')",
        "limit": "Max results (default 10, max 20)",
    },
}


def execute(arguments: dict) -> tuple:
    composition = arguments.get("composition", "")
    if not composition:
        return (ToolResult(ToolCall("search_perovskite_database", arguments), "", error="composition is required"), [])

    limit = min(int(arguments.get("limit", 10)), 20)

    try:
        # Query the Perovskite Database Project API
        params = urllib.parse.urlencode({
            "search": composition,
            "limit": limit,
        })
        url = f"{PDP_API}/v1/search/simple?{params}"

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())

        entries = data if isinstance(data, list) else data.get("data", data.get("results", []))
        if not entries:
            return (ToolResult(ToolCall("search_perovskite_database", arguments),
                               f"No device data found for '{composition}'. The database may not have this exact composition, or the API may be down."), [])

        output_lines = [f"Perovskite Database results for '{composition}' ({len(entries)} devices):"]
        property_filter = arguments.get("property", "")

        for i, entry in enumerate(entries[:limit]):
            # Extract key fields with graceful fallback
            pce = entry.get("pce", entry.get("efficiency", entry.get("PCE", "N/A")))
            voc = entry.get("voc", entry.get("Voc", entry.get("VOC", "N/A")))
            jsc = entry.get("jsc", entry.get("Jsc", entry.get("JSC", "N/A")))
            ff = entry.get("ff", entry.get("FF", entry.get("fill_factor", "N/A")))
            comp = entry.get("perovskite_composition", entry.get("composition", "N/A"))
            device_stack = entry.get("device_stack", entry.get("architecture", "N/A"))
            method = entry.get("deposition_method", entry.get("method", "N/A"))
            stability = entry.get("stability", entry.get("stability_test", "N/A"))

            # Apply property filter if specified
            if property_filter:
                row = f"PCE={pce} Voc={voc} Jsc={jsc} FF={ff} stability={stability}"
                if property_filter.lower() not in row.lower():
                    continue

            output_lines.append(
                f"[{i+1}] PCE: {pce}% | Voc: {voc}V | Jsc: {jsc}mA/cm² | FF: {ff}%\n"
                f"    Composition: {comp}\n"
                f"    Stack: {device_stack}\n"
                f"    Method: {method}\n"
                f"    Stability: {stability}"
            )

        return (ToolResult(ToolCall("search_perovskite_database", arguments), "\n".join(output_lines)), entries[:limit])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return (ToolResult(ToolCall("search_perovskite_database", arguments),
                               f"No data found for '{composition}' in the Perovskite Database."), [])
        return (ToolResult(ToolCall("search_perovskite_database", arguments), "", error=f"API error: HTTP {e.code}"), [])
    except urllib.error.URLError as e:
        return (ToolResult(ToolCall("search_perovskite_database", arguments),
                           "", error=f"API unavailable: {e.reason}"), [])
    except Exception as e:
        return (ToolResult(ToolCall("search_perovskite_database", arguments), "", error=str(e)), [])
