"""analyze_perovskite + search_materials — 材料科学工具。"""

from ...core.schemas import ToolCall, ToolResult
from ..materials_service import analyze_perovskite as _analyze, search_materials_project as _search


# ── analyze_perovskite ──

ANALYZE_SCHEMA = {
    "name": "analyze_perovskite",
    "description": (
        "Analyze a perovskite crystal structure using Pymatgen. "
        "Given a chemical formula (e.g., 'MAPbI3', 'Cs0.1FA0.9PbI3', 'CsSnI3'), "
        "computes the Goldschmidt tolerance factor, octahedral factor, and predicts "
        "the crystal system and structural stability. "
        "Use this to quickly assess whether a proposed perovskite composition "
        "is structurally viable before doing DFT or searching papers."
    ),
    "parameters": {
        "formula": "Perovskite chemical formula, ABX3 type (e.g., 'MAPbI3', 'Cs0.2FA0.8PbBr3')",
    },
}


def execute_analyze(arguments: dict) -> tuple:
    formula = arguments.get("formula", "")
    if not formula:
        return (ToolResult(ToolCall("analyze_perovskite", arguments), "", error="formula is required"), None)

    try:
        result = _analyze(formula)
        if "error" in result:
            return (ToolResult(ToolCall("analyze_perovskite", arguments), result["error"], error=result["error"]), None)

        output_lines = [
            f"Structure analysis for {result['formula']} ({result['reduced_formula']}):",
            f"  A-site: {', '.join(result['A_site'])} (R_A = {result['r_A']:.4f} Å)",
            f"  B-site: {', '.join(result['B_site'])} (R_B = {result['r_B']:.4f} Å)",
            f"  X-site: {', '.join(result['X_site'])} (R_X = {result['r_X']:.4f} Å)",
            f"  Goldschmidt tolerance factor t = {result['tolerance_factor']:.4f}",
            f"  Octahedral factor μ = {result['octahedral_factor']:.4f}",
            f"  Predicted crystal system: {result['predicted_crystal_system']}",
            f"  Likely stable: {'YES' if result['likely_stable'] else 'NO'}",
            f"  Issues: {'; '.join(result['issues'])}",
        ]
        return (ToolResult(ToolCall("analyze_perovskite", arguments), "\n".join(output_lines)), None)
    except Exception as e:
        return (ToolResult(ToolCall("analyze_perovskite", arguments), "", error=str(e)), None)


# ── search_materials ──

SEARCH_MAT_SCHEMA = {
    "name": "search_materials",
    "description": (
        "Search the Materials Project database (140k+ inorganic materials computed "
        "with DFT) for known data on a chemical formula. Returns bandgap, "
        "formation energy, crystal structure (cubic/tetragonal/orthorhombic), "
        "and previous experimental or computational data. "
        "Use this to check what is already known about a composition before "
        "running your own calculations."
    ),
    "parameters": {
        "formula": "Chemical formula to search (e.g., 'PbTiO3', 'CsPbI3')",
    },
}


def execute_search_materials(arguments: dict) -> tuple:
    formula = arguments.get("formula", "")
    if not formula:
        return (ToolResult(ToolCall("search_materials", arguments), "", error="formula is required"), None)

    try:
        result = _search(formula)
        if "error" in result:
            return (ToolResult(ToolCall("search_materials", arguments), result["error"]), None)

        if not result.get("results"):
            return (ToolResult(ToolCall("search_materials", arguments), result.get("message", "No results")), None)

        lines = [f"Materials Project results for '{result['formula']}':"]
        lines.append(f"Source: {result.get('source', 'MP')}")
        for r in result["results"]:
            lines.append(
                f"  [{r['material_id']}] {r['formula']} | "
                f"Bandgap: {r['band_gap_eV']} eV | "
                f"Formation energy: {r['formation_energy_eV_per_atom']} eV/atom | "
                f"Crystal: {r['crystal_system']}"
            )
        return (ToolResult(ToolCall("search_materials", arguments), "\n".join(lines)), None)
    except Exception as e:
        return (ToolResult(ToolCall("search_materials", arguments), "", error=str(e)), None)


SCHEMAS = [ANALYZE_SCHEMA, SEARCH_MAT_SCHEMA]
EXECUTOR_MAP = {
    "analyze_perovskite": execute_analyze,
    "search_materials": execute_search_materials,
}
