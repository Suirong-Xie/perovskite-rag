"""
PerovskiteGPT v1.5 — 材料科学服务
基于 Pymatgen 的钙钛矿晶体结构分析工具

提供的 Agent 工具:
  - analyze_perovskite:  容忍因子、八面体因子、键价和分析、稳定性评估
  - search_materials:    查询 Materials Project 数据库（可选，需要 MP_API_KEY）
"""

import math
import os
from typing import Optional

from pymatgen.core import Composition, Element, Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_prediction.substitution_probability import (
    SubstitutionProbability,
)


# ── Shannon 离子半径 (Å) ──
# 钙钛矿中关键的配位数: A位=12, B位=6, X位=6 (但卤素在钙钛矿中实际配位=2, 习惯用6)
SHANNON_RADII = {
    # A-site cations (CN=12)
    ("Cs", 12):  1.88,
    ("Rb", 12):  1.72,
    ("K", 12):   1.64,
    ("Na", 12):  1.39,
    ("MA", 12):  2.17,  # methylammonium, 文献估算值
    ("FA", 12):  2.53,  # formamidinium, 文献估算值
    ("EA", 12):  2.74,  # ethylammonium
    ("GA", 12):  2.78,  # guanidinium
    # B-site cations (CN=6)
    ("Pb", 6):   1.19,
    ("Sn", 6):   1.18,
    ("Ge", 6):   0.73,
    ("Mn", 6):   0.83,
    ("Cu", 6):   0.73,
    ("Zn", 6):   0.74,
    ("Mg", 6):   0.72,
    ("Ca", 6):   1.00,
    ("Sr", 6):   1.18,
    ("Ba", 6):   1.35,
    ("Ti", 6):   0.605,
    ("Zr", 6):   0.72,
    ("Bi", 6):   1.03,
    ("Sb", 6):   0.76,
    ("In", 6):   0.80,
    ("Cd", 6):   0.95,
    ("Fe", 6):   0.78,  # high-spin Fe2+
    ("Fe3", 6):  0.645,  # high-spin Fe3+
    ("Ni", 6):   0.69,
    ("Co", 6):   0.745,  # high-spin Co2+
    ("Pd", 6):   0.86,
    ("Pt", 6):   0.80,
    ("Ag", 6):   1.15,
    # X-site anions (CN=6, 习惯值用于容忍因子)
    ("F", 6):    1.33,
    ("Cl", 6):   1.81,
    ("Br", 6):   1.96,
    ("I", 6):    2.20,
    ("O", 6):    1.40,
    ("S", 6):    1.84,
    ("Se", 6):   1.98,
}

# A-site 配位数固定为12, B-site和X-site固定为6
A_CN = 12
B_CN = 6
X_CN = 6


def _get_shannon_radius(element_symbol: str, cn: int) -> Optional[float]:
    """获取指定离子在给定配位数下的 Shannon 半径。"""
    key = (element_symbol, cn)
    if key in SHANNON_RADII:
        return SHANNON_RADII[key]
    # 尝试从 pymatgen 获取（仅限元素周期表中的元素）
    try:
        el = Element(element_symbol)
        radii = el.ionic_radii
        if radii:
            # 取最接近 CN 的值
            best = min(radii.items(), key=lambda x: abs(x[0] - cn))
            return best[1]
    except Exception:
        pass
    return None


# pymatgen 不识别有机阳离子 MA/FA/EA/GA，需要用占位元素替换后再 parse
_ORGANIC_CATIONS = {
    "MA": "Cs",   # methylammonium → use Cs as ionic radius proxy (closest CN12 radius)
    "FA": "Rb",   # formamidinium → use Rb as proxy
    "EA": "K",    # ethylammonium
    "GA": "K",    # guanidinium
    "AC": "Cs",   # acetamidinium
}


def _sanitize_formula(formula: str) -> tuple[str, dict[str, str]]:
    """将有机阳离子替换为占位元素以便 pymatgen Composition 解析。
    Returns:
        (sanitized_formula, reverse_map) — reverse_map 将占位元素映射回原始符号
    """
    reverse_map = {}
    sanitized = formula
    for organic, placeholder in _ORGANIC_CATIONS.items():
        if organic in sanitized:
            reverse_map[placeholder] = organic
            sanitized = sanitized.replace(organic, placeholder)
    return sanitized, reverse_map


def _parse_perovskite_composition(formula: str) -> dict:
    """
    解析钙钛矿化学式为各占位组分。
    支持 ABX3 型: MAPbI3, Cs0.1FA0.9PbI3, CsPb(Br0.5I0.5)3 等。

    Returns:
        {"A": [(symbol, fraction), ...], "B": [...], "X": [...]}
    """
    sanitized, reverse_map = _sanitize_formula(formula)

    try:
        comp = Composition(sanitized)
    except Exception:
        # Fallback: 手动解析带有机阳离子的化学式
        return _manual_parse(formula)

    raw = comp.as_dict()  # {str: amount} in pymatgen-core

    # 恢复有机阳离子符号
    elements = {}
    for sym, amt in raw.items():
        actual = reverse_map.get(sym, sym)
        # 检查是否被 pymatgen 错误拆分（如 "MA" → "M" + "A"）
        if len(sym) == 1 and sym.isupper():
            # 单字母元素可能是拆分错误，尝试合并
            pass
        elements[actual] = elements.get(actual, 0.0) + amt

    # 按元素类型分类到 A / B / X 位
    a_elements = {"Cs", "Rb", "K", "Na", "MA", "FA", "EA", "GA"}
    b_elements = {"Pb", "Sn", "Ge", "Mn", "Cu", "Zn", "Mg", "Ca", "Sr", "Ba",
                  "Ti", "Zr", "Bi", "Sb", "In", "Cd", "Fe", "Ni", "Co", "Pd",
                  "Pt", "Ag"}
    x_elements = {"F", "Cl", "Br", "I", "O", "S", "Se"}

    # pymatgen 可能把有机阳离子拆分为单字母片段，合并它们
    a_candidates, b_candidates, x_candidates = [], [], []
    unknowns = []

    for sym, amt in elements.items():
        if sym in a_elements:
            a_candidates.append((sym, amt))
        elif sym in b_elements:
            b_candidates.append((sym, amt))
        elif sym in x_elements:
            x_candidates.append((sym, amt))
        else:
            # 检查是否是拆分的有机阳离子片段
            r6 = _get_shannon_radius(sym, 6)
            if r6 and r6 > 1.2:
                a_candidates.append((sym, amt))
            elif r6:
                b_candidates.append((sym, amt))
            else:
                # 可能是 pymatgen 拆分的有机阳离子片段，归入 A 位
                unknowns.append((sym, amt))

    # 如果 A 位为空但有未知片段，合并为 A 位（有机阳离子情况）
    a_list = a_candidates
    if not a_list and unknowns:
        # 还原原始有机阳离子名称
        a_list = _reconstruct_organic_cation(formula, unknowns)
    elif unknowns:
        a_list.extend(unknowns)

    # Normalize
    return {
        "A": a_list,
        "B": b_candidates,
        "X": x_candidates,
        "formula": formula,
    }


def _reconstruct_organic_cation(original_formula: str,
                                fragments: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """尝试从原始化学式中恢复有机阳离子。
    当 pymatgen 把 'MA' 拆成 M + A 时，从原始字符串推断真正的有机阳离子。
    """
    for organic in _ORGANIC_CATIONS:
        if organic in original_formula:
            amt = sum(f[1] for f in fragments)
            return [(organic, amt)]
    return fragments


def _manual_parse(formula: str) -> dict:
    """手动解析含有机阳离子的化学式（当 pymatgen Composition 失败时）。"""
    a_list, b_list, x_list = [], [], []
    a_elements = {"Cs", "Rb", "K", "Na", "MA", "FA", "EA", "GA"}
    b_elements = {"Pb", "Sn", "Ge", "Mn", "Cu", "Zn", "Mg", "Ca", "Sr", "Ba",
                  "Ti", "Zr", "Bi", "Sb", "In", "Cd", "Fe", "Ni", "Co", "Pd",
                  "Pt", "Ag"}
    x_elements = {"F", "Cl", "Br", "I", "O", "S", "Se"}

    for organic in _ORGANIC_CATIONS:
        if organic in formula:
            a_list.append((organic, 1.0))

    for el in a_elements:
        if el in formula and el not in _ORGANIC_CATIONS:
            a_list.append((el, 1.0))
    for el in b_elements:
        if el in formula:
            b_list.append((el, 1.0))
    for el in x_elements:
        if el in formula:
            x_list.append((el, 3.0 if not x_list else 0.0))

    # 处理小数占比 (Cs0.1, FA0.9 等)
    import re
    fractions = {}
    for match in re.finditer(r'([A-Z][a-z]?)(\d*\.?\d*)', formula):
        sym, num = match.group(1), match.group(2)
        if sym in a_elements:
            val = float(num) if num else 1.0
            fractions[sym] = fractions.get(sym, 0.0) + val

    if fractions:
        return {
            "A": [(s, fractions.get(s, 1.0)) for s, _ in a_list] if a_list else a_list,
            "B": b_list,
            "X": x_list if x_list else [("I", 3.0)],
            "formula": formula,
        }

    return {
        "A": a_list,
        "B": b_list,
        "X": x_list if x_list else [("I", 3.0)],
        "formula": formula,
    }


def _effective_radius(site_species: list[tuple[str, float]], cn: int) -> float:
    """计算混合占位的有效离子半径（加权平均）。"""
    total_frac = sum(f for _, f in site_species)
    if total_frac == 0:
        return 0.0
    weighted_sum = 0.0
    for sym, frac in site_species:
        r = _get_shannon_radius(sym, cn)
        if r is None:
            r = _get_shannon_radius(sym, 6) or 1.5  # fallback
        weighted_sum += r * frac
    return weighted_sum / total_frac


def compute_tolerance_factor(parsed: dict) -> dict:
    """计算 Goldschmidt 容忍因子和八面体因子。"""
    r_A = _effective_radius(parsed["A"], A_CN)
    r_B = _effective_radius(parsed["B"], B_CN)
    r_X = _effective_radius(parsed["X"], X_CN)

    if r_B == 0 or r_X == 0:
        return {"error": "Cannot compute: missing ionic radii for B or X site"}

    # Goldschmidt tolerance factor
    t = (r_A + r_X) / (math.sqrt(2) * (r_B + r_X))

    # Octahedral factor
    mu = r_B / r_X

    # 预期晶体结构
    if 0.9 <= t <= 1.0:
        crystal_system = "cubic (Pm-3m)"
    elif 0.8 <= t < 0.9:
        crystal_system = "tetragonal/orthorhombic (I4/mcm or Pnma)"
    elif 0.7 <= t < 0.8:
        crystal_system = "orthorhombic (Pnma) or non-perovskite"
    elif t > 1.0:
        crystal_system = "hexagonal (6H/4H polytype) — A too large for cubic"
    else:
        crystal_system = "non-perovskite — t < 0.7, likely ilmenite or other phase"

    # 稳定性评估
    issues = []
    if t < 0.71 or t > 1.0:
        issues.append("Tolerance factor out of ideal range (0.71–1.0)")
    if mu < 0.41:
        issues.append("B-site cation too small for octahedral coordination (μ < 0.41)")
    if mu > 0.90:
        issues.append("B-site cation too large for octahedral coordination (μ > 0.90)")

    stable = len(issues) == 0

    return {
        "tolerance_factor": round(t, 4),
        "octahedral_factor": round(mu, 4),
        "r_A": round(r_A, 4),
        "r_B": round(r_B, 4),
        "r_X": round(r_X, 4),
        "predicted_crystal_system": crystal_system,
        "likely_stable": stable,
        "issues": issues if issues else ["None — parameters in ideal range"],
    }


def analyze_perovskite(formula: str) -> dict:
    """
    分析钙钛矿组分的结构稳定性。

    Args:
        formula: 化学式, 如 "MAPbI3", "Cs0.1FA0.9PbI3", "CsPb(Br0.5I0.5)3"

    Returns:
        dict with tolerance_factor, octahedral_factor, crystal_system, etc.
    """
    parsed = _parse_perovskite_composition(formula)

    if not parsed["A"] or not parsed["B"] or not parsed["X"]:
        return {
            "error": (
                f"Could not identify all perovskite sites. "
                f"Found A={parsed['A']}, B={parsed['B']}, X={parsed['X']}."
                f"Make sure the formula is ABX3 type."
            ),
            "parsed_formula": parsed["formula"],
        }

    tf = compute_tolerance_factor(parsed)

    return {
        "formula": formula,
        "reduced_formula": parsed["formula"],
        "A_site": [f"{s}{f:.2f}" for s, f in parsed["A"]],
        "B_site": [f"{s}{f:.2f}" for s, f in parsed["B"]],
        "X_site": [f"{s}{f:.2f}" for s, f in parsed["X"]],
        **tf,
    }


def search_materials_project(formula: str) -> dict:
    """
    查询 Materials Project 数据库中已知的钙钛矿数据。
    需要环境变量 MP_API_KEY（https://materialsproject.org/api）。

    Args:
        formula: 化学式, 如 "PbTiO3", "MAPbI3"

    Returns:
        dict with MP entries (bandgap, formation_energy, structure, material_id)
    """
    api_key = os.getenv("MP_API_KEY", "")
    if not api_key:
        return {
            "error": (
                "MP_API_KEY not set. Get a free API key at https://materialsproject.org/api "
                "and add it to .env as MP_API_KEY=your_key"
            ),
        }

    try:
        from pymatgen.ext.matproj import MPRester
    except ImportError:
        return {"error": "pymatgen.ext.matproj not available in pymatgen-core. Install pymatgen[mp]."}

    try:
        with MPRester(api_key) as mpr:
            # 搜索钙钛矿结构
            entries = mpr.get_entries(
                formula,
                property_data=["band_gap", "formation_energy_per_atom"],
            )
            if not entries:
                return {
                    "formula": formula,
                    "results": [],
                    "message": f"No entries found in Materials Project for '{formula}'",
                }

            results = []
            for entry in entries[:10]:
                struct = entry.structure
                results.append({
                    "material_id": entry.entry_id,
                    "formula": struct.composition.reduced_formula,
                    "band_gap_eV": round(entry.data.get("band_gap", 0), 3),
                    "formation_energy_eV_per_atom": round(
                        entry.data.get("formation_energy_per_atom", 0), 4
                    ),
                    "crystal_system": struct.get_space_group_info()[0]
                    if hasattr(struct, "get_space_group_info") else "N/A",
                })

            return {
                "formula": formula,
                "results": results,
                "source": "Materials Project (DFT — GGA/GGA+U)",
            }
    except Exception as e:
        return {"error": f"Materials Project query failed: {str(e)}"}
