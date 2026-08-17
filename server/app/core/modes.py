"""
Agent Mode 配置 — 5 种科研模式 + Auto 自动判断。

每个 Mode 是一组预算和行为参数，驱动同一套状态机产生不同粒度、深度、风格的回答。
不新建状态机，仅调整 budget / state_path / prompt_variant / toolset。
"""

from dataclasses import dataclass


@dataclass
class AgentMode:
    """Agent 运行模式配置。"""
    key: str                   # API 传值: "chat" | "survey" | "deep" | "read" | "compute" | "auto"
    label: str                 # 中文展示名
    icon: str                  # emoji 图标
    search_budget: int         # RETRIEVE 阶段最多搜索轮数 (0=跳过)
    read_budget: int           # 最多阅读论文篇数
    min_papers: int            # 回答前最少全文论文数
    citation_level: str        # "optional" | "key_claims" | "every_claim" | "internal"
    skip_pre_search: bool      # 跳过 chat.py 预搜索
    state_path: str            # "direct" | "quick" | "full" | "flexible" | "adaptive"
    prompt_variant: str        # "chat" | "survey" | "deep" | "read" | "compute" | "default"
    allow_compute_tools: bool  # 是否开放 pymatgen/gaussian 等计算工具
    description: str           # 一行用例说明


MODES: dict[str, AgentMode] = {
    # ── 问答：不搜不读，直接回答 ──
    "chat": AgentMode(
        key="chat", label="问答", icon="💬",
        search_budget=0, read_budget=0, min_papers=0,
        citation_level="optional", skip_pre_search=True,
        state_path="direct", prompt_variant="chat",
        allow_compute_tools=False,
        description="直接问答，不查文献（定义/概念/闲聊/跟进）"),

    # ── 调研：快速概览，轻量搜索+略读 ──
    "survey": AgentMode(
        key="survey", label="调研", icon="🔍",
        search_budget=3, read_budget=6, min_papers=3,
        citation_level="key_claims", skip_pre_search=False,
        state_path="quick", prompt_variant="survey",
        allow_compute_tools=False,
        description="文献概览与快速总结（找论文/了解进展/对比）"),

    # ── 深度：全面调研，大量搜索+精读 ──
    "deep": AgentMode(
        key="deep", label="深度", icon="📚",
        search_budget=5, read_budget=15, min_papers=8,
        citation_level="every_claim", skip_pre_search=False,
        state_path="full", prompt_variant="deep",
        allow_compute_tools=False,
        description="全面调研与逐句引证（写综述/系统分析/复杂对比）"),

    # ── 精读：深度解读指定论文，不搜索 ──
    "read": AgentMode(
        key="read", label="精读", icon="📖",
        search_budget=0, read_budget=3, min_papers=1,
        citation_level="internal", skip_pre_search=True,
        state_path="quick", prompt_variant="read",
        allow_compute_tools=False,
        description="深度解读指定论文（方法分析/创新点评/图表数据）"),

    # ── 计算：材料计算，开放 pymatgen/gaussian 工具 ──
    "compute": AgentMode(
        key="compute", label="计算", icon="🧮",
        search_budget=2, read_budget=3, min_papers=0,
        citation_level="optional", skip_pre_search=True,
        state_path="flexible", prompt_variant="compute",
        allow_compute_tools=True,
        description="材料计算与数据查询（容忍因子/带隙预测/DFT/MP数据库）"),

    # ── 自动（默认）：智能判断 ──
    "auto": AgentMode(
        key="auto", label="自动", icon="🤖",
        search_budget=4, read_budget=8, min_papers=3,
        citation_level="key_claims", skip_pre_search=False,
        state_path="adaptive", prompt_variant="default",
        allow_compute_tools=False,
        description="智能判断最佳模式"),
}

DEFAULT_MODE = "auto"


def resolve_mode(mode_key: str | None) -> AgentMode:
    """解析 mode key，无效/空 → auto。"""
    if mode_key and mode_key in MODES:
        return MODES[mode_key]
    return MODES[DEFAULT_MODE]


def get_mode_labels() -> list[dict]:
    """返回前端可用的模式列表 [{key, label, icon, description}, ...]"""
    return [
        {"key": m.key, "label": m.label, "icon": m.icon, "description": m.description}
        for m in MODES.values()
    ]
