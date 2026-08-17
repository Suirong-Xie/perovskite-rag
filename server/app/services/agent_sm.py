"""
Agent State Machine v2 — 带反馈回路的 3 状态 Agent。

状态:
  RETRIEVE → READ → RESPOND
      ↑         │
      └─────────┘  (3次连续 read 失败 + 全文不足 → 回 RETRIEVE 重搜)

目标: 积累 ≥5 篇有全文的论文后再回答。无全文论文作为补充引用。

特性:
  - 工具失败不计入预算
  - 状态通过 SSE AgentEvent.state_change 暴露
  - 纯规则驱动
  - 最多 2 次回退 (防止无限循环)
"""

from __future__ import annotations

import json, re, time
from enum import Enum
from typing import AsyncGenerator
from dataclasses import dataclass, field

from ..core.config import AGENT_STATE_BUDGETS
from ..core.schemas import AgentEvent, ToolCall, ToolResult
from ..core.modes import AgentMode, resolve_mode


# ═══════════════════════════════════════════════════════════════════
# 状态 & 上下文
# ═══════════════════════════════════════════════════════════════════

class AgentState(Enum):
    RETRIEVE = "retrieve"
    QUICK_READ = "quick_read"       # 首轮快速阅读（1-3 篇）
    DEEP_READ = "deep_read"         # 跟进深入阅读（2-4 篇）
    RESPOND = "respond"


STATE_LABELS = {
    AgentState.RETRIEVE: "检索文献",
    AgentState.QUICK_READ: "快速阅读",
    AgentState.DEEP_READ: "深入阅读",
    AgentState.RESPOND: "生成回答",
}


@dataclass
class StateContext:
    """状态间共享上下文。"""
    # 论文全文状态
    fulltext_sources: set[str] = field(default_factory=set)    # 确认有 PDF
    nofulltext_sources: set[str] = field(default_factory=set)  # 确认无 PDF
    unknown_sources: set[str] = field(default_factory=set)     # 尚未尝试读取

    # 论文元数据（从搜索结果中保存，用于分类和学术引文）
    paper_meta: dict[str, dict] = field(default_factory=dict)
    # source → {title, journal, year, content_preview}

    # 分类
    question_type: str = "broad"           # "broad" | "specific"
    intent: str = "research"               # "chat" | "research"
    paper_type: dict[str, str] = field(default_factory=dict)
    # source → "review" | "experimental"

    # 论文池引用（会话级，跨轮复用）
    paper_pool: object | None = None  # SessionPaperPool

    # 计数器
    search_count: int = 0
    read_success_count: int = 0
    read_fail_count: int = 0
    retrieve_llm_calls: int = 0
    read_llm_calls: int = 0
    total_tool_calls: int = 0
    back_to_retrieve_count: int = 0  # READ 回退到 RETRIEVE 的次数
    rejected_tool_calls: int = 0  # READ 阶段违规调用非 READ 工具的次数

    # 可追踪性
    state_history: list[dict] = field(default_factory=list)

    def log_state(self, state: AgentState, action: str, detail: str = ""):
        self.state_history.append({
            "state": state.value, "action": action,
            "detail": detail, "timestamp": time.time(),
        })

    def state_summary(self) -> dict:
        current = self.state_history[-1] if self.state_history else {}
        return {
            "current_state": current.get("state", "init"),
            "current_state_label": STATE_LABELS.get(
                next((s for s in AgentState if s.value == current.get("state")), None), ""
            ),
            "searches_done": self.search_count,
            "papers_found": len(self.fulltext_sources) + len(self.unknown_sources),
            "papers_fulltext": len(self.fulltext_sources),
            "papers_nofulltext": len(self.nofulltext_sources),
            "papers_read": self.read_success_count,
            "total_tool_calls": self.total_tool_calls,
            "question_type": self.question_type,
        }


def _build_fallback_answer(fulltext: set, nofulltext: set) -> str:
    """当 LLM 完全没产出时，用已收集的来源生成兜底回答。"""
    lines = ["## 检索结果\n"]
    if fulltext:
        lines.append(f"已找到 **{len(fulltext)}** 篇有全文的论文：\n")
        for i, s in enumerate(sorted(fulltext)):
            file_id = s.replace('.pdf', '')
            lines.append(f"{i+1}. {s} — [📄](/api/pdf/{file_id})")
        lines.append("")
    if nofulltext:
        lines.append(f"另有 **{len(nofulltext)}** 篇仅有摘要的论文作为补充参考。\n")
    if not fulltext and not nofulltext:
        lines.append("未找到相关论文，请尝试更换搜索关键词。\n")
    lines.append("\n请查看上方的参考来源列表获取详细信息。")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 问题 & 论文分类 (纯启发式，不消耗 LLM 调用)
# ═══════════════════════════════════════════════════════════════════

# 问题分类关键词
_BROAD_QUESTION_KW = [
    "介绍", "概述", "综述", "进展", "现状", "有哪些", "是什么",
    "什么是", "定义", "概念", "列举", "推荐", "找几篇", "给我看",
    "总结", "归纳", "概括", "发展", "研究热点", "前沿", "方向",
    "survey", "review", "overview", "introduction", "summary",
    "summarize", "what is", "what are", "tell me about", "overview of",
]

_SPECIFIC_QUESTION_KW = [
    "对比", "比较", "差异", "区别", "哪个", "多少", "效率",
    "具体", "掺杂", "制备方法", "工艺参数", "活化能", "带隙",
    "组分", "配比", "浓度", "温度", "退火", "怎么", "如何",
    "vs", "versus", "difference", "compare", "comparison",
    "which", "how much", "what efficiency", "specific",
    "doping", "fabrication", "synthesis", "optimization",
]


# 强信号特定关键词: 出现即基本确定为 specific 类型
# 给予 2x 权重以打破与 broad 关键字的平局
_STRONG_SPECIFIC_KW = {
    "vs", "versus", "difference", "compare", "comparison",
    "which", "对比", "差异", "区别", "比较",
}


def _classify_question(text: str) -> str:
    """纯关键词匹配判断问题类型。默认 broad。

    使用词边界 (\b) 匹配 ASCII 短关键字 (≤4 字符)，避免
    如 "vs" 匹配 "perovskite" 的 substring 假阳性。
    CJK 关键字直接用子串匹配 (中文无词边界问题)。
    """
    text_lower = text.lower()

    def _match(kw: str) -> bool:
        kw_lower = kw.lower()
        # ASCII 短关键字 → 词边界匹配，防止 "vs" ⊂ "perovskite"
        if kw.isascii() and len(kw) <= 4:
            return bool(re.search(r'\b' + re.escape(kw_lower) + r'\b', text_lower))
        return kw_lower in text_lower

    broad_score = sum(1 for kw in _BROAD_QUESTION_KW if _match(kw))
    spec_score = sum(1 for kw in _SPECIFIC_QUESTION_KW if _match(kw))

    # 强信号特定关键词额外加权 => 打破平局
    for kw in _STRONG_SPECIFIC_KW:
        if _match(kw):
            spec_score += 1

    if spec_score > broad_score:
        return "specific"
    return "broad"


# ── 意图分类：闲聊 vs 需要文献的研究问题 ──

_CHAT_INTENT_KW = [
    "什么是", "是什么", "定义", "什么意思", "解释一下",
    "what is", "what are", "define", "definition", "meaning",
    "谢谢", "好的", "明白了", "换个说法", "简单说", "总结一下",
    "thanks", "thank you", "got it", "simpler terms",
]

_RESEARCH_INTENT_KW = [
    "找论文", "搜索", "文献", "最新", "进展", "研究",
    "paper", "search", "find", "recent", "latest", "literature",
    "对比", "比较", "差异", "区别", "哪个更好",
    "compare", "comparison", "difference", "which is better",
    "带隙", "掺杂", "退火", "组分", "稳定性", "数据",
    "不稳定", "降解", "衰减", "衰退",
    "bandgap", "doping", "stability", "efficiency",
    "degradation", "mechanism", "fabrication", "synthesis",
    "效率", "方法", "机理", "机制", "制备", "表征",
    "调控", "优化", "改善",
]


def _classify_intent(text: str, is_followup: bool = False,
                     has_pool: bool = False) -> str:
    """区分问题意图：chat vs research。

    纯对话/定义/短问题 → chat（跳过搜索，直接回答）
    有明确文献需求 → research（走 RETRIEVE→READ→RESPOND）
    """
    text_lower = text.lower()
    text_short = len(text) < 50

    chat_score = sum(1 for kw in _CHAT_INTENT_KW if kw in text_lower)
    research_score = sum(1 for kw in _RESEARCH_INTENT_KW if kw in text_lower)

    # 纯对话短句 → chat
    if text_short and research_score == 0:
        return "chat"

    # 跟进 + 论文池 → research（除非纯对话）
    if is_followup and has_pool and chat_score == 0:
        return "research"

    if research_score > chat_score:
        return "research"
    if chat_score > research_score:
        return "chat"

    # "what is" + 短问题 → 定义/概念问答，走 chat
    if text_short and ("what is" in text_lower or "什么是" in text_lower):
        return "chat"

    # 平局：科研关键词优先
    if research_score > 0:
        return "research"
    return "chat" if text_short else "research"


# ── 回答风格分类：判断用户想要什么类型的回答 ──

_DATA_LOOKUP_KW = [
    "多少", "多大", "多厚", "数值", "数据", "测过", "有没有人测",
    "具体数值", "具体数据", "实际值", "真实值", "量是多少",
    "有没有数据", "有没有报道", "有没有报告", "实验值",
    "how much", "what is the value", "measured value",
    "reported value", "specific data", "quantitative",
    "thickness", "thick", "concentration", "concentrations",
    "整理",  # "整理一个表格/数据"
]

_HOW_TO_KW = [
    "怎么", "如何", "怎样", "方法", "方案", "步骤", "流程",
    "设置", "选取", "选择什么", "用什么", "怎么做",
    "how to", "how do", "how can", "method", "approach",
    "procedure", "protocol", "guideline", "recommend",
]

_FORMAT_REQUEST_KW = [
    "表格", "列表", "列出", "列举", "清单", "整理成",
    "table", "list", "bullet", "summarize in",
    "整理一个", "给我一个", "列一个",
]

_COMPARE_KW = [
    "对比", "比较", "差异", "区别", "哪个更好", "哪个更",
    "优缺点", "优劣", "优劣对比",
    "compare", "comparison", "difference", "vs", "versus",
    "which is better", "which one", "pros and cons",
]

_FRUSTRATION_KW = [
    "不够具体", "不够详细", "不对",
    "不是这个意思", "你没理解",
    "no specific", "not what i", "not specific",
    "i meant", "what i meant", "that's not",
    "再查", "再搜", "再看看", "重新", "换一个",
    "还是没", "还没有", "不够",
]


def _classify_response_style(text: str) -> dict:
    """检测用户期望的回答风格和格式。

    Returns:
        {style, format_request, is_dissatisfied}
        style: "data_lookup" | "how_to" | "compare" | "overview"
        format_request: "table" | "list" | None
        is_dissatisfied: bool
    """
    text_lower = text.lower()

    scores = {
        "data_lookup": sum(1 for kw in _DATA_LOOKUP_KW if kw in text_lower),
        "how_to": sum(1 for kw in _HOW_TO_KW if kw in text_lower),
        "compare": sum(1 for kw in _COMPARE_KW if kw in text_lower),
    }
    best = max(scores, key=scores.get)
    style = best if scores[best] > 0 else "overview"

    # Format request detection
    format_request = None
    for kw in _FORMAT_REQUEST_KW:
        if kw in text_lower:
            if kw in ("表格", "table", "整理一个", "整理成"):
                format_request = "table"
            elif kw in ("列表", "列出", "列举", "清单", "list", "bullet", "列一个"):
                format_request = "list"
            elif kw in ("给我一个", "summarize in"):
                format_request = "table"  # default to table for "give me a ..."
            if format_request:
                break

    # Dissatisfaction detection
    is_dissatisfied = any(kw in text_lower for kw in _FRUSTRATION_KW)
    # "没有...吗/？" pattern: user is saying something is missing
    if not is_dissatisfied and "没有" in text_lower and ("吗" in text_lower or "？" in text_lower or "?" in text_lower):
        is_dissatisfied = True

    return {
        "style": style,
        "format_request": format_request,
        "is_dissatisfied": is_dissatisfied,
    }


# 综述关键词
_REVIEW_TITLE_KW = [
    "review", "progress", "advances", "survey", "overview",
    "comprehensive", "perspective", "roadmap", "retrospect",
    "tutorial", "state of the art", "state-of-the-art",
    "critical review", "mini review", "recent progress",
    "recent advances", "current status", "this review",
    "we review", "summarizes recent", "overview of",
    "综述", "进展", "回顾", "概述", "研究进展", "研究现状",
]


def _classify_paper(source: str, meta: dict) -> str:
    """判断论文是综述还是实验论文。默认实验论文。"""
    title = (meta.get("title", "") or "").lower()
    content = (meta.get("content_preview", "") or "").lower()
    source_lower = (source or "").lower()

    # 检查 title + content_preview + source 文件名
    for kw in _REVIEW_TITLE_KW:
        kw_lower = kw.lower()
        if kw_lower in title or kw_lower in content or kw_lower in source_lower:
            return "review"
    return "experimental"


# ═══════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════

MIN_FULLTEXT_PAPERS = 3       # 回答前最少全文论文数（从 8 降到 3，渐进式）
MIN_READ_CHARS = 100           # read_paper 有效内容最少字符数
MAX_BACK_TO_RETRIEVE = 1       # 最多回退次数（从 2 降到 1）
MAX_CONSECUTIVE_FAILS = 3      # 触发回退的连续失败数

BUDGETS = {
    "retrieve_llm": AGENT_STATE_BUDGETS.get("retrieve_llm", 2),
    "retrieve_search": AGENT_STATE_BUDGETS.get("retrieve_search", 2),
    "quick_read": AGENT_STATE_BUDGETS.get("quick_read", 3),
    "deep_read": AGENT_STATE_BUDGETS.get("deep_read", 4),
}

from .tools import RETRIEVE_TOOLS, READ_TOOLS, filter_tools as _filter_tools

MAX_TOTAL_TOOL_CALLS = 15      # 全局安全阀（从 35 降到 15）
MAX_REJECTED_IN_READ = 3       # READ 阶段违规调用非 READ 工具次数上限


# ═══════════════════════════════════════════════════════════════════
# 状态 Prompt
# ═══════════════════════════════════════════════════════════════════

RETRIEVE_PROMPT = """
## 文献检索 — 直接搜索，不解释

收集 ≥{min_fulltext} 篇有全文的论文。**立即调用 search_papers / search_arxiv / search_semantic_scholar，不要输出任何文字。**

### 规则
1. 用户消息中已附带预检索文献列表，先评估是否足够
   - 📄 标记的论文 = 有全文, 优先使用
   - 🔗 标记的论文 = 仅有摘要/元数据, 作为参考但不能打开全文
2. 使用 search_papers / search_arxiv / search_semantic_scholar 搜索
3. 🔗 论文的摘要信息已包含在搜索结果中，可以直接用于回答，不必为了"看内容"而反复搜索
4. 如果搜索结果不够 {min_fulltext} 篇有全文，换关键词从不同角度搜索（材料体系、制备工艺、表征手段、稳定性等维度）

### 已知无全文的论文（不要再搜）:
{nofulltext_list}

### 预算
- 最多 {max_llm} 轮搜索决策
- 收集到足够论文后自动进入阅读阶段
"""

# ── 新状态机 Prompts ──

QUICK_READ_PROMPT = """
## 当前阶段：快速阅读 (QUICK_READ)

**重要：直接调用 read_paper 工具，不要写任何解释或分析文字！**

从论文池中一次性选择最多 {max_papers} 篇论文，立即调用 read_paper 工具。

### 选择策略
- 综述型问题: 优先 read_paper 综述论文
- 具体型问题: 优先 read_paper 实验论文
- 同时调用多个 read_paper（系统会并行读取）

### 论文池
{paper_pool_summary}

- 读完即自动进入回答阶段
- 如果论文池无合适论文，不调用任何工具，直接进入回答
"""

DEEP_READ_PROMPT = """
## 当前阶段：深入阅读 (DEEP_READ)

**重要：直接调用 read_paper 工具，不要写任何解释或分析文字！**

跟进问题: {followup_question}

从论文池（共 {total_pool} 篇，已读 {read_count} 篇，未读 {unread_count} 篇）中，
一次性选择最多 {max_papers} 篇最相关的论文，立即调用 read_paper。

{paper_pool_summary}

- 同时调用多个 read_paper（系统会并行读取）
- 读完即进入回答阶段
- 如果已读论文足够回答，不调用工具直接进入回答
"""

RESPOND_PROMPT = """
## 当前阶段：生成回答

{context_hint}
{style_hint}

### 参考资料
**有全文 ({n_fulltext} 篇)**:
{fulltext_list}

**仅摘要 ({n_nofulltext} 篇)**:
{nofulltext_list}

### 回答要求
1. **每个事实陈述必须引用出处**：[📄](/api/pdf/FileID) 有全文，[🔗](https://doi.org/DOI) 仅摘要。引用时说明论文发现了什么，不要只放一个链接。
2. **有量化数据就给出具体数值**，不要只说"改善/提升/降低"。
3. **不要编造** File ID、数据、论文标题。
4. **不要提内部流程**（"我搜了X篇/读了Y篇"）。
5. **如果拿不到用户要的具体数据，诚实说明**，给出最接近的信息或建议如何查找。

{format_instruction}
"""

# 简化版 RESPOND prompt — 用于闲聊/简单问答（无需论文引用）
CHAT_RESPOND_PROMPT = """
## 当前阶段：直接回答

**重要**：这是一个简单问答或对话。如果对话历史中有上下文，请自然承接前文。

**规则**:
1. 用用户的语言回答，直接、简洁、准确
2. 如果你知道答案，基于你的知识直接回答
3. 如果问题涉及具体的实验数据、最新进展、或你不确定的事实，诚实说明需要查文献
4. 不要编造数据或引用
5. 不要使用工具调用标记
6. 不要提及内部流程（"我搜了/读了"等）
7. 如果对话历史中有相关论文引用，可以自然地提及
"""

# 调研模式 RESPOND prompt — 结构化概览，关键数据引用
SURVEY_RESPOND_PROMPT = """
## 当前阶段：生成调研回答

{context_hint}
{style_hint}

你已经检索并阅读了相关文献，请快速总结关键发现。

### 回答要求
1. **先框架后要点**：用1-2句概括整体情况，然后分点列出关键发现
2. **每个关键数据带引用**：有具体数值和出处
3. **控制长度**：500-1500字，不要展开过细
4. **引用格式**：[📄](/api/pdf/FileID) 有全文，[🔗](https://doi.org/DOI) 仅摘要

{format_instruction}
"""

# 精读模式 RESPOND prompt — 深度论文解读
READ_RESPOND_PROMPT = """
## 当前阶段：解读论文

你已阅读了用户指定的论文全文。请从以下维度进行深度分析：

### 回答框架

1. **核心创新**：这篇论文最关键的新发现或新方法是什么？与其他工作相比的独特性在哪？

2. **方法分析**：
   - 实验/计算方法是否合理？
   - 有什么潜在的局限或改进空间？
   - 样品的表征是否充分？

3. **关键数据**：
   - 列出最重要的量化结果（效率、稳定性、结构参数等）
   - 这些数据是否令人信服？统计显著性如何？

4. **与其他工作对比**：
   - 如果论文本身有对比，复述对比结果
   - 如果你知道相关的工作，可以提及

5. **启发与展望**：
   - 这篇工作对领域的贡献和影响
   - 后续可以探索的方向

### 禁止
- 不要使用工具调用
- 不要编造论文中没有的数据
- 明确区分"论文中报道的"和"你推断的"
"""

# 计算模式 RESPOND prompt — 材料计算步骤化
COMPUTE_RESPOND_PROMPT = """
## 当前阶段：材料计算

用户需要你进行材料科学计算或数据查询。请按以下方式呈现结果：

### 回答要求

1. **展示计算步骤**：
   - 列出使用的公式及参数来源
   - 逐步展示计算过程
   - 给出最终数值结果

2. **结果解读**：
   - 计算结果意味着什么？
   - 例如容忍因子 t 在什么范围 → 预测什么晶体结构 → 稳定性如何

3. **数据来源标注**：
   - 如果是查询 Materials Project 等数据库，标注数据出处
   - 如果使用经验公式，注明文献来源

4. **不确定性说明**：
   - 计算方法的局限性
   - 与实际实验值的典型偏差

### 可用工具

你可以直接调用以下工具：
- `analyze_perovskite`: 计算容忍因子、八面体因子、晶体结构预测
- `search_materials`: 查询 Materials Project DFT 数据库
- `search_papers`: 查找计算参数或公式出处
- `read_paper`: 阅读相关论文获取实验参考值

使用这些工具获取数据后，整理成清晰的回答。

### 禁止
- 不要在没有工具验证的情况下声称精确数值
- 不要混淆 DFT 计算值和实验测量值
"""


# ═══════════════════════════════════════════════════════════════════
# 状态机核心
# ═══════════════════════════════════════════════════════════════════

def log(msg: str):
    print(f"[SM] {msg}", flush=True)


class _StateComplete(Exception):
    def __init__(self, next_state: AgentState):
        self.next_state = next_state


class AgentStateMachine:

    def __init__(
        self, messages, task_id, use_native_tools,
        execute_tool_fn, run_native_round_fn, AGENT_SYSTEM_PROMPT,
        paper_pool=None, followup_question="", mode="auto",
    ):
        self.messages = messages
        self.task_id = task_id
        self.use_native_tools = use_native_tools
        self.execute_tool = execute_tool_fn
        self._run_native_round = run_native_round_fn
        self._system_prompt = AGENT_SYSTEM_PROMPT
        self.ctx = StateContext()
        self._safety_valve = MAX_TOTAL_TOOL_CALLS
        self.followup_question = followup_question
        self.mode = resolve_mode(mode)
        log(f"TASK {task_id} MODE: {self.mode.label} "
            f"(search={self.mode.search_budget}, read={self.mode.read_budget}, "
            f"min_papers={self.mode.min_papers}, path={self.mode.state_path})")

        # 论文池集成
        if paper_pool is not None:
            self.ctx.paper_pool = paper_pool
            log(f"TASK {task_id} POOL: integrated ({paper_pool.total_count} papers, "
                f"unread={paper_pool.unread_count})")

        # 从预搜索中提取初始论文
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m
                break
        if last_user:
            content = last_user.get("content", "")
            # 新格式: `file_id` (journal)
            # 旧格式: File ID: `file_id`
            for match in re.finditer(
                r'(?:File ID:\s*)?`([A-Za-z][A-Za-z0-9_\-]{8,})`', content
            ):
                self.ctx.unknown_sources.add(match.group(1))
            log(f"TASK {task_id} Pre-search: {len(self.ctx.unknown_sources)} papers "
                f"extracted from user message")
            if self.ctx.unknown_sources:
                self.ctx.search_count = 1

        # 从论文池补充（更可靠）
        if self.ctx.paper_pool is not None:
            for src in self.ctx.paper_pool.papers:
                self.ctx.unknown_sources.add(src)
            log(f"TASK {task_id} Pool provides +{self.ctx.paper_pool.total_count} sources")

    def _compress_retrieve_history(self):
        """进入 READ 前压缩 RETRIEVE 历史：去掉带 tool_calls 格式的消息，
        替换为干净的论文列表摘要。切断 DeepSeek 对 search_papers 的模式惯性。

        借鉴 RESPOND 阶段的 _extract_read_papers() 模式——用干净上下文
        替代脏 tool_call 历史。
        """
        # 找到 RETRIEVE_PROMPT 注入点（RETRIEVE 阶段开始的标记）
        retrieve_start = None
        for i, m in enumerate(self.messages):
            if m.get("role") == "system" and "当前阶段：文献检索" in (m.get("content") or ""):
                retrieve_start = i
                break

        if retrieve_start is None:
            log(f"TASK {self.task_id} READ compress: no RETRIEVE boundary found")
            return

        # 提取 RETRIEVE 阶段产生的论文信息（从 tool 消息中提取搜索结果）
        # 实际格式：search_local → "Source: xxx.pdf", search_arxiv → "Source: xxx"
        # 预搜索注入的用户消息 → "File ID: `xxx`"
        found_papers = []
        for m in self.messages[retrieve_start:]:
            content = m.get("content")
            if not content:
                continue  # assistant 消息的 content 为 None（仅有 tool_calls）
            # 匹配 "Source: xxx" 或 "File ID: `xxx`" 或 "File ID: xxx"
            for match in re.finditer(
                r'(?:Source:\s*|File ID:\s*`?)([\w\d_\-./]+)',
                content
            ):
                pid = match.group(1).rstrip('.')
                if pid not in found_papers:
                    found_papers.append(pid)

        # 构建干净摘要：按论文类型分组 + 元数据摘要
        unk_list = sorted(self.ctx.unknown_sources) if self.ctx.unknown_sources else []
        ft_list = sorted(self.ctx.fulltext_sources) if self.ctx.fulltext_sources else []
        nft_list = sorted(self.ctx.nofulltext_sources) if self.ctx.nofulltext_sources else []

        # 分组：综述 vs 实验
        review_papers = [s for s in (unk_list + ft_list)
                         if self.ctx.paper_type.get(s) == "review"]
        expt_papers = [s for s in (unk_list + ft_list)
                       if self.ctx.paper_type.get(s) != "review"]

        def _fmt_paper(pid):
            """带元数据的论文条目。"""
            meta = self.ctx.paper_meta.get(pid, {})
            j = meta.get("journal", "")
            y = meta.get("year", "")
            pre = (meta.get("content_preview", "") or "")[:120]
            yr_str = f"{y} | " if y else ""
            extra = f" — {yr_str}{j}" if j else ""
            line = f"- {pid}{extra}"
            if pre:
                line += f"\n  _{pre}..._"
            return line

        lines = [
            "## 检索阶段已完成",
            "",
            f"已执行 **{self.ctx.search_count}** 次搜索，收集到以下论文：",
            "",
        ]

        if review_papers:
            depth_hint = "**先从这里开始读**" if self.ctx.question_type == "broad" else "仅供参考"
            lines.append(f"### 📝 综述论文 ({len(review_papers)} 篇) — {depth_hint}")
            lines.append("")
            for pid in review_papers:
                lines.append(_fmt_paper(pid))
            lines.append("")

        if expt_papers:
            depth_hint = ("综述后根据引用选择" if self.ctx.question_type == "broad"
                          else "**优先阅读**")
            lines.append(f"### 🔬 实验论文 ({len(expt_papers)} 篇) — {depth_hint}")
            lines.append("")
            for pid in expt_papers:
                lines.append(_fmt_paper(pid))
            lines.append("")

        if nft_list:
            lines.append("### 🔗 无全文（跳过）")
            lines.append("")
            for pid in nft_list:
                lines.append(f"- {pid}")
            lines.append("")

        lines.extend([
            "---",
            "**可用工具**: read_paper, read_arxiv_paper, extract_data, compare_papers。",
            "**禁止**: search_papers, search_arxiv, search_semantic_scholar — 检索阶段已结束。",
            "连续 3 篇无全文 → 自动回到检索阶段。",
        ])

        summary = "\n".join(lines)

        # 删除 RETRIEVE 阶段的所有消息（from retrieve_start to end）
        del self.messages[retrieve_start:]

        # 追加干净的检索摘要
        self.messages.append({"role": "system", "content": summary})

        log(f"TASK {self.task_id} READ compress: {len(found_papers)} papers extracted, "
            f"RETRIEVE history collapsed into clean summary ({len(summary)} chars)")

    async def run(self) -> AsyncGenerator[AgentEvent, None]:
        max_loops = 8  # 全局安全阀

        # ── 判断入口路径（mode 优先，auto 时用 intent 分类）──
        pool = self.ctx.paper_pool
        has_pool = pool is not None and pool.unread_count > 0
        is_followup = bool(self.followup_question) and has_pool

        # 提取用户问题文本
        user_text = ""
        for m in reversed(self.messages):
            if m.get("role") == "user":
                raw = m.get("content", "") or ""
                idx = raw.find("系统已为你预检索")
                user_text = raw[:idx].strip() if idx != -1 else raw.strip()
                break

        intent = _classify_intent(user_text, is_followup=is_followup, has_pool=has_pool)
        self.ctx.intent = intent

        # ── Mode 驱动的路径选择 ──
        sp = self.mode.state_path

        if sp == "direct":
            state = AgentState.RESPOND
            log(f"TASK {self.task_id} PATH: mode={self.mode.key} → direct RESPOND")
        elif sp == "quick":
            state = AgentState.QUICK_READ
            log(f"TASK {self.task_id} PATH: mode={self.mode.key} → QUICK_READ → RESPOND")
        elif sp == "full":
            state = AgentState.RETRIEVE
            log(f"TASK {self.task_id} PATH: mode={self.mode.key} → "
                f"RETRIEVE → {'QUICK_READ → DEEP_READ → ' if self.mode.read_budget > 8 else 'QUICK_READ → '}RESPOND")
        elif sp == "flexible":
            # Calculate mode: check if question needs search
            if intent == "chat" and not has_pool:
                state = AgentState.RESPOND
            else:
                state = AgentState.RETRIEVE
            log(f"TASK {self.task_id} PATH: mode={self.mode.key} → flexible "
                f"(intent={intent}, resolved to {state.value})")
        else:  # "adaptive" or unknown — auto mode
            if is_followup and has_pool and intent != "chat":
                state = AgentState.DEEP_READ
                log(f"TASK {self.task_id} PATH: auto → follow-up → DEEP_READ")
            elif intent == "chat":
                state = AgentState.RESPOND
                log(f"TASK {self.task_id} PATH: auto → chat → RESPOND")
            else:
                state = AgentState.RETRIEVE
                log(f"TASK {self.task_id} PATH: auto → research → RETRIEVE")

        for _ in range(max_loops):
            try:
                if state == AgentState.RETRIEVE:
                    async for event in self._run_retrieve():
                        yield event
                elif state == AgentState.QUICK_READ:
                    async for event in self._run_quick_read():
                        yield event
                elif state == AgentState.DEEP_READ:
                    async for event in self._run_deep_read():
                        yield event
                elif state == AgentState.RESPOND:
                    async for event in self._run_respond():
                        yield event
                    return
            except _StateComplete as sc:
                state = sc.next_state

        # 安全阀触发 — 强制回答
        log(f"TASK {self.task_id} Safety valve, forcing RESPOND")
        async for event in self._run_respond():
            yield event

    # ── RETRIEVE ──

    async def _run_retrieve(self) -> AsyncGenerator[AgentEvent, None]:
        self.ctx.log_state(AgentState.RETRIEVE, "enter",
                           f"fulltext={len(self.ctx.fulltext_sources)}, "
                           f"unknown={len(self.ctx.unknown_sources)}")
        yield AgentEvent.state_change(AgentState.RETRIEVE.value, self.ctx.state_summary())

        # 检查是否已有足够的候选论文（预搜索 + 前序轮次积累）
        min_papers = self.mode.min_papers
        total_candidates = len(self.ctx.fulltext_sources) + len(self.ctx.unknown_sources)
        if min_papers > 0 and total_candidates >= min_papers:
            log(f"TASK {self.task_id} Already have {total_candidates} candidates "
                f"(fulltext={len(self.ctx.fulltext_sources)}, "
                f"unknown={len(self.ctx.unknown_sources)}), skip RETRIEVE")
            self.ctx.log_state(AgentState.RETRIEVE, "skip",
                               f"enough candidates ({total_candidates})")
            yield AgentEvent.state_change(AgentState.RETRIEVE.value, self.ctx.state_summary())
            raise _StateComplete(AgentState.QUICK_READ)

        # 注入 prompt
        nofulltext_str = "\n".join(f"  - {s}" for s in sorted(self.ctx.nofulltext_sources)) or "(尚无)"
        self.messages.append({
            "role": "system",
            "content": RETRIEVE_PROMPT.format(
                min_fulltext=self.mode.min_papers,
                max_llm=self.mode.search_budget,
                nofulltext_list=nofulltext_str,
            ),
        })

        llm_rounds = 0
        max_llm = max(self.mode.search_budget * 2, 2)  # 至少 2 轮
        search_attempts = 0
        max_search = self.mode.search_budget
        min_papers_plenty = self.mode.min_papers * 2

        while search_attempts < max_search and llm_rounds < max_llm:
            # 每次搜索前检查：是否已够
            if min_papers_plenty > 0 and len(self.ctx.fulltext_sources) + len(self.ctx.unknown_sources) >= min_papers_plenty:
                log(f"TASK {self.task_id} RETRIEVE: plenty of candidates, moving to QUICK_READ")
                break

            tool_calls_batch = []
            # compute 模式开放全部工具，其他模式按状态过滤
            if self.mode.allow_compute_tools:
                active_tools = None  # None = 全部工具
            else:
                active_tools = _filter_tools(RETRIEVE_TOOLS)

            async for event in self._run_native_round(
                self.task_id, self.ctx.retrieve_llm_calls + 1,
                self.messages, force_answer=False,
                allowed_tools=active_tools,
            ):
                if event.type == "_tool_call":
                    tool_calls_batch.append(event.data["tool_call"])
                elif event.type == "done":
                    pass
                else:
                    yield event

            self.ctx.retrieve_llm_calls += 1
            llm_rounds += 1

            if not tool_calls_batch:
                log(f"TASK {self.task_id} RETRIEVE: LLM done "
                    f"(fulltext={len(self.ctx.fulltext_sources)}, "
                    f"unknown={len(self.ctx.unknown_sources)})")
                break

            for tool_call in tool_calls_batch:
                if search_attempts >= max_search:
                    break

                if tool_call.name not in RETRIEVE_TOOLS:
                    log(f"TASK {self.task_id} RETRIEVE: unexpected {tool_call.name}, skip")
                    yield AgentEvent.tool_call(tool_call.name, tool_call.arguments)
                    self.messages.append({
                        "role": "system",
                        "content": f"⚠️ {tool_call.name} 不在检索阶段可用。请使用: {', '.join(sorted(RETRIEVE_TOOLS))}。",
                    })
                    continue

                self.ctx.total_tool_calls += 1
                search_attempts += 1
                log(f"SM TASK {self.task_id} RETRIEVE [{search_attempts}/{max_search}]: "
                    f"{tool_call.name}")

                yield AgentEvent.tool_call(tool_call.name, tool_call.arguments)
                result, raw_data = self.execute_tool(tool_call)
                log(f"SM TASK {self.task_id} RETRIEVE RESULT: {len(result.output)} chars")

                yield AgentEvent.tool_result(
                    tool_call.name,
                    result.output[:300] if result.output else "(empty)",
                    result.error,
                )

                if raw_data and isinstance(raw_data, list):
                    for item in raw_data:
                        src = item.get("source", "") if isinstance(item, dict) else ""
                        if src and src not in self.ctx.nofulltext_sources:
                            self.ctx.unknown_sources.add(src)
                        if isinstance(item, dict) and src:
                            self.ctx.paper_meta[src] = {
                                "title": item.get("title", ""),
                                "journal": item.get("journal_name", "") or item.get("venue", ""),
                                "year": item.get("year", "") or item.get("_s2_year", ""),
                                "content_preview": (
                                    item.get("content", "")
                                    or item.get("abstract", "")
                                    or item.get("summary", "")
                                    or ""
                                )[:500],
                            }
                    yield AgentEvent.search_results(raw_data)

                self.ctx.search_count += 1
                self._append_tool_result(tool_call, result)

                if self.ctx.total_tool_calls >= self._safety_valve:
                    break

            if self.ctx.total_tool_calls >= self._safety_valve:
                break

        self.ctx.log_state(AgentState.RETRIEVE, "exit",
                           f"fulltext={len(self.ctx.fulltext_sources)}, "
                           f"unknown={len(self.ctx.unknown_sources)}")
        yield AgentEvent.state_change(AgentState.RETRIEVE.value, self.ctx.state_summary())
        raise _StateComplete(AgentState.QUICK_READ)

    # ── QUICK_READ (首轮快速阅读) ──

    async def _run_quick_read(self) -> AsyncGenerator[AgentEvent, None]:
        """首轮快速阅读: 从论文池选代表性论文, 读完立即回答。"""
        max_papers = min(self.mode.read_budget, 6)  # 首次阅读最多 6 篇

        # 提取用户问题
        user_question = ""
        for m in reversed(self.messages):
            if m.get("role") == "user":
                raw = m.get("content", "") or ""
                idx = raw.find("系统已为你预检索")
                user_question = raw[:idx].strip() if idx != -1 else raw.strip()
                break
        self.ctx.question_type = _classify_question(user_question)

        # ── 从论文池选择代表性论文 ──
        pool = self.ctx.paper_pool
        if pool is not None and pool.unread_count > 0:
            _style = _classify_response_style(user_question)
            selected = pool.select_representative(
                n=max_papers, question=user_question,
                question_type=self.ctx.question_type,
                response_style=_style["style"],
            )
            log(f"TASK {self.task_id} QUICK_READ: selected {len(selected)} from pool "
                f"(type={self.ctx.question_type}, style={_style['style']})")
            # 同步到 ctx
            for p in selected:
                src = p["source"]
                if src not in self.ctx.fulltext_sources:
                    self.ctx.unknown_sources.add(src)
                if src not in self.ctx.paper_meta:
                    self.ctx.paper_meta[src] = {
                        "title": p.get("title", ""),
                        "journal": p.get("journal", ""),
                        "year": p.get("year", ""),
                        "content_preview": p.get("content_preview", ""),
                    }
                self.ctx.paper_type[src] = p.get("paper_type", "experimental")
        else:
            log(f"TASK {self.task_id} QUICK_READ: no pool or empty, using ctx.unknown_sources")
            selected = []

        self.ctx.log_state(AgentState.QUICK_READ, "enter",
                           f"candidates={len(self.ctx.unknown_sources)}")
        yield AgentEvent.state_change(AgentState.QUICK_READ.value, self.ctx.state_summary())

        # 如果没有候选论文，直接回答
        total_candidates = len(self.ctx.unknown_sources | self.ctx.fulltext_sources)
        if total_candidates == 0:
            log(f"TASK {self.task_id} QUICK_READ: no papers to read, skip to RESPOND")
            raise _StateComplete(AgentState.RESPOND)

        # ── 直接执行阅读（不调 LLM，用启发式选择的结果）──
        # QUICK_READ 中不再让 LLM 选择论文——直接用 select_representative() 的结果，
        # 然后用 ToolCall 包装成正常的 read_paper 执行。省掉 1 次 LLM 往返。
        papers_to_read = [p["source"] for p in selected] if selected else []
        if not papers_to_read:
            # fallback: 使用 ctx 中的 unknown_sources
            papers_to_read = list(self.ctx.unknown_sources)[:max_papers]

        log(f"TASK {self.task_id} QUICK_READ: executing reads for {papers_to_read}")

        # 压缩 RETRIEVE 历史
        self._compress_retrieve_history()

        for source in papers_to_read:
            if self.ctx.read_success_count >= max_papers:
                break
            if self.ctx.total_tool_calls >= self._safety_valve:
                break

            tool_call = ToolCall("read_paper", {"source": source})
            self.ctx.total_tool_calls += 1

            yield AgentEvent.tool_call("read_paper", {"source": source})
            result, raw_data = self.execute_tool(tool_call)

            result_chars = len(result.output) if result.output else 0
            is_success = (result_chars >= MIN_READ_CHARS
                          and not result.error
                          and "PDF not found" not in (result.output or "")
                          and "无全文" not in (result.output or ""))

            if is_success:
                self.ctx.read_success_count += 1
                self.ctx.fulltext_sources.add(source)
                self.ctx.unknown_sources.discard(source)
                if pool:
                    pool.mark_read(source, result.output)
            else:
                self.ctx.read_fail_count += 1
                self.ctx.nofulltext_sources.add(source)
                self.ctx.unknown_sources.discard(source)
                if pool:
                    pool.mark_nofulltext(source)

            yield AgentEvent.tool_result(
                "read_paper",
                result.output[:300] if result.output else "(empty)",
                result.error,
            )
            self._append_tool_result(tool_call, result)

        self.ctx.log_state(AgentState.QUICK_READ, "exit",
                           f"read={self.ctx.read_success_count}")
        yield AgentEvent.state_change(AgentState.QUICK_READ.value, self.ctx.state_summary())
        raise _StateComplete(AgentState.RESPOND)

    # ── DEEP_READ (跟进深入阅读) ──

    async def _run_deep_read(self) -> AsyncGenerator[AgentEvent, None]:
        """跟进深入阅读: 根据用户跟进问题，从论文池选最相关论文。"""
        max_papers = self.mode.read_budget
        pool = self.ctx.paper_pool

        log(f"TASK {self.task_id} DEEP_READ: followup='{self.followup_question[:60]}'")

        # 从论文池选择
        if pool is not None and pool.unread_count > 0:
            _style = _classify_response_style(self.followup_question)
            selected = pool.select_for_followup(
                n=max_papers, followup_question=self.followup_question,
                response_style=_style["style"],
            )
            log(f"TASK {self.task_id} DEEP_READ: selected {len(selected)} from pool "
                f"(style={_style['style']})")
            for p in selected:
                src = p["source"]
                if src not in self.ctx.paper_meta:
                    self.ctx.paper_meta[src] = {
                        "title": p.get("title", ""),
                        "journal": p.get("journal", ""),
                        "year": p.get("year", ""),
                        "content_preview": p.get("content_preview", ""),
                    }
                self.ctx.unknown_sources.add(src)
                self.ctx.paper_type[src] = p.get("paper_type", "experimental")
        else:
            log(f"TASK {self.task_id} DEEP_READ: no pool, fallback to RETRIEVE")
            self.ctx.log_state(AgentState.DEEP_READ, "no_pool",
                               "fallback to RETRIEVE")
            yield AgentEvent.state_change(AgentState.DEEP_READ.value, self.ctx.state_summary())
            raise _StateComplete(AgentState.RETRIEVE)  # 回退搜新论文

        self.ctx.log_state(AgentState.DEEP_READ, "enter",
                           f"unread={pool.unread_count if pool else 0}")
        yield AgentEvent.state_change(AgentState.DEEP_READ.value, self.ctx.state_summary())

        # ── 直接执行阅读（不调 LLM）──
        papers_to_read = [p["source"] for p in selected] if selected else []
        if not papers_to_read:
            papers_to_read = list(self.ctx.unknown_sources)[:max_papers]

        log(f"TASK {self.task_id} DEEP_READ: executing reads for {papers_to_read}")

        for source in papers_to_read:
            if self.ctx.read_success_count >= max_papers:
                break
            if self.ctx.total_tool_calls >= self._safety_valve:
                break

            tool_call = ToolCall("read_paper", {"source": source})
            self.ctx.total_tool_calls += 1

            yield AgentEvent.tool_call("read_paper", {"source": source})
            result, raw_data = self.execute_tool(tool_call)

            result_chars = len(result.output) if result.output else 0
            is_success = (result_chars >= MIN_READ_CHARS
                          and not result.error
                          and "PDF not found" not in (result.output or "")
                          and "无全文" not in (result.output or ""))

            if is_success:
                self.ctx.read_success_count += 1
                self.ctx.fulltext_sources.add(source)
                self.ctx.unknown_sources.discard(source)
                if pool:
                    pool.mark_read(source, result.output)
            else:
                self.ctx.read_fail_count += 1
                self.ctx.nofulltext_sources.add(source)
                self.ctx.unknown_sources.discard(source)
                if pool:
                    pool.mark_nofulltext(source)

            yield AgentEvent.tool_result(
                "read_paper",
                result.output[:300] if result.output else "(empty)",
                result.error,
            )
            self._append_tool_result(tool_call, result)

        self.ctx.log_state(AgentState.DEEP_READ, "exit",
                           f"read={self.ctx.read_success_count}")
        yield AgentEvent.state_change(AgentState.DEEP_READ.value, self.ctx.state_summary())
        raise _StateComplete(AgentState.RESPOND)

    # ── (废弃) 旧 READ ──

    async def _run_read(self) -> AsyncGenerator[AgentEvent, None]:
        # ── 问题 & 论文分类 ──
        # 提取用户原始问题（去掉系统预检索注入的内容）
        user_question = ""
        for m in reversed(self.messages):
            if m.get("role") == "user":
                raw = m.get("content", "") or ""
                # 截取 "系统已为你预检索" 之前的部分
                idx = raw.find("系统已为你预检索")
                user_question = raw[:idx].strip() if idx != -1 else raw.strip()
                break
        self.ctx.question_type = _classify_question(user_question)
        log(f"TASK {self.task_id} QUESTION: type={self.ctx.question_type}, "
            f"q='{user_question[:80]}'")

        # 分类所有论文
        for src in self.ctx.unknown_sources | self.ctx.fulltext_sources:
            meta = self.ctx.paper_meta.get(src, {})
            self.ctx.paper_type[src] = _classify_paper(src, meta)

        n_review = sum(1 for t in self.ctx.paper_type.values() if t == "review")
        n_expt = sum(1 for t in self.ctx.paper_type.values() if t == "experimental")
        log(f"TASK {self.task_id} PAPER TYPES: {n_review} reviews, {n_expt} experimental")

        # ── 动态预算 ──
        max_papers = BUDGETS["read_papers"]
        if self.ctx.question_type == "specific":
            max_reviews = 0
            max_experiments = min(max_papers, 8)
            min_fulltext = min(MIN_FULLTEXT_PAPERS, 5)
        else:  # broad
            max_reviews = min(max(1, max_papers // 3), 3)
            max_experiments = max_papers - max_reviews
            min_fulltext = MIN_FULLTEXT_PAPERS

        log(f"TASK {self.task_id} READ budget: reviews={max_reviews}, "
            f"experiments={max_experiments}, min_ft={min_fulltext}")

        self.ctx.log_state(AgentState.READ, "enter",
                           f"fulltext={len(self.ctx.fulltext_sources)}, "
                           f"unknown={len(self.ctx.unknown_sources)}, "
                           f"type={self.ctx.question_type}")
        yield AgentEvent.state_change(AgentState.READ.value, self.ctx.state_summary())

        # ── 压缩 RETRIEVE 历史：去掉 tool_calls 格式，切断 DeepSeek 模式惯性 ──
        self._compress_retrieve_history()

        # ── 注入阅读 prompt ──
        if self.ctx.question_type == "specific":
            prompt_text = READ_PROMPT_SPECIFIC.format(
                total_papers=max_papers,
                min_fulltext=min_fulltext,
            )
        else:
            prompt_text = READ_PROMPT_BROAD.format(
                max_reviews=max_reviews,
                max_experiments=max_experiments,
                total_papers=max_papers,
                min_fulltext=min_fulltext,
            )
        self.messages.append({"role": "system", "content": prompt_text})

        # ── 阶段追踪 ──
        reviews_read = 0
        experiments_read = 0
        current_phase = "review" if (max_reviews > 0) else "experiment"
        transition_done = (current_phase == "experiment")  # specific 跳过

        llm_rounds = 0
        max_llm_rounds = max_papers * 2
        consecutive_fails = 0

        while llm_rounds < max_llm_rounds:
            tool_call = None
            async for event in self._run_native_round(
                self.task_id, self.ctx.read_llm_calls + 1,
                self.messages, force_answer=False,
                allowed_tools=_filter_tools(READ_TOOLS),
            ):
                if event.type == "_tool_call":
                    tool_call = event.data["tool_call"]
                elif event.type == "done":
                    pass  # 忽略 — SM 自己管理状态转换，不让 done 穿透到 chat.py
                else:
                    yield event

            self.ctx.read_llm_calls += 1
            llm_rounds += 1

            if tool_call is None:
                log(f"TASK {self.task_id} READ: LLM done")
                break

            # 双重保护：API 层 + 运行时检查
            if tool_call.name not in READ_TOOLS:
                self.ctx.rejected_tool_calls += 1
                rc = self.ctx.rejected_tool_calls
                log(f"TASK {self.task_id} READ: unexpected {tool_call.name} "
                    f"(rejected #{rc}/{MAX_REJECTED_IN_READ}), skip")
                # 仍 yield tool_call 让 chat.py 清除 full_content 中的 <tool_calls> 残骸
                yield AgentEvent.tool_call(tool_call.name, tool_call.arguments)

                if rc >= MAX_REJECTED_IN_READ:
                    log(f"TASK {self.task_id} READ: {rc} rejected calls reached "
                        f"limit ({MAX_REJECTED_IN_READ}), forcing RESPOND")
                    self.messages.append({
                        "role": "system",
                        "content": (
                            f"❌ 严重警告：你已经 **{rc} 次** 试图调用搜索工具（{tool_call.name}），"
                            f"但搜索阶段已结束。检索结果已在上下文中列出。\n\n"
                            f"**系统强制切换到回答阶段。**请基于已阅读的论文直接生成回答。"
                        ),
                    })
                    self.ctx.log_state(AgentState.READ, "force_respond",
                                       f"rejected {rc}/{MAX_REJECTED_IN_READ}")
                    yield AgentEvent.state_change(AgentState.READ.value, self.ctx.state_summary())
                    raise _StateComplete(AgentState.RESPOND)

                self.messages.append({
                    "role": "system",
                    "content": (
                        f"❌ 错误：你调用了 {tool_call.name}，但搜索阶段已结束！\n"
                        f"检索结果已在上方列出，请直接使用 read_paper 阅读论文。\n"
                        f"这是第 **{rc}/{MAX_REJECTED_IN_READ}** 次违规。"
                        f"再违规 **{MAX_REJECTED_IN_READ - rc}** 次系统将强制切换到回答阶段。\n"
                        f"可用工具: {', '.join(sorted(READ_TOOLS))}。"
                    ),
                })
                continue

            self.ctx.total_tool_calls += 1
            source = tool_call.arguments.get("source", "unknown")
            log(f"SM TASK {self.task_id} READ [{llm_rounds}/{max_llm_rounds}]: {source}")

            yield AgentEvent.tool_call(tool_call.name, tool_call.arguments)
            result, raw_data = self.execute_tool(tool_call)

            result_chars = len(result.output) if result.output else 0
            is_success = (result_chars >= MIN_READ_CHARS
                          and not result.error
                          and "PDF not found" not in (result.output or "")
                          and "无全文" not in (result.output or ""))

            if is_success:
                self.ctx.read_success_count += 1
                self.ctx.fulltext_sources.add(source)
                self.ctx.unknown_sources.discard(source)
                consecutive_fails = 0

                # ── 阶段追踪 ──
                ptype = self.ctx.paper_type.get(source, "experimental")
                if ptype == "review":
                    reviews_read += 1
                else:
                    experiments_read += 1

                # ── 阶段切换检测 (broad 模式) ──
                if (self.ctx.question_type == "broad"
                        and not transition_done
                        and (reviews_read >= max_reviews
                             or not any(
                            self.ctx.paper_type.get(s) == "review"
                            for s in self.ctx.unknown_sources))):
                    transition_done = True
                    current_phase = "experiment"
                    remaining = max_papers - reviews_read - experiments_read
                    # 构建实验论文列表（带元数据摘要）
                    expt_lines = []
                    for s in sorted(self.ctx.unknown_sources):
                        if self.ctx.paper_type.get(s) == "experimental":
                            meta = self.ctx.paper_meta.get(s, {})
                            j = meta.get("journal", "?")
                            y = meta.get("year", "")
                            yr = f"{y} | " if y else ""
                            expt_lines.append(f"- {s} ({yr}{j})")
                    expt_list = "\n".join(expt_lines) if expt_lines else "(无)"
                    self.messages.append({
                        "role": "system",
                        "content": READ_TRANSITION_PROMPT.format(
                            reviews_read=reviews_read,
                            remaining=remaining,
                            experiment_list=expt_list,
                        ),
                    })
                    log(f"TASK {self.task_id} READ: Phase transition review→experiment "
                        f"(reviews={reviews_read}, remaining={remaining})")
            else:
                self.ctx.read_fail_count += 1
                consecutive_fails += 1
                self.ctx.nofulltext_sources.add(source)
                self.ctx.unknown_sources.discard(source)

            log(f"SM TASK {self.task_id} READ: {'OK' if is_success else 'FAIL'} "
                f"({result_chars} chars, consec_fails={consecutive_fails})")

            yield AgentEvent.tool_result(
                tool_call.name,
                result.output[:300] if result.output else "(empty)",
                result.error,
            )
            if raw_data and isinstance(raw_data, dict):
                yield AgentEvent.search_results([raw_data])
            self._append_tool_result(tool_call, result)

            # ── 回退判断 ──
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                if len(self.ctx.fulltext_sources) < min_fulltext:
                    if self.ctx.back_to_retrieve_count < MAX_BACK_TO_RETRIEVE:
                        self.ctx.back_to_retrieve_count += 1
                        log(f"TASK {self.task_id} READ: {consecutive_fails} consecutive "
                            f"fails, fulltext={len(self.ctx.fulltext_sources)}/{MIN_FULLTEXT_PAPERS}, "
                            f"back to RETRIEVE (#{self.ctx.back_to_retrieve_count})")

                        # 注入回退提示
                        self.messages.append({
                            "role": "system",
                            "content": READ_FAILED_PROMPT.format(
                                fails=consecutive_fails,
                                nofulltext_list="\n".join(f"  - {s}" for s in sorted(self.ctx.nofulltext_sources)),
                                fulltext_list="\n".join(f"  - {s}" for s in sorted(self.ctx.fulltext_sources)) or "(尚无)",
                                needed=min_fulltext - len(self.ctx.fulltext_sources),
                            ),
                        })

                        yield AgentEvent.state_change(
                            AgentState.READ.value, self.ctx.state_summary())
                        raise _StateComplete(AgentState.RETRIEVE)
                    else:
                        log(f"TASK {self.task_id} READ: max back-to-RETRIEVE "
                            f"({MAX_BACK_TO_RETRIEVE}) reached, accepting current papers")
                break

            if self.ctx.total_tool_calls >= self._safety_valve:
                break

        self.ctx.log_state(AgentState.READ, "exit",
                           f"fulltext={len(self.ctx.fulltext_sources)}, "
                           f"nofulltext={len(self.ctx.nofulltext_sources)}, "
                           f"back_count={self.ctx.back_to_retrieve_count}")
        yield AgentEvent.state_change(AgentState.READ.value, self.ctx.state_summary())
        raise _StateComplete(AgentState.RESPOND)

    # ── RESPOND ──

    async def _run_respond(self) -> AsyncGenerator[AgentEvent, None]:
        self.ctx.log_state(AgentState.RESPOND, "enter", "")
        yield AgentEvent.state_change(AgentState.RESPOND.value, self.ctx.state_summary())

        # ── 从消息历史中提取已读论文的正文 ──
        paper_contents = self._extract_read_papers()

        # ── 构建干净的上下文（无 tool_call 格式）──
        # 找到用户原始问题
        user_msg = None
        for m in reversed(self.messages):
            if m.get("role") == "user":
                user_msg = m
                break

        # 构建论文参考列表
        fulltext_list = "\n".join(
            f"  [{i+1}] {s}" for i, s in enumerate(sorted(self.ctx.fulltext_sources))
        ) or "(无)"
        nofulltext_list = "\n".join(
            f"  [{i+1}] {s}" for i, s in enumerate(sorted(self.ctx.nofulltext_sources))
        ) or "(无)"

        # 未读论文摘要（用于生成后续建议）
        pool = self.ctx.paper_pool
        if pool is not None and pool.unread_count > 0:
            unread_summary = pool.unread_summary_for_suggestions(max_papers=10)
        else:
            unread_summary = "(无)"
        log(f"TASK {self.task_id} RESPOND: unread_summary={len(unread_summary)} chars")

        # ── 提取用户问题文本（用于风格分类）──
        user_text = ""
        if user_msg:
            raw = user_msg.get("content", "") or ""
            idx = raw.find("系统已为你预检索")
            user_text = raw[:idx].strip() if idx != -1 else raw.strip()

        # ── 回答风格自适应 ──
        style_info = _classify_response_style(user_text)
        is_dissatisfied = style_info["is_dissatisfied"]
        is_followup = bool(self.followup_question)

        # 上下文提示：跟进/不满
        if is_dissatisfied:
            context_hint = (
                "⚠️ 用户对上一轮回答不满意。请直接、具体地切入问题核心。"
                "如果手头论文没有用户要的精确数据，诚实说明并给出最接近的信息或建议如何查找。"
            )
        elif is_followup:
            context_hint = "这是对上一轮讨论的跟进追问，请自然承接前文，直接切入问题核心。"
        else:
            context_hint = ""

        # 风格提示：按问题类型
        _style_map = {
            "data_lookup": "**回答风格**：用户问的是具体数据/数值。直接列出数据，有就有、没有就没有。不需要展开框架分析或长篇背景介绍。",
            "how_to": "**回答风格**：用户问的是方法/方案。给出具体步骤和参数建议，引用文献作为依据。",
            "compare": "**回答风格**：用户在做对比。并列展示不同选项的差异，用数据支撑对比结论。",
            "overview": "",
        }
        style_hint = _style_map.get(style_info["style"], "")

        # 格式指令：表格/列表
        _fmt_map = {
            "table": "**格式要求**：请使用Markdown表格组织数据，让信息一目了然。",
            "list": "**格式要求**：请使用清晰的编号列表组织内容。",
        }
        format_instruction = _fmt_map.get(style_info["format_request"] or "", "")

        log(f"TASK {self.task_id} RESPOND: style={style_info['style']} "
            f"fmt={style_info['format_request']} "
            f"dissatisfied={is_dissatisfied} followup={is_followup}")

        # ── 根据 mode 选择 prompt ──
        has_papers = bool(paper_contents) or len(self.ctx.fulltext_sources) > 0
        pv = self.mode.prompt_variant

        if pv == "chat" or (pv == "default" and self.ctx.intent == "chat" and not has_papers):
            respond_prompt = CHAT_RESPOND_PROMPT
            log(f"TASK {self.task_id} RESPOND: prompt=chat")
        elif pv == "survey":
            respond_prompt = SURVEY_RESPOND_PROMPT.format(
                context_hint=context_hint,
                style_hint=style_hint,
                format_instruction=format_instruction,
            )
            log(f"TASK {self.task_id} RESPOND: prompt=survey")
        elif pv == "read":
            respond_prompt = READ_RESPOND_PROMPT
            log(f"TASK {self.task_id} RESPOND: prompt=read")
        elif pv == "compute":
            respond_prompt = COMPUTE_RESPOND_PROMPT
            log(f"TASK {self.task_id} RESPOND: prompt=compute")
        else:
            # "deep" or "default" with papers → full RESPOND_PROMPT
            respond_prompt = RESPOND_PROMPT.format(
                context_hint=context_hint,
                style_hint=style_hint,
                n_fulltext=len(self.ctx.fulltext_sources),
                fulltext_list=fulltext_list,
                n_nofulltext=len(self.ctx.nofulltext_sources),
                nofulltext_list=nofulltext_list,
                format_instruction=format_instruction,
            )
            log(f"TASK {self.task_id} RESPOND: prompt={'deep' if pv == 'deep' else 'default'} "
                f"(fulltext={len(self.ctx.fulltext_sources)}, "
                f"nofulltext={len(self.ctx.nofulltext_sources)})")

        # 干净的上下文：无 tool_call / tool_result 格式，LLM 不会产生 XML 条件反射
        clean_messages = [{"role": "system", "content": self._system_prompt}]

        # ── 注入对话历史（user/assistant 轮次，不含状态机 system 消息）──
        # 让 LLM 感知当前 session 的上下文，实现连贯的多轮对话
        conv_turns = [m for m in self.messages if m.get("role") in ("user", "assistant")]
        if len(conv_turns) > 1:
            clean_messages.extend(conv_turns)
            log(f"TASK {self.task_id} RESPOND: injected {len(conv_turns)} conversation turns "
                f"for context continuity")

        if paper_contents:
            # 把所有已读论文正文拼成一个参考资料块（带引文元数据）
            ref_blocks = []
            for i, pc in enumerate(paper_contents):
                meta_line = []
                if pc.get("year"):
                    meta_line.append(str(pc["year"]))
                if pc.get("journal"):
                    meta_line.append(pc["journal"])
                if pc.get("title"):
                    meta_line.append(f"《{pc['title']}》")
                meta_str = " | ".join(meta_line) if meta_line else ""
                header = f"### [{i+1}] {pc['source']}"
                if meta_str:
                    header += f"\n**{meta_str}**"
                ref_blocks.append(
                    f"{header}\n{pc['content']}\n---"
                )
            clean_messages.append({
                "role": "system",
                "content": (
                    "## 已读论文全文参考资料\n\n"
                    "以下是你已阅读的论文正文。回答中的所有事实必须来自这些资料。\n"
                    "每篇论文标注了年份、期刊和标题（如有），请在引用时使用这些信息。\n\n"
                    + "\n\n".join(ref_blocks)
                ),
            })

        clean_messages.append({"role": "system", "content": respond_prompt})
        clean_messages.append(user_msg or {"role": "user", "content": "请基于以上论文回答。"})

        resp_text = ""
        async for event in self._run_native_round(
            self.task_id, 0, clean_messages, force_answer=True,
        ):
            if event.type == "_tool_call":
                log(f"TASK {self.task_id} RESPOND: LLM emitted tool_call, "
                    f"discarding {len(resp_text)} chars, retrying with stronger prompt")
                clean_messages.append({
                    "role": "system",
                    "content": "❌ 刚才你输出了工具调用标记。你现在必须在回答阶段，严禁调用任何工具。"
                               "请直接输出纯文本回答，不要使用 <tool_call> 或任何 XML 标签。",
                })
                resp_text = ""
                async for event2 in self._run_native_round(
                    self.task_id, 0, clean_messages, force_answer=True,
                ):
                    if event2.type == "_tool_call":
                        log(f"TASK {self.task_id} RESPOND: retry also failed")
                        resp_text = ""
                        break
                    if event2.type == "text":
                        chunk = event2.data.get("content", "")
                        resp_text += chunk
                        yield event2
                break
            if event.type == "text":
                chunk = event.data.get("content", "")
                resp_text += chunk
                yield event

        if resp_text.strip():
            # 检测 XML 污染
            xml_tags = ["<tool_call", "<tool_calls>", "antha:tool_call",
                        "<｜｜DSML｜｜tool_call", "<invoke"]
            if not any(tag in resp_text for tag in xml_tags):
                return
            log(f"TASK {self.task_id} RESPOND: text contains XML pollution, stripping")
            resp_text = re.sub(
                r'<(?:anth:)?tool_calls?>.*?</(?:anth:)?tool_calls?>',
                '', resp_text, flags=re.DOTALL
            ).strip()
            if resp_text:
                yield AgentEvent.text(resp_text)
                return

        # 兜底
        log(f"TASK {self.task_id} RESPOND: all attempts failed, using fallback")
        fallback = _build_fallback_answer(
            self.ctx.fulltext_sources, self.ctx.nofulltext_sources)
        yield AgentEvent.text(fallback)

    def _extract_read_papers(self) -> list[dict]:
        """从消息历史中提取 read_paper 的返回内容，去除 tool_call 格式。
        同时从 paper_meta 中补齐期刊/年份/标题等学术引文元数据。"""
        papers = []
        for m in self.messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            # read_paper 的输出以 "Content of " 开头
            if not content.startswith("Content of "):
                continue
            # 提取 source 和正文
            first_newline = content.find("\n")
            if first_newline == -1:
                continue
            header = content[:first_newline]  # "Content of Nature_2024_xxx.pdf (first 5000 chars):"
            body = content[first_newline + 1:].strip()
            source = header.replace("Content of ", "").split(" (first")[0]
            if len(body) >= MIN_READ_CHARS:
                meta = self.ctx.paper_meta.get(source, {})
                papers.append({
                    "source": source,
                    "content": body[:5000],
                    "journal": meta.get("journal", ""),
                    "year": meta.get("year", ""),
                    "title": meta.get("title", ""),
                })
        log(f"TASK {self.task_id} RESPOND: extracted {len(papers)} read papers "
            f"({sum(1 for p in papers if p.get('year'))} with year metadata)")
        return papers

    async def _try_respond_clean(self, messages, label) -> str:
        """调用 LLM 生成回答，本地缓冲并检测 <tool_calls> 污染。
        只在内容干净时才返回，否则返回空字符串。
        不 yield 任何 text 事件 — 调用者负责在确认干净后推送。
        """
        resp_text = ""
        async for event in self._run_native_round(
            self.task_id, 0, messages, force_answer=True,
        ):
            if event.type == "_tool_call":
                log(f"TASK {self.task_id} RESPOND {label}: LLM emitted tool_call, discarding")
                return ""
            if event.type == "text":
                resp_text += event.data.get("content", "")

        if not resp_text.strip():
            return ""
        if any(tag in resp_text for tag in ("<tool_call", "<tool_calls>", "antha:tool_call", "<｜｜DSML｜｜tool_call", "<invoke")):
            log(f"TASK {self.task_id} RESPOND {label}: text contains <tool_calls> XML, "
                f"discarding ({len(resp_text)} chars)")
            return ""
        log(f"TASK {self.task_id} RESPOND {label}: clean answer ({len(resp_text)} chars)")
        return resp_text

    # ── 辅助 ──

    def _append_tool_result(self, tool_call: ToolCall, result: ToolResult):
        tool_call_id = tool_call.arguments.pop(
            "_tool_call_id", f"call_{self.ctx.total_tool_calls}")
        self.messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": tool_call_id, "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                },
            }],
        })
        result_content = result.output
        if result.error:
            result_content += f"\nError: {result.error}"
        self.messages.append({
            "role": "tool", "tool_call_id": tool_call_id, "content": result_content,
        })
