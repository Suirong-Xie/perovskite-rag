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


# ═══════════════════════════════════════════════════════════════════
# 状态 & 上下文
# ═══════════════════════════════════════════════════════════════════

class AgentState(Enum):
    RETRIEVE = "retrieve"
    READ = "read"
    RESPOND = "respond"


STATE_LABELS = {
    AgentState.RETRIEVE: "检索文献",
    AgentState.READ: "深度阅读",
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
    paper_type: dict[str, str] = field(default_factory=dict)
    # source → "review" | "experimental"

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

MIN_FULLTEXT_PAPERS = 8       # 回答前最少全文论文数
MIN_READ_CHARS = 100           # read_paper 有效内容最少字符数
MAX_BACK_TO_RETRIEVE = 2       # 最多回退次数
MAX_CONSECUTIVE_FAILS = 3      # 触发回退的连续失败数

BUDGETS = {
    "retrieve_llm": AGENT_STATE_BUDGETS.get("retrieve_llm", 2),
    "retrieve_search": AGENT_STATE_BUDGETS.get("retrieve_search", 3),
    "read_papers": AGENT_STATE_BUDGETS.get("read_papers", 3),
}

from .tools import RETRIEVE_TOOLS, READ_TOOLS, filter_tools as _filter_tools

MAX_TOTAL_TOOL_CALLS = 35
MAX_REJECTED_IN_READ = 3  # READ 阶段违规调用非 READ 工具次数上限


# ═══════════════════════════════════════════════════════════════════
# 状态 Prompt
# ═══════════════════════════════════════════════════════════════════

RETRIEVE_PROMPT = """
## 当前阶段：文献检索 (RETRIEVE)

目标是收集 **≥{min_fulltext} 篇有全文的论文**，覆盖研究问题的不同侧面。

### 快速通道
如果用户只是要求展示/列举论文（如"给我看几篇XX的论文"、"有哪些关于YY的研究"），搜索一轮后即可直接进入回答阶段，无需逐篇阅读全文。

### 规则
1. 如果用户消息中已附带系统预检索的文献列表，先评估
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

READ_PROMPT_BROAD = """
## 当前阶段：深度阅读 — 综述优先模式

你的问题是综述/概述型的。请采用**两阶段结构化阅读**策略。

### 📚 Phase 1: 先读综述论文（最多 {max_reviews} 篇）
优先阅读下方「📝 综述论文」列表中的论文。读完每篇综述后，留意：
- 哪些实验论文被频繁引用？它们的核心发现是什么？
- 关键实验数据的出处（期刊、年份、研究组）
- 该领域公认的基准结果和典型实验方法

**读完综述后，系统会自动引导你进入 Phase 2。**

### 🔬 Phase 2: 再读代表性实验论文（最多 {max_experiments} 篇）
根据综述中提取的关键参考文献，从「🔬 实验论文」列表中选择最有代表性的实验论文深入阅读。优先选：
- 综述中频繁引用的论文
- 提供量化数据（效率、稳定性、组分、器件结构）的论文
- 不同研究组的代表性工作（避免全读同一课题组）
- 高影响力期刊的论文

### 规则
1. **必须从综述开始** — 综述提供概念框架和研究全景
2. 读完综述后，根据其引用的实验论文来选择下一步
3. 跳过标记为 🔗 的论文（无全文）
4. 不要过早结束 — 读满预算才能产生深入回答

### 预算
- Phase 1 综述: 最多 {max_reviews} 篇
- Phase 2 实验: 最多 {max_experiments} 篇
- 总计: 最多 {total_papers} 篇
- 至少 {min_fulltext} 篇后进入回答阶段
"""

READ_PROMPT_SPECIFIC = """
## 当前阶段：深度阅读 — 聚焦模式

你的问题是具体/技术型的。**跳过综述论文**，直接阅读实验研究论文。

### 阅读策略
- 优先读与你问题最直接相关的实验论文
- 寻找提供量化数据的论文（具体效率值、稳定性指标、组分、工艺参数等）
- 如果有多个研究组报道了相关结果，做交叉对比
- 综述论文（📝 标记）仅供参考，不要浪费配额去读

### 规则
1. 聚焦与你问题直接相关的实验数据
2. 跳过综述论文
3. 标记为 🔗 的论文无全文，跳过
4. 不要过早结束

### 预算
- 最多 {total_papers} 篇
- 至少 {min_fulltext} 篇后进入回答阶段
"""

READ_TRANSITION_PROMPT = """
## ⏭️ 阶段切换：综述 → 实验论文

你已读完 {reviews_read} 篇综述。现在根据已读综述中的参考文献，从下方实验论文列表中选择最具代表性的论文。

选择标准：
1. **被综述频繁引用**的实验论文
2. 提供了**关键量化数据**（效率记录、稳定性里程碑、经典组分/工艺）
3. 代表**不同研究组/方法**的论文（多样性 > 同质化）

剩余阅读配额：{remaining} 篇。

### 🔬 可读实验论文:
{experiment_list}
"""

READ_FAILED_PROMPT = """
## ⚠️ 连续 {fails} 篇论文没有全文！

系统将回到检索阶段。请用**不同的搜索策略**重新搜索：
- 尝试不同的出版源（Nature → ACS → RSC → Wiley）
- 使用不同的关键词角度
- 避免搜索已被标记为"无全文"的论文

已知无全文: {nofulltext_list}
当前全文: {fulltext_list}
还需: {needed} 篇
"""

RESPOND_PROMPT = """
## 当前阶段：生成回答 (RESPOND)

你已阅读了以下论文的全文。回答中**每一个事实性陈述都必须有具体论文支撑**。

### 参考资料

**已读全文 ({n_fulltext} 篇)** — 回答的主要依据:
{fulltext_list}

**未读全文 ({n_nofulltext} 篇)** — 仅作补充引用:
{nofulltext_list}

### 学术引文格式 (严格遵守)

每条关键陈述必须使用以下格式精确引用：

```
在[Year]年[Journal]的[Title/标识]中[明确报道/发现/证明/提出]：[具体数据或结论]。[📄](/api/pdf/FileID)
```

**要求**：
1. **必须包含**: 年份、期刊、论文标识（标题/第一作者/文件名中的 ID）
2. **必须陈述具体发现**: 什么被观测/测量/证明了，附带量化数据
3. **区分强度**: "明确指出"（强结论）、"报道"（数据点）、"发现"（新现象）、"提出"（理论/模型）
4. **每个自然段至少一个引用**，每个数据点必须有出处
5. 无全文的论文仅作补充引用: [🔗](https://doi.org/DOI)

**正确示例**:
> 在2024年Nature Energy的《Bandgap tuning in mixed-halide perovskites》中，Smith等人明确报道了Br含量从0%增加到20%时带隙从1.55 eV线性增加到1.72 eV，其依据是对30组不同组分的UV-Vis吸收光谱的系统测量。[📄](/api/pdf/Nature_2024_s41586-024-06121-1)
>
> Zhang等人进一步发现，湿度超过60%时MAPbI₃钙钛矿在24h内完全分解为PbI₂，XRD显示(110)钙钛矿峰强度衰减~80%。[📄](/api/pdf/NatEnergy_2023_s41560-023-01234-5)

**错误示例 (绝对禁止)**:
> ❌ "多项研究表明钙钛矿的稳定性可以通过组分调控改善" — 太笼统，无引用，无数据
> ❌ "相关实验表明..." — 完全没有引用
> ❌ "[📄](/api/pdf/xxx)" 单独出现 — 没有说明论文实际发现了什么

### 回答结构
1. **总览**: 用综述论文建立全景框架（如有读综述）
2. **分述**: 每个专题/维度用具体论文支撑，附数据和引用
3. **交叉对比**: 如果不同论文对同一现象有不同结论，明确标注分歧
4. **量化优先**: 只要有数据就不要只说"改善/提升/降低"，给出具体数值

### 禁止
- **禁止调用任何工具** — 纯文本输出
- **禁止编造 File ID、数据、论文标题**
- **禁止提及"我读了X篇论文"之类的内部流程**
- **禁止"多项研究/大量文献/广泛认为"等无引用的笼统表述**
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
    ):
        self.messages = messages
        self.task_id = task_id
        self.use_native_tools = use_native_tools
        self.execute_tool = execute_tool_fn
        self._run_native_round = run_native_round_fn
        self._system_prompt = AGENT_SYSTEM_PROMPT
        self.ctx = StateContext()
        self._safety_valve = MAX_TOTAL_TOOL_CALLS

        # 从预搜索中提取初始论文
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m
                break
        if last_user and "系统已为你预检索" in (last_user.get("content") or ""):
            content = last_user.get("content", "")
            for match in re.finditer(r'File ID:\s*`?([^`\s]+)`?', content):
                self.ctx.unknown_sources.add(match.group(1))
            log(f"TASK {task_id} Pre-search: {len(self.ctx.unknown_sources)} papers")
            self.ctx.search_count = 1

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
        state = AgentState.RETRIEVE
        max_loops = 10  # 全局安全阀

        for _ in range(max_loops):
            try:
                if state == AgentState.RETRIEVE:
                    async for event in self._run_retrieve():
                        yield event
                elif state == AgentState.READ:
                    async for event in self._run_read():
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

        # 检查是否已满足全文要求
        if len(self.ctx.fulltext_sources) >= MIN_FULLTEXT_PAPERS:
            log(f"TASK {self.task_id} Already have {len(self.ctx.fulltext_sources)} "
                f"fulltext papers, skip RETRIEVE")
            self.ctx.log_state(AgentState.RETRIEVE, "skip",
                               f"enough fulltext ({len(self.ctx.fulltext_sources)})")
            yield AgentEvent.state_change(AgentState.RETRIEVE.value, self.ctx.state_summary())
            raise _StateComplete(AgentState.READ)

        # 注入 prompt
        nofulltext_str = "\n".join(f"  - {s}" for s in sorted(self.ctx.nofulltext_sources)) or "(尚无)"
        self.messages.append({
            "role": "system",
            "content": RETRIEVE_PROMPT.format(
                min_fulltext=MIN_FULLTEXT_PAPERS,
                max_llm=BUDGETS["retrieve_llm"],
                nofulltext_list=nofulltext_str,
            ),
        })

        llm_rounds = 0
        max_llm = BUDGETS["retrieve_llm"] * 2
        search_attempts = 0
        max_search = BUDGETS["retrieve_search"]

        while search_attempts < max_search and llm_rounds < max_llm:
            # 每次搜索前检查：是否已够
            if len(self.ctx.fulltext_sources) + len(self.ctx.unknown_sources) >= MIN_FULLTEXT_PAPERS * 2:
                log(f"TASK {self.task_id} RETRIEVE: plenty of candidates, moving to READ")
                break

            tool_call = None
            async for event in self._run_native_round(
                self.task_id, self.ctx.retrieve_llm_calls + 1,
                self.messages, force_answer=False,
                allowed_tools=_filter_tools(RETRIEVE_TOOLS),
            ):
                if event.type == "_tool_call":
                    tool_call = event.data["tool_call"]
                elif event.type == "done":
                    pass  # 忽略 — SM 自己管理状态转换
                else:
                    yield event

            self.ctx.retrieve_llm_calls += 1
            llm_rounds += 1

            if tool_call is None:
                log(f"TASK {self.task_id} RETRIEVE: LLM done "
                    f"(fulltext={len(self.ctx.fulltext_sources)}, "
                    f"unknown={len(self.ctx.unknown_sources)})")
                break

            if tool_call.name not in RETRIEVE_TOOLS:
                log(f"TASK {self.task_id} RETRIEVE: unexpected {tool_call.name}, skip")
                # 仍 yield tool_call 让 chat.py 清除 full_content 中的 <tool_calls> 残骸
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
                    # 保存元数据（归一化三类搜索工具的字段名）
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

        self.ctx.log_state(AgentState.RETRIEVE, "exit",
                           f"fulltext={len(self.ctx.fulltext_sources)}, "
                           f"unknown={len(self.ctx.unknown_sources)}")
        yield AgentEvent.state_change(AgentState.RETRIEVE.value, self.ctx.state_summary())
        raise _StateComplete(AgentState.READ)

    # ── READ ──

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

        respond_prompt = RESPOND_PROMPT.format(
            n_fulltext=len(self.ctx.fulltext_sources),
            fulltext_list=fulltext_list,
            n_nofulltext=len(self.ctx.nofulltext_sources),
            nofulltext_list=nofulltext_list,
        )

        # 干净的上下文：无 tool_call / tool_result 格式，LLM 不会产生 XML 条件反射
        clean_messages = [{"role": "system", "content": self._system_prompt}]

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

        resp_text = await self._try_respond_clean(clean_messages, "respond")
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
