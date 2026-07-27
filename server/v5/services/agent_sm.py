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

    # 计数器
    search_count: int = 0
    read_success_count: int = 0
    read_fail_count: int = 0
    retrieve_llm_calls: int = 0
    read_llm_calls: int = 0
    total_tool_calls: int = 0
    back_to_retrieve_count: int = 0  # READ 回退到 RETRIEVE 的次数

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

READ_PROMPT = """
## 当前阶段：深度阅读 (READ)

从检索结果中逐一阅读论文全文。目标是**尽可能多地读**有全文的论文，积累丰富的实验数据和细节。

### 规则
1. **优先阅读标记了 📄 或 PDF 链接的论文**（这些有本地全文）
2. 标记了 🔗 或仅有 DOI 的论文无法打开全文，跳过它们
3. 优先读顶刊论文（Nature/Science/NatEnergy 等），但也要覆盖不同期刊和研究组
4. 系统会自动追踪哪些有全文、哪些没有
5. **连续 3 篇没有全文 → 自动回到检索阶段，换其他论文**
6. 读完一篇后继续读下一篇，不要过早结束 — 更全面的阅读会产生更深入的回答
7. **你的目标是读完所有可读的论文**，而不只是 1~2 篇

### 预算
- 本阶段最多可读 {max_papers} 篇
- 至少读完 {min_fulltext} 篇才能进入回答
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

### 参考资料

**有全文的论文 ({n_fulltext} 篇)** — 主要依据:
{fulltext_list}

**无全文的论文 ({n_nofulltext} 篇)** — 补充引用，提供链接供用户自行查阅:
{nofulltext_list}

### 规则
1. **回答中的信息必须来自有全文的论文**（📄 标记），不能将仅有摘要的论文作为事实来源
2. 仅有摘要的论文作为补充信息，**使用 DOI 链接引用**: [🔗](https://doi.org/XXX)
3. 有全文的论文使用 PDF 链接引用: [📄](/api/pdf/FileID)
4. 每个关键数据点后附加引用链接
5. **绝对禁止编造 File ID 或数据**
6. 先框架后细节，结构清晰

### 禁止
- **禁止调用任何工具** — 这是纯文本输出阶段
- **禁止提到内部流程**
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
        self.ctx.log_state(AgentState.READ, "enter",
                           f"fulltext={len(self.ctx.fulltext_sources)}, "
                           f"unknown={len(self.ctx.unknown_sources)}")
        yield AgentEvent.state_change(AgentState.READ.value, self.ctx.state_summary())

        max_papers = BUDGETS["read_papers"]
        self.messages.append({
            "role": "system",
            "content": READ_PROMPT.format(
                max_papers=max_papers,
                min_fulltext=MIN_FULLTEXT_PAPERS,
            ),
        })

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
                else:
                    yield event

            self.ctx.read_llm_calls += 1
            llm_rounds += 1

            if tool_call is None:
                log(f"TASK {self.task_id} READ: LLM done")
                break

            # 双重保护：API 层 + 运行时检查
            if tool_call.name not in READ_TOOLS:
                log(f"TASK {self.task_id} READ: unexpected {tool_call.name}, skip")
                # 仍 yield tool_call 让 chat.py 清除 full_content 中的 <tool_calls> 残骸
                yield AgentEvent.tool_call(tool_call.name, tool_call.arguments)
                self.messages.append({
                    "role": "system",
                    "content": f"⚠️ 阅读阶段不可用 {tool_call.name}。可用工具: {', '.join(sorted(READ_TOOLS))}。",
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
                if len(self.ctx.fulltext_sources) < MIN_FULLTEXT_PAPERS:
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
                                needed=MIN_FULLTEXT_PAPERS - len(self.ctx.fulltext_sources),
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
            # 把所有已读论文正文拼成一个参考资料块
            ref_blocks = []
            for i, pc in enumerate(paper_contents):
                ref_blocks.append(
                    f"### [{i+1}] {pc['source']}\n{pc['content']}\n"
                    f"---"
                )
            clean_messages.append({
                "role": "system",
                "content": (
                    "## 已读论文全文参考资料\n\n"
                    "以下是你已经阅读过的论文正文。回答中的所有事实必须来自这些资料。\n\n"
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
        """从消息历史中提取 read_paper 的返回内容，去除 tool_call 格式。"""
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
                papers.append({"source": source, "content": body[:5000]})
        log(f"TASK {self.task_id} RESPOND: extracted {len(papers)} read papers")
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
