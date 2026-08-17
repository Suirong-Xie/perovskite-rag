"""
PerovskiteGPT V5 — Agent Loop Engine
ReAct-style agent: Think → Tool Call → Observe → Repeat → Answer

工具通过 LLM 文本输出中的 <tool_call> JSON 块来调用,
兼容任何 OpenAI-compatible gateway，不依赖原生 function calling API。

## 如何添加新 Skill

1. 在 TOOLS 列表中定义工具元数据（name, description, parameters）
2. 编写 executor 函数，签名为 (arguments: dict) -> ToolResult
3. 在 TOOL_EXECUTORS 中注册 name → executor
4. 在 AGENT_SYSTEM_PROMPT 的"可用工具"部分提及新工具

示例:
    TOOLS.append({
        "name": "my_tool",
        "description": "What this tool does...",
        "parameters": {"param1": "description"},
    })
    def execute_my_tool(args): ...
    TOOL_EXECUTORS["my_tool"] = execute_my_tool
"""
import json
import re
import os
import subprocess
from typing import AsyncGenerator, Optional, Callable
from ..core.config import AGENT_MAX_ROUNDS, LLM_BACKEND
from ..core.schemas import AgentEvent, ToolCall, ToolResult
from ..core.llm import chat_completion_stream, chat_completion_with_tools
from .retrieval import search_papers
from .tools import ALL_TOOLS, EXECUTORS as TOOL_EXECUTORS, RETRIEVE_TOOLS, READ_TOOLS, filter_tools
from .tools.paper_utils import find_pdf_path, find_pdf_fast, JOURNAL_DIR_MAP

# PyMuPDF4LLM 可选依赖（read_paper 优先使用）
try:
    import pymupdf4llm  # noqa: F401
    _HAS_PYMUPDF4LLM = True
except ImportError:
    _HAS_PYMUPDF4LLM = False

# 复用 chunking 管线的 markdown 清洗函数
_clean_md = None


def _get_clean_md():
    global _clean_md
    if _clean_md is None:
        import sys
        _pipeline = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pipeline')
        if _pipeline not in sys.path:
            sys.path.insert(0, _pipeline)
        from s2_chunk_and_embed import clean_markdown_text
        _clean_md = clean_markdown_text
    return _clean_md
from .gaussian_service import submit_job, check_job
from .materials_service import analyze_perovskite, search_materials_project
from .arxiv_service import search_arxiv, download_arxiv_pdf, clean_paper_text
from .semantic_scholar_service import search_semantic_scholar

# ── 工具定义 (从 tools/ 包导入) ──

TOOLS = ALL_TOOLS

# ── Agent 系统 prompt ──

AGENT_SYSTEM_PROMPT = """你是 Sunny，钙钛矿太阳能电池领域的 AI 研究助手。

## 知识来源

- 📄 本地全文库：Nature/Science 系列 + S2 全文论文（12,000+ PDF）
- 🔗 S2 摘要库：18,000+ 篇仅有摘要/元数据的论文（不可打开全文）
- arXiv 预印本、Semantic Scholar、Materials Project DFT 数据库

## 核心规则

1. **用用户的语言回答**（中文问→中文答，英文问→英文答）
2. **📄 标记的论文 = 有全文 PDF**，可作为事实来源，引用格式：`[📄](/api/pdf/FileID)`
3. **🔗 标记的论文 = 仅摘要/元数据**，不能作为主要事实依据，引用格式：`[🔗](https://doi.org/DOI)`
4. **禁止编造 File ID 或数据** — 只引用搜索结果中实际出现的论文
5. **简单展示类问题直接回答**（如"给我看几篇XX论文"），不需要逐篇阅读全文

## 搜索策略

- 始终用英文关键词搜索
- 优先使用系统预检索结果（如果有），只对缺失维度补充
- 结果不理想时换关键词，不要反复搜同一个角度
- 对比类问题分开搜索

## 回答风格

- 先框架后细节，每个数据点附加引用
- 专业、基于数据，不凭空发挥
- 不要描述"我搜了什么/用了什么工具"，直接给答案
- 不要在末尾追加来源列表"""

# 注意：工具列表、调用格式、预算控制由状态机动态注入，不在此处硬编码。

# ── 工具执行 ──


def log(msg: str):
    print(f"[V5:Agent] {msg}", flush=True)


# ── Context 压缩 ──

# 当消息历史超过此 token 数时触发压缩
CONTEXT_COMPRESS_THRESHOLD = 6000


def _estimate_tokens(messages: list[dict]) -> int:
    """粗略估算 token 数（4 字符 ≈ 1 token）。"""
    return sum(len(m.get("content", "") or "") for m in messages) // 4


def _compress_context(messages: list[dict], max_tokens: int = CONTEXT_COMPRESS_THRESHOLD):
    """压缩消息历史中的旧工具结果。

    策略：找到最早的 system 消息（工具结果），如果它超过 500 字符，
    截断为前 300 字符的摘要，保留关键信息（论文来源和数量）。
    """
    if _estimate_tokens(messages) < max_tokens:
        return

    for i, m in enumerate(messages):
        if m.get("role") != "system":
            continue
        content = m.get("content", "")
        if len(content) <= 500:
            continue

        # 提取关键信息：前 200 字符 + 末尾论文计数
        lines = content.split("\n")
        first_line = lines[0] if lines else ""  # "Found N results..."
        snippet = content[:300].replace("\n", " ")
        m["content"] = (
            f"[压缩] {first_line}\n"
            f"  摘要: {snippet}...\n"
            f"  (完整结果已省略，核心信息在前序轮次中已使用)"
        )
        log(f"Compressed system msg #{i}: {len(content)}→{len(m['content'])} chars")
        break  # 每次只压缩一条，下一轮再压缩下一条


def register_tool(name: str, executor: Callable[[dict], tuple],
                  description: str = "", parameters: dict = None):
    """注册一个新的 Agent skill。新 skill 会自动出现在可用工具列表中。"""
    TOOL_EXECUTORS[name] = executor
    if description:
        TOOLS.append({
            "name": name,
            "description": description,
            "parameters": parameters or {},
        })


def execute_tool(tool_call: ToolCall) -> tuple:
    """分发工具调用到对应的执行器。
    Returns:
        (ToolResult, raw_data) — raw_data 是工具特定的结构化数据
        （search_papers 返回原始结果列表，read_paper 返回 source info dict）
    """
    executor = TOOL_EXECUTORS.get(tool_call.name)
    if executor is None:
        return (ToolResult(tool_call, "", error=f"Unknown tool: {tool_call.name}"), None)
    try:
        return executor(tool_call.arguments)
    except Exception as e:
        return (ToolResult(tool_call, "", error=str(e)), None)


# ── Agent 循环 ──

# 用于检测 <tool_call> JSON 块的正则
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*\n?(.*?)\n?\s*</tool_call>', re.DOTALL
)


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """从文本中提取第一个 <tool_call> JSON 块并解析为 ToolCall。
    如果解析失败返回 None。
    """
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None

    json_str = match.group(1).strip()
    try:
        data = json.loads(json_str)
        name = data.get("name", "")
        arguments = data.get("arguments", {})
        if not name:
            return None
        # 确保 arguments 是 dict
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(name, arguments)
    except (json.JSONDecodeError, TypeError) as e:
        log(f"Failed to parse tool_call JSON: {e}")
        log(f"Raw JSON string: {json_str[:200]}")
        return None


def extract_text_without_tool_call(text: str) -> str:
    """移除 <tool_call> 块，返回剩余文本"""
    return TOOL_CALL_PATTERN.sub('', text).strip()


async def run_agent_loop(
    task_id: str,
    sid: str,
    user_message: str,
    history: list[dict],
    pool=None,
    followup_question: str = "",
    mode: str = "auto",
) -> AsyncGenerator[AgentEvent, None]:
    """
    Agent 循环 — 状态机驱动 (v5.3 多模式)。

    状态机流程由 mode 决定:
      auto:    _classify_intent → chat|research
      chat:    DIRECT → RESPOND
      survey:  RETRIEVE → QUICK_READ → RESPOND
      deep:    RETRIEVE → QUICK_READ → DEEP_READ → RESPOND
      read:    QUICK_READ → RESPOND (no RETRIEVE)
      compute: flexible (search optional, tools open)

    Yields:
        AgentEvent of types: thinking, tool_call, tool_result, text, done, error, state
    """
    from .agent_sm import AgentStateMachine

    # 构建初始消息列表
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    # 添加历史（最近 10 条，过滤掉 _task_id 标记）
    for msg in history[-10:]:
        if msg.get("_task_id"):
            continue
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 添加用户消息
    messages.append({"role": "user", "content": user_message})

    # 根据 LLM 后端选择工具调用方式
    use_native_tools = (LLM_BACKEND == "deepseek")
    if use_native_tools:
        messages[0]["content"] += (
            "\n\n你可以直接调用可用的函数（search_papers / search_arxiv / "
            "search_semantic_scholar / read_paper / read_arxiv_paper 等）来检索和"
            "阅读文献。系统会根据当前阶段自动引导你的行为。"
        )

    # ── 状态机驱动 ──
    sm = AgentStateMachine(
        messages=messages,
        task_id=task_id,
        use_native_tools=use_native_tools,
        execute_tool_fn=_execute_tool_for_sm,
        run_native_round_fn=_run_native_round_for_sm,
        AGENT_SYSTEM_PROMPT=AGENT_SYSTEM_PROMPT,
        paper_pool=pool,
        followup_question=followup_question,
        mode=mode,
    )

    # 委托给状态机，直接转发所有事件
    async for event in sm.run():
        yield event

    # 正常结束
    yield AgentEvent.done()


# ── 状态机适配器 ──

def _execute_tool_for_sm(tool_call: ToolCall) -> tuple:
    """适配 execute_tool 给状态机使用。"""
    return execute_tool(tool_call)


async def _run_native_round_for_sm(
    task_id: str, round_label, messages: list[dict], force_answer: bool = False,
    allowed_tools: list = None, max_tokens: int = None,
) -> AsyncGenerator[AgentEvent, None]:
    """适配 _run_native_round 给状态机使用。"""
    round_num = 0 if isinstance(round_label, str) else round_label
    async for event in _run_native_round(task_id, round_num, messages,
                                          force_answer=force_answer,
                                          allowed_tools=allowed_tools,
                                          max_tokens=max_tokens):
        yield event


async def _run_native_round(
    task_id: str, round_num: int, messages: list[dict],
    force_answer: bool = False,
    allowed_tools: list = None,
    max_tokens: int = None,
) -> AsyncGenerator[AgentEvent, None]:
    """使用原生 Function Calling 执行一轮 LLM 调用。
    实时流式推送 text content，完成后检测 tool_calls。
    如果有 tool_calls，yield 内部 _tool_call 事件；否则 yield done。

    Args:
        force_answer: 为 True 时不传 tools，强制纯文本回答
        allowed_tools: 限制可用工具列表，None 表示使用全部 TOOLS
    """
    full_response = ""
    tool_calls = None
    if force_answer:
        active_tools = []
    elif allowed_tools is not None:
        active_tools = allowed_tools
    else:
        active_tools = TOOLS
    try:
        async for event in chat_completion_with_tools(messages, active_tools, max_tokens=max_tokens):
            if event["type"] == "text":
                full_response += event["content"]
                yield AgentEvent.text(event["content"])
            elif event["type"] == "done":
                tool_calls = event["tool_calls"]
    except Exception as e:
        log(f"TASK {task_id} Round {round_num} LLM error: {e}")
        yield AgentEvent.error(f"LLM error in round {round_num}: {e}")
        return

    if not tool_calls and not full_response.strip():
        log(f"TASK {task_id} Round {round_num} empty response")
        yield AgentEvent.error("Empty response from LLM")
        return

    if tool_calls:
        names = [tc.get("name", "?") for tc in tool_calls]
        log(f"TASK {task_id} Round {round_num} TOOLS (native): {names}")

        # 返回所有 tool_calls（状态机会批量并行执行）
        for tc in tool_calls:
            tc["arguments"]["_tool_call_id"] = tc.get("id", f"call_{round_num}")
            yield AgentEvent("_tool_call", {"tool_call": ToolCall(tc["name"], tc["arguments"])})
        return

    # 没有 tool_call → 最终回答（文本已在流式推送中实时发送）
    log(f"TASK {task_id} Round {round_num} FINAL ANSWER: {len(full_response)} chars")
    yield AgentEvent.done()


async def _run_text_round(
    task_id: str, round_num: int, messages: list[dict],
) -> AsyncGenerator[AgentEvent, None]:
    """使用文本 <tool_call> 解析执行一轮 LLM 调用（OpenClaw fallback）。
    实时流式推送 text chunks，检测 <tool_call> 块。
    """
    full_response = ""
    maybe_tool = False
    try:
        async for chunk in chat_completion_stream(messages):
            full_response += chunk
            if not maybe_tool:
                stripped = full_response.lstrip()
                if "<tool_call>" in full_response:
                    maybe_tool = True
                elif stripped and "<tool_call>".startswith(stripped):
                    maybe_tool = True
                else:
                    yield AgentEvent.text(chunk)
    except Exception as e:
        log(f"TASK {task_id} Round {round_num} LLM error: {e}")
        yield AgentEvent.error(f"LLM error in round {round_num}: {e}")
        return

    if not full_response.strip():
        log(f"TASK {task_id} Round {round_num} empty response")
        yield AgentEvent.error("Empty response from LLM")
        return

    tool_call = parse_tool_call(full_response)

    if tool_call is not None:
        log(f"TASK {task_id} Round {round_num} TOOL (text): {tool_call.name}")

        thinking_text = extract_text_without_tool_call(full_response)
        if thinking_text and maybe_tool:
            yield AgentEvent.thinking(thinking_text)

        tool_call.arguments["_full_response"] = full_response.strip()
        yield AgentEvent("_tool_call", {"tool_call": tool_call})
        return

    # 没有 tool_call → 最终回答
    log(f"TASK {task_id} Round {round_num} FINAL ANSWER: {len(full_response)} chars")

    if maybe_tool:
        chunk_size = 20
        for i in range(0, len(full_response), chunk_size):
            yield AgentEvent.text(full_response[i:i + chunk_size])
    # else: 文本已在流式推送中实时发送

    yield AgentEvent.done()
