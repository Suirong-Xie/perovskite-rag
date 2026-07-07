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
from ..core.config import AGENT_MAX_ROUNDS, PAPERS_DIR, JOURNALS_PDF_DIR, LLM_BACKEND
from ..core.schemas import AgentEvent, ToolCall, ToolResult
from ..core.llm import chat_completion_stream, chat_completion_with_tools
from .retrieval import search_papers

# ── 工具定义 ──

TOOLS = [
    {
        "name": "search_papers",
        "description": (
            "Search the perovskite solar cell research paper database for papers "
            "matching a scientific query. Returns ranked results with journal name, "
            "source filename, and content snippets. Use this to find relevant papers "
            "before answering."
        ),
        "parameters": {
            "query": "English search query string (e.g., 'inverted perovskite solar cell stability')",
            "top_k": "Number of results to return (default 5, max 10)",
        },
    },
    {
        "name": "read_paper",
        "description": (
            "Read the full text of a specific paper given its source filename. "
            "Use this when search results are insufficient and you need to read "
            "a paper in detail. Source filenames look like 'Nature_2021_s41467-021-26121-1.pdf'."
        ),
        "parameters": {
            "source": "Paper source filename from search results (e.g., 'Nature_2021_xxx.pdf')",
        },
    },
]

# ── Agent 系统 prompt ──

AGENT_SYSTEM_PROMPT = """你是 Sunny，钙钛矿太阳能电池领域的 AI 研究助手。
你是一个具备文献检索能力的科研 Agent，你的知识来源是钙钛矿论文数据库（chunked_v3，收录 Nature 系列期刊论文）。

## 工作方式

对于用户的每个问题，你必须：

1. **分析问题** — 这个问题涉及哪些方面？哪些关键词最相关？
2. **搜索文献** — 使用 search_papers 工具从数据库中检索相关论文
3. **深入阅读**（如果需要）— 用 read_paper 读取关键论文的全文
4. **综合回答** — 基于文献数据给出带引用的答案

## 搜索策略（重要）

- 始终使用英文进行搜索，关键词要精确（如 "hole transport layer stability" 而不是 "improve solar cells"）
- 复杂问题要拆解：如果一个问題有多个维度（效率、稳定性、工艺），分别搜索
- 如果第一次搜索结果不够理想，尝试换个角度重新搜索
- 搜索后必须引用具体文献，不要凭空回答

## 可用工具

要调用工具，输出一个 JSON 块（每次只能调用一个）：

<tool_call>
{"name": "search_papers", "arguments": {"query": "英文搜索查询", "top_k": 5}}
</tool_call>

<tool_call>
{"name": "read_paper", "arguments": {"source": "论文文件名.pdf"}}
</tool_call>

规则：
- 调用工具时，只输出 <tool_call> 块，不要写其他内容
- 获得足够信息后，直接输出最终回答，不要输出 <tool_call>
- 回答中不要说你搜了什么、查了什么，直接给答案

## 引用规则（极其重要，违反将导致引用失效）

- 你只能用搜索工具返回的 **File ID** 来构建引用链接
- 搜索结果的每条记录都有 File ID 字段（如 `Nature_2021_s41467-021-26121-1`）
- 引用格式：`[📄](/api/pdf/文件ID)` — 把 File ID 原样填入
- **绝对禁止自己编造 File ID**，哪怕你知道某篇论文的内容，只要搜索结果里没有，就不能引用
- 如果搜索结果不够，请再次搜索或换关键词搜索，不要凭空引用
- 正确示例：搜索返回 File ID: `NatComm_2014_ncomms4461` → 引用写作 `[📄](/api/pdf/NatComm_2014_ncomms4461)`
- 错误示例：自己编一个 `Nature_2023_On-device_lead_sequestration...` → 这种文件不存在！

## 回答风格

- 专业、直接、基于数据
- 先给框架和结论，再给具体数据点
- 每个数据点后面必须附加引用链接
- 不要在末尾额外追加来源列表，引用在正文中就够"""

# ── 工具执行 ──


def log(msg: str):
    print(f"[V5:Agent] {msg}", flush=True)


# journal_name（搜索结果）→ journals_pdf 子目录映射，用于 O(1) PDF 查找
JOURNAL_DIR_MAP = {
    "Nature": "Nature",
    "NatEnergy": "NatEnergy",
    "NatMater": "NatMater",
    "NatPhoton": "NatPhoton",
    "NatNanotech": "NatNanotech",
    "NatComm": "NatComm",
    "Science": "Science",
}


def find_pdf_fast(source: str, journal_name: str = "") -> Optional[str]:
    """按 source 文件名 + journal_name 查找 PDF。
    先用 journal_name 做 O(1) 查找 journals_pdf/{journal}/，
    fallback 到扫描 papers_pdf/{year}/{month}/。
    """
    # 1. journals_pdf/{journal}/{source} — O(1) with journal_name
    journal_dir_name = JOURNAL_DIR_MAP.get(journal_name, "")
    if journal_dir_name:
        pdf_file = JOURNALS_PDF_DIR / journal_dir_name / source
        if pdf_file.exists():
            return str(pdf_file)
    # 2. journals_pdf/*/ — scan all journal dirs (fallback)
    if JOURNALS_PDF_DIR.exists():
        for journal_dir in JOURNALS_PDF_DIR.iterdir():
            if not journal_dir.is_dir():
                continue
            if journal_dir.name == journal_dir_name:
                continue  # already checked above
            pdf_file = journal_dir / source
            if pdf_file.exists():
                return str(pdf_file)
    # 3. papers_pdf/{year}/{month}/{source} — 旧 arXiv 数据
    for year_dir in sorted(PAPERS_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            pdf_file = month_dir / source
            if pdf_file.exists():
                return str(pdf_file)
    return None


def find_pdf_path(source: str) -> Optional[str]:
    """按 source 文件名查找 PDF（兼容旧接口，内部委托给 find_pdf_fast）"""
    return find_pdf_fast(source)


def execute_search_tool(arguments: dict) -> tuple:
    """执行 search_papers 工具。
    Returns:
        (ToolResult, raw_results) — raw_results 是原始搜索结果列表，
        供上层（chat.py）累积到 TaskInfo.sources 用于 PDF 寻回和高亮。
    """
    query = arguments.get("query", "")
    top_k = min(int(arguments.get("top_k", 5)), 10)
    if not query:
        return (ToolResult(
            ToolCall("search_papers", arguments),
            "", error="query is required"
        ), [])

    results = search_papers(query, top_k=top_k)
    if not results:
        return (ToolResult(
            ToolCall("search_papers", arguments),
            "No results found for this query.",
        ), [])

    # 格式化结果为紧凑但信息丰富的文本
    output_lines = [f"Found {len(results)} results for '{query}':\n"]
    for i, r in enumerate(results):
        file_id = r.get("source", "").replace(".pdf", "")
        output_lines.append(
            f"[{i+1}] {r.get('journal_name', 'Unknown')} | "
            f"Similarity: {r.get('similarity', 0):.3f} | "
            f"Source: {r.get('source', 'N/A')} | "
            f"File ID: {file_id}\n"
            f"    Content: {r.get('content', '')[:600]}"
        )
    return (ToolResult(
        ToolCall("search_papers", arguments),
        "\n".join(output_lines),
    ), results)


def execute_read_tool(arguments: dict) -> tuple:
    """执行 read_paper 工具。
    Returns:
        (ToolResult, raw_info) — raw_info 是包含 source/journal 的 dict，
        供上层累积到 TaskInfo.sources。
    """
    source = arguments.get("source", "")
    if not source:
        return (ToolResult(
            ToolCall("read_paper", arguments),
            "", error="source is required"
        ), {})

    pdf_path = find_pdf_path(source)
    if not pdf_path:
        return (ToolResult(
            ToolCall("read_paper", arguments),
            f"PDF not found for source: {source}",
        ), {})

    try:
        proc = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return (ToolResult(
                ToolCall("read_paper", arguments),
                "", error=f"pdftotext error: {proc.stderr[:200]}"
            ), {})
        text = proc.stdout[:5000]  # 限制长度避免 token 爆炸
        return (ToolResult(
            ToolCall("read_paper", arguments),
            f"Content of {source} (first 5000 chars):\n\n{text}",
        ), {"source": source, "content": text[:600]})
    except subprocess.TimeoutExpired:
        return (ToolResult(
            ToolCall("read_paper", arguments),
            "", error="pdftotext timed out after 30s"
        ), {})
    except Exception as e:
        return (ToolResult(
            ToolCall("read_paper", arguments),
            "", error=str(e)
        ), {})


# ── Skill 注册表 ──
# 添加新 skill: TOOL_EXECUTORS["name"] = executor_function
# 每个 executor 返回 (ToolResult, raw_data) 元组
TOOL_EXECUTORS: dict[str, Callable[[dict], tuple]] = {
    "search_papers": execute_search_tool,
    "read_paper": execute_read_tool,
}


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
) -> AsyncGenerator[AgentEvent, None]:
    """
    ReAct Agent 循环 — async generator。

    每个 round:
      1. 调用 LLM（streaming，实时推送 text chunks）
      2. 检测 tool_call（原生 function calling 或文本 <tool_call> 解析）
         - 有 → 执行工具，yield AgentEvent.tool_call + tool_result，进入下一轮
         - 无 → yield AgentEvent.text，结束

    Yields:
        AgentEvent of types: thinking, tool_call, tool_result, text, done, error
    """
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
        # 为原生 function calling 添加提示（工具通过 API tools 参数传入）
        messages[0]["content"] += (
            "\n\n你可以直接调用可用的函数（search_papers / read_paper）来检索和阅读文献。"
            "请先用 search_papers 检索相关论文，必要时用 read_paper 深入阅读，"
            "然后基于文献数据给出带引用的回答。"
        )

    # Agent 循环
    for round_num in range(1, AGENT_MAX_ROUNDS + 1):
        log(f"TASK {task_id} Round {round_num}/{AGENT_MAX_ROUNDS} "
            f"({'native' if use_native_tools else 'text'} tools)")

        if use_native_tools:
            # ── 原生 Function Calling 路径 ──
            gen = _run_native_round(task_id, round_num, messages)
        else:
            # ── 文本 <tool_call> 解析路径 ──
            gen = _run_text_round(task_id, round_num, messages)

        tool_call = None
        async for event in gen:
            if event.type == "_tool_call":
                # 内部信号：有工具调用需要执行
                tool_call = event.data["tool_call"]
            else:
                yield event

        if tool_call is None:
            # 没有工具调用 → 最终回答（done 已在 gen 中 yield）
            return

        # 执行工具
        log(f"TASK {task_id} Round {round_num} TOOL: {tool_call.name}")

        yield AgentEvent.tool_call(tool_call.name, tool_call.arguments)

        result, raw_data = execute_tool(tool_call)
        log(f"TASK {task_id} Round {round_num} TOOL RESULT: "
            f"{tool_call.name} → {len(result.output)} chars"
            f"{' ERROR: ' + result.error if result.error else ''}")

        yield AgentEvent.tool_result(
            tool_call.name,
            result.output[:300] if result.output else "(empty)",
            result.error,
        )

        # 将结构化原始数据通过 search_results 事件传出
        if tool_call.name == "search_papers" and raw_data:
            yield AgentEvent.search_results(raw_data)
        elif tool_call.name == "read_paper" and raw_data:
            yield AgentEvent.search_results([raw_data])

        # 追加工具调用和结果到对话历史
        if use_native_tools:
            # Native tools 格式：assistant 消息含 tool_calls + tool 角色消息
            tool_call_id = tool_call.arguments.pop("_tool_call_id", f"call_{round_num}")
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    },
                }],
            })
            result_content = result.output
            if result.error:
                result_content += f"\nError: {result.error}"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_content,
            })
        else:
            # Text tools 格式：assistant 消息 + system 消息
            messages.append({
                "role": "assistant",
                "content": tool_call.arguments.pop("_full_response", ""),
            })
            result_content = f"Tool result for {tool_call.name}:\n{result.output}"
            if result.error:
                result_content += f"\nError: {result.error}"
            messages.append({"role": "system", "content": result_content})

        # 继续下一轮

    # 达到最大轮数
    log(f"TASK {task_id} Max rounds ({AGENT_MAX_ROUNDS}) reached, forcing final answer")
    yield AgentEvent.error(
        f"Reached maximum of {AGENT_MAX_ROUNDS} tool-calling rounds. "
        "Please try a more specific question."
    )


async def _run_native_round(
    task_id: str, round_num: int, messages: list[dict],
) -> AsyncGenerator[AgentEvent, None]:
    """使用原生 Function Calling 执行一轮 LLM 调用。
    实时流式推送 text content，完成后检测 tool_calls。
    如果有 tool_calls，yield 内部 _tool_call 事件；否则 yield done。
    """
    full_response = ""
    tool_calls = None
    try:
        async for event in chat_completion_with_tools(messages, TOOLS):
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
        # 有工具调用：已实时推送的 text 是 thinking（chat.py 会在 tool_call 时重置）
        log(f"TASK {task_id} Round {round_num} TOOL (native): {tool_calls[0]['name']}")

        # 取第一个 tool_call（后续可扩展并行执行）
        tc = tool_calls[0]
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
