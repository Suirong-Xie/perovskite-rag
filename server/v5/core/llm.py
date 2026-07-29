"""
PerovskiteGPT V5 — LLM Client 抽象层
支持 OpenClaw Gateway（Sunny agent）和 DeepSeek API
"""
import json
import httpx
from .config import (
    LLM_BACKEND,
    OPENCLAW_GATEWAY_URL, OPENCLAW_GATEWAY_TOKEN, OPENCLAW_MODEL,
    DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
)

# ── LLM 调用抽象 ──


def _get_backend_config():
    """根据 LLM_BACKEND 返回 (url, token, model)"""
    if LLM_BACKEND == "deepseek":
        # DeepSeek base_url 已含 /v1，只需追加 /chat/completions
        return (
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            DEEPSEEK_API_KEY,
            DEEPSEEK_MODEL,
        )
    else:
        # OpenClaw gateway: host + /v1/chat/completions
        return (
            f"{OPENCLAW_GATEWAY_URL}/v1/chat/completions",
            OPENCLAW_GATEWAY_TOKEN,
            OPENCLAW_MODEL,
        )


async def chat_completion_stream(
    messages: list[dict],
    model: str = None,
    gateway_url: str = None,
    api_token: str = None,
    timeout: float = 120.0,
):
    """
    向 LLM 平台发送流式 chat completion 请求。
    后端由 LLM_BACKEND 配置决定（默认 deepseek）。
    使用 async generator yield 每个 text delta。
    """
    if gateway_url and api_token:
        url = gateway_url
        token = api_token
        model_name = model or DEEPSEEK_MODEL
    else:
        url, token, model_name = _get_backend_config()
        if model:
            model_name = model

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", url, json=payload, headers=headers,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"LLM API error: HTTP {resp.status_code} — {body[:300]!r}")

            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError):
                        pass


async def chat_completion_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str = None,
    timeout: float = 120.0,
    max_tokens: int = None,
):
    """
    使用原生 function calling 的流式 chat completion。
    返回 (text_content: str, tool_calls: list[dict] | None)。

    tool_calls 格式: [{"id": "call_xxx", "name": "search_papers", "arguments": {...}}]

    目前仅支持 DeepSeek 后端（OpenAI-compatible tools API）。
    """
    url, token, model_name = _get_backend_config()
    if model:
        model_name = model

    # 将内部工具定义转为 OpenAI tools 格式
    openai_tools = []
    for t in tools:
        properties = {}
        required = []
        for param_name, param_desc in t.get("parameters", {}).items():
            # 简单推断类型：含 "Number" 或 "top_k" → integer，否则 string
            param_type = "integer" if ("top_k" in param_name or "number" in param_desc.lower()) else "string"
            properties[param_name] = {"type": param_type, "description": param_desc}
            required.append(param_name)

        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })

    payload = {
        "model": model_name,
        "messages": messages,
        "tools": openai_tools,
        "stream": True,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    text_content = ""
    tool_calls_acc: dict[int, dict] = {}  # index → {id, name, arguments_str}

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", url, json=payload, headers=headers,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"LLM API error: HTTP {resp.status_code} — {body[:300]!r}")

            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})

                        # 文本内容
                        if "content" in delta and delta["content"]:
                            text_content += delta["content"]
                            yield {"type": "text", "content": delta["content"]}

                        # 工具调用（流式累积）
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": tc.get("id", ""),
                                        "name": "",
                                        "arguments_str": "",
                                    }
                                if "id" in tc and tc["id"]:
                                    tool_calls_acc[idx]["id"] = tc["id"]
                                func = tc.get("function", {})
                                if "name" in func and func["name"]:
                                    tool_calls_acc[idx]["name"] = func["name"]
                                if "arguments" in func:
                                    tool_calls_acc[idx]["arguments_str"] += func["arguments"]
                    except (json.JSONDecodeError, KeyError):
                        pass

    # 解析累积的 tool_calls
    tool_calls = []
    for idx in sorted(tool_calls_acc.keys()):
        tc = tool_calls_acc[idx]
        try:
            args = json.loads(tc["arguments_str"]) if tc["arguments_str"].strip() else {}
        except json.JSONDecodeError:
            args = {}
        if tc["name"]:
            tool_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "arguments": args,
            })

    # 最终 yield: 工具调用结果（如果有）
    yield {"type": "done", "tool_calls": tool_calls if tool_calls else None}


async def chat_completion(
    messages: list[dict],
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    max_tokens: int = 512,
    timeout: float = 60.0,
) -> str:
    """
    向 DeepSeek API 发送非流式 chat completion 请求。
    用于翻译等短任务。
    """
    url = f"{base_url or DEEPSEEK_BASE_URL}/chat/completions"
    key = api_key or DEEPSEEK_API_KEY
    model_name = model or DEEPSEEK_MODEL

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek API error: HTTP {resp.status_code} — {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]
