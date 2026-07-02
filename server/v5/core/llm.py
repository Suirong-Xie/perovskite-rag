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
