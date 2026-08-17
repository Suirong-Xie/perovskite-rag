"""
PerovskiteGPT v1.5 — 翻译服务
中文 → 英文（用于检索 query 优化）
"""
from ..core.llm import chat_completion


async def translate_to_english(text: str) -> str:
    """
    检测中文输入并翻译为英文。
    如果输入不含中文则原样返回。
    """
    if not any('一' <= c <= '鿿' for c in text[:10]):
        return text  # 非中文，不需要翻译

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a translator. Translate the following Chinese perovskite "
                    "research question to English. Output ONLY the English translation, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ]
        result = await chat_completion(messages, max_tokens=100)
        translated = result.strip().strip('"')
        if translated:
            print(f"[v1.5] TRANSLATE: '{text}' → '{translated}'", flush=True)
            return translated
    except Exception as exc:
        print(f"[v1.5] TRANSLATE error: {exc}", flush=True)

    return text  # 翻译失败则返回原文
