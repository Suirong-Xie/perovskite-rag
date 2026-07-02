"""
PerovskiteGPT V5 — 检索服务
通过直接 Python import 调用 search_tool（替代 subprocess）
"""
import sys
import json
import time
from pathlib import Path
from typing import Optional
from ..core.config import SUNNY_RAG_DIR, SEARCH_DATA_VERSION, SEARCH_DEFAULT_TOP_K


# 将 sunny-rag/scripts/ 加入 sys.path，便于直接 import search_tool
_SEARCH_SCRIPTS_DIR = str(SUNNY_RAG_DIR / "scripts")
if _SEARCH_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SEARCH_SCRIPTS_DIR)

# 延迟导入，避免循环依赖
_search_module = None


def _get_search_module():
    """延迟加载 search_tool 模块"""
    global _search_module
    if _search_module is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "search_tool",
            SUNNY_RAG_DIR / "scripts" / "search_tool.py",
        )
        _search_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_search_module)
    return _search_module


# 缓存搜索结果
_cache: dict = {}


def search_papers(query: str, top_k: int = None, data_version: str = None,
                  clear_cache: bool = False) -> list:
    """
    语义搜索论文。
    复用 sunny-rag/scripts/search_tool.py 的 search() 函数。
    """
    if clear_cache:
        _cache.clear()

    top_k = top_k or SEARCH_DEFAULT_TOP_K
    data_version = data_version or SEARCH_DATA_VERSION

    cache_key = f"{query}:{top_k}:{data_version}"
    if cache_key in _cache:
        return _cache[cache_key]

    module = _get_search_module()
    start = time.time()
    results = module.search(query, top_k=top_k, journal_boost=True, data_version=data_version)
    elapsed = time.time() - start
    print(f"[V5] SEARCH: '{query[:60]}' → {len(results)} results in {elapsed:.2f}s", flush=True)

    _cache[cache_key] = results
    return results


def clear_cache():
    """清除搜索缓存"""
    _cache.clear()
