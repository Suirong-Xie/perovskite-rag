"""
Agent Skill 注册表 — 自包含的工具模块。

每个模块导出:
  SCHEMA: dict — OpenAI function calling 格式的工具定义
  execute(arguments) -> tuple[ToolResult, any]: 工具执行函数

添加新 Skill:
  1. 在此目录创建 tools/new_skill.py
  2. 定义 SCHEMA 和 execute()
  3. 在下方 MODULES 列表中添加 import
"""

from . import search_local, search_arxiv, search_s2
from . import read_paper, extract_data
from . import compute
from . import materials
from . import citations
from . import perovskite_db
from . import compare

# 按加载顺序排列 (用于 prompt 中展示顺序)
MODULES = [
    search_local,
    search_arxiv,
    search_s2,
    read_paper,
    extract_data,
    compute,
    materials,
    citations,
    perovskite_db,
    compare,
]

ALL_TOOLS = []
EXECUTORS = {}

for m in MODULES:
    # 支持模块导出多个 schema (如 read_paper 导出 read_paper + read_arxiv_paper)
    if hasattr(m, 'SCHEMAS'):
        for schema in m.SCHEMAS:
            name = schema.get("name", "unknown")
            ALL_TOOLS.append(schema)
            EXECUTORS[name] = m.EXECUTOR_MAP[name]
    elif hasattr(m, 'SCHEMA'):
        name = m.SCHEMA.get("name", "unknown")
        ALL_TOOLS.append(m.SCHEMA)
        EXECUTORS[name] = m.execute

# 按状态限制可用工具
RETRIEVE_TOOLS = {
    "search_papers", "search_arxiv", "search_semantic_scholar",
    "get_citations", "get_references", "search_perovskite_database",
}
READ_TOOLS = {
    "read_paper", "read_arxiv_paper", "extract_data", "compare_papers",
}


def filter_tools(names: set[str]) -> list[dict]:
    """从完整工具列表中过滤出指定名称的工具。"""
    return [t for t in ALL_TOOLS if t["name"] in names]
