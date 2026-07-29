"""
PerovskiteGPT V5 — Pydantic 数据模型
"""
from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    paper_id: Optional[str] = None


class TaskInfo:
    """生成任务的状态信息"""
    def __init__(self, sid: str):
        self.sid = sid
        self.chunks: list[str] = []
        self.done = False
        self.error: Optional[str] = None
        self.cancelled = False
        # 累积 Agent 循环中所有 search_papers / read_paper 的原始结果
        self.sources: list[dict] = []
        # 经过 find_pdf_path 验证确实存在的 file_id 集合
        self.pdfs_validated: set[str] = set()
        # 验证后的 sources 列表的 JSON 字符串（Agent 完成后设置，SSE 端点推送）
        self.sources_json: Optional[str] = None
        # Agent 状态机当前状态（供 status API 和 SSE 暴露）
        self.agent_state: Optional[dict] = None
        # 后续建议列表的 JSON 字符串（Agent 完成后设置）
        self.suggestions_json: Optional[str] = None


class SessionSummary:
    def __init__(self, sid: str, title: str = "", message_count: int = 0):
        self.id = sid
        self.title = title
        self.message_count = message_count


class SearchResult:
    def __init__(self, data: dict):
        self.rank: int = data.get("rank", 0)
        self.similarity: float = data.get("similarity", 0.0)
        self.journal_rank: int = data.get("journal_rank", 7)
        self.journal_name: str = data.get("journal_name", "Other")
        self.source: str = data.get("source", "")
        self.path: str = data.get("path", "")
        self.content: str = data.get("content", "")
        self.idx: int = data.get("idx", 0)

    @property
    def file_id(self) -> str:
        return self.source.replace(".pdf", "")


class ToolCall:
    """Agent 工具调用"""
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}


class ToolResult:
    """工具调用结果"""
    def __init__(self, tool_call: ToolCall, output: str, error: Optional[str] = None):
        self.tool_call = tool_call
        self.output = output
        self.error = error

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_call.name,
            "output": self.output[:500],
            "error": self.error,
        }


class AgentEvent:
    """Agent 循环产生的事件，通过 SSE 流式推送到前端"""
    def __init__(self, event_type: str, data: dict):
        self.type = event_type
        self.data = data

    @classmethod
    def thinking(cls, content: str) -> "AgentEvent":
        return cls("thinking", {"content": content})

    @classmethod
    def tool_call(cls, name: str, arguments: dict) -> "AgentEvent":
        return cls("tool_call", {"name": name, "arguments": arguments})

    @classmethod
    def tool_result(cls, name: str, summary: str, error: Optional[str] = None) -> "AgentEvent":
        return cls("tool_result", {"name": name, "summary": summary, "error": error})

    @classmethod
    def text(cls, content: str) -> "AgentEvent":
        return cls("text", {"content": content})

    @classmethod
    def done(cls) -> "AgentEvent":
        return cls("done", {})

    @classmethod
    def search_results(cls, data: list[dict]) -> "AgentEvent":
        """搜索结果的结构化数据（Agent 循环内部使用，由 chat.py 消费）"""
        return cls("search_results", {"results": data})

    @classmethod
    def error(cls, message: str) -> "AgentEvent":
        return cls("error", {"message": message})

    @classmethod
    def state_change(cls, state: str, summary: dict) -> "AgentEvent":
        """Agent 状态机状态切换事件，供前端展示进度。"""
        return cls("state", {"current_state": state, **summary})

    @classmethod
    def suggestions(cls, items: list[str]) -> "AgentEvent":
        """后续研究建议列表。"""
        return cls("suggestions", {"items": items})
