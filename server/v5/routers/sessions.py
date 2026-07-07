"""
PerovskiteGPT V5 — 会话管理 API Router
"""
from fastapi import APIRouter
from ..services.session_store import store

router = APIRouter()


@router.get("/api/sessions")
def list_sessions():
    """列出所有会话"""
    return {"sessions": store.list_all(), "order": store.order}


@router.post("/api/sessions")
def create_session():
    """创建新会话"""
    sid = store.create()
    return {"id": sid, "title": store.get(sid).get("title", "")}


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    store.delete(session_id)
    return {"ok": True}


@router.put("/api/sessions/{session_id}")
def rename_session(session_id: str, data: dict):
    store.rename(session_id, data.get("title", ""))
    return {"ok": True}


@router.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """获取会话的消息历史"""
    from ..routers.chat import _tasks
    if session_id not in store.sessions:
        return {"messages": []}
    msgs = store.get_history(session_id)
    # 检查是否有活跃的生成任务（即使尚无输出，也告诉前端有任务在跑）
    for tid, t in _tasks.items():
        if t.sid == session_id and not t.done:
            full = "".join(t.chunks)
            msgs = msgs + [{
                "role": "assistant",
                "content": full + ("\n\n_(回答未完成)_" if full else ""),
                "_task_id": tid,
            }]
            break
    return {"messages": msgs}
