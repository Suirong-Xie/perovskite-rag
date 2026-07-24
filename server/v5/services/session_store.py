"""
PerovskiteGPT V5 — 会话持久化
管理 session 元数据和消息历史
"""
import json
import os
import uuid
from pathlib import Path
from ..core.config import SESSIONS_FILE, SESSIONS_DIR


class SessionStore:
    """会话的 CRUD 封装"""

    def __init__(self):
        self.sessions: dict = {}
        self.order: list = []

    def load(self):
        """从磁盘加载所有 session 元数据"""
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        if SESSIONS_FILE.exists():
            with open(SESSIONS_FILE) as f:
                data = json.load(f)
            sessions_in = data.get("sessions", {})
            self.order = data.get("order", [])
            migrated = False
            self.sessions = {}
            for sid, s in sessions_in.items():
                if "messages" in s and s["messages"]:
                    # 旧格式迁移：messages 从 sessions.json 内嵌 → 独立 history.json
                    session_dir = SESSIONS_DIR / sid
                    session_dir.mkdir(parents=True, exist_ok=True)
                    history_file = session_dir / "history.json"
                    if not history_file.exists():
                        with open(history_file, "w") as f:
                            json.dump(s["messages"], f, ensure_ascii=False)
                            f.flush()
                            os.fsync(f.fileno())
                    self.sessions[sid] = {"title": s.get("title", ""), "message_count": len(s["messages"])}
                    migrated = True
                else:
                    session_dir = SESSIONS_DIR / sid
                    history_file = session_dir / "history.json"
                    mc = s.get("message_count", 0)
                    if history_file.exists():
                        with open(history_file) as f:
                            mc = len(json.load(f))
                    self.sessions[sid] = {"title": s.get("title", ""), "message_count": mc}
            if migrated:
                self._save()
        else:
            self.sessions = {}

    def _save(self):
        """持久化 session 元数据"""
        SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        summary = {}
        for sid, s in self.sessions.items():
            summary[sid] = {"title": s.get("title", ""), "message_count": s.get("message_count", 0)}
        with open(SESSIONS_FILE, "w") as f:
            json.dump({"sessions": summary, "order": self.order}, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def create(self, title: str = "") -> str:
        """创建新 session，返回 session_id"""
        sid = uuid.uuid4().hex[:12]
        self.sessions[sid] = {"title": title or f"会话 {len(self.sessions) + 1}", "message_count": 0}
        self.order.insert(0, sid)
        self._save()
        return sid

    def exists(self, sid: str) -> bool:
        return sid in self.sessions

    def get(self, sid: str) -> dict:
        return self.sessions.get(sid, {})

    def delete(self, sid: str):
        if sid in self.sessions:
            del self.sessions[sid]
            if sid in self.order:
                self.order.remove(sid)
            import shutil
            session_dir = SESSIONS_DIR / sid
            if session_dir.exists():
                shutil.rmtree(session_dir)
            self._save()

    def rename(self, sid: str, title: str):
        if sid in self.sessions and title.strip():
            self.sessions[sid]["title"] = title.strip()[:60]
            self._save()

    def append_message(self, sid: str, role: str, content: str, sources: list = None,
                       thinking_chain: str = None):
        """追加消息到 session 历史，可选附带参考来源列表和思考链路"""
        session_dir = SESSIONS_DIR / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        history_file = session_dir / "history.json"
        msgs = []
        if history_file.exists():
            with open(history_file) as f:
                msgs = json.load(f)
        msg = {"role": role, "content": content}
        if sources:
            msg["sources"] = sources
        if thinking_chain:
            msg["thinking_chain"] = thinking_chain
        msgs.append(msg)
        with open(history_file, "w") as f:
            json.dump(msgs, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if sid not in self.sessions:
            self.sessions[sid] = {"title": content[:50] if role == "user" else "", "message_count": 0}
        self.sessions[sid]["message_count"] = len(msgs)
        # 自动更新标题（取第一条用户消息）
        if role == "user" and len([m for m in msgs if m["role"] == "user"]) <= 1:
            title = content[:40]
            if len(content) > 40:
                title += "......"
            self.sessions[sid]["title"] = title
        self._save()

    def get_history(self, sid: str) -> list:
        """获取 session 的消息历史"""
        history_file = SESSIONS_DIR / sid / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                return json.load(f)
        return []

    def list_all(self) -> list:
        """按 order 列出所有 session 摘要"""
        result = []
        for sid in self.order:
            if sid in self.sessions:
                s = self.sessions[sid]
                result.append({
                    "id": sid,
                    "title": s.get("title", ""),
                    "message_count": s.get("message_count", 0),
                })
        return result


# 全局单例
store = SessionStore()
