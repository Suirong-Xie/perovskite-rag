"""Session persistence: load, save, manage conversation history."""

import json
import os
from typing import List, Dict
from collections import OrderedDict

from config import SESSION_DIR, MAX_HISTORY_ROUNDS, LRU_CACHE_SIZE

# ── LRU cache for hot sessions ──
_session_cache: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()


def _get_session_path(session_id: str) -> str:
    return os.path.join(SESSION_DIR, f"{session_id}.json")


def load_session(session_id: str) -> List[Dict[str, str]]:
    """Load session from disk, with LRU caching."""
    path = _get_session_path(session_id)
    if not os.path.exists(path):
        return []
    
    # Check cache first
    if session_id in _session_cache:
        _session_cache.move_to_end(session_id)
        return _session_cache[session_id]
    
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    # Cache it
    _session_cache[session_id] = history
    if len(_session_cache) > LRU_CACHE_SIZE:
        _session_cache.popitem(last=False)
    
    return history


def save_session(session_id: str, history: List[Dict[str, str]]):
    """Save session to disk."""
    path = _get_session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    # Update cache
    _session_cache[session_id] = history


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Get history for prompt assembly (last N rounds)."""
    history = load_session(session_id)
    return history[-MAX_HISTORY_ROUNDS * 2:]  # each round = (user + assistant)


def update_session_history(session_id: str, history: List[Dict[str, str]]):
    """Save updated history."""
    save_session(session_id, history)


def get_history_text(session_id: str) -> str:
    """Format recent history as text for prompt."""
    history = get_session_history(session_id)
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{'Human' if role == 'user' else 'Assistant'}: {content}")
    return "\n".join(lines)


def add_to_history(session_id: str, question: str, answer: str, sources: List[str] = None):
    """Append a QA pair to session history."""
    history = load_session(session_id)
    history.append({"role": "user", "content": question})
    
    answer_entry = {"role": "assistant", "content": answer}
    if sources:
        answer_entry["sources"] = sources
    history.append(answer_entry)
    
    save_session(session_id, history)


def list_all_sessions() -> List[dict]:
    """List all sessions with metadata."""
    sessions = []
    if not os.path.exists(SESSION_DIR):
        return sessions
    for fname in sorted(os.listdir(SESSION_DIR)):
        if fname.endswith(".json"):
            sid = fname.replace(".json", "")
            path = os.path.join(SESSION_DIR, fname)
            size = os.path.getsize(path)
            mtime = os.path.getmtime(path)
            # Peek at first message for title
            history = load_session(sid)
            title = sid
            for msg in history:
                if msg.get("role") == "user":
                    title = msg["content"][:60]
                    break
            sessions.append({
                "id": sid,
                "title": title,
                "message_count": len(history) // 2,
                "size": size,
                "mtime": mtime,
            })
    return sorted(sessions, key=lambda x: x["mtime"], reverse=True)


def delete_session(session_id: str):
    """Delete a session file."""
    path = _get_session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
    _session_cache.pop(session_id, None)


def rename_session(session_id: str, new_title: str):
    """Rename is stored as a special entry in the session file."""
    # We store the title as the first message's content if it's a title marker
    history = load_session(session_id)
    # Title is stored as a separate marker entry at position 0
    if history and history[0].get("_type") == "title":
        history[0]["content"] = new_title
    else:
        history.insert(0, {"_type": "title", "content": new_title})
    save_session(session_id, history)
