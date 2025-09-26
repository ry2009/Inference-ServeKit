import time
import uuid
from typing import Dict, Optional


class SessionManager:
    """Tracks RL episode sessions and KV residency metadata."""

    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def start(self, env_id: str, model: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "env_id": env_id,
            "model": model,
            "last_t": time.time(),
            "kv_bytes": 0,
            "engine_session_id": None,
            "tokens": [],
            "accepted_mask": [],
            "tools": [],
            "meta": {},
        }
        return session_id

    def bind_engine(self, session_id: str, engine_session_id: str) -> None:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session: {session_id}")
        self.sessions[session_id]["engine_session_id"] = engine_session_id

    def touch(self, session_id: str, kv_bytes: int = 0) -> None:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session: {session_id}")
        entry = self.sessions[session_id]
        entry["last_t"] = time.time()
        entry["kv_bytes"] = kv_bytes

    def get(self, session_id: str) -> Optional[dict]:
        return self.sessions.get(session_id)

    def record_tokens(self, session_id: str, tokens: list[str], accepted_mask: list[bool]):
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session: {session_id}")
        entry = self.sessions[session_id]
        entry["tokens"].extend(tokens)
        entry["accepted_mask"].extend(accepted_mask)

    def record_tool(self, session_id: str, tool_call: dict):
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session: {session_id}")
        self.sessions[session_id]["tools"].append(tool_call)

    def set_meta(self, session_id: str, **kwargs):
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session: {session_id}")
        self.sessions[session_id]["meta"].update(kwargs)

    def end(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def stats(self):
        return self.sessions.copy()

    def trace(self, session_id: str) -> dict:
        entry = self.sessions.get(session_id)
        if not entry:
            raise KeyError(f"Unknown session: {session_id}")
        return {
            "env_id": entry["env_id"],
            "model": entry["model"],
            "tokens": list(entry["tokens"]),
            "accepted_mask": list(entry["accepted_mask"]),
            "kv_bytes": entry["kv_bytes"],
            "tools": list(entry["tools"]),
            "meta": dict(entry.get("meta", {})),
        }
