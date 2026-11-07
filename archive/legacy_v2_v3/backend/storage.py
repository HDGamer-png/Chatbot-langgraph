import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAT_DIR = REPO_ROOT / "chat_history"
USER_DATA_FILE = REPO_ROOT / "user_data.json"
CHAT_DIR.mkdir(parents=True, exist_ok=True)


class ChatHistoryStore:
    def __init__(self, chat_dir: Path = CHAT_DIR):
        self.chat_dir = chat_dir
        self.chat_dir.mkdir(parents=True, exist_ok=True)

    def session_path(self, session_id: str) -> Path:
        return self.chat_dir / f"session_{session_id}.json"

    def create_session(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "user_id": user_id or "",
            "turns": [],
            "agent_stats": {},
        }
        self.save_session(session_id, data)
        return data

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        for path in sorted(self.chat_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            sessions.append({
                "session_id": path.name[len("session_"):-len(".json")],
                "modified": path.stat().st_mtime,
            })
        return sessions

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self.session_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        path = self.session_path(session_id)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_turn(self, session_id: str, turn: Dict[str, Any]) -> None:
        data = self.load_session(session_id)
        if data is None:
            data = self.create_session(session_id)
        turns = data.setdefault("turns", [])
        turns.append(turn)
        data["updated_at"] = datetime.now().isoformat()
        self.save_session(session_id, data)

    def get_history_text(self, session_id: str, n: int = 6) -> str:
        data = self.load_session(session_id)
        if not data:
            return ""
        turns = data.get("turns", [])[-n:]
        lines = []
        for t in turns:
            lines.append(f"Người dùng: {t.get('user_query', '')}")
            lines.append(f"Chatbot: {t.get('final_answer', '')[:200]}")
        return "\n".join(lines)

    def get_process(self, session_id: str, turn_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        data = self.load_session(session_id)
        if not data:
            return None
        turns = data.get("turns", [])
        if not turns:
            return {"session_id": session_id, "turns": []}
        if turn_id is None:
            turn = turns[-1]
        else:
            turn = next((t for t in turns if t.get("turn_id") == turn_id), None)
            if turn is None:
                return None
        return {
            "session_id": session_id,
            "turn_id": turn.get("turn_id"),
            "timestamp": turn.get("timestamp"),
            "user_query": turn.get("user_query"),
            "intent": turn.get("intent"),
            "final_answer": turn.get("final_answer"),
            "agent_timings": turn.get("agent_timings", []),
            "call_graph": turn.get("call_graph", []),
            "log": turn.get("log", []),
        }


class UserDataStore:
    def __init__(self, data_file: Path = USER_DATA_FILE):
        self.data_file = data_file
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            self.data_file.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
        self._load()

    def _load(self) -> None:
        try:
            self._data = json.loads(self.data_file.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}

    def _save(self) -> None:
        self.data_file.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_user(self, user_id: str) -> Dict[str, Any]:
        return self._data.get(user_id, {})

    def save_user(self, user_id: str, profile: Dict[str, Any]) -> None:
        profile = dict(profile)
        profile["updated_at"] = datetime.now().isoformat()
        self._data[user_id] = profile
        self._save()

    def list_users(self) -> List[Dict[str, Any]]:
        return [
            {"user_id": user_id, **profile}
            for user_id, profile in self._data.items()
        ]
