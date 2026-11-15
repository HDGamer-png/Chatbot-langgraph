import json
import sqlite3
from datetime import datetime
from typing import Optional


class ChatDatabase:
    # Khởi tạo kết nối SQLite và tạo schema bảng nếu chưa tồn tại
    def __init__(self, db_path: str = "chatbot.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    # Tạo bảng sessions, turns, agent_timings và users nếu chưa có
    def _create_tables(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    user_id     TEXT DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id       TEXT    NOT NULL,
                    turn_id          INTEGER NOT NULL,
                    timestamp        TEXT    NOT NULL,
                    user_query       TEXT    NOT NULL,
                    final_answer     TEXT    DEFAULT '',
                    intent           TEXT    DEFAULT '',
                    confidence_score REAL    DEFAULT 0.0,
                    attachments_json TEXT    DEFAULT '[]',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_timings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_id     INTEGER NOT NULL,
                    session_id  TEXT    NOT NULL,
                    agent_name  TEXT    NOT NULL,
                    called_by   TEXT    DEFAULT '',
                    duration    REAL    DEFAULT 0.0,
                    calls_json  TEXT    DEFAULT '[]'
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id     TEXT PRIMARY KEY,
                    name        TEXT DEFAULT '',
                    email       TEXT DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
            """)

    # Tạo phiên hội thoại mới hoặc trả về phiên hiện có nếu đã tồn tại
    def create_session(self, session_id: str, user_id: str = "") -> dict:
        now = datetime.now().isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, user_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, user_id, now, now),
            )
        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }

    # Tải chi tiết phiên hội thoại gồm danh sách các lượt hỏi đáp và timings
    def load_session(self, session_id: str) -> Optional[dict]:
        cursor = self.conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        session_row = cursor.fetchone()
        if session_row is None:
            return None

        cursor = self.conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id ASC", (session_id,)
        )
        turn_rows = cursor.fetchall()
        turns = []
        for row in turn_rows:
            turn_id = row["turn_id"]
            timing_cursor = self.conn.execute(
                "SELECT agent_name, called_by, duration, calls_json FROM agent_timings WHERE session_id = ? AND turn_id = ?",
                (session_id, turn_id),
            )
            timings = [
                {
                    "agent": t["agent_name"],
                    "called_by": t["called_by"],
                    "duration": t["duration"],
                    "calls": json.loads(t["calls_json"]),
                }
                for t in timing_cursor.fetchall()
            ]
            turns.append({
                "turn_id": turn_id,
                "timestamp": row["timestamp"],
                "user_query": row["user_query"],
                "final_answer": row["final_answer"],
                "intent": row["intent"],
                "confidence_score": row["confidence_score"],
                "attachments": json.loads(row["attachments_json"]),
                "agent_timings": timings,
            })

        return {
            "session_id": session_row["session_id"],
            "user_id": session_row["user_id"],
            "created_at": session_row["created_at"],
            "updated_at": session_row["updated_at"],
            "turns": turns,
        }

    # Liệt kê tất cả các phiên hội thoại sắp xếp theo thời gian tạo mới nhất
    def list_sessions(self) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT session_id, created_at, updated_at FROM sessions ORDER BY created_at DESC"
        )
        return [
            {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in cursor.fetchall()
        ]

    # Lưu một lượt hỏi đáp mới và cập nhật thời gian hoạt động của phiên
    def append_turn(self, session_id: str, turn_data: dict) -> None:
        now = datetime.now().isoformat()
        turn_id = turn_data.get("turn_id", 0)
        timestamp = turn_data.get("timestamp", now)
        user_query = turn_data.get("user_query", "")
        final_answer = turn_data.get("final_answer", "")
        intent = turn_data.get("intent", "")
        confidence_score = turn_data.get("confidence_score", 0.0)
        attachments = turn_data.get("attachments", [])
        agent_timings = turn_data.get("agent_timings", [])

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO turns (session_id, turn_id, timestamp, user_query, final_answer, intent, confidence_score, attachments_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_id,
                    timestamp,
                    user_query,
                    final_answer,
                    intent,
                    confidence_score,
                    json.dumps(attachments, ensure_ascii=False),
                ),
            )
            for timing in agent_timings:
                self.conn.execute(
                    """
                    INSERT INTO agent_timings (turn_id, session_id, agent_name, called_by, duration, calls_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn_id,
                        session_id,
                        timing.get("agent", "Unknown"),
                        timing.get("called_by", ""),
                        timing.get("duration", 0.0),
                        json.dumps(timing.get("calls", []), ensure_ascii=False),
                    ),
                )
            self.conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    # Lấy chuỗi văn bản lịch sử n lượt hỏi đáp gần nhất
    def get_history_text(self, session_id: str, n: int = 6) -> str:
        cursor = self.conn.execute(
            "SELECT user_query, final_answer FROM turns WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?",
            (session_id, n),
        )
        rows = cursor.fetchall()
        if not rows:
            return ""

        rows = list(reversed(rows))
        lines = []
        for row in rows:
            lines.append(f"User: {row['user_query']}")
            lines.append(f"Bot: {row['final_answer'] if row['final_answer'] else '(chưa trả lời)'}")
            lines.append("---")

        if lines and lines[-1] == "---":
            lines.pop()
        return "\n".join(lines)

    # Lấy danh sách n lượt hội thoại gần nhất dưới dạng danh sách dict
    def get_recent_turns(self, session_id: str, limit: int = 5) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT turn_id, user_query, final_answer, intent, confidence_score, timestamp FROM turns WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?",
            (session_id, limit),
        )
        return [dict(r) for r in reversed(cursor.fetchall())]

    # Lấy thông tin hồ sơ người dùng theo user_id
    def get_user_profile(self, user_id: str) -> Optional[dict]:
        if not user_id:
            return None
        cursor = self.conn.execute(
            "SELECT user_id, name, email, updated_at FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "user_id": row["user_id"],
            "profile": {
                "name": row["name"],
                "email": row["email"],
                "updated_at": row["updated_at"],
            },
        }

    # Lưu hoặc cập nhật thông tin hồ sơ người dùng
    def save_user_profile(self, user_id: str, profile: dict) -> dict:
        now = datetime.now().isoformat()
        name = profile.get("name", "").strip() if isinstance(profile, dict) else ""
        email = profile.get("email", "").strip() if isinstance(profile, dict) else ""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO users (user_id, name, email, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name = excluded.name,
                    email = excluded.email,
                    updated_at = excluded.updated_at
                """,
                (user_id, name, email, now, now),
            )
        return {
            "user_id": user_id,
            "profile": {
                "name": name,
                "email": email,
                "updated_at": now,
            },
        }

    # Lấy thông tin hiệu năng và chuỗi thực thi của các Agent trong lượt xử lý
    def get_process(self, session_id: str, turn_id: Optional[int] = None) -> Optional[dict]:
        if turn_id is None:
            cursor = self.conn.execute(
                "SELECT MAX(turn_id) as max_turn FROM turns WHERE session_id = ?", (session_id,)
            )
            result = cursor.fetchone()
            if result is None or result["max_turn"] is None:
                return None
            turn_id = result["max_turn"]

        cursor = self.conn.execute(
            "SELECT agent_name, called_by, duration, calls_json FROM agent_timings WHERE session_id = ? AND turn_id = ? ORDER BY id ASC",
            (session_id, turn_id),
        )
        rows = cursor.fetchall()
        if not rows:
            return None

        agents = []
        total_duration = 0.0
        for row in rows:
            duration = row["duration"]
            total_duration += duration
            agents.append({
                "agent_name": row["agent_name"],
                "called_by": row["called_by"],
                "duration": duration,
                "calls": json.loads(row["calls_json"]),
            })

        return {
            "session_id": session_id,
            "turn_id": turn_id,
            "agents": agents,
            "total_duration": round(total_duration, 4),
            "agent_count": len(agents),
        }

    # Đóng kết nối SQLite an toàn
    def close(self) -> None:
        if self.conn:
            self.conn.close()

    # Đảm bảo đóng kết nối SQLite khi giải phóng đối tượng
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # Trả về biểu diễn chuỗi của đối tượng ChatDatabase
    def __repr__(self) -> str:
        return f"ChatDatabase(db_path='{self.db_path}')"
