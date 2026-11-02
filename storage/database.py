"""
storage/database.py — Module quản lý cơ sở dữ liệu SQLite cho hệ thống Chatbot.

Thay thế toàn bộ hệ thống lưu trữ file JSON bằng SQLite, cung cấp:
  - Lưu trữ phiên hội thoại (sessions)
  - Lưu trữ lượt hỏi-đáp (turns) với đầy đủ metadata
  - Lưu trữ thông tin hiệu năng Agent (agent_timings)
  - Truy vấn lịch sử hội thoại & đồ thị gọi hàm (call graph)

Ưu điểm so với JSON:
  - Thread-safe (check_same_thread=False + context manager)
  - ACID transactions (auto-commit qua `with self.conn`)
  - Truy vấn nhanh hơn với index
  - Không bị mất dữ liệu khi crash giữa chừng
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional


class ChatDatabase:
    """
    Lớp quản lý cơ sở dữ liệu SQLite cho hệ thống Multi-Agent Chatbot.

    Quản lý 3 bảng chính:
      - sessions      : Thông tin phiên hội thoại.
      - turns          : Từng lượt hỏi-đáp trong phiên.
      - agent_timings  : Thời gian xử lý và cây gọi hàm của mỗi Agent.

    Usage:
        db = ChatDatabase("chatbot.db")
        db.create_session("session-001", user_id="user-abc")
        db.append_turn("session-001", turn_data={...})
        history = db.get_history_text("session-001", n=6)
    """

    def __init__(self, db_path: str = "chatbot.db"):
        """
        Khởi tạo kết nối SQLite và tạo bảng nếu chưa tồn tại.

        Args:
            db_path: Đường dẫn tới file cơ sở dữ liệu SQLite.
                     Mặc định: "chatbot.db" tại thư mục gốc của dự án.
        """
        self.db_path = db_path

        # Kết nối SQLite với check_same_thread=False để hỗ trợ đa luồng
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )

        # Trả về kết quả dạng Row (truy cập theo tên cột)
        self.conn.row_factory = sqlite3.Row

        # Tạo các bảng cần thiết
        self._create_tables()

    # -----------------------------------------------------------------------
    # Khởi tạo schema — Tạo bảng
    # -----------------------------------------------------------------------

    def _create_tables(self) -> None:
        """
        Tạo các bảng SQLite nếu chưa tồn tại.

        Bảng:
          - sessions      : Lưu thông tin phiên hội thoại.
          - turns          : Lưu từng lượt hỏi-đáp.
          - agent_timings  : Lưu thông tin hiệu năng Agent.
        """
        with self.conn:
            # --- Bảng sessions ---
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    user_id     TEXT DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT
                )
            """)

            # --- Bảng turns ---
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

            # --- Bảng agent_timings ---
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

    # -----------------------------------------------------------------------
    # Session management — Quản lý phiên hội thoại
    # -----------------------------------------------------------------------

    def create_session(self, session_id: str, user_id: str = "") -> dict:
        """
        Tạo phiên hội thoại mới trong database.

        Nếu session_id đã tồn tại, phương thức sẽ bỏ qua (INSERT OR IGNORE)
        và trả về thông tin phiên hiện có.

        Args:
            session_id : ID duy nhất của phiên.
            user_id    : ID người dùng (tùy chọn).

        Returns:
            dict với thông tin phiên:
            {
                "session_id": str,
                "user_id": str,
                "created_at": str (ISO format),
                "updated_at": str or None,
                "turns": []
            }
        """
        now = datetime.now().isoformat()

        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, user_id, now, now),
            )

        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }

    def load_session(self, session_id: str) -> Optional[dict]:
        """
        Tải toàn bộ phiên hội thoại từ database.

        Trả về định dạng tương thích với hệ thống JSON cũ để đảm bảo
        backward compatibility.

        Args:
            session_id: ID phiên cần tải.

        Returns:
            dict tương thích JSON cũ:
            {
                "session_id": str,
                "user_id": str,
                "created_at": str,
                "updated_at": str,
                "turns": [
                    {
                        "turn_id": int,
                        "timestamp": str,
                        "user_query": str,
                        "final_answer": str,
                        "intent": str,
                        "confidence_score": float,
                        "attachments": list,
                        "agent_timings": list[dict]
                    },
                    ...
                ]
            }
            Hoặc None nếu session_id không tồn tại.
        """
        # Truy vấn thông tin phiên
        cursor = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        session_row = cursor.fetchone()

        if session_row is None:
            return None

        # Truy vấn tất cả turns thuộc phiên này (sắp xếp theo turn_id)
        cursor = self.conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id ASC",
            (session_id,),
        )
        turn_rows = cursor.fetchall()

        # Xây dựng danh sách turns
        turns = []
        for row in turn_rows:
            turn_id = row["turn_id"]

            # Truy vấn agent_timings cho turn này
            timing_cursor = self.conn.execute(
                """
                SELECT agent_name, called_by, duration, calls_json
                FROM agent_timings
                WHERE session_id = ? AND turn_id = ?
                """,
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

    def list_sessions(self) -> list[dict]:
        """
        Liệt kê tất cả các phiên hội thoại.

        Returns:
            Danh sách dict, mỗi phần tử gồm:
            {
                "session_id": str,
                "created_at": str,
                "updated_at": str
            }
        """
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

    # -----------------------------------------------------------------------
    # Turn management — Quản lý lượt hỏi-đáp
    # -----------------------------------------------------------------------

    def append_turn(self, session_id: str, turn_data: dict) -> None:
        """
        Thêm một lượt hỏi-đáp mới vào phiên.

        Phương thức này cũng tự động cập nhật updated_at của session
        và lưu agent_timings nếu có.

        Args:
            session_id : ID phiên cần thêm turn.
            turn_data  : dict chứa thông tin lượt, gồm:
                - turn_id          : int — Số thứ tự lượt.
                - timestamp        : str — Thời điểm (ISO format). Tự tạo nếu thiếu.
                - user_query       : str — Câu hỏi của người dùng.
                - final_answer     : str — Câu trả lời cuối cùng.
                - intent           : str — Intent nhận diện được.
                - confidence_score : float — Độ tin cậy.
                - attachments      : list — Danh sách file đính kèm.
                - agent_timings    : list[dict] — Thông tin hiệu năng Agent.
        """
        now = datetime.now().isoformat()

        # Trích xuất dữ liệu từ turn_data, dùng giá trị mặc định an toàn
        turn_id = turn_data.get("turn_id", 0)
        timestamp = turn_data.get("timestamp", now)
        user_query = turn_data.get("user_query", "")
        final_answer = turn_data.get("final_answer", "")
        intent = turn_data.get("intent", "")
        confidence_score = turn_data.get("confidence_score", 0.0)
        attachments = turn_data.get("attachments", [])
        agent_timings = turn_data.get("agent_timings", [])

        with self.conn:
            # --- Chèn turn mới ---
            self.conn.execute(
                """
                INSERT INTO turns
                    (session_id, turn_id, timestamp, user_query, final_answer,
                     intent, confidence_score, attachments_json)
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

            # --- Chèn agent_timings (nếu có) ---
            for timing in agent_timings:
                self.conn.execute(
                    """
                    INSERT INTO agent_timings
                        (turn_id, session_id, agent_name, called_by, duration, calls_json)
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

            # --- Cập nhật updated_at cho session ---
            self.conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    # -----------------------------------------------------------------------
    # History retrieval — Truy xuất lịch sử hội thoại
    # -----------------------------------------------------------------------

    def get_history_text(self, session_id: str, n: int = 6) -> str:
        """
        Lấy lịch sử n lượt hỏi-đáp gần nhất dưới dạng văn bản.

        Định dạng trả về phù hợp để nhúng vào prompt cho LLM/Agent:
            User: <câu hỏi>
            Bot: <câu trả lời>
            ---
            User: <câu hỏi>
            Bot: <câu trả lời>

        Args:
            session_id : ID phiên cần lấy lịch sử.
            n          : Số lượt gần nhất cần lấy (mặc định: 6).

        Returns:
            Chuỗi văn bản chứa lịch sử hội thoại. Chuỗi rỗng nếu không có dữ liệu.
        """
        # Lấy n turns gần nhất (ORDER BY turn_id DESC, LIMIT n, rồi đảo ngược)
        cursor = self.conn.execute(
            """
            SELECT user_query, final_answer
            FROM turns
            WHERE session_id = ?
            ORDER BY turn_id DESC
            LIMIT ?
            """,
            (session_id, n),
        )
        rows = cursor.fetchall()

        if not rows:
            return ""

        # Đảo ngược để lịch sử hiển thị theo thứ tự thời gian (cũ → mới)
        rows = list(reversed(rows))

        # Xây dựng chuỗi văn bản
        lines = []
        for row in rows:
            lines.append(f"User: {row['user_query']}")
            answer = row["final_answer"] if row["final_answer"] else "(chưa trả lời)"
            lines.append(f"Bot: {answer}")
            lines.append("---")

        # Bỏ dấu "---" cuối cùng
        if lines and lines[-1] == "---":
            lines.pop()

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Process / Agent timings — Dashboard data
    # -----------------------------------------------------------------------

    def get_process(
        self, session_id: str, turn_id: Optional[int] = None
    ) -> Optional[dict]:
        """
        Lấy thông tin hiệu năng Agent và đồ thị gọi hàm (call graph).

        Dùng cho dashboard theo dõi hoạt động của hệ thống Multi-Agent.

        Args:
            session_id : ID phiên.
            turn_id    : Nếu chỉ định, chỉ lấy thông tin của turn đó.
                         Nếu None, lấy turn mới nhất.

        Returns:
            dict gồm:
            {
                "session_id": str,
                "turn_id": int,
                "agents": [
                    {
                        "agent_name": str,
                        "called_by": str,
                        "duration": float,
                        "calls": list
                    },
                    ...
                ],
                "total_duration": float,
                "agent_count": int
            }
            Hoặc None nếu không tìm thấy dữ liệu.
        """
        # Nếu không chỉ định turn_id, lấy turn mới nhất
        if turn_id is None:
            cursor = self.conn.execute(
                """
                SELECT MAX(turn_id) as max_turn
                FROM turns
                WHERE session_id = ?
                """,
                (session_id,),
            )
            result = cursor.fetchone()
            if result is None or result["max_turn"] is None:
                return None
            turn_id = result["max_turn"]

        # Truy vấn agent_timings cho turn được chỉ định
        cursor = self.conn.execute(
            """
            SELECT agent_name, called_by, duration, calls_json
            FROM agent_timings
            WHERE session_id = ? AND turn_id = ?
            ORDER BY id ASC
            """,
            (session_id, turn_id),
        )
        rows = cursor.fetchall()

        if not rows:
            return None

        # Xây dựng danh sách agents
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

    # -----------------------------------------------------------------------
    # Lifecycle — Đóng kết nối
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Đóng kết nối SQLite một cách an toàn."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Đảm bảo kết nối được đóng khi object bị hủy."""
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"ChatDatabase(db_path='{self.db_path}')"
