"""
core/agents/base.py — Lớp nền tảng cho tất cả 10 Agent.

Cung cấp 3 cơ chế cốt lõi:
  1. Đo thời gian tự động (Auto Timing)
  2. Ghi nhận cây gọi hàm (Call Graph Recording)
  3. Cô lập lỗi (Error Isolation): 1 Agent gặp lỗi không làm crash toàn hệ thống
"""

import time
from datetime import datetime
from typing import Any


class AgentBase:
    """Lớp cha cho mọi Agent trong hệ thống Multi-Agent."""

    NAME: str = "Base"

    def __init__(self, called_by: str = "System"):
        self.called_by = called_by
        self._calls: list[dict] = []
        self._start_time: float = 0.0
        self._duration: float = 0.0

    def _record_call(self, target_agent: str) -> None:
        """Ghi nhận việc Agent này gọi đến một Agent khác (Call Graph Edge)."""
        self._calls.append({
            "from": self.NAME,
            "to": target_agent,
            "ts": datetime.now().isoformat(),
        })

    def _log(self, message: str) -> str:
        """Tạo bản ghi nhật ký có dấu thời gian."""
        return f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [{self.NAME}] {message}"

    def get_timing(self) -> dict:
        """Trả về thông tin hiệu năng của Agent sau khi chạy xong."""
        return {
            "agent": self.NAME,
            "called_by": self.called_by,
            "duration": round(self._duration, 4),
            "calls": [c["to"] for c in self._calls],
        }

    def get_call_edges(self) -> list[dict]:
        """Trả về danh sách các cạnh trong cây gọi hàm."""
        return list(self._calls)

    def run(self, **kwargs) -> dict:
        """Override trong lớp con: Logic chính của Agent."""
        raise NotImplementedError(f"{self.NAME}.run() chưa được cài đặt.")

    def safe_run(self, **kwargs) -> dict:
        """
        Chạy Agent với cơ chế đo thời gian và cô lập lỗi.
        Đảm bảo nếu Agent gặp ngoại lệ, hệ thống vẫn tiếp tục hoạt động.
        """
        self._start_time = time.time()
        self._calls = []
        try:
            result = self.run(**kwargs)
        except Exception as exc:
            result = {
                "error": str(exc),
                "message": f"[{self.NAME}] Lỗi khi xử lý: {exc}",
            }
        finally:
            self._duration = time.time() - self._start_time
        return result

