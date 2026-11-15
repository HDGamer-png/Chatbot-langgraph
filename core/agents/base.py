import time
from datetime import datetime

class AgentBase:
    NAME: str = "Base"

    # Khởi tạo tác tử cơ sở với nguồn gọi tương ứng.
    def __init__(self, called_by: str = "System"):
        self.called_by = called_by
        self._calls: list[dict] = []
        self._start_time: float = 0.0
        self._duration: float = 0.0

    # Ghi nhận cạnh liên kết gọi hàm giữa tác tử hiện tại và tác tử đích.
    def _record_call(self, target_agent: str) -> None:
        self._calls.append({
            "from": self.NAME,
            "to": target_agent,
            "ts": datetime.now().isoformat(),
        })

    # Tạo dòng nhật ký hoạt động kèm thời gian thực thi.
    def _log(self, message: str) -> str:
        return f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [{self.NAME}] {message}"

    # Trả về số liệu thời gian thực thi và danh sách tác tử đã gọi.
    def get_timing(self) -> dict:
        return {
            "agent": self.NAME,
            "called_by": self.called_by,
            "duration": round(self._duration, 4),
            "calls": [c["to"] for c in self._calls],
        }

    # Lấy danh sách các liên kết trong cây điều phối tác tử.
    def get_call_edges(self) -> list[dict]:
        return list(self._calls)

    # Phương thức trừu tượng thực thi logic tác tử cần được lớp con cài đặt.
    def run(self, **kwargs) -> dict:
        raise NotImplementedError(f"{self.NAME}.run() chưa được cài đặt.")

    # Thực thi an toàn tác tử kèm đo lường hiệu năng và bắt ngoại lệ tự động.
    def safe_run(self, **kwargs) -> dict:
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
