"""
core/agents/datetime_agent.py — DateTimeAgent
Xử lý thời gian, ngày tháng, múi giờ, đếm ngược sự kiện.

Hoạt động 100% nội bộ bằng thư viện chuẩn Python (datetime, zoneinfo).
Hiểu ngôn ngữ tự nhiên tiếng Việt về thời gian.
"""

import re
from datetime import datetime, timedelta
from .base import AgentBase

# Tên thứ tiếng Việt
_WEEKDAYS_VI = [
    "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm",
    "Thứ Sáu", "Thứ Bảy", "Chủ Nhật",
]

# Tên tháng tiếng Việt
_MONTHS_VI = [
    "", "tháng 1", "tháng 2", "tháng 3", "tháng 4", "tháng 5", "tháng 6",
    "tháng 7", "tháng 8", "tháng 9", "tháng 10", "tháng 11", "tháng 12",
]

# Ánh xạ múi giờ phổ biến (tên tiếng Việt → UTC offset hours)
_TIMEZONE_MAP = {
    "việt nam": 7, "vn": 7, "hà nội": 7, "sài gòn": 7, "hồ chí minh": 7,
    "mỹ": -5, "new york": -5, "washington": -5, "los angeles": -8, "california": -8,
    "london": 0, "anh": 0, "uk": 0,
    "paris": 1, "pháp": 1, "đức": 1, "berlin": 1,
    "tokyo": 9, "nhật": 9, "nhật bản": 9, "japan": 9,
    "hàn quốc": 9, "seoul": 9, "korea": 9,
    "bắc kinh": 8, "trung quốc": 8, "shanghai": 8, "china": 8,
    "singapore": 8, "úc": 10, "sydney": 10, "australia": 10,
    "ấn độ": 5, "india": 5, "moscow": 3, "nga": 3,
    "dubai": 4, "thái lan": 7, "bangkok": 7,
}

# Ngày lễ quan trọng (tháng, ngày)
_HOLIDAYS = {
    "tết dương lịch": (1, 1),
    "năm mới": (1, 1),
    "valentine": (2, 14),
    "quốc tế phụ nữ": (3, 8),
    "giỗ tổ hùng vương": (4, 18),   # Âm lịch 10/3, ước lượng
    "giải phóng miền nam": (4, 30),
    "quốc tế lao động": (5, 1),
    "quốc khánh": (9, 2),
    "phụ nữ việt nam": (10, 20),
    "nhà giáo": (11, 20),
    "giáng sinh": (12, 25),
    "noel": (12, 25),
    "halloween": (10, 31),
}


def _format_date_vi(dt: datetime) -> str:
    """Định dạng ngày tháng kiểu tiếng Việt."""
    weekday = _WEEKDAYS_VI[dt.weekday()]
    return f"{weekday}, ngày {dt.day:02d} {_MONTHS_VI[dt.month]} năm {dt.year}"


def _format_time_vi(dt: datetime) -> str:
    """Định dạng giờ phút kiểu tiếng Việt."""
    return f"{dt.hour:02d} giờ {dt.minute:02d} phút"


class DateTimeAgent(AgentBase):
    """
    Tác tử thời gian — Xử lý ngày tháng, múi giờ, đếm ngược.
    100% Python nội bộ, không cần API.
    """

    NAME = "DateTime"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    def run(self, query: str = "") -> dict:
        """
        Phân tích câu hỏi về thời gian và trả về kết quả.

        Returns:
            {"datetime_result": str, "message": str}
        """
        now = datetime.now()
        q = query.lower().strip() if query else ""

        # Nếu không có query cụ thể, trả về thông tin hiện tại
        if not q:
            return self._current_datetime(now)

        # Kiểm tra từng loại câu hỏi
        # 1. Múi giờ
        tz_result = self._check_timezone(q, now)
        if tz_result:
            return tz_result

        # 2. Đếm ngược đến sự kiện / ngày lễ
        countdown_result = self._check_countdown(q, now)
        if countdown_result:
            return countdown_result

        # 3. Ngày mai / Hôm qua / Tuần sau
        relative_result = self._check_relative_date(q, now)
        if relative_result:
            return relative_result

        # 4. Tháng này có bao nhiêu ngày
        if "bao nhiêu ngày" in q and "tháng" in q:
            return self._days_in_month(now)

        # Default: Trả về ngày giờ hiện tại
        return self._current_datetime(now)

    def _current_datetime(self, now: datetime) -> dict:
        """Trả về thông tin ngày giờ hiện tại."""
        date_str = _format_date_vi(now)
        time_str = _format_time_vi(now)
        result = (
            f"🕐 Bây giờ là: {time_str} (Giờ Việt Nam, UTC+7)\n"
            f"📅 Hôm nay là: {date_str}\n"
            f"📆 Tuần thứ {now.isocalendar()[1]} của năm {now.year}"
        )
        return {"datetime_result": result, "message": "Đã lấy thông tin ngày giờ hiện tại."}

    def _check_timezone(self, q: str, now: datetime) -> dict | None:
        """Kiểm tra và trả lời câu hỏi về múi giờ."""
        tz_keywords = ["giờ ở", "giờ tại", "múi giờ", "bây giờ ở", "mấy giờ ở"]
        if not any(kw in q for kw in tz_keywords):
            return None

        for name, offset in _TIMEZONE_MAP.items():
            if name in q:
                target_time = now + timedelta(hours=offset - 7)  # VN là UTC+7
                time_str = _format_time_vi(target_time)
                date_str = _format_date_vi(target_time)
                utc_label = f"UTC{'+' if offset >= 0 else ''}{offset}"
                result = (
                    f"🌍 Giờ tại {name.title()}: {time_str} ({utc_label})\n"
                    f"📅 Ngày: {date_str}\n"
                    f"🇻🇳 Giờ Việt Nam (UTC+7): {_format_time_vi(now)}\n"
                    f"⏱ Chênh lệch: {'+' if offset - 7 >= 0 else ''}{offset - 7} giờ so với Việt Nam"
                )
                return {"datetime_result": result, "message": f"Đã tra cứu múi giờ {name}."}

        return None

    def _check_countdown(self, q: str, now: datetime) -> dict | None:
        """Đếm ngược ngày đến sự kiện / ngày lễ."""
        countdown_keywords = ["còn bao nhiêu ngày", "còn mấy ngày", "đến khi nào",
                              "bao lâu nữa", "bao giờ"]
        if not any(kw in q for kw in countdown_keywords):
            return None

        # Tết Nguyên Đán (ước tính — thường rơi vào cuối tháng 1 hoặc đầu tháng 2)
        if "tết" in q and "dương" not in q:
            # Ước tính Tết âm lịch (1/1 Âm lịch ≈ 29/01 dương lịch hàng năm)
            tet_estimate = datetime(now.year + 1, 1, 29)
            if now.month >= 2:
                tet_estimate = datetime(now.year + 1, 1, 29)
            else:
                tet_estimate = datetime(now.year, 1, 29)
            days_left = (tet_estimate - now).days
            if days_left < 0:
                tet_estimate = datetime(now.year + 1, 1, 29)
                days_left = (tet_estimate - now).days
            result = (
                f"🧧 Tết Nguyên Đán (ước tính): còn khoảng {days_left} ngày\n"
                f"📅 Dự kiến: {_format_date_vi(tet_estimate)}\n"
                f"⚠ Lưu ý: Ngày Tết chính xác phụ thuộc vào Lịch Âm."
            )
            return {"datetime_result": result, "message": "Đã tính đếm ngược đến Tết."}

        # Các ngày lễ khác
        for holiday_name, (month, day) in _HOLIDAYS.items():
            if holiday_name in q:
                target = datetime(now.year, month, day)
                if target < now:
                    target = datetime(now.year + 1, month, day)
                days_left = (target - now).days
                result = (
                    f"🎉 {holiday_name.title()}: còn {days_left} ngày\n"
                    f"📅 Ngày: {_format_date_vi(target)}"
                )
                return {"datetime_result": result, "message": f"Đã đếm ngược đến {holiday_name}."}

        return None

    def _check_relative_date(self, q: str, now: datetime) -> dict | None:
        """Xử lý ngày tương đối: hôm qua, ngày mai, tuần sau..."""
        if "ngày mai" in q:
            dt = now + timedelta(days=1)
            result = f"📅 Ngày mai là: {_format_date_vi(dt)}"
            return {"datetime_result": result, "message": "Ngày mai."}

        if "hôm qua" in q:
            dt = now - timedelta(days=1)
            result = f"📅 Hôm qua là: {_format_date_vi(dt)}"
            return {"datetime_result": result, "message": "Hôm qua."}

        if "tuần sau" in q or "tuần tới" in q:
            dt = now + timedelta(weeks=1)
            result = (
                f"📅 Tuần sau (cùng thứ): {_format_date_vi(dt)}\n"
                f"📆 Thứ Hai tuần sau: {_format_date_vi(now + timedelta(days=(7 - now.weekday())))}"
            )
            return {"datetime_result": result, "message": "Tuần sau."}

        if "tuần trước" in q:
            dt = now - timedelta(weeks=1)
            result = f"📅 Tuần trước (cùng thứ): {_format_date_vi(dt)}"
            return {"datetime_result": result, "message": "Tuần trước."}

        if "tháng sau" in q or "tháng tới" in q:
            month = now.month + 1
            year = now.year
            if month > 12:
                month = 1
                year += 1
            result = f"📅 Tháng sau: {_MONTHS_VI[month]} năm {year}"
            return {"datetime_result": result, "message": "Tháng sau."}

        return None

    def _days_in_month(self, now: datetime) -> dict:
        """Tính số ngày trong tháng hiện tại."""
        import calendar
        days = calendar.monthrange(now.year, now.month)[1]
        result = (
            f"📅 {_MONTHS_VI[now.month].title()} năm {now.year} có {days} ngày.\n"
            f"📆 Hôm nay là ngày {now.day}/{now.month}, còn {days - now.day} ngày nữa hết tháng."
        )
        return {"datetime_result": result, "message": f"Tháng {now.month} có {days} ngày."}

