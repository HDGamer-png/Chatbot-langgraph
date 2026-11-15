from datetime import datetime, timedelta
import calendar
from .base import AgentBase

_WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
_MONTHS_VI = ["", "tháng 1", "tháng 2", "tháng 3", "tháng 4", "tháng 5", "tháng 6", "tháng 7", "tháng 8", "tháng 9", "tháng 10", "tháng 11", "tháng 12"]

_TIMEZONE_MAP = {
    "việt nam": 7, "vn": 7, "hà nội": 7, "sài gòn": 7, "hồ chí minh": 7,
    "mỹ": -5, "new york": -5, "washington": -5, "los angeles": -8, "california": -8,
    "london": 0, "anh": 0, "uk": 0, "paris": 1, "pháp": 1, "đức": 1, "berlin": 1,
    "tokyo": 9, "nhật": 9, "nhật bản": 9, "japan": 9, "hàn quốc": 9, "seoul": 9, "korea": 9,
    "bắc kinh": 8, "trung quốc": 8, "shanghai": 8, "china": 8, "singapore": 8,
    "úc": 10, "sydney": 10, "australia": 10, "ấn độ": 5, "india": 5, "moscow": 3, "nga": 3,
    "dubai": 4, "thái lan": 7, "bangkok": 7,
}

_HOLIDAYS = {
    "tết dương lịch": (1, 1), "năm mới": (1, 1), "valentine": (2, 14), "quốc tế phụ nữ": (3, 8),
    "giỗ tổ hùng vương": (4, 18), "giải phóng miền nam": (4, 30), "quốc tế lao động": (5, 1),
    "quốc khánh": (9, 2), "phụ nữ việt nam": (10, 20), "nhà giáo": (11, 20),
    "giáng sinh": (12, 25), "noel": (12, 25), "halloween": (10, 31),
}

# Định dạng đối tượng datetime thành chuỗi ngày tháng tiếng Việt.
def _format_date_vi(dt: datetime) -> str:
    return f"{_WEEKDAYS_VI[dt.weekday()]}, ngày {dt.day:02d} {_MONTHS_VI[dt.month]} năm {dt.year}"

# Định dạng đối tượng datetime thành chuỗi giờ phút tiếng Việt.
def _format_time_vi(dt: datetime) -> str:
    return f"{dt.hour:02d} giờ {dt.minute:02d} phút"

class DateTimeAgent(AgentBase):
    NAME = "DateTime"

    # Khởi tạo tác tử xử lý ngày giờ và múi giờ.
    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    # Phân tích câu hỏi thời gian và trả về kết quả thời gian, múi giờ hoặc đếm ngược tương ứng.
    def run(self, query: str = "", **kwargs) -> dict:
        now = datetime.now()
        q = query.lower().strip() if query else ""
        if not q:
            return self._current_datetime(now)
        tz = self._check_timezone(q, now)
        if tz:
            return tz
        cd = self._check_countdown(q, now)
        if cd:
            return cd
        rel = self._check_relative_date(q, now)
        if rel:
            return rel
        if "bao nhiêu ngày" in q and "tháng" in q:
            return self._days_in_month(now)
        return self._current_datetime(now)

    # Tổng hợp thông tin ngày, giờ, thứ và tuần hiện tại theo giờ Việt Nam.
    def _current_datetime(self, now: datetime) -> dict:
        res = f"🕐 Bây giờ là: {_format_time_vi(now)} (Giờ Việt Nam, UTC+7)\n📅 Hôm nay là: {_format_date_vi(now)}\n📆 Tuần thứ {now.isocalendar()[1]} của năm {now.year}"
        return {"datetime_result": res, "message": "Đã lấy thông tin ngày giờ hiện tại."}

    # Tra cứu và quy đổi giờ hiện tại theo các múi giờ quốc tế.
    def _check_timezone(self, q: str, now: datetime) -> dict | None:
        if not any(kw in q for kw in ["giờ ở", "giờ tại", "múi giờ", "bây giờ ở", "mấy giờ ở"]):
            return None
        for name, offset in _TIMEZONE_MAP.items():
            if name in q:
                target = now + timedelta(hours=offset - 7)
                diff = offset - 7
                res = (
                    f"🌍 Giờ tại {name.title()}: {_format_time_vi(target)} (UTC{'+' if offset >= 0 else ''}{offset})\n"
                    f"📅 Ngày: {_format_date_vi(target)}\n"
                    f"🇻🇳 Giờ Việt Nam (UTC+7): {_format_time_vi(now)}\n"
                    f"⏱ Chênh lệch: {'+' if diff >= 0 else ''}{diff} giờ so với Việt Nam"
                )
                return {"datetime_result": res, "message": f"Đã tra cứu múi giờ {name}."}
        return None

    # Tính toán số ngày đếm ngược đến các ngày lễ và Tết Nguyên Đán.
    def _check_countdown(self, q: str, now: datetime) -> dict | None:
        if not any(kw in q for kw in ["còn bao nhiêu ngày", "còn mấy ngày", "đến khi nào", "bao lâu nữa", "bao giờ"]):
            return None
        if "tết" in q and "dương" not in q:
            target_year = now.year + 1 if now.month >= 2 else now.year
            tet = datetime(target_year, 1, 29)
            days = (tet - now).days
            if days < 0:
                tet = datetime(now.year + 1, 1, 29)
                days = (tet - now).days
            res = f"🧧 Tết Nguyên Đán (ước tính): còn khoảng {days} ngày\n📅 Dự kiến: {_format_date_vi(tet)}\n⚠ Lưu ý: Ngày Tết chính xác phụ thuộc vào Lịch Âm."
            return {"datetime_result": res, "message": "Đã tính đếm ngược đến Tết."}
        for name, (m, d) in _HOLIDAYS.items():
            if name in q:
                tgt = datetime(now.year, m, d)
                if tgt < now:
                    tgt = datetime(now.year + 1, m, d)
                days = (tgt - now).days
                res = f"🎉 {name.title()}: còn {days} ngày\n📅 Ngày: {_format_date_vi(tgt)}"
                return {"datetime_result": res, "message": f"Đã đếm ngược đến {name}."}
        return None

    # Xác định ngày theo các mốc thời gian tương đối như hôm qua, ngày mai, tuần tới.
    def _check_relative_date(self, q: str, now: datetime) -> dict | None:
        if "ngày mai" in q:
            return {"datetime_result": f"📅 Ngày mai là: {_format_date_vi(now + timedelta(days=1))}", "message": "Ngày mai."}
        if "hôm qua" in q:
            return {"datetime_result": f"📅 Hôm qua là: {_format_date_vi(now - timedelta(days=1))}", "message": "Hôm qua."}
        if "tuần sau" in q or "tuần tới" in q:
            res = f"📅 Tuần sau (cùng thứ): {_format_date_vi(now + timedelta(weeks=1))}\n📆 Thứ Hai tuần sau: {_format_date_vi(now + timedelta(days=(7 - now.weekday())))}"
            return {"datetime_result": res, "message": "Tuần sau."}
        if "tuần trước" in q:
            return {"datetime_result": f"📅 Tuần trước (cùng thứ): {_format_date_vi(now - timedelta(weeks=1))}", "message": "Tuần trước."}
        if "tháng sau" in q or "tháng tới" in q:
            m = now.month + 1 if now.month < 12 else 1
            y = now.year if now.month < 12 else now.year + 1
            return {"datetime_result": f"📅 Tháng sau: {_MONTHS_VI[m]} năm {y}", "message": "Tháng sau."}
        return None

    # Tính toán tổng số ngày trong tháng hiện tại và số ngày còn lại đến hết tháng.
    def _days_in_month(self, now: datetime) -> dict:
        days = calendar.monthrange(now.year, now.month)[1]
        res = f"📅 {_MONTHS_VI[now.month].title()} năm {now.year} có {days} ngày.\n📆 Hôm nay là ngày {now.day}/{now.month}, còn {days - now.day} ngày nữa hết tháng."
        return {"datetime_result": res, "message": f"Tháng {now.month} có {days} ngày."}
