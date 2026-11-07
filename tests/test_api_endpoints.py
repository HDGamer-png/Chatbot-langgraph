"""
tests/test_api_endpoints.py — Kiểm tra toàn bộ REST API endpoints của Flask app qua test_client.
Không yêu cầu server background đang chạy.
"""

import sys
import unittest
from pathlib import Path

# Thêm root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app, db


class TestApiEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_index_page(self):
        """GET / phải trả về 200 và chứa giao diện HTML."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"html", res.data.lower())

    def test_02_health_endpoint(self):
        """GET /api/health trả về status: ok và thông số hệ thống."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("agents_count", data)

    def test_03_session_lifecycle(self):
        """Tạo session mới qua POST /api/session/new và lấy danh sách qua GET /api/sessions."""
        res = self.client.post("/api/session/new", json={"user_id": "tester_01"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        session_id = data.get("session_id")
        self.assertTrue(session_id)

        # Kiểm tra danh sách sessions
        res_list = self.client.get("/api/sessions")
        self.assertEqual(res_list.status_code, 200)
        sessions = res_list.get_json().get("sessions", [])
        self.assertTrue(any(s.get("session_id") == session_id for s in sessions))

    def test_04_chat_endpoint(self):
        """POST /api/chat xử lý câu hỏi qua Multi-Agent pipeline và trả về reply."""
        res = self.client.post(
            "/api/chat",
            json={
                "message": "tính 10 + 20",
                "session_id": "test_api_session",
                "user_id": "tester_01",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("reply", data)
        self.assertEqual(data.get("intent"), "calculate")
        self.assertIn("30", data.get("reply", ""))

    def test_05_user_profile_crud(self):
        """GET và POST /api/users/<user_id> lưu trữ và trích xuất hồ sơ người dùng."""
        user_id = "test_user_api"
        # 1. Lưu hồ sơ
        post_res = self.client.post(
            f"/api/users/{user_id}",
            json={"name": "Nguyễn Văn Test", "email": "test@example.com"},
        )
        self.assertEqual(post_res.status_code, 200)

        # 2. Đọc lại hồ sơ
        get_res = self.client.get(f"/api/users/{user_id}")
        self.assertEqual(get_res.status_code, 200)
        user_data = get_res.get_json()
        profile = user_data.get("profile", {})
        self.assertEqual(profile.get("name"), "Nguyễn Văn Test")
        self.assertEqual(profile.get("email"), "test@example.com")

    def test_06_process_dashboard_page(self):
        """GET /process/<session_id> trả về giao diện Dashboard SVG Call Graph."""
        res = self.client.get("/process/test_api_session")
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
