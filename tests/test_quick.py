import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from core.graph import handle_query


class TestQuickQueries(unittest.TestCase):
    # Kiểm tra nhanh các truy vấn cơ bản và xác thực phản hồi của hệ thống Multi-Agent
    def test_quick_queries(self):
        queries = [
            ("tính 25 * 4 + 10", "calculate"),
            ("bây giờ là mấy giờ", "datetime"),
            ("xin chào bạn", "greet"),
            ("langgraph là gì", "rag"),
        ]
        for q, exp_intent in queries:
            res = handle_query(session_id="test_quick", user_query=q)
            self.assertEqual(res.get("intent"), exp_intent)
            self.assertTrue(len(res.get("final_answer", "")) > 0)
            self.assertIn("confidence_score", res)


if __name__ == "__main__":
    unittest.main()
