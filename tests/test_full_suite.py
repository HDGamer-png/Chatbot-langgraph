"""
tests/test_full_suite.py — Bộ kiểm thử tự động toàn diện (Unit & Integration Tests)
Dự án: LangGraph Multi-Agent Chatbot v4.0 (Pure Python Local Engine)
"""

import sys
import unittest
from pathlib import Path

# Thêm root vào sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from core.graph import handle_query, build_graph
from core.agents.intent import IntentClassifierAgent
from core.agents.math_agent import CalculatorAgent
from core.agents.datetime_agent import DateTimeAgent
from core.agents.rag import KnowledgeBaseAgent
from core.agents.validator import ValidatorAgent
from storage.database import ChatDatabase
from app import app


class TestMultiAgentSystem(unittest.TestCase):

    def setUp(self):
        self.db = ChatDatabase(str(BASE_DIR / "chatbot.db"))
        self.app = app.test_client()

    def test_01_intent_classification(self):
        """Kiểm tra độ chính xác phân loại ý định."""
        agent = IntentClassifierAgent()
        cases = [
            ("tính 15 * 6", "calculate"),
            ("bây giờ là mấy giờ", "datetime"),
            ("xin chào bot", "greet"),
            ("langgraph là gì", "rag"),
            ("thời tiết hôm nay", "search"),
        ]
        for query, expected_intent in cases:
            res = agent.run(query=query)
            self.assertEqual(res["intent"], expected_intent, f"Query '{query}' expected '{expected_intent}' but got '{res['intent']}'")

    def test_02_calculator_agent(self):
        """Kiểm tra tính toán đại số và số học an toàn."""
        agent = CalculatorAgent()
        # Phép tính cơ bản
        res1 = agent.run(query="tính 50 * 2 + 25")
        self.assertEqual(res1["calc_result"], "125")

        # Căn bậc hai
        res2 = agent.run(query="căn bậc hai của 144")
        self.assertEqual(res2["calc_result"], "12")

        # Giải phương trình
        res3 = agent.run(query="giải phương trình 2x + 6 = 16")
        self.assertIn("5", res3["calc_result"])

    def test_03_datetime_agent(self):
        """Kiểm tra thời gian và ngôn ngữ tự nhiên."""
        agent = DateTimeAgent()
        res = agent.run(query="hôm nay là thứ mấy")
        self.assertTrue(len(res["datetime_result"]) > 10)
        self.assertIn("Thứ", res["datetime_result"])

    def test_04_knowledge_base_rag(self):
        """Kiểm tra tra cứu tri thức nội bộ BM25."""
        agent = KnowledgeBaseAgent()
        res = agent.run(query="langgraph multi agent", intent="rag")
        self.assertTrue(len(res["kb_context"]) > 20)
        self.assertIn("LangGraph", res["kb_context"])

    def test_05_database_sqlite(self):
        """Kiểm tra lưu trữ và truy vấn SQLite."""
        import uuid
        session_id = f"test_unit_sess_{uuid.uuid4().hex[:8]}"
        self.db.create_session(session_id, user_id="unit_tester")
        self.db.append_turn(session_id, {
            "turn_id": 1,
            "timestamp": "2026-11-02T12:00:00",
            "user_query": "Unit test query",
            "final_answer": "Unit test answer",
            "intent": "general",
            "confidence_score": 0.95,
            "agent_timings": [{"agent": "IntentClassifier", "duration": 0.01}],
        })
        loaded = self.db.load_session(session_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded["turns"]), 1)
        self.assertEqual(loaded["turns"][0]["user_query"], "Unit test query")

    def test_06_flask_api_endpoints(self):
        """Kiểm tra các REST API của Flask."""
        import uuid
        test_sess_id = f"flask_test_{uuid.uuid4().hex[:8]}"

        # 1. Health endpoint
        res_health = self.app.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json["status"], "ok")

        # 2. Chat endpoint
        res_chat = self.app.post("/api/chat", json={
            "message": "tính 10 + 20",
            "session_id": test_sess_id,
        })
        self.assertEqual(res_chat.status_code, 200)
        data = res_chat.json
        self.assertEqual(data["intent"], "calculate")
        self.assertIn("30", data["reply"])
        self.assertTrue(data["elapsed"] < 2.0)  # Cực nhanh dưới 2s

        # 3. History endpoint
        res_hist = self.app.get(f"/api/history/{test_sess_id}")
        self.assertEqual(res_hist.status_code, 200)
        self.assertEqual(len(res_hist.json["turns"]), 1)

        # 4. Process dashboard page
        res_proc = self.app.get(f"/process/{test_sess_id}")
        self.assertEqual(res_proc.status_code, 200)
        self.assertIn("html", res_proc.data.decode("utf-8").lower())


if __name__ == "__main__":
    unittest.main()
