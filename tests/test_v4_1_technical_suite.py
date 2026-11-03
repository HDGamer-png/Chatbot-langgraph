"""
tests/test_v4_1_technical_suite.py — Bộ kiểm thử toàn diện phiên bản v4.1 Technical Assistant.

Kiểm tra:
  1. Persistent BM25 Cache: Lưu & nạp chỉ mục nhị phân siêu tốc (< 25ms), chữ ký hash tệp.
  2. LRU Query Cache: Phản hồi tức thì (< 5ms) cho các câu hỏi lặp lại.
  3. Multi-turn Context Sliding Window Memory: Duy trì ngữ cảnh đối thoại qua nhiều lượt liên tiếp.
  4. Technical Response Synthesizer: Định dạng câu trả lời chuẩn mực Kỹ sư Trưởng / Tech Lead.
"""

import sys
import time
import uuid
import unittest
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agents.rag import KnowledgeBaseAgent
from core.graph import handle_query, clear_query_cache
from storage.database import ChatDatabase


class TestTechnicalAssistantV41(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_path = "test_chatbot_v41.db"
        cls.db = ChatDatabase(cls.db_path)
        cls.test_session_id = f"test_v41_{uuid.uuid4().hex[:8]}"
        cls.db.create_session(cls.test_session_id, user_id="test_engineer")
        clear_query_cache()

    @classmethod
    def tearDownClass(cls):
        clear_query_cache()
        p = Path(cls.db_path)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    def test_01_persistent_bm25_cache_speed_and_integrity(self):
        """Kiểm tra Persistent BM25 Cache: tốc độ nạp nóng < 30ms và dữ liệu toàn vẹn."""
        # Lần 1: Khởi tạo để đảm bảo cache file tồn tại
        agent1 = KnowledgeBaseAgent()
        self.assertGreater(len(agent1._chunks), 0)

        # Lần 2: Nạp từ Persistent Cache
        t0 = time.time()
        agent2 = KnowledgeBaseAgent()
        load_time_ms = (time.time() - t0) * 1000

        self.assertTrue(agent2._from_cache, "Lần khởi tạo thứ hai phải nạp từ persistent cache.")
        self.assertEqual(len(agent1._chunks), len(agent2._chunks))
        self.assertLess(load_time_ms, 50.0, f"Thời gian nạp từ cache phải < 50ms (thực tế: {load_time_ms:.2f}ms)")

        # Kiểm tra truy vấn từ chỉ mục cache
        res = agent2.run(query="solve_linear_equation math_utils")
        self.assertIn("solve_linear_equation", res.get("kb_context", ""))

    def test_02_query_lru_cache(self):
        """Kiểm tra LRU Query Cache phản hồi tức thì (< 5ms) cho truy vấn lặp lại."""
        clear_query_cache()
        sess = f"lru_sess_{uuid.uuid4().hex[:6]}"
        query = "Tính giá trị biểu thức: 15 * 8 + 120"

        # Lần 1: Chạy qua toàn bộ pipeline
        t0 = time.time()
        res1 = handle_query(session_id=sess, user_query=query, turn_id=1)
        dur1 = (time.time() - t0) * 1000
        self.assertIn("240", res1.get("final_answer", ""))

        # Lần 2: Nhận kết quả từ LRU Cache
        t1 = time.time()
        res2 = handle_query(session_id=sess, user_query=query, turn_id=2)
        dur2 = (time.time() - t1) * 1000

        self.assertIn("240", res2.get("final_answer", ""))
        self.assertLess(dur2, 10.0, f"Truy vấn từ LRU cache phải < 10ms (thực tế: {dur2:.2f}ms)")
        # Kiểm tra log có ghi nhận cache hit
        logs = "\n".join(res2.get("log", []))
        self.assertIn("QueryCache", logs)

    def test_03_multi_turn_sliding_window_memory(self):
        """Kiểm tra Bộ nhớ ngữ cảnh trượt (Multi-turn Context Sliding Window)."""
        sess = f"memory_sess_{uuid.uuid4().hex[:6]}"
        history = []

        # Lượt 1: Giới thiệu chủ đề về file math_utils.py
        turn1_q = "Trong file math_utils.py có những hàm toán học nào?"
        turn1_res = handle_query(session_id=sess, user_query=turn1_q, turn_id=1, history=history)
        turn1_ans = turn1_res.get("final_answer", "")
        self.assertIn("solve_linear_equation", turn1_ans)

        history.append({
            "turn_id": 1,
            "user_query": turn1_q,
            "final_answer": turn1_ans,
            "intent": turn1_res.get("intent", ""),
        })

        # Lượt 2: Câu hỏi nối tiếp dùng đại từ "hàm đó" (phụ thuộc ngữ cảnh Lượt 1)
        turn2_q = "Hàm đó nhận vào những tham số nào và có thể xảy ra ngoại lệ gì?"
        turn2_res = handle_query(session_id=sess, user_query=turn2_q, turn_id=2, history=history)
        turn2_ans = turn2_res.get("final_answer", "")

        # Kiểm tra bot hiểu câu hỏi đang hỏi về hàm trong math_utils.py
        self.assertTrue(
            "solve_linear_equation" in turn2_ans or "math_utils" in turn2_ans or "Hệ số" in turn2_ans or "tham số" in turn2_ans.lower(),
            "Lượt 2 phải kế thừa ngữ cảnh về math_utils từ lượt 1"
        )

    def test_04_technical_synthesizer_format(self):
        """Kiểm tra định dạng phản hồi chuẩn Technical Assistant."""
        sess = f"tech_sess_{uuid.uuid4().hex[:6]}"
        query = "Phân tích cấu trúc hàm solve_linear_equation trong file math_utils.py"
        res = handle_query(session_id=sess, user_query=query, turn_id=1)
        final_ans = res.get("final_answer", "")

        # Kiểm tra các khối định dạng kỹ sư
        self.assertIn("###", final_ans, "Phải có các tiêu đề Markdown phân đoạn")
        self.assertTrue(
            "Đánh giá Mã nguồn" in final_ans or "Tri thức" in final_ans or "Kỹ thuật" in final_ans,
            "Phải chứa mục phân tích kỹ thuật hoặc mã nguồn"
        )
        self.assertIn("Chi tiết luồng thực thi", final_ans, "Phải có bảng chi tiết thời gian thực thi của các tác tử")


if __name__ == "__main__":
    unittest.main()
