"""
tests/test_domain_training_pipeline.py — Bộ kiểm thử tự động quy trình Huấn luyện & Chuyển đổi Chuyên ngành.

Kiểm tra:
  1. Train Intent Classifier với 2 động cơ: Approach A (Fast) và Approach B (Dense SVD).
  2. Nạp mô hình nhị phân .joblib và siêu dữ liệu .json vào IntentClassifierAgent.
  3. Trình biên dịch BM25 Index Builder (build_knowledge_index.py) và Persistent Cache theo domain.
  4. Cơ chế mở rộng truy vấn (Query Expansion) qua từ điển đồng nghĩa synonyms.json và chống match giả.
  5. Thẩm định toàn diện độ chính xác và độ trễ qua eval_pipeline.py.
  6. Trình kích hoạt và chuyển đổi domain switch_domain.py và phục hồi trạng thái toàn vẹn.
"""

import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.agents.intent import IntentClassifierAgent
from core.agents.rag import KnowledgeBaseAgent
from scripts.train_intent import train_and_export
from scripts.build_knowledge_index import build_and_export_index
from scripts.eval_pipeline import load_qa_pairs, run_evaluation
from scripts.switch_domain import list_available_domains, validate_domain_structure, activate_domain


class TestDomainTrainingPipeline(unittest.TestCase):

    # Dọn dẹp các tệp báo cáo thử nghiệm tạm thời sau khi kết thúc bộ kiểm thử.
    @classmethod
    def tearDownClass(cls):
        for fname in ["eval_report__template.md", "eval_results__template.json"]:
            p = BASE_DIR / fname
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    # Kiểm tra huấn luyện và xuất mô hình cho cả 2 kiến trúc Fast và Dense.
    def test_01_train_fast_and_dense_engines(self):
        # 1. Huấn luyện Approach A (Fast)
        meta_fast = train_and_export(engine="fast", n_splits=3)
        self.assertIn("accuracy", meta_fast["metrics"])
        self.assertGreaterEqual(meta_fast["metrics"]["accuracy"], 0.90)
        self.assertTrue((BASE_DIR / "models" / "intent_model.joblib").exists())
        self.assertTrue((BASE_DIR / "models" / "intent_meta.json").exists())

        # 2. Huấn luyện Approach B (Dense)
        meta_dense = train_and_export(engine="dense", n_splits=3)
        self.assertIn("f1_macro", meta_dense["metrics"])
        self.assertGreaterEqual(meta_dense["metrics"]["f1_macro"], 0.88)

    # Kiểm tra nạp mô hình nhị phân và suy luận chính xác trên IntentClassifierAgent.
    def test_02_intent_classifier_loaded_model(self):
        train_and_export(engine="fast", n_splits=3)
        agent = IntentClassifierAgent()
        agent.reload()
        test_queries = [
            ("tính 100 + 25", "calculate"),
            ("bây giờ là mấy giờ", "datetime"),
            ("xin chào bot", "greet"),
            ("langgraph là gì", "rag"),
            ("thời tiết hôm nay", "search"),
            ("phân tích ưu nhược điểm của python", "analyze"),
            ("lập kế hoạch học máy trong 3 tháng", "plan"),
        ]
        for q, expected in test_queries:
            res = agent.run(query=q)
            self.assertEqual(res["intent"], expected, f"Query '{q}' expected '{expected}' but got '{res['intent']}'")
            self.assertGreater(res["confidence"], 0.5)

    # Kiểm tra trình biên dịch BM25 Index và tốc độ nạp nóng từ persistent cache.
    def test_03_bm25_index_builder_and_cache(self):
        info_default = build_and_export_index()
        self.assertGreater(info_default["total_chunks"], 0)
        self.assertGreater(info_default["vocab_size"], 0)
        self.assertLess(info_default["hot_load_ms"], 50.0)
        self.assertTrue(Path(info_default["cache_file"]).exists())

        # Kiểm tra biên dịch chỉ mục cho domain _template và xác thực nạp tài liệu riêng của domain
        info_domain = build_and_export_index(domain_name="_template")
        self.assertGreater(info_domain["total_chunks"], info_default["total_chunks"])
        self.assertGreater(info_domain["cache_size_kb"], 0.0)
        domain_cache = Path(info_domain["cache_file"])
        self.assertTrue(domain_cache.exists())

    # Kiểm tra cơ chế mở rộng truy vấn qua từ điển từ đồng nghĩa synonyms.
    def test_04_rag_synonyms_expansion(self):
        agent = KnowledgeBaseAgent()
        tokens = agent._expand_query_tokens("mã nguồn math_utils")
        self.assertTrue("code" in tokens or "coding" in tokens or "lập" in tokens)
        res = agent.run(query="mã nguồn math_utils")
        self.assertTrue(len(res["sources"]) > 0)

        # Kiểm tra ngăn chặn match nhầm từ con (false positive boundary matching)
        agent_boundary = KnowledgeBaseAgent()
        agent_boundary._synonyms = {"ai": ["trí tuệ nhân tạo"]}
        tokens_no_fp = agent_boundary._expand_query_tokens("ngày mai")
        self.assertNotIn("trí", tokens_no_fp)
        self.assertNotIn("nhân", tokens_no_fp)

    # Kiểm tra bộ thẩm định hiệu năng eval_pipeline trên tập qa_pairs.
    def test_05_eval_pipeline(self):
        qa_file = BASE_DIR / "data" / "domains" / "_template" / "qa_pairs.jsonl"
        pairs = load_qa_pairs(qa_file)
        self.assertGreater(len(pairs), 0)
        report = run_evaluation(pairs, domain_name="_template")
        self.assertGreaterEqual(report["intent_accuracy_percent"], 85.0)
        self.assertGreaterEqual(report["recall_hit_rate_percent"], 85.0)

    # Kiểm tra cấu trúc domain template và chức năng chuyển đổi domain.
    def test_06_domain_switcher(self):
        domains = list_available_domains()
        self.assertTrue(any(d["folder"] == "_template" for d in domains))
        valid, msgs = validate_domain_structure(BASE_DIR / "data" / "domains" / "_template")
        self.assertTrue(valid)

        active_file = BASE_DIR / "data" / ".active_domain"
        prev_domain = active_file.read_text(encoding="utf-8").strip() if active_file.exists() else None

        try:
            # Kích hoạt domain _template (không chạy test con để tránh lặp)
            success = activate_domain("_template", engine="fast", run_tests=False)
            self.assertTrue(success)
            self.assertTrue(active_file.exists())
            self.assertEqual(active_file.read_text(encoding="utf-8").strip(), "_template")
        finally:
            # Khôi phục trạng thái ban đầu của hệ thống
            if prev_domain and prev_domain not in ("default", "base"):
                activate_domain(prev_domain, engine="fast", run_tests=False)
            else:
                activate_domain("default", engine="fast", run_tests=False)


if __name__ == "__main__":
    unittest.main()
