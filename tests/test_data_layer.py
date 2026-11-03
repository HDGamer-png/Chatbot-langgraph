"""
tests/test_data_layer.py — Bộ kiểm thử toàn diện khả năng xử lý Dữ liệu Đa định dạng v4.0.

Kiểm tra:
  1. Đọc hiểu mã nguồn mọi ngôn ngữ (.py, .js, .sql, .java).
  2. Đọc hiểu tài liệu văn phòng (.docx, .xlsx, .md).
  3. Đọc hiểu toàn bộ thư mục đính kèm (recursive folder).
  4. Nén và trích xuất tệp .zip.
  5. Tác tử RAG (KnowledgeBaseAgent) tự động nạp đệ quy và truy xuất chính xác bằng BM25.
"""

import sys
import unittest
import zipfile
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.file_reader import (
    extract_text,
    extract_text_from_code,
    extract_text_from_docx,
    extract_text_from_xlsx,
    extract_text_from_folder,
    extract_text_from_zip,
    scan_documents
)
from core.agents.rag import KnowledgeBaseAgent
from core.state import AgentState


class TestDataLayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.docs_dir = Path("data/docs")
        cls.code_dir = cls.docs_dir / "ma_nguon_mau"

        # Tạo file zip thử nghiệm để test trích xuất nén
        cls.test_zip_path = Path("tests/test_sample_folder.zip")
        with zipfile.ZipFile(cls.test_zip_path, "w") as z:
            z.writestr("project/main.py", "def run():\n    print('Hello Multi-Agent')\n")
            z.writestr("project/config.yaml", "app:\n  name: TestApp\n  port: 8080\n")
            z.writestr("project/readme.txt", "Tài liệu dự án mẫu đóng gói trong file zip.")

    @classmethod
    def tearDownClass(cls):
        if cls.test_zip_path.exists():
            cls.test_zip_path.unlink()

    def test_01_code_extraction_multi_language(self):
        """Kiểm tra trích xuất mã nguồn đa ngôn ngữ (Python, JS, SQL, Java)."""
        py_file = self.code_dir / "math_utils.py"
        js_file = self.code_dir / "api_client.js"
        sql_file = self.code_dir / "schema_internship.sql"
        java_file = self.code_dir / "AgentPayload.java"

        txt_py = extract_text(py_file)
        self.assertIsNotNone(txt_py)
        self.assertIn("solve_linear_equation", txt_py)

        txt_js = extract_text(js_file)
        self.assertIsNotNone(txt_js)
        self.assertIn("ChatbotClient", txt_js)

        txt_sql = extract_text(sql_file)
        self.assertIsNotNone(txt_sql)
        self.assertIn("internship_students", txt_sql)

        txt_java = extract_text(java_file)
        self.assertIsNotNone(txt_java)
        self.assertIn("AgentPayload", txt_java)

    def test_02_office_extraction_docx_and_xlsx(self):
        """Kiểm tra trích xuất văn bản Word (.docx) và Excel (.xlsx)."""
        docx_file = self.docs_dir / "quy_che_va_ke_hoach_thuc_tap.docx"
        xlsx_file = self.docs_dir / "bang_diem_danh_gia_thuc_tap.xlsx"

        txt_docx = extract_text(docx_file)
        self.assertIsNotNone(txt_docx)
        self.assertIn("QUY CHẾ VÀ TIÊU CHÍ ĐÁNH GIÁ", txt_docx)
        self.assertIn("Kiến trúc Multi-Agent & LangGraph", txt_docx)

        txt_xlsx = extract_text(xlsx_file)
        self.assertIsNotNone(txt_xlsx)
        self.assertIn("TC01", txt_xlsx)
        self.assertIn("Kiến trúc LangGraph Multi-Agent 2 Pha", txt_xlsx)

    def test_03_folder_extraction_recursive(self):
        """Kiểm tra trích xuất toàn bộ cấu trúc và nội dung thư mục."""
        folder_text = extract_text_from_folder(self.code_dir)
        self.assertIsNotNone(folder_text)
        self.assertIn("Thư mục dự án: ma_nguon_mau", folder_text)
        self.assertIn("math_utils.py", folder_text)
        self.assertIn("api_client.js", folder_text)
        self.assertIn("schema_internship.sql", folder_text)

    def test_04_zip_extraction(self):
        """Kiểm tra giải nén trong bộ nhớ và đọc cây file từ zip."""
        zip_text = extract_text_from_zip(self.test_zip_path)
        self.assertIsNotNone(zip_text)
        self.assertIn("project/main.py", zip_text)
        self.assertIn("Hello Multi-Agent", zip_text)
        self.assertIn("TestApp", zip_text)

    def test_05_scan_documents_and_rag_bm25(self):
        """Kiểm tra Agent RAG (KnowledgeBaseAgent) lập chỉ mục BM25 từ toàn bộ tài liệu."""
        rag_agent = KnowledgeBaseAgent()
        self.assertFalse(rag_agent._using_fallback, "RAG phải nạp dữ liệu từ data/docs, không được dùng fallback")
        self.assertGreaterEqual(len(rag_agent._chunks), 4, "Phải chia thành nhiều chunks từ tài liệu đa định dạng")

        # Truy vấn thông tin trong file markdown
        res_md = rag_agent.run(query="Hoàng Minh Đức làm đề tài thực tập gì?")
        kb_text = res_md.get("kb_context", "")
        self.assertIn("Hoàng Minh Đức", kb_text)

        # Truy vấn thông tin trong file docx
        res_docx = rag_agent.run(query="Trọng số đánh giá kiến trúc Multi-Agent là bao nhiêu?")
        self.assertIn("30%", res_docx.get("kb_context", ""))


if __name__ == "__main__":
    unittest.main()
