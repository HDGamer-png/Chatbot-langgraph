# Hướng Dẫn Kiến Trúc & Vận Hành Multi-Agent Chuyên Ngành

## 1. Tổng Quan Kiến Trúc
Hệ thống Multi-Agent Chatbot vận hành với 10 tác tử chuyên biệt, hoạt động hoàn toàn bằng Python logic cục bộ không phụ thuộc API đám mây.
Các tác tử phối hợp gồm:
- IntentClassifierAgent: Phân loại ý định người dùng với mô hình máy học siêu nhẹ.
- KnowledgeBaseAgent: Truy xuất tri thức nội bộ bằng thuật toán BM25 kết hợp mở rộng từ đồng nghĩa.
- CalculatorAgent: Đo lường số học và giải phương trình đại số với SymPy.
- DateTimeAgent: Tra cứu ngày giờ và lịch trình sự kiện thời gian thực.
- WebSearchAgent: Thu thập tài liệu và tiêu chuẩn kỹ thuật.
- AnalystAgent: Phân tích cấu trúc và ngữ nghĩa văn bản bằng thuật toán TextRank.
- PlannerAgent: Lập lộ trình kỹ thuật và phân rã nhiệm vụ.
- ValidatorAgent: Thẩm định chéo nguồn dữ liệu và tính toán điểm tin cậy tổng thể.
- ResponseGeneratorAgent: Tổng hợp báo cáo Markdown hoàn chỉnh chuẩn mực Tech Lead.

## 2. Quy Trình Chuyên Biệt Hóa Lĩnh Vực
Để huấn luyện hệ thống cho một lĩnh vực mới:
1. Đặt tài liệu nghiệp vụ (.md, .txt, .docx, .xlsx, mã nguồn) vào thư mục `kb_docs/`.
2. Bổ sung các mẫu câu ý định đặc thù vào `intents.jsonl`.
3. Định nghĩa từ điển thuật ngữ đồng nghĩa trong `synonyms.json`.
4. Cung cấp bộ câu hỏi kiểm thử tiêu chuẩn trong `qa_pairs.jsonl`.
5. Kích hoạt và huấn luyện tự động thông qua `scripts/switch_domain.py`.
