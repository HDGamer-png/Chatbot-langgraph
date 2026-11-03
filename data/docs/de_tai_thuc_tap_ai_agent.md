# ĐỀ TÀI THỰC TẬP TỐT NGHIỆP: MULTI-AGENT CHATBOT V4.0

## 1. Thông tin sinh viên
- Họ và tên: Hoàng Minh Đức
- Vị trí thực tập: Kỹ sư AI Agent & Xử lý ngôn ngữ tự nhiên (NLP)
- Năm thực hiện: 2026
- Đơn vị thực tập: Trung tâm R&D Trí tuệ Nhân tạo

## 2. Mục tiêu nghiên cứu & phát triển
Xây dựng một hệ thống Chatbot Trí tuệ nhân tạo Đa tác tử (Multi-Agent System) dựa trên LangGraph:
- Hoạt động 100% độc lập cục bộ (Local Pure Python), không phụ thuộc vào bất kỳ API thương mại tốn phí nào (No Groq, No OpenAI, No Anthropic).
- Kiến trúc 10 AI Agents chuyên biệt hóa cao:
  1. IntentClassifierAgent: Phân loại ý định người dùng bằng TF-IDF kết hợp Multinomial Naive Bayes.
  2. KnowledgeBaseAgent (RAG): Truy xuất tài liệu nội bộ bằng giải thuật xếp hạng BM25 Okapi nguyên bản.
  3. CalculatorAgent: Giải toán học, phương trình đại số bậc nhất, hàm lượng giác và biểu thức giải tích qua SymPy & AST.
  4. DateTimeAgent: Xử lý thời gian, ngày lễ Việt Nam (Tết Nguyên Đán, Giỗ Tổ Hùng Vương, Quốc Khánh 2/9) và múi giờ quốc tế.
  5. WebSearchAgent: Tìm kiếm thông tin trực tuyến không cần API key thông qua DuckDuckGo và từ điển công nghệ ngoại tuyến.
  6. AnalystAgent: Tóm tắt văn bản và trích xuất từ khóa bằng giải thuật đồ thị TextRank (PageRank for NLP).
  7. PlannerAgent: Phân tích mục tiêu người dùng thành kế hoạch thực thi 3-5 bước logic có ràng buộc phụ thuộc.
  8. ValidatorAgent: Kiểm định tính đúng đắn của dữ liệu, kiểm tra phép tính toán học và tính điểm tin cậy (Confidence Score).
  9. RouterAgent: Điều phối thực thi song song (Fan-out) qua ThreadPoolExecutor và gom nhánh (Fan-in) với Barrier Synchronization.
  10. ResponseGeneratorAgent: Tổng hợp câu trả lời theo chuẩn Markdown chuyên nghiệp, hiển thị bảng biểu và đồ thị Call Graph SVG.

## 3. Khả năng xử lý Dữ liệu Đa định dạng (Data Layer)
Hệ thống có khả năng đọc hiểu và trích xuất mọi định dạng:
- Mã nguồn: Tất cả các ngôn ngữ lập trình (Python, JS, TS, Java, C++, C#, Go, Rust, PHP, SQL, Shell, Dockerfile, v.v.).
- Tài liệu văn phòng: PDF (PyMuPDF / pypdf), Word (.docx), Excel (.xlsx), PowerPoint (.pptx).
- Hình ảnh: Phân tích siêu dữ liệu, kích thước, EXIF và nhận diện chữ quang học OCR.
- Thư mục dự án: Cho phép đính kèm cả một thư mục dự án hoặc tập tin nén .zip, tự động duyệt cây thư mục và đọc nội dung toàn bộ tệp bên trong.
