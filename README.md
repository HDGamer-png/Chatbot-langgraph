# 🤖 LangGraph Multi-Agent Chatbot v4.0

> **Hệ sinh thái Trợ lý AI Đa tác tử Thông minh — Kiến trúc 10 Tác tử Thuần Python (100% On-Premise & No-API-Key)**  
> **Tác giả:** Hoàng Minh Đức — Thực tập sinh AI Agent  
> **Phiên bản:** `v4.0 Production-Grade`

---

## 🎯 Giới thiệu Đề tài

Dự án nghiên cứu và hiện thực hóa một hệ thống **Chatbot Đa tác tử (Multi-Agent System)** hoàn chỉnh dựa trên khung điều phối **LangGraph StateGraph**. 

Khác biệt hoàn toàn với các chatbot đơn tác tử (Single-Agent) hoặc các hệ thống chỉ đóng vai trò "gọi API dịch vụ bên ngoài", phiên bản **v4.0** được tái thiết kế toàn diện theo nguyên lý:
1. **Thực thi Hai pha (Two-Phase Execution):** **Ưu tiên chạy Song song trước (Parallel Fan-out)** để thu thập và tính toán dữ liệu $\rightarrow$ sau đó **Xử lý Tuần tự (Sequential Pipeline)** để kiểm định, bắt lỗi ảo giác và tổng hợp câu trả lời.
2. **100% Logic Python Nội bộ (No-API-Key):** Loại bỏ hoàn toàn sự phụ thuộc vào các dịch vụ đám mây thương mại (Groq, Anthropic, OpenAI). Không phát sinh chi phí token, triệt tiêu lỗi giới hạn hạn ngạch (Rate-Limit 429), bảo mật tuyệt đối dữ liệu nội bộ.
3. **Tốc độ Siêu tốc:** Thời gian phản hồi chỉ **~16ms – 93ms** (nhanh gấp **~30 lần** so với mô hình tuần tự 2.5s – 3.5s cũ).

---

## 🏗️ Sơ đồ Kiến trúc Điều phối 2 Pha (LangGraph StateGraph)

```
                            ┌────────────────────────┐
                            │   User Input / Query   │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │ [0] IntentClassifier   │ (1ms - Scikit-Learn TF-IDF + NB)
                            └───────────┬────────────┘
                                        │
════════════════════════════════════════╪════════════════════════════════════════
 PHA 1: THỰC THI ĐỒNG THỜI (PARALLEL FAN-OUT - ThreadPool 6 Workers)
  Các tác tử chạy độc lập, cùng lúc lấy ngữ cảnh câu hỏi, không phụ thuộc nhau:
════════════════════════════════════════╪════════════════════════════════════════
          ┌─────────────────┬───────────┴───────────┬─────────────────┐
          ▼                 ▼                       ▼                 ▼
   ┌─────────────┐   ┌─────────────┐         ┌─────────────┐   ┌─────────────┐
   │[1] Local RAG│   │[2]Calculator│         │[3] DateTime │   │[4]Web Search│
   │ (BM25/Docs) │   │   (SymPy)   │         │ (NLP Lịch)  │   │(DuckDuckGo) │
   └──────┬──────┘   └──────┬──────┘         └──────┬──────┘   └──────┬──────┘
          │                 │                       │                 │
          └─────────────────┼───────────┬───────────┴─────────────────┘
                            │           │
                            │           ▼
                            │    ┌─────────────┐
                            │    │ [5] Analyst │  (TextRank NLP Tóm tắt & Trích xuất)
                            │    └──────┬──────┘
                            │           │
                            ▼           ▼
             ══════════════════════════════════════════════
              BARRIER SYNCHRONIZATION (GOM TẤT CẢ KẾT QUẢ)
              Lưu toàn bộ kết quả vào AgentState Blackboard
             ══════════════════════════════════════════════
                                        │
════════════════════════════════════════╪════════════════════════════════════════
 PHA 2: XỬ LÝ TUẦN TỰ (SEQUENTIAL PIPELINE - FAN-IN & VERIFICATION)
  Các tác tử cấp cao xử lý có thứ tự logic bắt buộc:
════════════════════════════════════════╪════════════════════════════════════════
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │   [6] Planner Agent    │
                            │ (Lập cấu trúc lời giải)│
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │  [7] Validator Agent   │
                            │ (Kiểm tra chéo kết quả │
                            │   RAG, đối chiếu Toán, │
                            │  chấm Confidence Score)│
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │[8] ResponseGenerator   │
                            │ (Đóng gói Markdown,    │
                            │  Call Graph SVG Trace) │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │  Final Answer to User  │
                            └────────────────────────┘
```

---

## 🤖 Danh mục 10 Tác tử Chuyên biệt

| STT | Tác tử (Agent) | Phân tầng thực thi | Thuật toán / Công nghệ cốt lõi | Chức năng chi tiết |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **IntentClassifier** | Pha 0 (Cửa ngõ) | Scikit-Learn (`TF-IDF` + `MultinomialNB`) | Phân loại câu hỏi vào 8 nhóm ý định trong **1ms**, độ chính xác 100%. |
| **2** | **Router / Dispatcher** | Điều phối trung tâm | Python `ThreadPoolExecutor` (6 workers) | Kích hoạt worker song song, bọc timeout 8s, đồng bộ hóa State an toàn. |
| **3** | **KnowledgeBase (RAG)** | Pha 1 (Song song) | **BM25Okapi** tự cài đặt từ đầu | Tự động đọc tài liệu trong `data/docs/`, cắt chunk 300 từ, tra cứu ngữ nghĩa. |
| **4** | **Calculator** | Pha 1 (Song song) | **SymPy** & Safe Python AST Eval | Giải phương trình đại số, lượng giác, tích phân, số học chính xác 100%. |
| **5** | **DateTime** | Pha 1 (Song song) | `datetime` & Vietnamese NLP Parser | Ngày giờ hiện tại, đếm ngược ngày lễ, múi giờ quốc tế bằng tiếng Việt. |
| **6** | **WebSearch** | Pha 1 (Song song) | DuckDuckGo HTTP + Offline Tech KB | Tìm kiếm tin tức miễn phí 0đ, fallback kho tri thức công nghệ offline. |
| **7** | **Analyst** | Pha 1 (Song song) | Thuật toán **TextRank NLP** | Phân tích cấu trúc ngữ nghĩa, tóm tắt tự động, trích xuất từ khóa trọng tâm. |
| **8** | **Planner** | Pha 2 (Tuần tự) | Goal-Dependency Decomposition | Phân rã mục tiêu thành lộ trình 3-5 bước thực hiện logic rõ ràng. |
| **9** | **Validator** | Pha 2 (Tuần tự) | Fact-Checking & Jaccard Overlap | Kiểm định chéo kết quả, đối chiếu toán, chống ảo giác, tính Confidence Score. |
| **10** | **ResponseGenerator** | Pha 2 (Tuần tự) | Multi-tier Markdown Synthesizer | Tổng hợp câu trả lời Markdown chuyên nghiệp, đính kèm Call Graph và Metrics. |

---

## 📈 Bảng So sánh Hiệu năng Thực nghiệm (Benchmark)

*Số liệu được đo đạc tự động từ kịch bản kiểm thử [`scripts/benchmark.py`](file:///d:/Python/Thực%20tập/Thuc_tap/Chatbot%20AI/scripts/benchmark.py) và lưu tại [`benchmark_results.json`](file:///d:/Python/Thực%20tập/Thuc_tap/Chatbot%20AI/benchmark_results.json):*

| Tiêu chí đánh giá | v1 (Tuần tự cũ) | v3 (API Cloud cũ) | **v4 (Pure Python Local mới)** |
| :--- | :---: | :---: | :---: |
| **Thời gian phản hồi trung bình** | 2.500 ms – 3.500 ms | 1.200 ms – 1.800 ms | **89.7 ms (Nhanh gấp ~30 lần)** |
| **Thời gian phản hồi nhanh nhất** | 1.800 ms | 950 ms | **15.9 ms** |
| **Độ chính xác phân loại ý định** | 84.6% (Regex) | 92.3% (LLM Prompt) | **100.0% (TF-IDF + Naive Bayes)** |
| **Chi phí duy trì token** | Trả phí theo token | Trả phí theo API Key | **0 VNĐ (Hoàn toàn miễn phí)** |
| **Nguy cơ lỗi Rate-Limit (429)** | Thường xuyên bị chặn | Dễ hết quota ngày/phút | **0% (Không bao giờ bị chặn)** |
| **Bảo mật dữ liệu (Data Privacy)** | Gửi ra máy chủ nước ngoài | Gửi ra Cloud quốc tế | **100% On-Premise nội bộ** |
| **Cơ sở dữ liệu lưu trữ** | File JSON rời rạc | File JSON rời rạc | **SQLite Database (`chatbot.db`)** |
| **Trích xuất tệp đính kèm** | Chưa đầy đủ | Bị ngắt ở endpoint chat | **Đầy đủ: PDF, Word, Excel, OCR** |
| **Dashboard giám sát tiến trình** | Bảng HTML thô sơ | Bản demo tĩnh | **SVG Call Graph tương tác thực tế** |

---

## ⚡ Hướng dẫn Cài đặt & Vận hành

### 1. Yêu cầu môi trường
- Python 3.10 trở lên (khuyến nghị Python 3.11 – 3.14).
- Hệ điều hành: Windows, macOS hoặc Linux.

### 2. Cài đặt thư viện
```bash
git clone https://github.com/HDGamer-png/Chatbot-langgraph.git
cd Chatbot-langgraph
pip install -r requirements.txt
```

### 3. Vận hành hệ thống

#### Cách A: Chế độ Chat tương tác trên Terminal (CLI)
```bash
python chatbot.py
```
*(Gõ trực tiếp các câu hỏi tính toán, giải phương trình, hỏi thời gian, tra cứu tài liệu để thấy phản hồi trong 20ms)*.

#### Cách B: Chế độ Web Server (Giao diện Dark Theme & Dashboard)
```bash
python app.py
```
- Mở trình duyệt truy cập: **`http://127.0.0.1:5000`**
- Xem Dashboard đồ họa SVG của phiên chat: Bấm nút **"Xem tiến trình"** hoặc mở đường dẫn **`http://127.0.0.1:5000/process/<session_id>`**.

---

## 📂 Cấu trúc Thư mục Dự án

```text
Chatbot AI/
├── app.py                      # Server Flask v4.0 (REST API, SSE Streaming, SQLite)
├── chatbot.py                  # Điểm khởi chạy chính (CLI Interactive hoặc Web)
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── chatbot.db                  # CSDL quan hệ SQLite (tự sinh khi khởi chạy)
├── benchmark_results.json      # Báo cáo kết quả đo đạc độ trễ và độ chính xác
├── BaoCao_LangGraph_HoangMinhDuc.docx # Báo cáo nghiên cứu thực tập chính thức
├── core/                       # LÕI MULTI-AGENT ENGINE
│   ├── __init__.py
│   ├── state.py                # AgentState TypedDict (Blackboard)
│   ├── router.py               # Router điều phối Fan-out / Fan-in song song
│   ├── graph.py                # LangGraph StateGraph kết nối 2 pha
│   └── agents/                 # 10 TÁC TỬ CHUYÊN BIỆT
│       ├── base.py             # AgentBase (Auto-timing, Call Graph, Safe Run)
│       ├── intent.py           # Agent 1: Scikit-learn Intent Classifier
│       ├── rag.py              # Agent 2: Local RAG (BM25Okapi)
│       ├── math_agent.py       # Agent 3: SymPy & AST Calculator
│       ├── datetime_agent.py   # Agent 4: NLP DateTime & Calendar
│       ├── search.py           # Agent 5: DuckDuckGo Search Engine
│       ├── analyst.py          # Agent 6: TextRank NLP Summarizer
│       ├── planner.py          # Agent 7: Goal-Dependency Planner
│       ├── validator.py        # Agent 8: Fact-Checker & Confidence Scorer
│       └── response.py         # Agent 9: Multi-tier Markdown Synthesizer
├── storage/                    # TẦNG LƯU TRỮ BỀN VỮNG
│   ├── __init__.py
│   ├── database.py             # Quản lý SQLite (sessions, turns, agent_timings)
│   └── file_reader.py          # Trích xuất PDF, DOCX, XLSX, TXT, Ảnh OCR
├── data/
│   └── docs/                   # Thư mục chứa tài liệu nội bộ cho Local RAG
├── templates/
│   ├── index.html              # Giao diện Web Chat Dark Theme
│   └── agent_dashboard.html    # Dashboard Call Graph SVG tương tác
├── static/
│   ├── app.js                  # Frontend Controller (Streaming SSE, Attachments)
│   └── uploads/                # Thư mục lưu tệp tải lên
├── tests/                      # KIỂM THỬ TỰ ĐỘNG
│   ├── test_full_suite.py      # Bộ Unit & Integration Test (6/6 Pass)
│   └── test_quick.py           # Kiểm tra nhanh chức năng cơ bản
└── scripts/
    ├── benchmark.py            # Đo đạc độ trễ và xuất bảng so sánh
    └── update_docx.py          # Script tự động cập nhật tài liệu Báo cáo Word
```

---

## 🧪 Kiểm thử Tự động & Benchmark

Chạy bộ kiểm thử tự động toàn diện:
```bash
python tests/test_full_suite.py
```
*Kết quả:* `Ran 6 tests in 1.056s. OK.`

Chạy script đo đạc độ trễ và xuất báo cáo:
```bash
python scripts/benchmark.py
```

---

## 📜 Giấy phép & Thông tin Báo cáo

- **Đề tài tốt nghiệp / thực tập:** *Nghiên cứu mô hình điều phối AI Agent và xây dựng Chatbot thông minh với LangGraph*.
- **Người thực hiện:** Hoàng Minh Đức.
- **Báo cáo chi tiết kèm theo:** [`BaoCao_LangGraph_HoangMinhDuc.docx`](file:///d:/Python/Thực%20tập/Thuc_tap/Chatbot%20AI/BaoCao_LangGraph_HoangMinhDuc.docx).
