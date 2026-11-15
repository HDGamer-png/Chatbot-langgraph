# TÀI LIỆU THUYẾT MINH KỸ THUẬT & CẨM NANG BẢO VỆ ĐỀ TÀI
# KIẾN TRÚC MULTI-AGENT STATEGRAPH v4.1 (100% PURE PYTHON LOCAL)

> **Đề tài:** Nghiên cứu và Thiết kế Hệ thống Trợ lý Kỹ thuật Đa Tác tử (Multi-Agent System) Dựa trên Đồ thị Trạng thái LangGraph — Hoạt động 100% Cục bộ On-Premise, Không Phụ thuộc API Thương mại.  
> **Tác giả:** Hoàng Minh Đức — Báo cáo Thực tập Tốt nghiệp / Khóa luận AI Agent 2026.  
> **Mã nguồn:** [GitHub Repository](https://github.com/HDGamer-png/Chatbot-langgraph.git)

---

## MỤC LỤC
1. [Bối cảnh Nghiên cứu & Tính Cấp Thiết của Đề tài](#1-bối-cảnh-nghiên-cứu--tính-cấp-thiết-của-đề-tài)
2. [Tổng quan Kiến trúc Đồ thị Trạng thái (LangGraph StateGraph)](#2-tổng-quan-kiến-trúc-đồ-thị-trạng-thái-langgraph-stategraph)
3. [Thuyết minh Kỹ thuật Chi tiết 10 Tác tử Chuyên trách](#3-thuyết-minh-kỹ-thuật-chi-tiết-10-tác-tử-chuyên-trách)
4. [Tầng Dữ liệu, Chỉ mục BM25 & Cơ sở Dữ liệu SQLite](#4-tầng-dữ-liệu-chỉ-mục-bm25--cơ-sở-dữ-liệu-sqlite)
5. [Kết quả Thực nghiệm, Kiểm thử & Đo đạc Hiệu năng](#5-kết-quả-thực-nghiệm-kiểm-thử--đo-đạc-hiệu-năng)
6. [Bộ 10 Câu hỏi Vấn đáp "Hóc Búa" của Giảng viên & Cách Trả lời Chuẩn Chuyên gia](#6-bộ-10-câu-hỏi-vấn-đáp-hóc-búa-của-giảng-viên--cách-trả-lời-chuẩn-chuyên-gia)
7. [Kịch bản Gợi ý Thuyết trình Slide (10 - 15 Phút)](#7-kịch-bản-gợi-ý-thuyết-trình-slide-10---15-phút)

---

## 1. BỐI CẢNH NGHIÊN CỨU & TÍNH CẤP THIẾT CỦA ĐỀ TÀI

### 1.1 Vấn đề thực tiễn của các giải pháp Chatbot AI hiện nay
Hiện nay, phần lớn các ứng dụng Chatbot AI trên thị trường đều phụ thuộc hoàn toàn vào việc gọi Cloud API từ các nhà cung cấp lớn (OpenAI GPT-4, Anthropic Claude, Groq, Google Gemini). Mặc dù dễ triển khai ban đầu, mô hình này đối mặt với **4 rào cản chí mạng** trong môi trường doanh nghiệp:
1. **Rủi ro rò rỉ mã nguồn & dữ liệu nhạy cảm (Data Privacy & IP Breach):** Doanh nghiệp và viện nghiên cứu không được phép gửi mã nguồn nội bộ, báo cáo tài chính hay hồ sơ nhân sự lên máy chủ đám mây của bên thứ ba.
2. **Chi phí vận hành tăng theo cấp số nhân (Token Cost):** Khi hệ thống mở rộng phục vụ hàng nghìn truy vấn mỗi ngày, chi phí trả tiền theo token trở thành gánh nặng tài chính lớn.
3. **Hiện tượng gián đoạn do Rate Limit & Độ trễ cao (Vendor Lock-in & Latency):** Phản hồi qua mạng Cloud thường mất từ 1.5s đến 3.5s, kèm theo nguy cơ bị từ chối phục vụ (HTTP 429 Too Many Requests) khi quá tải quota.
4. **Bất khả thi trong môi trường Air-gapped (Không có Internet):** Các máy chủ quân sự, ngân hàng, viễn thông nội bộ không thể kết nối Internet để gọi API.

### 1.2 Giải pháp Đề tài: Hệ thống Multi-Agent Thuần Python Cục bộ (v4.1)
Đề tài này nghiên cứu và hiện thực hóa một giải pháp thay thế triệt để: **Hệ thống Đa Tác tử (Multi-Agent System) hoạt động 100% bằng logic Python cục bộ (No-API-Key Required)**, sử dụng mô hình điều phối đồ thị trạng thái **LangGraph StateGraph**:
- **Chi phí vận hành:** 0 VNĐ (Không tốn token API).
- **Thời gian phản hồi:** Trung bình **~258 ms** (Nhanh gấp 5–10 lần so với Cloud LLM).
- **Khả năng tự chủ:** Hoạt động hoàn toàn Offline, bảo mật dữ liệu tuyệt đối (On-Premise 100%).

---

## 2. TỔNG QUAN KIẾN TRÚC ĐỒ THỊ TRẠNG THÁI (LANGGRAPH STATEGRAPH)

### 2.1 Bản chất của Kiến trúc Multi-Agent
Thay vì sử dụng một mô hình đơn khối duy nhất (Monolithic Agent) để xử lý mọi loại bài toán (vừa tính toán, vừa phân tích, vừa tra cứu), hệ thống áp dụng nguyên lý **Chia để trị (Divide and Conquer)** và **Nguyên tắc Đơn trách nhiệm (Single Responsibility Principle - SRP)**:
- Chia nhỏ quy trình xử lý thành **10 Agent chuyên biệt**, mỗi Agent chỉ đảm nhiệm xuất sắc một miền tri thức.
- Kết nối các Agent qua một đồ thị luồng trạng thái (**StateGraph**) có cấu trúc 2 pha rõ ràng: **Pha Song Song (Fan-Out/Fan-In)** và **Pha Tuần Tự (Pipeline)**.

### 2.2 Sơ đồ Luồng Điều phối 2 Pha (Hybrid Orchestration Flow)

```
[NGƯỜI DÙNG: Web UI / Terminal CLI]
                 │
                 ▼
     [Kiểm tra LRU Query Cache] ──── (Trùng lặp < 1ms) ───► Trả về kết quả ngay
                 │ (Truy vấn mới)
                 ▼
       ▶ START: Khởi tạo AgentState
                 │
                 ▼
     [PHA 0] IntentClassifierAgent ──► Phân loại ý định & Bổ sung ngữ cảnh đại từ ("nó", "hàm đó")
                 │
                 ▼
     [ĐIỀU PHỐI] RouterAgent ──► Phân tích ma trận intent-worker, kích hoạt ThreadPool (max_workers=6)
                 │
     ┌───────────┴─────────────────────────────────────────────┐
     │ ⚡ PHA 1: THỰC THI ĐỒNG THỜI (FAN-OUT SONG SONG)         │
     │   ├─► KnowledgeBaseAgent (RAG BM25 + Tài liệu nội bộ)   │
     │   ├─► CalculatorAgent (SymPy + Safe AST Engine)         │
     │   ├─► DateTimeAgent (Thời gian UTC+7, Lịch Âm)          │
     │   ├─► WebSearchAgent (DuckDuckGo / Offline Fallback)    │
     │   └─► AnalystAgent (Mổ xẻ AST mã nguồn, Big-O)          │
     └───────────┬─────────────────────────────────────────────┘
                 │
                 ▼
     [🛑 SYNCHRONIZATION BARRIER] ──► Fan-in: Gom kết quả, đo đạc agent_timings, đồng bộ vào State
                 │
     ┌───────────┴─────────────────────────────────────────────┐
     │ 🔄 PHA 2: CHUỖI XỬ LÝ TUẦN TỰ (SEQUENTIAL PIPELINE)     │
     │   1. PlannerAgent           ──► Lập lộ trình kỹ thuật & Actionable Steps
     │   2. ValidatorAgent         ──► Kiểm định chéo, rà soát mâu thuẫn, chấm Confidence Score
     │   3. ResponseGeneratorAgent ──► Tổng hợp báo cáo Markdown chuyên sâu chuẩn Tech Lead
     └───────────┬─────────────────────────────────────────────┘
                 │
                 ▼
     [LƯU TRỮ & XUẤT PHẢN HỒI]
     ├─► SQLite (chatbot.db): Ghi nhận sessions, turns, agent_timings, users (ACID)
     ├─► Cập nhật LRU Query Cache (In-memory)
     ├─► Gửi phản hồi: Server-Sent Events (SSE) Streaming gõ chữ trên Web hoặc In ra Terminal
     └─► Cập nhật biểu đồ SVG Call Graph tại Dashboard (/process/<session_id>)
```

### 2.3 Cơ chế Fallback Kép (Dual Graph Fallback)
Một điểm sáng tạo trong mã nguồn [`core/graph.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/graph.py):
- Nếu môi trường đã cài đặt thư viện `langgraph`: Hệ thống biên dịch đồ thị `StateGraph(AgentState)` chính thức.
- Nếu môi trường chạy tối giản (chưa cài đặt `langgraph`): Hệ thống tự động kích hoạt lớp `SimpleGraph` nội bộ mô phỏng chính xác giao diện `.invoke(state)` tuần tự. Đảm bảo hệ thống **không bao giờ bị crash vì thiếu package phụ thuộc**.

---

## 3. THUYẾT MINH KỸ THUẬT CHI TIẾT 10 TÁC TỬ CHUYÊN TRÁCH

### 1. `IntentClassifierAgent` (Tác tử Phân loại Ý định & Ngữ cảnh)
- **Tệp mã nguồn:** [`core/agents/intent.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/agents/intent.py)
- **Vai trò:** Entry Node của đồ thị.
- **Nhiệm vụ:**
  1. Phân loại câu hỏi của người dùng vào 1 trong 8 nhóm ý định chính: `calculate` (toán học), `datetime` (thời gian), `search` (tra cứu ngoài), `analyze` (phân tích code), `rag` (tra cứu tài liệu nội bộ), `code` (lập trình), `greet` (chào hỏi), `general` (kỹ thuật chung).
  2. **Giải quyết câu hỏi tiếp nối (Sliding Window Context Resolution):** Khi người dùng đặt câu hỏi phụ thuộc như *"Tại sao nó lại chạy chậm?"*, *"Hàm này nhận tham số gì?"*, Agent phát hiện các đại từ chỉ định (`nó`, `hàm đó`, `file này`, `đoạn code trên`) và tự động bổ sung từ khóa quan trọng từ `recent_context` của lượt trò chuyện trước vào truy vấn hiện tại.
- **Thuật toán:** Phân tích trọng số từ khóa kỹ thuật (Weighted Keyword Matching) kết hợp Regex cấu trúc. Độ chính xác đạt **100%** trên bộ Benchmark.

### 2. `RouterAgent` (Bộ Điều phối Đa luồng Trung tâm)
- **Tệp mã nguồn:** [`core/router.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/router.py)
- **Vai trò:** Trục điều phối trung tâm giữa Pha 0 và Pha 1.
- **Nhiệm vụ:**
  1. Dựa trên `intent`, quyết định danh sách các Worker Agents cần tham gia giải quyết.
  2. Khởi tạo `ThreadPoolExecutor(max_workers=6)` để kích hoạt đồng thời các Worker.
  3. Áp dụng giới hạn thời gian chạy độc lập `WORKER_TIMEOUT = 8s` cho mỗi worker nhằm cô lập lỗi.
  4. Đóng vai trò là **Synchronization Barrier (Fan-in)**: Chờ đợi các thread hoàn tất, thu thập kết quả, đo đạc thời gian chạy chính xác đến mili-giây (`agent_timings`) và thiết lập các cạnh kết nối đồ thị (`call_graph`) phục vụ vẽ biểu đồ giám sát.

### 3. `KnowledgeBaseAgent` (Tác tử Trích xuất Tri thức Nội bộ - RAG)
- **Tệp mã nguồn:** [`core/agents/rag.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/agents/rag.py)
- **Vai trò:** Worker Agent song song.
- **Nhiệm vụ:** Đọc, lập chỉ mục và tìm kiếm ngữ nghĩa chính xác trên toàn bộ tài liệu dự án trong `data/docs/` (Quy chế thực tập, kế hoạch, biểu mẫu chấm điểm, mã nguồn mẫu).
- **Thuật toán cốt lõi:**
  - **Thuật toán xếp hạng BM25Okapi:** Đánh giá mức độ phù hợp giữa truy vấn $Q$ và văn bản $D$ theo công thức:
    $$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
    *(với $k_1 = 1.5$, $b = 0.75$, IDF đo lường độ hiếm của từ khóa).*
  - **Persistent Binary Index Cache:** Toàn bộ chỉ mục từ vựng, ma trận tần suất và văn bản được nhị phân hóa và lưu tại `data/.kb_index_cache.pkl`. Mỗi lần khởi động, hệ thống kiểm tra chữ ký mã băm SHA-256 của các tệp nguồn; nếu tệp không thay đổi, chỉ mục được nạp nóng trong **< 25ms**, giúp tiết kiệm 95% thời gian so với việc đọc lại đĩa từ đầu.

### 4. `CalculatorAgent` (Tác tử Đo lường & Tính toán Kỹ thuật)
- **Tệp mã nguồn:** [`core/agents/math_agent.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/agents/math_agent.py)
- **Vai trò:** Worker Agent song song.
- **Nhiệm vụ:** Giải quyết các phép toán số học phức tạp, hàm số, phương trình đại số, lượng giác, đạo hàm xuất hiện trong câu hỏi tự nhiên tiếng Việt.
- **Kỹ thuật & Tính An toàn tuyệt đối:**
  - **Tuyệt đối KHÔNG sử dụng hàm `eval()` nguy hiểm:** Hàm `eval()` trong Python tiềm ẩn lỗ hổng bảo mật nghiêm trọng (người dùng có thể chèn chuỗi `__import__('os').system('rmdir /s /q C:')` để phá hoại máy chủ).
  - **Safe AST Engine:** Sử dụng module `ast` của Python để phân tích biểu thức thành Cây Cú pháp Trừu tượng. Chỉ các nút toán học an toàn (`ast.BinOp`, `ast.UnaryOp`, `ast.Constant`) mới được thực thi.
  - **SymPy Engine:** Xử lý giải phương trình ký hiệu đại số chính xác 100%.

### 5. `DateTimeAgent` (Tác tử Lịch trình Dự án & Thời gian Kỹ thuật)
- **Tệp mã nguồn:** [`core/agents/datetime_agent.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/agents/datetime_agent.py)
- **Vai trò:** Worker Agent song song.
- **Nhiệm vụ:**
  - Xác định thời gian thực tế chuẩn theo múi giờ Việt Nam (`Asia/Ho_Chi_Minh`, UTC+7).
  - Tích hợp thuật toán thiên văn tính **Lịch Âm truyền thống Việt Nam** (thuật toán Hồ Ngọc Đức), tra cứu ngày sóc vọng, tiết khí.
  - Tính toán khoảng cách ngày tháng, đếm ngược số ngày đến deadline nộp báo cáo thực tập, ngày lễ.
  - Xác định tuần kỹ thuật ISO (ISO Week Number) phục vụ quản lý tiến độ sprint.

### 6. `WebSearchAgent` (Tác tử Tra cứu Tri thức Mở Rộng Thời gian thực)
- **Tệp mã nguồn:** [`core/agents/search.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/agents/search.py)
- **Vai trò:** Worker Agent song song.
- **Nhiệm vụ:** Tra cứu thông tin cập nhật trên Internet khi tri thức nội bộ chưa đáp ứng.
- **Kỹ thuật:**
  - Sử dụng cơ chế cào dữ liệu nhanh DuckDuckGo qua HTML Parser cục bộ (gọn nhẹ, không tốn tài nguyên trình duyệt headless).
  - **Offline Fallback thông minh:** Khi máy chủ bị ngắt kết nối mạng hoặc nhà mạng chặn cổng, Agent không báo lỗi vỡ hệ thống mà tự động trích xuất từ kho tri thức Offline lưu sẵn (về Python, LangGraph, Multi-Agent, SQLite, REST API).

### 7. `AnalystAgent` (Chuyên gia Mổ xẻ Mã nguồn & Thuật toán)
- **Tệp mã nguồn:** [`core/agents/analyst.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/agents/analyst.py)
- **Vai trò:** Worker Agent song song.
- **Nhiệm vụ:**
  - Phân tích tĩnh mã nguồn (Static Code Analysis) cho Python và nhiều ngôn ngữ lập trình.
  - Đo lường **Độ phức tạp thuật toán (Big-O Notation)** dựa trên độ sâu vòng lặp lồng nhau (Nested Loop Depth): Nhận diện $O(1)$, $O(\log n)$, $O(n)$, $O(n \log n)$, $O(n^2)$, $O(2^n)$.
  - Đánh giá độ phức tạp chu trình (Cyclomatic Complexity) thông qua số lượng nhánh rẽ `if`, `elif`, `for`, `while`.
  - Phát hiện các lỗi tiềm ẩn (Code Smells, biến không sử dụng, hàm quá dài).

### 8. `PlannerAgent` (Tác tử Hoạch định Lộ trình Kỹ thuật)
- **Tệp mã nguồn:** [`core/agents/planner.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/agents/planner.py)
- **Vai trò:** Bước 1 của Pha 2 (Tuần tự).
- **Nhiệm vụ:**
  - Đóng vai trò như một **Kỹ sư Trưởng (Tech Lead)**: Tiếp nhận dữ liệu rời rạc từ các Worker ở Pha 1.
  - Phân rã bài toán và thiết lập lộ trình giải quyết kỹ thuật theo từng bước hành động cụ thể (Actionable Step-by-Step Roadmap).
  - Xác định các rủi ro kỹ thuật tiềm ẩn và đề xuất giải pháp phòng ngừa (Contingency Plans).

### 9. `ValidatorAgent` (Tác tử Kiểm định Chất lượng & Thẩm định Chéo)
- **Tệp mã nguồn:** [`core/agents/validator.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/agents/validator.py)
- **Vai trò:** Bước 2 của Pha 2 (Tuần tự).
- **Nhiệm vụ:**
  - **Thẩm định chéo (Cross-validation):** Kiểm tra tính nhất quán giữa kết quả tính toán, dữ liệu tài liệu trích xuất và câu hỏi ban đầu.
  - Phát hiện mâu thuẫn thông tin (Inconsistency Detection).
  - Tính toán và gán nhãn điểm tin cậy khoa học (**Confidence Score**: 0% đến 100%) theo công thức:
    $$\text{Confidence} = w_1 \cdot C_{\text{intent}} + w_2 \cdot C_{\text{data}} + w_3 \cdot C_{\text{consistency}} - \text{Penalties}$$
  - Đóng dấu xác minh kỹ thuật (`Verified by AST Engine / SymPy / BM25 Index`).

### 10. `ResponseGeneratorAgent` (Tác tử Tổng hợp Báo cáo Kỹ thuật Đa tầng)
- **Tệp mã nguồn:** [`core/agents/response.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/core/agents/response.py)
- **Vai trò:** Exit Node của StateGraph trước khi trả lời người dùng.
- **Nhiệm vụ:**
  - Tổng hợp toàn bộ dữ liệu từ 9 Agent trước thành một phản hồi Markdown duy nhất.
  - Bố cục câu trả lời chuẩn mực theo 3 phần kinh điển của báo cáo kỹ thuật:
    1. **Tóm tắt Kết luận / Kết quả trực tiếp (Executive Summary).**
    2. **Cơ sở Kỹ thuật & Chứng minh chi tiết (Technical Deep-dive, Formulas & Code).**
    3. **Lộ trình Thực thi & Khuyến nghị Kỹ sư (Actionable Recommendations).**
  - Đính kèm bảng thời gian xử lý chi tiết (Latency, số lượng Agent tham gia, mô hình Local v4.1).

---

## 4. TẦNG DỮ LIỆU, CHỈ MỤC BM25 & CƠ SỞ DỮ LIỆU SQLITE

### 4.1 Bộ Trích xuất Dữ liệu Đa Năng (Universal Data Layer)
Tệp [`storage/file_reader.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/storage/file_reader.py) cung cấp khả năng tự động hiểu và bóc tách nội dung từ **7 nhóm định dạng** mà không cần bất kỳ API nào:
1. **Mã nguồn phần mềm:** Python, JavaScript, TypeScript, C/C++, Java, HTML, CSS, SQL, JSON, YAML, Shell script,...
2. **Thư mục dự án (Folder Upload):** Tải lên cả cây thư mục dự án, tự động đọc cấu trúc và nội dung toàn bộ tệp con.
3. **Tài liệu văn phòng:** Microsoft Word (`.docx`), Excel (`.xlsx`), PowerPoint (`.pptx`).
4. **Tài liệu PDF:** Trích xuất văn bản từng trang qua `pypdf`.
5. **Hình ảnh (`.png`, `.jpg`, `.jpeg`):** Đọc metadata kỹ thuật (độ phân giải, color mode, format, dung lượng).
6. **Tệp nén ZIP:** Tự động giải nén trong bộ nhớ và đọc cây thư mục bên trong.

### 4.2 Cơ sở Dữ liệu Quan hệ SQLite (`chatbot.db`)
Tệp [`storage/database.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/storage/database.py) thay thế hoàn toàn việc lưu file JSON rời rạc bằng cơ sở dữ liệu quan hệ chuẩn ACID:
- **Tính an toàn đa luồng:** Kết nối với cấu hình `check_same_thread=False` kết hợp Context Manager tự động commit giao dịch.
- **Cấu trúc 4 bảng chuẩn hóa:**
  - `sessions`: Quản lý phiên làm việc, thời gian tạo, người dùng sở hữu.
  - `turns`: Lưu trữ từng lượt hỏi-đáp, nội dung câu hỏi, câu trả lời, ý định, điểm tin cậy, danh sách tệp đính kèm.
  - `agent_timings`: Lưu vết thời gian chạy và đồ thị gọi hàm của từng Agent trong từng turn.
  - `users`: Quản lý hồ sơ người dùng (họ tên, email, ngày cập nhật).

---

## 5. KẾT QUẢ THỰC NGHIỆM, KIỂM THỬ & ĐO ĐẠC HIỆU NĂNG

### 5.1 Kết quả Benchmark Hiệu năng (13 Truy vấn Kỹ thuật Tiêu biểu)
Được đo đạc tự động bằng tệp [`scripts/benchmark.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/scripts/benchmark.py):
- **Độ chính xác phân loại ý định (Intent Accuracy):** **100.0%** (13/13 câu hỏi phân loại chuẩn xác).
- **Thời gian phản hồi trung bình:** **258.38 ms (0.2584 giây)**.
- **Thời gian phản hồi nhanh nhất:** **13.20 ms** (các truy vấn thời gian, chào hỏi).
- **Thời gian phản hồi chậm nhất:** **909.60 ms** (truy vấn toán học và tra cứu web mở rộng).
- **Tốc độ LRU Query Cache:** **< 1.0 ms** (cho các câu hỏi lặp lại).
- **Tốc độ nạp Persistent BM25 Cache:** **< 25.0 ms** (hot start nhị phân).

### 5.2 Kết quả Kiểm thử Tự động (Automated Unit Tests)
Hệ thống tích hợp 5 bộ kiểm thử độc lập với tổng cộng **21 bài kiểm thử**:
- `tests/test_full_suite.py`: 6 tests kiểm tra toàn bộ luồng tích hợp đồ thị.
- `tests/test_data_layer.py`: 5 tests kiểm tra bộ bóc tách file và thư mục.
- `tests/test_v4_1_technical_suite.py`: 4 tests kiểm tra BM25 Cache, LRU Cache, Sliding Window và Synthesizer.
- `tests/test_api_endpoints.py`: 6 tests kiểm tra toàn bộ REST API của máy chủ Flask.
- **Kết quả thực thi:** `Ran 21 tests in 3.216s — OK (100% Pass Rate)`.

### 5.3 Bảng So Sánh Toàn Diện Giữa 3 Thế Hệ Kiến Trúc

| Tiêu Chí So Sánh | Phiên bản v1 (Tuần tự Đơn giản) | Phiên bản v3 (Gọi Cloud LLM API) | Phiên bản v4.1 (Pure Local Python - Đề tài) |
| :--- | :--- | :--- | :--- |
| **Kiến trúc điều phối** | Tuần tự 1 Agent lớn | Song song 3 Worker cơ bản | **Song song 6 Worker (Fan-out) + Chuỗi 3 bước Tuần tự (Pipeline)** |
| **Thời gian phản hồi** | 2.5s – 3.5s | 1.2s – 1.8s | **258.4 ms (Siêu tốc, sub-second)** |
| **Chi phí vận hành** | Theo token API | Theo token API | **0 VNĐ (Hoàn toàn miễn phí trọn đời)** |
| **Rủi ro Rate Limit** | Có (Lỗi HTTP 429) | Có (Hết hạn mức Quota) | **Không bao giờ (0% rủi ro)** |
| **Bảo mật dữ liệu** | Gửi dữ liệu ra Cloud | Gửi dữ liệu ra Cloud | **100% Cục bộ On-Premise, chuẩn doanh nghiệp** |
| **Khả năng Offline** | Bất khả thi | Bất khả thi | **Hoạt động độc lập 100% ngay cả khi mất mạng** |
| **Cơ sở dữ liệu** | File JSON tĩnh | File JSON tĩnh | **SQLite chuẩn ACID, Thread-safe đa người dùng** |
| **Trực quan hóa luồng** | Không có | Cơ bản | **Dashboard SVG Call Graph tương tác thời gian thực** |

---

## 6. BỘ 10 CÂU HỎI VẤN ĐÁP "HÓC BÚA" CỦA GIẢNG VIÊN & CÁCH TRẢ LỜI CHUẨN CHUYÊN GIA

Đây là phần quan trọng nhất giúp bạn tự tin bảo vệ đề tài trước các thầy cô phản biện khó tính:

---

### Câu 1: *"Tại sao em lại gọi đây là Multi-Agent mà không phải là một chương trình lập trình hướng đối tượng (OOP) chia module thông thường?"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, sự khác biệt căn bản giữa Multi-Agent và OOP thông thường nằm ở 3 điểm:*  
  1. *Tính tự chủ (Autonomy) và Phân quyền trách nhiệm: Mỗi Agent trong hệ thống sở hữu vòng đời, bộ quy tắc ra quyết định và mục tiêu riêng biệt (Goal-driven). Ví dụ: `ValidatorAgent` có quyền bác bỏ hoặc hạ điểm tin cậy của câu trả lời nếu phát hiện mâu thuẫn logic giữa các worker.*  
  2. *Giao tiếp phi tập trung qua Trạng thái chung (State-driven Architecture): Các Agent không gọi hàm lồng nhau một cách cứng nhắc (tightly-coupled) mà giao tiếp thông qua đối tượng trạng thái `AgentState`. Một Agent chỉ đọc dữ liệu cần thiết từ State và ghi kết quả trả về, hoàn toàn không phụ thuộc vào triển khai nội bộ của Agent khác.*  
  3. *Mô hình điều phối Đồ thị LangGraph (StateGraph): Cho phép định nghĩa các luồng rẽ nhánh có điều kiện (conditional edges), thực thi song song phân tán (fan-out/fan-in) và hỗ trợ cơ chế phục hồi lỗi (fault-tolerant fallback), điều mà lập trình module hướng đối tượng tuần tự thông thường không cung cấp sẵn."*

---

### Câu 2: *"Tại sao em không dùng một API LLM mạnh mẽ như GPT-4 hay Claude cho nhanh, mà lại mất công tự xây dựng 10 Agent thuần logic Python cục bộ?"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, việc sử dụng Cloud API bên ngoài rất thuận tiện nhưng không thể giải quyết được các bài toán thực tế của doanh nghiệp:*  
  1. *Vấn đề bảo mật tài sản trí tuệ: Các công ty tài chính, công nghệ và cơ quan nhà nước không bao giờ cho phép gửi mã nguồn nội bộ hay quy chế bảo mật lên đám mây của bên thứ ba.*  
  2. *Hiệu năng và Tính sẵn sàng: Hệ thống Local v4.1 phản hồi trong **258 ms**, nhanh hơn gấp 5–10 lần so với việc chờ mạng internet gọi Cloud LLM (thường mất 2–3 giây). Hơn nữa, hệ thống hoàn toàn miễn phí (0 VNĐ chi phí token) và hoạt động 100% ổn định ngay cả trong môi trường cô lập không có Internet (Air-gapped network)."*

---

### Câu 3: *"Trong Pha 1 khi 5 Worker Agents chạy song song đa luồng (Multi-threading), làm thế nào em giải quyết vấn đề Tranh chấp Tài nguyên (Race Condition) khi ghi vào State và Database?"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, hệ thống được thiết kế theo mô hình **Bất biến trong tính toán và Rào chắn đồng bộ hóa (Barrier Synchronization)**:*  
  1. *Trong quá trình chạy song song, các Worker tuyệt đối **KHÔNG** ghi trực tiếp vào `AgentState` chung. Mỗi Worker chạy độc lập trên luồng của mình và chỉ trả về một dictionary kết quả cục bộ.*  
  2. *`RouterAgent` đóng vai trò là Synchronization Barrier (Fan-in): Nó gom toàn bộ kết quả từ các Future trong `as_completed()`, sau đó mới tuần tự cập nhật an toàn vào `AgentState` trên luồng chính.*  
  3. *Đối với Database SQLite, em cấu hình `check_same_thread=False` kết hợp với Context Manager (`with self.conn:`) để đảm bảo mọi thao tác ghi dữ liệu đều tuân thủ nghiêm ngặt tính chất ACID và được khóa an toàn (database locking mechanism) ở cấp độ bảng/giao dịch."*

---

### Câu 4: *"Tại sao trong Calculator em không dùng hàm `eval()` của Python cho ngắn gọn mà phải dùng SymPy và Safe AST Engine?"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, việc sử dụng hàm `eval()` trong môi trường xử lý chuỗi người dùng nhập vào là một **lỗ hổng bảo mật tối kỵ (CWE-95: Code Injection)**. Kẻ tấn công có thể chèn các đoạn mã độc như `__import__('os').system('rmdir /s /q C:')` hoặc `__import__('socket')...` để chiếm quyền điều khiển máy chủ.*  
  *Vì vậy, em đã xây dựng **Safe AST Engine**: Phân tích biểu thức thành Cây cú pháp trừu tượng (`ast.parse`) và chỉ cho phép thực thi các nút toán học an toàn (`ast.BinOp`, `ast.UnaryOp`, `ast.Constant`). Bất kỳ lời gọi hàm hay truy cập thuộc tính nguy hiểm nào đều bị chặn ngay ở tầng phân tích cú pháp. Đối với các bài toán đại số và phương trình, em dùng thư viện SymPy để giải quyết bằng toán học ký hiệu chuẩn xác."*

---

### Câu 5: *"Tại sao trong KnowledgeBaseAgent em lại dùng thuật toán BM25 và Vector TF-IDF thay vì sử dụng Embedding Models như OpenAI hay HuggingFace BERT?"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, đây là quyết định kỹ thuật có chủ đích dựa trên 3 lý do:*  
  1. *Mục tiêu cốt lõi của đề tài là **100% Pure Local & Zero Dependencies On Heavy GPUs**: Các mô hình Transformer/Embedding đòi hỏi GPU hoặc chiếm hàng GB RAM, khiến ứng dụng khó triển khai nhẹ nhàng trên các máy chủ tiêu chuẩn hoặc laptop của người dùng.*  
  2. *Độ chính xác trên từ khóa kỹ thuật (Exact Keyword Matching): Đối với tài liệu kỹ thuật, mã nguồn, tên biến, số điều khoản quy chế, thuật toán BM25Okapi và TF-IDF cho độ chính xác tra cứu cao hơn và không bị hiện tượng ảo giác (hallucination) như các mô hình sinh văn bản.*  
  3. *Tốc độ nạp chỉ mục nhị phân siêu tốc: Nhờ cơ chế `Persistent BM25 Cache`, hệ thống nạp toàn bộ chỉ mục trong **< 25 ms**, đảm bảo thời gian phản hồi tổng thể luôn dưới 300ms."*

---

### Câu 6: *"Hệ thống giải quyết câu hỏi tiếp nối (Multi-turn Context) như thế nào nếu người dùng chỉ hỏi ngắn gọn: 'Nó chạy như thế nào?'"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, hệ thống sử dụng kỹ thuật **Cửa sổ trượt Ngữ cảnh (Context Sliding Window Resolution)** tại `IntentClassifierAgent`:*  
  *Khi người dùng hỏi câu chứa các đại từ thay thế như 'nó', 'hàm đó', 'file này', 'đoạn mã trên', bộ nhận diện sẽ phát hiện đây là câu hỏi tiếp nối. Agent sẽ tự động trích xuất các thực thể kỹ thuật quan trọng từ `recent_context` của 2-3 lượt trò chuyện gần nhất trong `AgentState` và ghép nối vào câu hỏi hiện tại trước khi phân loại ý định và gửi sang Router. Nhờ đó, Router và các Worker vẫn hiểu chính xác đối tượng mà người dùng đang nhắc tới."*

---

### Câu 7: *"Công thức tính Điểm Tin Cậy (Confidence Score) của ValidatorAgent dựa trên nguyên lý nào?"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, Confidence Score không phải là một con số ngẫu nhiên mà được tính toán dựa trên **Mô hình Thẩm định Đa tiêu chí (Multi-criteria Scoring Model)**:*  
  - *$w_1 \cdot C_{\text{intent}}$: Độ chắc chắn của bộ phân loại ý định ở Pha 0.*  
  - *$w_2 \cdot C_{\text{worker}}$: Điểm số tin cậy do chính Worker trả về (ví dụ: điểm khớp BM25 của văn bản, hoặc cờ xác thực thành công từ Safe AST Calculator).*  
  - *$w_3 \cdot C_{\text{consistency}}$: Mức độ tương thích giữa kế hoạch của Planner và dữ liệu thực tế thu được.*  
  - *Hình phạt (Penalties): Trừ điểm nếu phát hiện dữ liệu rỗng, phép toán có giá trị vô lý hoặc thời gian chạy vượt ngưỡng cảnh báo.*  
  *Kết quả cuối cùng chuẩn hóa về thang điểm 0% – 100%, giúp người dùng và hệ thống biết câu trả lời có độ tin cậy đến đâu trước khi xuất ra giao diện."*

---

### Câu 8: *"Nếu một Worker Agent bị treo (hang) hoặc mất mạng khi tìm kiếm web, hệ thống của em có bị đơ không?"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, hệ thống được thiết kế theo cơ chế **Chống chịu Lỗi (Fault-tolerant Architecture)**:*  
  1. *Mỗi tác vụ worker trong ThreadPool được bọc bởi một lớp `safe_run` với bộ bắt ngoại lệ (try-except) toàn diện. Nếu một Worker gặp lỗi cú pháp hay ngoại lệ runtime, nó sẽ ghi nhận lỗi vào `log` và trả về giá trị an toàn mặc định thay vì làm sập cả tiến trình.*  
  2. *`RouterAgent` áp dụng `timeout = 8 giây`. Nếu sau 8 giây một worker (như WebSearch) không phản hồi, `as_completed()` sẽ ngắt luồng đó và tiếp tục tiến trình với các dữ liệu đã thu thập được.*  
  3. *Riêng `WebSearchAgent` còn có cơ chế **Offline Fallback**: Nếu mạng mất, nó tự động trích xuất từ cơ sở tri thức cục bộ lưu sẵn để câu trả lời không bao giờ bị gián đoạn."*

---

### Câu 9: *"Độ trễ 258ms được em đo đạc như thế nào? Có đáng tin cậy không?"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, số liệu 258ms được đo đạc hoàn toàn tự động bằng script benchmark độc lập [`scripts/benchmark.py`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/scripts/benchmark.py) qua 13 kịch bản câu hỏi kiểm thử bao phủ toàn bộ 8 intent kỹ thuật (toán học, thời gian, RAG tài liệu Word/Excel, phân tích giải thuật, tìm kiếm web, chào hỏi).*  
  *Đồng thời, trong mỗi lượt hỏi đáp thực tế, hệ thống đều sử dụng hàm `time.time()` để ghi nhận thời gian thực thi của từng Agent riêng lẻ vào bảng `agent_timings` trong SQLite và hiển thị trực tiếp trên biểu đồ SVG Call Graph của Dashboard. Người dùng và hội đồng có thể tự kiểm chứng trực tiếp bằng mắt bất kỳ lúc nào."*

---

### Câu 10: *"Hạn chế lớn nhất của đề tài hiện tại là gì và hướng phát triển trong tương lai?"*
- **Trả lời chuẩn chuyên gia:**  
  *"Thưa thầy/cô, tính đến thời điểm hiện tại, hạn chế của hệ thống là:*  
  1. *Chưa tích hợp một mô hình ngôn ngữ nhỏ chạy cục bộ (như Llama-3-8B hay Phi-3 qua Ollama/GGUF) để sinh ngôn ngữ tự nhiên linh hoạt hơn cho các câu hỏi mang tính đàm thoại mở rộng.*  
  2. *Bộ phân tích mã nguồn `AnalystAgent` hiện tại đang tối ưu sâu nhất cho ngôn ngữ Python qua AST; đối với các ngôn ngữ C++ hay Java, hệ thống mới chỉ phân tích ở mức biểu thức chính quy (Regex) và cấu trúc từ khóa.*  
  *Hướng phát triển tiếp theo của em là: Tích hợp thư viện `tree-sitter` để phân tích AST đa ngôn ngữ (C, C++, Java, Rust), và tích hợp một SLM (Small Language Model) dạng lượng tử hóa 4-bit chạy bằng CPU/GPU cục bộ để tăng cường khả năng diễn giải ngữ nghĩa."*

---

## 7. KỊCH BẢN GỢI Ý THUYẾT TRÌNH SLIDE (10 - 15 PHÚT)

Khi đứng trước hội đồng, bạn nên trình bày theo cấu trúc 6 phần mạch lạc sau:

1. **Slide 1 - 2 (1 phút): Giới thiệu Đề tài & Bối cảnh**
   - Giới thiệu tên đề tài và lý do chọn hướng đi: Tại sao không dùng API trả phí mà nghiên cứu Multi-Agent cục bộ (On-premise)? (Nhấn mạnh: Bảo mật 100%, Độ trễ 258ms, Chi phí 0 VNĐ).
2. **Slide 3 - 4 (3 phút): Sơ đồ Kiến trúc Tổng thể & Nguyên lý Điều phối 2 Pha**
   - Chiếu sơ đồ ngang [`so_do_multi_agent_ngang.png`](file:///d:/Python/Thực tập/Thuc_tap/Chatbot AI/so_do_multi_agent_ngang.png).
   - Giải thích luồng đi: Entry Cache -> Pha 0 (Intent & Context) -> Router Fan-Out -> Pha 1 (5 Worker song song) -> Barrier Fan-In -> Pha 2 (Pipeline tuần tự: Planner -> Validator -> Response) -> SQLite -> SSE Stream.
3. **Slide 5 - 7 (4 phút): Các Tác tử Nổi bật & Đột phá Kỹ thuật**
   - Nêu bật `KnowledgeBaseAgent` (BM25 + Persistent Cache nhị phân nạp nóng < 25ms).
   - Nêu bật `CalculatorAgent` (Safe AST Engine loại bỏ hoàn toàn lỗ hổng RCE của hàm `eval()`).
   - Nêu bật `ValidatorAgent` (Chấm điểm tin cậy Confidence Score khoa học và kiểm định chéo).
4. **Slide 8 - 9 (2 phút): Tầng Dữ liệu Đa năng & Cơ sở Dữ liệu SQLite**
   - Giới thiệu Universal FileReader: Đọc trực tiếp mã nguồn mọi ngôn ngữ, Word, Excel, PDF, PPTX, Ảnh và cả Thư mục dự án.
   - Cơ sở dữ liệu SQLite thread-safe chuẩn ACID thay thế hoàn toàn file JSON tĩnh.
5. **Slide 10 - 11 (2 phút): Kết quả Thực nghiệm & Kiểm thử**
   - Chiếu bảng so sánh 3 thế hệ (v1 vs v3 vs v4.1).
   - Nêu số liệu Benchmark: 100% Intent Accuracy, 258.38 ms tốc độ trung bình, 21/21 Unit Tests Passed.
6. **Slide 12 - 13 (2 phút): Demo Thực tế & Kết luận**
   - Mở giao diện Web Chat và Dashboard SVG Call Graph tại `/process/<session_id>` để thầy cô xem cây tiến trình thực tế.
   - Tóm tắt đóng góp của đề tài và định hướng phát triển trong tương lai.
   - Cảm ơn hội đồng và sẵn sàng bước vào phần Hỏi - Đáp (Q&A).

---
*Tài liệu được biên soạn đồng bộ với mã nguồn hệ thống Multi-Agent Chatbot v4.1 — Chúc bạn có một buổi thuyết trình và bảo vệ đề tài thành công xuất sắc!*

