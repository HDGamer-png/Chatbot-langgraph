# LangGraph Multi-Agent Chatbot v3.0
**Tác giả:** Hoàng Minh Đức — Thực tập sinh AI Agent 2026

---

## Điểm mới so với v2

| Tính năng | v2 | v3 |
|---|---|---|
| Agent execution | Hàm Python inline | **Class riêng + ThreadPool/ProcessPool** |
| LLM | MockLLM fallback | **Claude API bắt buộc (Haiku + Sonnet)** |
| Lưu dữ liệu | Không | **JSON file mỗi session** |
| Timing analysis | Log text đơn giản | **Gantt chart + Call graph + Cumulative stats** |
| Dashboard | HTML static | **HTML dashboard đọc JSON session** |

---

## Kiến trúc Agent

```
[User Input]
     │
     ▼
① IntentClassifierAgent   ← Claude Haiku (fast, cheap)
     │  classify intent → 12 nhãn
     ▼
② KnowledgeBaseAgent      ← Pure Python, ProcessPool
     │  RAG từ KB nội bộ
     ▼
③ RouterAgent             ← ThreadPoolExecutor dispatcher
     │  ┌──────────────────────────────────┐
     ├──► WebSearchAgent   (I/O, Thread)   │
     ├──► CalculatorAgent  (CPU, Process)  │ ← Song song
     ├──► DateTimeAgent    (Python inline) │
     └──► AnalystAgent     (LLM, Thread)   │
          └──────────────────────────────────┘
     ▼
④ ResponseGeneratorAgent  ← Claude Sonnet (main)
     │  tổng hợp context → câu trả lời
     ▼
⑤ PersistenceNode         ← Ghi JSON file
     │  chat_history/session_<timestamp>.json
     ▼
[END]
```

### Tại sao mỗi Agent là Process/Thread riêng?

- **IntentClassifier**: Thread (I/O-bound — gọi Claude Haiku API)
- **KnowledgeBase**: Inline Python (pure CPU, không cần isolation)
- **Calculator**: `ProcessPoolExecutor` — CPU-bound `eval()`, cần isolation để tránh crash main process
- **WebSearch**: Thread (I/O-bound — HTTP request Tavily)
- **Analyst**: Thread (I/O-bound — gọi Claude Sonnet API)
- **ResponseGenerator**: Thread (I/O-bound — gọi Claude Sonnet API)

> **Router dispatch song song**: khi intent=`analyze`, Router tạo 2 futures trong ThreadPool — `WebSearch` và `Analyst` chạy **đồng thời**, giảm latency tổng.

---

## Luồng gọi nhau giữa các Agent

```
GraphEntryPoint
  └─► IntentClassifier (Claude Haiku)
        └─► KnowledgeBase (Python)
              └─► Router (Dispatcher)
                    ├─► WebSearch (Tavily API)    ┐ song song
                    ├─► Calculator (subprocess)   │ nếu cần
                    ├─► DateTime (inline)          │
                    └─► Analyst (Claude Sonnet)   ┘
                          └─► ResponseGenerator (Claude Sonnet)
                                └─► Persistence (JSON write)
```

**Agent kêu gọi LLM riêng (sub-call):**
- `IntentClassifier` → `claude-haiku-4-5-20251001` (classify)
- `Analyst` → `claude-sonnet-4-20250514` (phân tích sâu)
- `ResponseGenerator` → `claude-sonnet-4-20250514` (tổng hợp cuối)

---

## Cài đặt

```bash
pip install langgraph langchain-anthropic langchain-core \
            tavily-python rich python-dotenv
```

File `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...    # bắt buộc
TAVILY_API_KEY=tvly-...         # tùy chọn (web search)
```

---

## Chạy

```bash
python multi_agent_chatbot_v3.py
```

### Lệnh trong chat:

| Lệnh | Mô tả |
|---|---|
| `graph` | Sơ đồ kiến trúc Nodes |
| `timing` | Phân tích timing turn vừa rồi |
| `stats` | Thống kê tổng hợp từ file JSON |
| `help` | Gợi ý câu hỏi |
| `quit` | Thoát, hiển thị stats tổng |

---

## File JSON lưu ra

Mỗi session tạo file `chat_history/session_YYYYMMDD_HHMMSS.json`:

```json
{
  "session_id": "20260522_120000",
  "created_at": "2026-05-22T12:00:00",
  "model_fast": "claude-haiku-4-5-20251001",
  "model_main": "claude-sonnet-4-20250514",
  "turns": [
    {
      "turn_id": 1,
      "timestamp": "2026-05-22T12:00:05",
      "user_query": "LangGraph là gì?",
      "intent": "langgraph",
      "final_answer": "...",
      "agent_timings": [
        {
          "agent": "IntentClassifier",
          "start_ts": 1716379205.0,
          "end_ts": 1716379205.21,
          "duration": 0.21,
          "called_by": "GraphEntry",
          "calls": []
        }
      ],
      "call_graph": [
        {"from": "GraphEntry", "to": "IntentClassifier", "ts": 1716379205.0}
      ]
    }
  ],
  "agent_stats": {
    "IntentClassifier": {
      "calls": 3,
      "total_time": 0.63,
      "avg_time": 0.21,
      "max_time": 0.25
    }
  }
}
```

---

## Dashboard HTML

Mở `agent_dashboard.html` trong trình duyệt, kéo thả file JSON session vào để xem:

- **Gantt chart**: timeline song song của từng Agent
- **Call graph SVG**: sơ đồ Agent kêu gọi nhau
- **Cumulative stats**: bảng thống kê calls/avg/max time
- **Chat history**: toàn bộ hội thoại kèm timing

---

## Phân tích Timing — Ví dụ thực tế

```
Turn: "Phân tích ưu nhược điểm Multi-Agent" (intent=analyze)

Agent              Duration   Called By       Calls
─────────────────────────────────────────────────────
IntentClassifier   0.210s     GraphEntry      []
KnowledgeBase      0.018s     IntentClassifier []
Router             0.042s     KnowledgeBase   [WebSearch, Analyst]
WebSearch          1.850s     Router          []   ┐ song song
Analyst            2.100s     Router          [Claude-Sonnet] ┘
ResponseGenerator  1.900s     Router          [Claude-Sonnet]
Persistence        0.031s     ResponseGenerator []

Tổng: ~4.15s  (không song song sẽ là ~6.15s)
```

> **Bottleneck**: `ResponseGenerator` và `Analyst` đều gọi Claude Sonnet. Nếu muốn nhanh hơn: dùng `claude-haiku` cho Analyst.
