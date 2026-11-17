# Báo Cáo Thẩm Định Hiệu Năng Hệ Thống Multi-Agent (_TEMPLATE)

- **Thời điểm kiểm định:** `2026-11-18T06:49:40.515422`
- **Lĩnh vực hoạt động:** `_template`
- **Số lượng mẫu thẩm định:** `8` câu hỏi

## 1. Bảng Chỉ Số Cốt Lõi (Key Performance Indicators)

| Chỉ số đo lường | Kết quả đạt được | Mục tiêu tiêu chuẩn | Trạng thái |
|---|---|---|---|
| **Độ chính xác Ý định (Intent Accuracy)** | **100.00%** | $\ge 90\%$ | ✅ Đạt |
| **Độ phủ Tri thức (Recall / Hit Rate)** | **100.00%** | $\ge 85\%$ | ✅ Đạt |
| **Độ trễ trung bình (Avg Latency)** | **340.5 ms** | $< 350\text{ ms}$ | ✅ Đạt |
| **Độ trễ P95 (Tail Latency)** | **1036.2 ms** | $< 600\text{ ms}$ | ⚠️ Cần cải thiện |
| **Điểm tin cậy trung bình** | **74.23%** | $\ge 75\%$ | ⚠️ Cần cải thiện |

## 2. Chi Tiết Độ Trễ Từng Tác Tử (Agent Latency Breakdown)

| Tác tử (AI Agent) | Thời gian xử lý trung bình | Thời gian đỉnh (Max) |
|---|---|---|
| `IntentClassifier` | 43.39 ms | 183.10 ms |
| `Router` | 293.98 ms | 1011.10 ms |
| `Calculator` | 0.80 ms | 1.50 ms |
| `Planner` | 0.10 ms | 0.20 ms |
| `Validator` | 0.54 ms | 1.60 ms |
| `ResponseGenerator` | 0.06 ms | 0.20 ms |
| `DateTime` | 0.10 ms | 0.10 ms |
| `Analyst` | 0.23 ms | 0.90 ms |
| `KnowledgeBase` | 0.67 ms | 1.90 ms |
| `WebSearch` | 776.50 ms | 1010.00 ms |

## 3. Bảng Kiểm Thử Chi Tiết Từng Câu Hỏi

| # | Câu hỏi thẩm định | Ý định kỳ vọng | Dự đoán | Khớp ý định | Trúng tri thức | Độ tin cậy | Độ trễ |
|---|---|---|---|---|---|---|---|
| 1 | tính 50 * 2 + 25 | `calculate` | `calculate` | ✅ | ✅ | 0.74 | 196.5 ms |
| 2 | hôm nay là thứ mấy | `datetime` | `datetime` | ✅ | ✅ | 1.00 | 25.4 ms |
| 3 | xin chào bot | `greet` | `greet` | ✅ | ✅ | 1.00 | 25.2 ms |
| 4 | langgraph là gì | `rag` | `rag` | ✅ | ✅ | 0.60 | 28.9 ms |
| 5 | thời tiết hôm nay | `search` | `search` | ✅ | ✅ | 0.61 | 1036.2 ms |
| 6 | phân tích ưu nhược điểm của Python | `analyze` | `analyze` | ✅ | ✅ | 0.69 | 722.6 ms |
| 7 | lập kế hoạch học Python trong 3 tháng | `plan` | `plan` | ✅ | ✅ | 0.68 | 660.0 ms |
| 8 | kiến trúc hệ thống RAG hoạt động như thế nào | `rag` | `rag` | ✅ | ✅ | 0.61 | 29.2 ms |
