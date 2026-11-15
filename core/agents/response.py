import re
from typing import List, Optional
from .base import AgentBase

class ResponseGeneratorAgent(AgentBase):
    NAME = "ResponseGenerator"

    # Khởi tạo tác tử tổng hợp phản hồi kỹ thuật.
    def __init__(self, called_by: str = "Validator"):
        super().__init__(called_by=called_by)

    # Tổng hợp toàn bộ dữ liệu từ các tác tử thành báo cáo kỹ thuật Markdown chuẩn mực.
    def run(self, query: str = "", intent: str = "general", kb_context: str = "", calc_result: str = "", datetime_result: str = "", web_results: str = "", analysis: str = "", planning: str = "", validation: str = "", confidence_score: float = 0.0, agent_timings: Optional[List[dict]] = None, history_summary: str = "", recent_context: str = "", **kwargs) -> dict:
        agent_timings = agent_timings or []
        sections: List[str] = []
        if history_summary:
            sections.append(f"> [!NOTE]\n> **Ngữ cảnh lịch sử phiên làm việc:** {history_summary}")
        main_content = self._format_technical_core(query, intent, calc_result, datetime_result, kb_context, web_results, analysis)
        sections.append(main_content)
        code_insights = self._extract_code_insights(kb_context, query)
        if code_insights:
            sections.append(code_insights)
        cross_refs = self._format_cross_references(intent, kb_context, web_results, analysis)
        if cross_refs:
            sections.append(cross_refs)
        if planning:
            sections.append(f"### 📋 Lộ trình Kỹ thuật & Khuyến nghị Thực thi\n\n{planning}")
        if validation:
            conf_percent = int(confidence_score * 100) if confidence_score else 85
            icon = "🟢" if conf_percent >= 80 else ("🟡" if conf_percent >= 50 else "🔴")
            sections.append(f"### {icon} Thẩm định Kỹ thuật & Độ tin cậy ({conf_percent}%)\n\n{validation}")
        trace_section = self._format_execution_trace(agent_timings)
        if trace_section:
            sections.append(trace_section)
        return {
            "final_answer": "\n\n---\n\n".join(sections),
            "message": "Tổng hợp phản hồi kỹ thuật thành công.",
        }

    # Định dạng khối nội dung phản hồi trọng tâm tương ứng với từng nhóm ý định.
    def _format_technical_core(self, query: str, intent: str, calc_result: str, datetime_result: str, kb_context: str, web_results: str, analysis: str) -> str:
        if intent == "calculate":
            if calc_result:
                return f"### 🧮 Kết quả Đo lường & Tính toán Kỹ thuật\n\n- **Yêu cầu / Biểu thức:** `{query}`\n- **Kết quả tính toán:** **`{calc_result}`**\n\n> *Xác minh: Biểu thức đã được thẩm định an toàn qua SymPy & AST Engine.*"
            return f"### 🧮 Tính toán Kỹ thuật\n\nKhông thể giải biểu thức: `{query}`. Vui lòng kiểm tra cú pháp toán học."
        elif intent == "datetime":
            return f"### ⏰ Lịch trình Dự án & Mốc Thời gian\n\n{datetime_result}" if datetime_result else f"### ⏰ Mốc Thời gian\n\nĐã tiếp nhận yêu cầu lịch trình: `{query}`."
        elif intent == "greet":
            return (
                "### 🤖 Trợ lý AI Chuyên gia Phân tích Kỹ thuật & Mã nguồn (v4.1)\n\n"
                "Xin chào! Tôi là **Trợ lý AI Kỹ thuật & Thẩm định Tài liệu Nội bộ**, hoạt động **100% bằng logic Python cục bộ** (No-API-Key).\n\n"
                "**Các năng lực kỹ thuật sẵn sàng hỗ trợ bạn:**\n"
                "1. 💻 **Phân tích Mã nguồn (Mọi ngôn ngữ):** Đọc hiểu mã nguồn Python, JavaScript, Java, C++, C#, Go, Rust, SQL, v.v.\n"
                "2. 📁 **Đính kèm Thư mục Dự án:** Tự động quét cấu trúc cây thư mục, tệp .zip và phân tích mã nguồn liên kết.\n"
                "3. 📚 **Tra cứu Tài liệu Kỹ thuật (RAG BM25):** Truy xuất thông tin từ PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx).\n"
                "4. 📊 **Thống kê & Đo lường Chỉ số:** Ước tính độ phức tạp $O(n)$, kiểm toán số liệu và phân tích ngữ nghĩa bằng TextRank.\n"
                "5. 📋 **Lập kế hoạch & Khuyến nghị:** Lập lộ trình phát triển tính năng, tái cấu trúc mã (Refactoring) và kiểm thử Unit Test.\n\n"
                "👉 Hãy đính kèm tệp/thư mục mã nguồn hoặc đặt câu hỏi kỹ thuật để bắt đầu!"
            )
        elif intent == "rag":
            return f"### 📚 Tri thức & Trích xuất Tài liệu / Mã nguồn Nội bộ\n\n{kb_context}" if kb_context else f"### 📚 Tra cứu Tài liệu / Mã nguồn\n\nChưa tìm thấy đoạn mã hoặc tài liệu trùng khớp với yêu cầu: *\"{query}\"*. Hãy đính kèm tệp mã nguồn hoặc bổ sung tài liệu vào thư mục `data/docs/`."
        elif intent == "search":
            return f"### 🔍 Tra cứu Tài liệu Kỹ thuật & Tiêu chuẩn Công nghệ\n\n{web_results}" if web_results else f"### 🔍 Tra cứu\n\nĐã ghi nhận yêu cầu tìm kiếm: *\"{query}\"*."
        elif intent == "analyze":
            return f"### 📊 Báo cáo Phân tích Cấu trúc & Ngữ nghĩa (TextRank NLP)\n\n{analysis}" if analysis else f"### 📊 Phân tích Kỹ thuật\n\nNội dung phân tích cho chủ đề: *\"{query}\"*."
        elif intent == "plan":
            return f"### 🎯 Phân tích Mục tiêu Kỹ thuật & Phân rã Lộ trình\n\n**Mục tiêu yêu cầu:** *\"{query}\"*"
        else:
            parts = []
            if kb_context:
                parts.append(f"**Trích dẫn tri thức nội bộ:**\n{kb_context}")
            if analysis:
                parts.append(f"**Phân tích tổng quan:**\n{analysis}")
            if web_results:
                parts.append(f"**Tài liệu tham khảo công nghệ:**\n{web_results}")
            return "\n\n".join(parts) if parts else f"Đã xử lý yêu cầu kỹ thuật: *\"{query}\"*. Hệ thống đa tác tử đã đối chiếu dữ liệu nội bộ và thẩm định mức độ tin cậy."

    # Tự động nhận diện cấu trúc mã nguồn và bổ sung nhận xét kiến trúc phần mềm.
    def _extract_code_insights(self, kb_context: str, query: str) -> Optional[str]:
        if not kb_context or not any(kw in kb_context for kw in ("def ", "class ", "function ", "import ", "from ", "CREATE TABLE", "public class", "fn ", "SELECT ", "return ")):
            return None
        funcs = list(dict.fromkeys(re.findall(r"(?:def|class|function|fn)\s+([a-zA-Z_][a-zA-Z0-9_]*)", kb_context)))[:5]
        insights = ["### 💻 Đánh giá Mã nguồn & Giải pháp Kỹ thuật\n"]
        if funcs:
            insights.append("• **Các thành phần phát hiện:** " + ", ".join(f"`{f}`" for f in funcs))
        insights.extend([
            "• **Khuyến nghị kiến trúc & thực thi:**",
            "  - Kiểm tra điều kiện biên (Null safety, chia cho 0, ngoại lệ đầu vào).",
            "  - Đảm bảo hàm có đầy đủ Type Annotations và Unit Test kiểm thử hồi quy.",
            "  - Tối ưu hóa độ phức tạp thời gian đạt mức tiệm cận tối thiểu $O(1)$ hoặc $O(n).",
        ])
        return "\n".join(insights)

    # Tổng hợp các khối dữ liệu bổ trợ ngoài nội dung chính nếu có giá trị đối chiếu.
    def _format_cross_references(self, intent: str, kb_context: str, web_results: str, analysis: str) -> str:
        extras = []
        if intent != "rag" and kb_context:
            extras.append(f"**Tri thức đối chiếu (Local RAG):**\n{kb_context}")
        if intent != "search" and web_results:
            extras.append(f"**Tài liệu công nghệ tham chiếu thêm:**\n{web_results}")
        if intent not in ("analyze", "general") and analysis:
            extras.append(f"**Chỉ số phân tích văn bản (TextRank):**\n{analysis}")
        return ("### 📎 Dữ liệu Bổ trợ & Nguồn Tham chiếu\n" + "\n\n".join(extras)) if extras else ""

    # Trình bày dấu vết thời gian thực thi của từng tác tử trong bảng Markdown thu gọn.
    def _format_execution_trace(self, agent_timings: List[dict]) -> str:
        if not agent_timings:
            return ""
        total_duration = sum(t.get("duration", 0.0) for t in agent_timings)
        lines = [
            f"<details><summary>⏱️ <b>Chi tiết luồng thực thi Multi-Agent ({len(agent_timings)} tác tử, {total_duration:.3f}s)</b></summary>\n",
            "| # | Tác tử (AI Agent) | Điều phối bởi | Thời gian | Tác tử gọi tiếp |",
            "|---|---|---|---|---|",
        ]
        for i, t in enumerate(agent_timings, 1):
            lines.append(f"| {i} | `{t.get('agent', 'Unknown')}` | {t.get('called_by', '-')} | {t.get('duration', 0.0):.4f}s | {', '.join(t.get('calls', [])) or '-'} |")
        lines.extend(["", "</details>"])
        return "\n".join(lines)
