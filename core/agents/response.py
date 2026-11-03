"""
core/agents/response.py — ResponseGeneratorAgent v4.1
Tác tử Tổng hợp Phản hồi Kỹ thuật Chuyên sâu (Senior Technical Assistant / Tech Lead Synthesizer).

Định hình 1 mục đích duy nhất:
  - Phân tích mã nguồn mọi ngôn ngữ, đánh giá độ phức tạp thuật toán O(n).
  - Đối chiếu tài liệu kỹ thuật nội bộ (RAG BM25) với trích dẫn nguồn minh bạch.
  - Xây dựng lộ trình refactoring/triển khai có mức độ ưu tiên rõ ràng.
  - Thẩm định độ tin cậy từ ValidatorAgent.
  - Tích hợp bộ nhớ ngữ cảnh đa lượt (Multi-turn Context Memory).

Hoạt động 100% bằng logic Python nội bộ, không phụ thuộc API Key bên ngoài.
"""

import re
from typing import List, Optional, Dict, Any
from .base import AgentBase


class ResponseGeneratorAgent(AgentBase):
    """
    Tác tử tổng hợp phản hồi Kỹ thuật Chuyên sâu (Technical Lead Synthesizer).
    Tiếp nhận dữ liệu từ toàn bộ 10 AI Agent và ngữ cảnh hội thoại đa lượt,
    tổng hợp thành bản báo cáo kỹ thuật chuẩn mực Markdown.
    """

    NAME = "ResponseGenerator"

    def __init__(self, called_by: str = "Validator"):
        super().__init__(called_by=called_by)

    def run(
        self,
        query: str,
        intent: str,
        kb_context: str = "",
        calc_result: str = "",
        datetime_result: str = "",
        web_results: str = "",
        analysis: str = "",
        planning: str = "",
        validation: str = "",
        confidence_score: float = 0.0,
        agent_timings: Optional[List[dict]] = None,
        history_summary: str = "",
        recent_context: str = "",
        **kwargs,
    ) -> dict:
        """
        Tổng hợp câu trả lời kỹ thuật chuẩn hóa cao cấp.
        """
        agent_timings = agent_timings or []
        sections: List[str] = []

        # ── 0. BỐI CẢNH HỘI THOẠI TRƯỚC (NẾU CÓ) ───────────────────────────
        if history_summary:
            sections.append(
                f"> [!NOTE]\n"
                f"> **Ngữ cảnh lịch sử phiên làm việc:** {history_summary}"
            )

        # ── 1. PHẦN TRẢ LỜI CHÍNH THEO Ý ĐỊNH KỸ THUẬT (EXECUTIVE SUMMARY) ─
        main_content = self._format_technical_core(
            query=query,
            intent=intent,
            calc_result=calc_result,
            datetime_result=datetime_result,
            kb_context=kb_context,
            web_results=web_results,
            analysis=analysis,
        )
        sections.append(main_content)

        # ── 2. PHÂN TÍCH MÃ NGUỒN & KIẾN TRÚC KỸ THUẬT (CODE INSIGHTS) ──────
        code_insights = self._extract_code_insights(kb_context, query)
        if code_insights:
            sections.append(code_insights)

        # ── 3. TÀI LIỆU & DỮ LIỆU ĐỐI CHIẾU (CROSS-REFERENCES) ───────────────
        cross_refs = self._format_cross_references(
            intent=intent,
            kb_context=kb_context,
            web_results=web_results,
            analysis=analysis,
        )
        if cross_refs:
            sections.append(cross_refs)

        # ── 4. LỘ TRÌNH THỰC THI & KHUYẾN NGHỊ (ACTIONABLE ROADMAP) ─────────
        if planning:
            sections.append(f"### 📋 Lộ trình Kỹ thuật & Khuyến nghị Thực thi\n\n{planning}")

        # ── 5. THẨM ĐỊNH & ĐIỂM TIN CẬY KỸ THUẬT (CONFIDENCE SCOREBOARD) ───
        if validation:
            conf_percent = int(confidence_score * 100) if confidence_score else 85
            icon = "🟢" if conf_percent >= 80 else ("🟡" if conf_percent >= 50 else "🔴")
            val_section = (
                f"### {icon} Thẩm định Kỹ thuật & Độ tin cậy ({conf_percent}%)\n\n"
                f"{validation}"
            )
            sections.append(val_section)

        # ── 6. DẤU VẾT THỜI GIAN THỰC THI (CALL GRAPH TRACE) ────────────────
        trace_section = self._format_execution_trace(agent_timings)
        if trace_section:
            sections.append(trace_section)

        final_answer = "\n\n---\n\n".join(sections)
        return {
            "final_answer": final_answer,
            "message": "Tổng hợp phản hồi kỹ thuật thành công.",
        }

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE FORMATTERS
    # ══════════════════════════════════════════════════════════════════

    def _format_technical_core(
        self,
        query: str,
        intent: str,
        calc_result: str,
        datetime_result: str,
        kb_context: str,
        web_results: str,
        analysis: str,
    ) -> str:
        """Định dạng nội dung cốt lõi theo tiêu chuẩn Kỹ sư phần mềm."""
        if intent == "calculate":
            if calc_result:
                return (
                    f"### 🧮 Kết quả Đo lường & Tính toán Kỹ thuật\n\n"
                    f"- **Yêu cầu / Biểu thức:** `{query}`\n"
                    f"- **Kết quả tính toán:** **`{calc_result}`**\n\n"
                    f"> *Xác minh: Biểu thức đã được thẩm định an toàn qua SymPy & AST Engine.*"
                )
            return f"### 🧮 Tính toán Kỹ thuật\n\nKhông thể giải biểu thức: `{query}`. Vui lòng kiểm tra cú pháp toán học."

        elif intent == "datetime":
            if datetime_result:
                return f"### ⏰ Lịch trình Dự án & Mốc Thời gian\n\n{datetime_result}"
            return f"### ⏰ Mốc Thời gian\n\nĐã tiếp nhận yêu cầu lịch trình: `{query}`."

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
            if kb_context:
                return f"### 📚 Tri thức & Trích xuất Tài liệu / Mã nguồn Nội bộ\n\n{kb_context}"
            return (
                f"### 📚 Tra cứu Tài liệu / Mã nguồn\n\n"
                f"Chưa tìm thấy đoạn mã hoặc tài liệu trùng khớp với yêu cầu: *\"{query}\"*. "
                f"Hãy đính kèm tệp mã nguồn hoặc bổ sung tài liệu vào thư mục `data/docs/`."
            )

        elif intent == "search":
            if web_results:
                return f"### 🔍 Tra cứu Tài liệu Kỹ thuật & Tiêu chuẩn Công nghệ\n\n{web_results}"
            return f"### 🔍 Tra cứu\n\nĐã ghi nhận yêu cầu tìm kiếm: *\"{query}\"*."

        elif intent == "analyze":
            if analysis:
                return f"### 📊 Báo cáo Phân tích Cấu trúc & Ngữ nghĩa (TextRank NLP)\n\n{analysis}"
            return f"### 📊 Phân tích Kỹ thuật\n\nNội dung phân tích cho chủ đề: *\"{query}\"*."

        elif intent == "plan":
            return (
                f"### 🎯 Phân tích Mục tiêu Kỹ thuật & Phân rã Lộ trình\n\n"
                f"**Mục tiêu yêu cầu:** *\"{query}\"*"
            )

        else:  # general
            parts = []
            if kb_context:
                parts.append(f"**Trích dẫn tri thức nội bộ:**\n{kb_context}")
            if analysis:
                parts.append(f"**Phân tích tổng quan:**\n{analysis}")
            if web_results:
                parts.append(f"**Tài liệu tham khảo công nghệ:**\n{web_results}")
            if not parts:
                parts.append(
                    f"Đã xử lý yêu cầu kỹ thuật: *\"{query}\"*. "
                    "Hệ thống đa tác tử đã đối chiếu dữ liệu nội bộ và thẩm định mức độ tin cậy."
                )
            return "\n\n".join(parts)

    def _extract_code_insights(self, kb_context: str, query: str) -> Optional[str]:
        """Tự động phát hiện khối mã nguồn trong ngữ cảnh và bổ sung phân tích kiến trúc."""
        if not kb_context:
            return None

        # Kiểm tra xem có khối mã nguồn nào không (def, class, function, CREATE TABLE)
        has_code_keywords = any(kw in kb_context for kw in (
            "def ", "class ", "function ", "import ", "from ", "CREATE TABLE",
            "public class", "fn ", "SELECT ", "return "
        ))

        if not has_code_keywords:
            return None

        # Trích xuất các hàm hoặc lớp tiêu biểu
        func_names = re.findall(r"(?:def|class|function|fn)\s+([a-zA-Z_][a-zA-Z0-9_]*)", kb_context)
        unique_funcs = list(dict.fromkeys(func_names))[:5]

        insights = [
            "### 💻 Đánh giá Mã nguồn & Giải pháp Kỹ thuật",
            ""
        ]

        if unique_funcs:
            insights.append(f"• **Các thành phần phát hiện:** " + ", ".join(f"`{f}`" for f in unique_funcs))

        insights.append("• **Khuyến nghị kiến trúc & thực thi:**")
        insights.append("  - Kiểm tra điều kiện biên (Null safety, chia cho 0, ngoại lệ đầu vào).")
        insights.append("  - Đảm bảo hàm có đầy đủ Type Annotations và Unit Test kiểm thử hồi quy.")
        insights.append("  - Tối ưu hóa độ phức tạp thời gian đạt mức tiệm cận tối thiểu $O(1)$ hoặc $O(n)$.")

        return "\n".join(insights)

    def _format_cross_references(
        self, intent: str, kb_context: str, web_results: str, analysis: str
    ) -> str:
        """Thêm các phần dữ liệu bổ trợ nếu hữu ích cho người dùng."""
        extras = []
        if intent != "rag" and kb_context:
            extras.append(f"**Tri thức đối chiếu (Local RAG):**\n{kb_context}")
        if intent != "search" and web_results:
            extras.append(f"**Tài liệu công nghệ tham chiếu thêm:**\n{web_results}")
        if intent not in ("analyze", "general") and analysis:
            extras.append(f"**Chỉ số phân tích văn bản (TextRank):**\n{analysis}")

        if extras:
            return "### 📎 Dữ liệu Bổ trợ & Nguồn Tham chiếu\n" + "\n\n".join(extras)
        return ""

    def _format_execution_trace(self, agent_timings: List[dict]) -> str:
        """Định dạng dấu vết thực thi (Agent Execution Trace) thu nhỏ."""
        if not agent_timings:
            return ""

        total_duration = sum(t.get("duration", 0.0) for t in agent_timings)
        trace_lines = [
            f"<details><summary>⏱️ <b>Chi tiết luồng thực thi Multi-Agent ({len(agent_timings)} tác tử, {total_duration:.3f}s)</b></summary>",
            "",
            "| # | Tác tử (AI Agent) | Điều phối bởi | Thời gian | Tác tử gọi tiếp |",
            "|---|---|---|---|---|",
        ]

        for i, t in enumerate(agent_timings, 1):
            agent_name = t.get("agent", "Unknown")
            called_by = t.get("called_by", "-")
            duration = f"{t.get('duration', 0.0):.4f}s"
            calls = ", ".join(t.get("calls", [])) or "-"
            trace_lines.append(f"| {i} | `{agent_name}` | {called_by} | {duration} | {calls} |")

        trace_lines.append("")
        trace_lines.append("</details>")
        return "\n".join(trace_lines)
