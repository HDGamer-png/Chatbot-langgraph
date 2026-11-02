"""
core/agents/response.py — ResponseGeneratorAgent
Tác tử tổng hợp cuối cùng — Đóng gói toàn bộ kết quả thành câu trả lời Markdown chuẩn.

Hoạt động 100% nội bộ bằng Động cơ Template Động đa tầng (Multi-tier Synthesis Engine).
Không phụ thuộc API Key bên ngoài.
"""

from typing import List, Optional
from .base import AgentBase


class ResponseGeneratorAgent(AgentBase):
    """
    Tác tử tổng hợp (Coordinator / Response Generator).
    Tiếp nhận dữ liệu từ tất cả các Agent con (Pha Song song và Pha Tuần tự),
    tổng hợp thành một bản phản hồi Markdown hoàn chỉnh, có cấu trúc chuyên nghiệp.
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
        **kwargs,
    ) -> dict:
        """
        Tổng hợp câu trả lời theo intent và dữ liệu thu thập được.
        """
        agent_timings = agent_timings or []
        sections: List[str] = []

        # ── 1. PHẦN TRẢ LỜI CHÍNH THEO Ý ĐỊNH (EXECUTIVE CONTENT) ──────────
        main_content = self._format_main_content(
            query=query,
            intent=intent,
            calc_result=calc_result,
            datetime_result=datetime_result,
            kb_context=kb_context,
            web_results=web_results,
            analysis=analysis,
        )
        sections.append(main_content)

        # ── 2. CHI TIẾT PHÂN TÍCH / TÀI LIỆU BỔ TRỢ (NẾU CÓ) ───────────────
        extra_content = self._format_extra_content(
            intent=intent,
            kb_context=kb_context,
            web_results=web_results,
            analysis=analysis,
        )
        if extra_content:
            sections.append(extra_content)

        # ── 3. KẾ HOẠCH HÀNH ĐỘNG (PLANNING TỪ PLANNER AGENT) ─────────────
        if planning and intent in ("plan", "analyze", "general"):
            sections.append(f"### 📋 Kế hoạch & Lộ trình thực hiện\n{planning}")

        # ── 4. THÔNG TIN KIỂM ĐỊNH & ĐỘ TIN CẬY (VALIDATOR AGENT) ──────────
        if validation:
            conf_percent = int(confidence_score * 100) if confidence_score else 85
            icon = "🟢" if conf_percent >= 80 else ("🟡" if conf_percent >= 50 else "🔴")
            val_section = (
                f"### {icon} Đánh giá & Kiểm định độ tin cậy ({conf_percent}%)\n"
                f"{validation}"
            )
            sections.append(val_section)

        # ── 5. NGUỒN THAM KHẢO & DẤU VẾT THỜI GIAN (METRICS TRACING) ──────
        trace_section = self._format_execution_trace(agent_timings)
        if trace_section:
            sections.append(trace_section)

        final_answer = "\n\n---\n\n".join(sections)
        return {
            "final_answer": final_answer,
            "message": "Tổng hợp câu trả lời thành công.",
        }

    def _format_main_content(
        self,
        query: str,
        intent: str,
        calc_result: str,
        datetime_result: str,
        kb_context: str,
        web_results: str,
        analysis: str,
    ) -> str:
        """Định dạng câu trả lời cốt lõi theo từng nhóm ý định."""
        if intent == "calculate":
            if calc_result:
                return (
                    f"### 🧮 Kết quả tính toán\n\n"
                    f"**Biểu thức / Bài toán:** `{query}`\n\n"
                    f"👉 **Đáp số:** **`{calc_result}`**"
                )
            return f"### 🧮 Tính toán\n\nKhông thể giải biểu thức: `{query}`. Vui lòng kiểm tra lại định dạng phép tính."

        elif intent == "datetime":
            if datetime_result:
                return f"### ⏰ Thông tin thời gian\n\n{datetime_result}"
            return f"### ⏰ Thời gian\n\nĐã tiếp nhận yêu cầu thời gian cho: `{query}`."

        elif intent == "greet":
            return (
                "👋 **Xin chào! Tôi là Trợ lý AI Đa tác tử (LangGraph Multi-Agent Assistant v4.0).**\n\n"
                "Hệ thống vận hành **100% bằng logic Python nội bộ** với cơ chế điều phối song song kết hợp kiểm định chéo. "
                "Tôi có thể hỗ trợ bạn:\n"
                "- 🧮 **Tính toán & Giải toán:** biểu thức đại số, giải phương trình, lượng giác\n"
                "- 📚 **Tra cứu tài liệu RAG:** trích xuất kiến thức chuyên sâu từ file PDF, DOCX, TXT\n"
                "- ⏰ **Thời gian & Múi giờ:** ngày giờ hiện tại, đếm ngược ngày lễ, múi giờ quốc tế\n"
                "- 🔍 **Tìm kiếm tri thức:** tra cứu khái niệm công nghệ và thông tin tổng hợp\n"
                "- 📊 **Phân tích & Tóm tắt:** trích xuất ý chính văn bản bằng TextRank NLP\n"
                "- 📋 **Lập kế hoạch tác vụ:** phân rã mục tiêu thành các bước hành động cụ thể\n\n"
                "Hôm nay tôi có thể giúp gì cho bạn?"
            )

        elif intent == "rag":
            if kb_context:
                return f"### 📚 Tri thức trích xuất từ tài liệu nội bộ\n\n{kb_context}"
            return (
                f"### 📚 Tra cứu tri thức nội bộ\n\n"
                f"Chưa tìm thấy đoạn trích phù hợp cho câu hỏi: *\"{query}\"*. "
                f"Bạn có thể bổ sung tài liệu liên quan vào thư mục `data/docs/` hoặc đính kèm tệp trực tiếp vào phiên chat."
            )

        elif intent == "search":
            if web_results:
                return f"### 🔍 Kết quả tra cứu thông tin\n\n{web_results}"
            return f"### 🔍 Tra cứu\n\nĐã ghi nhận yêu cầu tìm kiếm cho: *\"{query}\"*."

        elif intent == "analyze":
            if analysis:
                return f"### 📊 Kết quả phân tích chuyên sâu\n\n{analysis}"
            return f"### 📊 Phân tích\n\nNội dung phân tích cho chủ đề: *\"{query}\"*."

        elif intent == "plan":
            return f"### 🎯 Phân tích mục tiêu & Lập kế hoạch\n\nYêu cầu: **{query}**"

        else:  # general
            ans = []
            if analysis:
                ans.append(f"**Phân tích tổng quan:**\n{analysis}")
            if kb_context:
                ans.append(f"**Thông tin liên quan:**\n{kb_context}")
            if web_results:
                ans.append(f"**Dữ liệu tham khảo:**\n{web_results}")
            if not ans:
                ans.append(
                    f"Đã xử lý yêu cầu của bạn: *\"{query}\"*. "
                    "Hệ thống đa tác tử đã phối hợp thu thập và xác thực dữ liệu qua đồ thị LangGraph."
                )
            return "\n\n".join(ans)

    def _format_extra_content(
        self, intent: str, kb_context: str, web_results: str, analysis: str
    ) -> str:
        """Thêm các phần dữ liệu bổ trợ nếu hữu ích cho người dùng."""
        extras = []
        if intent != "rag" and kb_context:
            extras.append(f"**Tri thức đối chiếu (Local RAG):**\n{kb_context}")
        if intent != "search" and web_results:
            extras.append(f"**Dữ liệu tham chiếu thêm:**\n{web_results}")
        if intent not in ("analyze", "general") and analysis:
            extras.append(f"**Điểm phân tích bổ trợ:**\n{analysis}")

        if extras:
            return "### 📎 Dữ liệu bổ trợ thu thập từ các tác tử\n" + "\n\n".join(extras)
        return ""

    def _format_execution_trace(self, agent_timings: List[dict]) -> str:
        """Định dạng dấu vết thực thi (Agent Execution Trace) thu nhỏ."""
        if not agent_timings:
            return ""

        total_duration = sum(t.get("duration", 0.0) for t in agent_timings)
        trace_lines = [
            f"<details><summary>⏱️ <b>Chi tiết luồng thực thi ({len(agent_timings)} tác tử, {total_duration:.3f}s)</b></summary>",
            "",
            "| # | Tác tử (Agent) | Điều phối bởi | Thời gian | Tác tử gọi tiếp |",
            "|---|----------------|---------------|-----------|-----------------|",
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

