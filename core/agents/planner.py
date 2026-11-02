"""
core/agents/planner.py — PlannerAgent: Lập kế hoạch hành động dựa trên ý định và dữ liệu thu thập.

Agent này chạy ĐẦU TIÊN trong pipeline tuần tự (sequential), ngay sau khi
pha song song (parallel phase) hoàn tất. Nó nhận toàn bộ dữ liệu đã thu thập
từ các agent song song và tạo ra một kế hoạch thực thi có cấu trúc gồm 3-5 bước.

Chiến lược lập kế hoạch:
  - Goal-Dependency Decomposition: phân tích thông tin hiện có, thông tin còn thiếu,
    và cách cấu trúc câu trả lời tối ưu.
  - Mỗi loại ý định (intent) có bản kế hoạch riêng phù hợp.
"""

from .base import AgentBase


class PlannerAgent(AgentBase):
    """
    Agent lập kế hoạch hành động cho pipeline tuần tự.

    Phân tích ý định (intent) và ngữ cảnh (context) đã thu thập được,
    sau đó sinh ra một danh sách các bước thực thi có thứ tự (3-5 bước)
    để các agent tiếp theo trong pipeline có thể đối chiếu, kiểm tra,
    và tổng hợp câu trả lời cuối cùng một cách chính xác.

    Attributes:
        NAME (str): Tên định danh của agent trong hệ thống — "Planner".
    """

    NAME = "Planner"

    # ------------------------------------------------------------------ #
    #  Bản đồ kế hoạch mặc định cho từng loại ý định (intent)           #
    # ------------------------------------------------------------------ #
    _PLAN_TEMPLATES: dict[str, list[str]] = {
        "calculate": [
            "Xác minh biểu thức toán học từ truy vấn gốc",
            "Thực hiện tính toán và kiểm tra các trường hợp biên",
            "So sánh kết quả tính toán với dữ liệu ngữ cảnh (nếu có)",
            "Định dạng kết quả cuối cùng với giải thích từng bước",
        ],
        "datetime": [
            "Phân tích tham chiếu thời gian trong truy vấn",
            "Tính toán ngày/giờ tương ứng dựa trên tham chiếu",
            "Định dạng kết quả theo tiếng Việt (ngày-tháng-năm, thứ, v.v.)",
            "Bổ sung thông tin liên quan (múi giờ, ngày lễ nếu có)",
        ],
        "rag": [
            "Xác định chủ đề chính và các từ khóa then chốt",
            "Tìm kiếm thông tin trong cơ sở tri thức nội bộ",
            "Trích xuất các đoạn văn bản liên quan nhất",
            "Tổng hợp thông tin và ghi chú nguồn trích dẫn",
            "Kiểm tra tính nhất quán giữa các nguồn",
        ],
        "search": [
            "Xây dựng truy vấn tìm kiếm tối ưu từ câu hỏi gốc",
            "Thu thập kết quả tìm kiếm từ nhiều nguồn",
            "Lọc và xếp hạng kết quả theo mức độ liên quan",
            "Tóm tắt thông tin chính kèm liên kết tham khảo",
        ],
        "analyze": [
            "Phân tích cú pháp và ngữ nghĩa đầu vào",
            "Xác định các điểm chính và luận điểm cần phân tích",
            "Đối chiếu chéo với dữ liệu từ nhiều nguồn",
            "Tổng hợp phân tích với kết luận có cơ sở",
        ],
        "plan": [
            "Phân rã mục tiêu thành các nhiệm vụ con",
            "Xác định tài nguyên và ràng buộc cần thiết",
            "Lập lịch trình thực hiện theo thứ tự ưu tiên",
            "Đánh giá tính khả thi và rủi ro tiềm ẩn",
            "Đề xuất phương án dự phòng",
        ],
        "greet": [
            "Nhận diện loại chào hỏi (xã giao, hỏi thăm, giới thiệu)",
            "Tạo phản hồi phù hợp với ngữ cảnh giao tiếp",
            "Bổ sung gợi ý hành động tiếp theo cho người dùng",
        ],
        "general": [
            "Đánh giá độ phức tạp của câu hỏi",
            "Xác định phương pháp trả lời tối ưu",
            "Thu thập và tổ chức thông tin liên quan",
            "Soạn câu trả lời toàn diện và dễ hiểu",
        ],
    }

    # ------------------------------------------------------------------ #
    #  Phương thức chính                                                  #
    # ------------------------------------------------------------------ #

    def run(self, query: str, intent: str, context: str = "") -> dict:
        """
        Sinh kế hoạch hành động dựa trên ý định và dữ liệu thu thập.

        Quy trình Goal-Dependency Decomposition:
          1. Kiểm tra thông tin nào đã có (context).
          2. Xác định thông tin nào còn thiếu.
          3. Chọn bản kế hoạch phù hợp với intent.
          4. Tùy chỉnh các bước dựa trên dữ liệu thực tế.

        Args:
            query (str): Câu hỏi / truy vấn gốc của người dùng.
            intent (str): Ý định đã phân loại (calculate, rag, search, …).
            context (str): Dữ liệu ngữ cảnh tổng hợp từ pha song song
                           (kết quả RAG, tính toán, tìm kiếm web, …).

        Returns:
            dict: {
                "planning": str   — Mô tả tổng quan kế hoạch dạng văn bản,
                "steps":    list  — Danh sách các bước thực thi,
                "message":  str   — Nhật ký Agent.
            }
        """

        # ---------- 1. Phân tích thông tin hiện có ---------- #
        has_context = bool(context and context.strip())
        available_info = self._analyze_available_info(context)

        # ---------- 2. Lấy bản kế hoạch mẫu theo intent ---------- #
        raw_steps = self._get_plan_for_intent(intent)

        # ---------- 3. Tùy chỉnh bước dựa trên dữ liệu thực tế ---------- #
        steps = self._customize_steps(raw_steps, intent, has_context, available_info)

        # ---------- 4. Sinh mô tả tổng quan ---------- #
        planning_summary = self._build_summary(query, intent, steps, available_info)

        return {
            "planning": planning_summary,
            "steps": steps,
            "message": self._log(
                f"Đã lập kế hoạch {len(steps)} bước cho intent='{intent}' "
                f"| context_available={has_context}"
            ),
        }

    # ------------------------------------------------------------------ #
    #  Phương thức hỗ trợ nội bộ                                         #
    # ------------------------------------------------------------------ #

    def _analyze_available_info(self, context: str) -> dict[str, bool]:
        """
        Phân tích ngữ cảnh để xác định loại thông tin nào đã có sẵn.

        Dựa trên các từ khóa đặc trưng trong chuỗi context để đoán nhận
        xem dữ liệu từ RAG, tính toán, web search hay phân tích đã có chưa.

        Args:
            context (str): Chuỗi ngữ cảnh tổng hợp.

        Returns:
            dict[str, bool]: Bản đồ {tên_nguồn: đã_có_hay_chưa}.
        """
        ctx_lower = context.lower() if context else ""
        return {
            "kb_data": any(
                kw in ctx_lower for kw in ["knowledge", "tài liệu", "cơ sở", "rag"]
            ),
            "calc_data": any(
                kw in ctx_lower for kw in ["kết quả", "result", "=", "tính"]
            ),
            "web_data": any(
                kw in ctx_lower for kw in ["http", "www", "web", "tìm kiếm", "search"]
            ),
            "analysis_data": any(
                kw in ctx_lower for kw in ["phân tích", "analysis", "analyze", "nhận xét"]
            ),
        }

    def _get_plan_for_intent(self, intent: str) -> list[str]:
        """
        Tra cứu bản kế hoạch mẫu cho loại ý định nhất định.

        Nếu intent không nằm trong bản đồ, sử dụng kế hoạch 'general'.

        Args:
            intent (str): Loại ý định.

        Returns:
            list[str]: Danh sách bước kế hoạch mẫu.
        """
        # Chuẩn hóa intent về dạng lowercase để khớp template
        normalized = intent.strip().lower()
        return list(self._PLAN_TEMPLATES.get(normalized, self._PLAN_TEMPLATES["general"]))

    def _customize_steps(
        self,
        steps: list[str],
        intent: str,
        has_context: bool,
        available_info: dict[str, bool],
    ) -> list[str]:
        """
        Tùy chỉnh các bước kế hoạch dựa trên trạng thái dữ liệu thực tế.

        Ví dụ: Nếu không có dữ liệu web → thêm bước cảnh báo thiếu nguồn.
               Nếu có kết quả tính toán → thêm bước xác minh chéo.

        Args:
            steps (list[str]): Danh sách bước gốc từ template.
            intent (str): Loại ý định.
            has_context (bool): Có ngữ cảnh hay không.
            available_info (dict[str, bool]): Bản đồ thông tin đã có.

        Returns:
            list[str]: Danh sách bước đã tùy chỉnh.
        """
        customized = list(steps)

        if not has_context:
            # Không có context nào cả → thêm bước cảnh báo ở đầu
            customized.insert(0, "⚠ Lưu ý: Chưa có dữ liệu ngữ cảnh — trả lời dựa trên kiến thức chung")

        # Thêm bước xác minh nếu có nhiều nguồn dữ liệu
        sources_count = sum(1 for v in available_info.values() if v)
        if sources_count >= 2:
            customized.append("Đối chiếu chéo giữa các nguồn dữ liệu để đảm bảo nhất quán")

        # Với intent tính toán: nhấn mạnh xác minh nếu đã có kết quả
        if intent.strip().lower() == "calculate" and available_info.get("calc_data"):
            customized.append("Kiểm tra lại kết quả tính toán bằng phương pháp thay thế")

        # Giữ tối đa 5 bước để kế hoạch gọn gàng
        return customized[:5]

    def _build_summary(
        self,
        query: str,
        intent: str,
        steps: list[str],
        available_info: dict[str, bool],
    ) -> str:
        """
        Xây dựng chuỗi mô tả tổng quan kế hoạch dưới dạng Markdown.

        Args:
            query (str): Câu hỏi gốc.
            intent (str): Loại ý định.
            steps (list[str]): Các bước đã tùy chỉnh.
            available_info (dict[str, bool]): Thông tin đã có.

        Returns:
            str: Mô tả kế hoạch dạng Markdown.
        """
        # Liệt kê nguồn dữ liệu đã có
        sources = [name for name, present in available_info.items() if present]
        sources_str = ", ".join(sources) if sources else "không có nguồn nào"

        # Xây dựng mô tả từng bước
        steps_text = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(steps))

        summary = (
            f"**Kế hoạch xử lý cho intent '{intent}'**\n"
            f"Truy vấn: {query}\n"
            f"Nguồn dữ liệu khả dụng: {sources_str}\n"
            f"Số bước thực thi: {len(steps)}\n\n"
            f"Các bước:\n{steps_text}"
        )
        return summary
