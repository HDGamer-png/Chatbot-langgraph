from .base import AgentBase

class PlannerAgent(AgentBase):
    NAME = "Planner"

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

    # Hoạch định các bước xử lý kỹ thuật dựa trên ý định truy vấn và ngữ cảnh dữ liệu hiện có.
    def run(self, query: str = "", intent: str = "general", context: str = "", **kwargs) -> dict:
        has_context = bool(context and context.strip())
        available_info = self._analyze_available_info(context)
        raw_steps = self._get_plan_for_intent(intent)
        steps = self._customize_steps(raw_steps, intent, has_context, available_info)
        planning_summary = self._build_summary(query, intent, steps, available_info)
        return {
            "planning": planning_summary,
            "steps": steps,
            "message": self._log(f"Đã lập kế hoạch {len(steps)} bước cho intent='{intent}' | context_available={has_context}"),
        }

    # Phân tích ngữ cảnh đầu vào để xác định các nguồn dữ liệu đã sẵn sàng.
    def _analyze_available_info(self, context: str) -> dict[str, bool]:
        ctx_lower = context.lower() if context else ""
        return {
            "kb_data": any(kw in ctx_lower for kw in ["knowledge", "tài liệu", "cơ sở", "rag"]),
            "calc_data": any(kw in ctx_lower for kw in ["kết quả", "result", "=", "tính"]),
            "web_data": any(kw in ctx_lower for kw in ["http", "www", "web", "tìm kiếm", "search"]),
            "analysis_data": any(kw in ctx_lower for kw in ["phân tích", "analysis", "analyze", "nhận xét"]),
        }

    # Lấy khuôn mẫu kế hoạch tương ứng với ý định của người dùng.
    def _get_plan_for_intent(self, intent: str) -> list[str]:
        return list(self._PLAN_TEMPLATES.get(intent.strip().lower(), self._PLAN_TEMPLATES["general"]))

    # Điều chỉnh linh hoạt các bước thực thi dựa trên tính sẵn sàng của dữ liệu thu thập.
    def _customize_steps(self, steps: list[str], intent: str, has_context: bool, available_info: dict[str, bool]) -> list[str]:
        customized = list(steps)
        if not has_context:
            customized.insert(0, "⚠ Lưu ý: Chưa có dữ liệu ngữ cảnh — trả lời dựa trên kiến thức chung")
        if sum(1 for v in available_info.values() if v) >= 2:
            customized.append("Đối chiếu chéo giữa các nguồn dữ liệu để đảm bảo nhất quán")
        if intent.strip().lower() == "calculate" and available_info.get("calc_data"):
            customized.append("Kiểm tra lại kết quả tính toán bằng phương pháp thay thế")
        return customized[:5]

    # Tổng hợp mô tả kế hoạch xử lý thành văn bản Markdown có cấu trúc.
    def _build_summary(self, query: str, intent: str, steps: list[str], available_info: dict[str, bool]) -> str:
        sources = [name for name, present in available_info.items() if present]
        sources_str = ", ".join(sources) if sources else "không có nguồn nào"
        steps_text = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(steps))
        return (
            f"**Kế hoạch xử lý cho intent '{intent}'**\n"
            f"Truy vấn: {query}\n"
            f"Nguồn dữ liệu khả dụng: {sources_str}\n"
            f"Số bước thực thi: {len(steps)}\n\n"
            f"Các bước:\n{steps_text}"
        )
