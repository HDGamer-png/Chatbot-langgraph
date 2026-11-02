"""
core/agents/validator.py — ValidatorAgent: Kiểm tra chéo kết quả và chấm điểm tin cậy.

Agent này chạy SAU PlannerAgent trong pipeline tuần tự. Nhiệm vụ chính:
  1. Source Coverage Check  — Độ phủ nguồn bằng Jaccard similarity.
  2. Math Verification      — Xác minh lại kết quả toán học.
  3. Consistency Check      — Kiểm tra tính nhất quán giữa các nguồn.
  4. Completeness Check     — Đánh giá mức độ đầy đủ thông tin.

Đầu ra là confidence_score (0.0-1.0) và danh sách cảnh báo (warnings).
"""

import re
from .base import AgentBase


class ValidatorAgent(AgentBase):
    """
    Agent kiểm tra chéo và chấm điểm độ tin cậy cho kết quả thu thập được.

    Nhận đầu vào từ tất cả các agent pha song song (RAG, Calculator, WebSearch,
    Analyzer) cùng kế hoạch từ PlannerAgent, sau đó thực hiện 4 bước kiểm tra
    để đánh giá chất lượng dữ liệu trước khi chuyển cho ResponseGenerator.

    Attributes:
        NAME (str): Tên định danh — "Validator".
    """

    NAME = "Validator"

    # ------------------------------------------------------------------ #
    #  Trọng số cho từng bước kiểm tra (tổng = 1.0)                      #
    # ------------------------------------------------------------------ #
    _WEIGHT_SOURCE_COVERAGE = 0.30
    _WEIGHT_MATH_VERIFY = 0.25
    _WEIGHT_CONSISTENCY = 0.20
    _WEIGHT_COMPLETENESS = 0.25

    # ------------------------------------------------------------------ #
    #  Phương thức chính                                                  #
    # ------------------------------------------------------------------ #

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
        **kwargs,
    ) -> dict:
        """
        Thực hiện 4 bước kiểm tra chéo và tính điểm tin cậy.

        Quy trình:
          1. Source Coverage  — So sánh token truy vấn với dữ liệu thu thập.
          2. Math Verification — Kiểm tra lại biểu thức toán (nếu intent=calculate).
          3. Consistency      — Đối chiếu RAG ↔ Web tìm mâu thuẫn.
          4. Completeness     — Đếm số nguồn dữ liệu có nội dung.

        Args:
            query (str): Câu hỏi gốc của người dùng.
            intent (str): Ý định đã phân loại.
            kb_context (str): Kết quả từ RAGAgent (cơ sở tri thức).
            calc_result (str): Kết quả từ CalculatorAgent.
            web_results (str): Kết quả từ WebSearchAgent.
            analysis (str): Kết quả từ AnalyzerAgent.
            planning (str): Kế hoạch từ PlannerAgent.

        Returns:
            dict: {
                "validation":       str   — Báo cáo kiểm tra dạng văn bản,
                "confidence_score": float — Điểm tin cậy 0.0-1.0,
                "warnings":         list  — Danh sách cảnh báo,
                "message":          str   — Nhật ký Agent.
            }
        """

        warnings: list[str] = []
        scores: dict[str, float] = {}

        # ---------- 1. Source Coverage Check ---------- #
        scores["source_coverage"] = self._check_source_coverage(
            query, kb_context, web_results, analysis, warnings
        )

        # ---------- 2. Math Verification ---------- #
        scores["math_verify"] = self._check_math(
            intent, calc_result, query, warnings
        )

        # ---------- 3. Consistency Check ---------- #
        scores["consistency"] = self._check_consistency(
            kb_context, web_results, warnings
        )

        # ---------- 4. Completeness Check ---------- #
        scores["completeness"] = self._check_completeness(
            intent, kb_context, calc_result, datetime_result, web_results, analysis, warnings
        )

        # ---------- Tính điểm tổng hợp ---------- #
        confidence_score = self._compute_confidence(scores, intent)

        # ---------- Cảnh báo mức tổng thể ---------- #
        if confidence_score < 0.5:
            warnings.append(
                f"⚠ Điểm tin cậy thấp ({confidence_score:.2f}). "
                "Câu trả lời có thể chưa đủ chính xác."
            )

        # ---------- Sinh báo cáo ---------- #
        validation_report = self._build_report(scores, confidence_score, warnings)

        return {
            "validation": validation_report,
            "confidence_score": round(confidence_score, 4),
            "warnings": warnings,
            "message": self._log(
                f"Validation hoàn tất | confidence={confidence_score:.4f} | "
                f"warnings={len(warnings)}"
            ),
        }

    # ================================================================== #
    #  BỘ KIỂM TRA (Private helpers)                                     #
    # ================================================================== #

    def _tokenize(self, text: str) -> set[str]:
        """
        Tách chuỗi thành tập hợp các token chữ thường, loại bỏ ký tự đặc biệt.

        Args:
            text (str): Chuỗi đầu vào.

        Returns:
            set[str]: Tập hợp các token đã chuẩn hóa (lowercase, chỉ chữ và số).
        """
        # Chỉ giữ lại chữ cái (bao gồm Unicode) và số, phân tách bằng khoảng trắng
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = {t for t in cleaned.split() if len(t) > 1}
        return tokens

    # ---- 1. Source Coverage ---- #

    def _check_source_coverage(
        self,
        query: str,
        kb_context: str,
        web_results: str,
        analysis: str,
        warnings: list[str],
    ) -> float:
        """
        Tính tỷ lệ phủ nguồn bằng Jaccard similarity giữa truy vấn và dữ liệu.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|.
        Ở đây A = tokens(query), B = tokens(tất cả nguồn).

        Args:
            query (str): Truy vấn gốc.
            kb_context (str): Dữ liệu từ RAG.
            web_results (str): Dữ liệu từ web.
            analysis (str): Dữ liệu từ analyzer.
            warnings (list[str]): Danh sách cảnh báo (có thể thêm vào).

        Returns:
            float: Điểm Jaccard (0.0-1.0).
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        # Gộp tất cả dữ liệu thu thập thành 1 tập token
        combined = " ".join([kb_context, web_results, analysis])
        source_tokens = self._tokenize(combined)

        if not source_tokens:
            warnings.append("Không có dữ liệu nguồn nào được thu thập.")
            return 0.0

        # Jaccard similarity
        intersection = query_tokens & source_tokens
        union = query_tokens | source_tokens
        score = len(intersection) / len(union) if union else 0.0

        if score < 0.1:
            warnings.append(
                f"Độ phủ nguồn rất thấp (Jaccard={score:.2f}). "
                "Dữ liệu thu thập có thể không liên quan đến truy vấn."
            )

        return score

    # ---- 2. Math Verification ---- #

    def _check_math(
        self,
        intent: str,
        calc_result: str,
        query: str,
        warnings: list[str],
    ) -> float:
        """
        Xác minh kết quả toán học bằng cách tách biểu thức và tính lại.

        Chỉ kích hoạt khi intent == "calculate" và có calc_result.
        Sử dụng đánh giá an toàn (chỉ cho phép số và phép toán cơ bản).

        Args:
            intent (str): Loại ý định.
            calc_result (str): Kết quả từ CalculatorAgent.
            query (str): Truy vấn gốc (có thể chứa biểu thức).
            warnings (list[str]): Danh sách cảnh báo.

        Returns:
            float: 1.0 nếu đúng hoặc không áp dụng, 0.0-0.5 nếu có vấn đề.
        """
        normalized_intent = intent.strip().lower()

        # Nếu không phải intent tính toán → bỏ qua, trả 1.0 (neutral)
        if normalized_intent != "calculate":
            return 1.0

        if not calc_result or not calc_result.strip():
            warnings.append("Intent là 'calculate' nhưng không có kết quả tính toán.")
            return 0.0

        # Thử trích xuất biểu thức "expr = result" từ calc_result
        match = re.search(r"(.+?)\s*=\s*(.+)", calc_result)
        if not match:
            # Không tìm thấy dạng "expr = result" → kiểm tra tổng quát
            # Nếu calc_result có chứa số → coi là đã có kết quả
            if re.search(r"\d", calc_result):
                return 0.8  # Có kết quả nhưng không xác minh được
            warnings.append("Không thể phân tích kết quả tính toán để xác minh.")
            return 0.3

        expression_str = match.group(1).strip()
        result_str = match.group(2).strip()

        # Thử tính lại biểu thức một cách an toàn
        try:
            # Chỉ cho phép số, dấu cách, và toán tử cơ bản
            safe_pattern = re.compile(r'^[\d\s\+\-\*/\.\(\)\%\*\^]+$')
            # Chuẩn hóa: thay ** hoặc ^ bằng **
            eval_expr = expression_str.replace("^", "**")

            if not safe_pattern.match(eval_expr):
                # Biểu thức phức tạp (có hàm, biến) → tin tưởng CalculatorAgent
                return 0.7

            # Đánh giá an toàn — chỉ cho phép toán tử, không cho phép builtin
            computed = eval(eval_expr, {"__builtins__": {}}, {})  # noqa: S307

            # So sánh kết quả
            try:
                expected = float(result_str.replace(",", "").strip())
                if abs(computed - expected) < 1e-6:
                    return 1.0  # Kết quả khớp hoàn toàn
                else:
                    warnings.append(
                        f"Kết quả tính toán không khớp: "
                        f"tính lại={computed}, báo cáo={expected}."
                    )
                    return 0.2
            except (ValueError, TypeError):
                # Không parse được kết quả thành số
                return 0.5
        except Exception:
            # Lỗi khi tính lại → không xác minh được
            return 0.6

    # ---- 3. Consistency Check ---- #

    def _check_consistency(
        self,
        kb_context: str,
        web_results: str,
        warnings: list[str],
    ) -> float:
        """
        Kiểm tra tính nhất quán giữa kết quả RAG và Web search.

        Sử dụng token-level overlap: nếu hai nguồn đều có nội dung nhưng
        overlap quá thấp → cảnh báo mâu thuẫn tiềm ẩn.

        Args:
            kb_context (str): Dữ liệu từ RAG.
            web_results (str): Dữ liệu từ web.
            warnings (list[str]): Danh sách cảnh báo.

        Returns:
            float: Điểm nhất quán (0.0-1.0).
        """
        kb_tokens = self._tokenize(kb_context)
        web_tokens = self._tokenize(web_results)

        # Nếu một hoặc cả hai nguồn trống → không đủ dữ liệu để đối chiếu
        if not kb_tokens or not web_tokens:
            # Không có đủ 2 nguồn để so sánh → neutral
            return 0.8

        # Tính overlap giữa hai nguồn
        intersection = kb_tokens & web_tokens
        smaller_set_size = min(len(kb_tokens), len(web_tokens))
        overlap_ratio = len(intersection) / smaller_set_size if smaller_set_size > 0 else 0.0

        if overlap_ratio < 0.05:
            warnings.append(
                "Dữ liệu từ cơ sở tri thức và web search có rất ít trùng lặp. "
                "Có thể có thông tin mâu thuẫn."
            )
            return 0.3

        if overlap_ratio < 0.15:
            warnings.append(
                "Mức độ nhất quán giữa các nguồn ở mức trung bình."
            )
            return 0.6

        return min(1.0, 0.5 + overlap_ratio)

    # ---- 4. Completeness Check ---- #

    def _check_completeness(
        self,
        intent: str,
        kb_context: str,
        calc_result: str,
        datetime_result: str,
        web_results: str,
        analysis: str,
        warnings: list[str],
    ) -> float:
        normalized_intent = intent.strip().lower()

        # Xác định nguồn nào đã có nội dung thực sự (không rỗng)
        has_kb = bool(kb_context and kb_context.strip())
        has_calc = bool(calc_result and calc_result.strip())
        has_dt = bool(datetime_result and datetime_result.strip())
        has_web = bool(web_results and web_results.strip())
        has_analysis = bool(analysis and analysis.strip())

        # Bản đồ yêu cầu: intent → danh sách (tên_nguồn, đã_có)
        requirements: dict[str, list[tuple[str, bool]]] = {
            "calculate": [("calc_result", has_calc)],
            "datetime": [("datetime_result", has_dt)],
            "rag": [("kb_context", has_kb)],
            "search": [("web_results", has_web)],
            "analyze": [("analysis", has_analysis), ("kb_context", has_kb)],
            "plan": [("analysis", has_analysis)],
            "greet": [],  # Không cần nguồn bổ sung
            "general": [],  # Linh hoạt
        }

        required_sources = requirements.get(normalized_intent, [])

        if not required_sources:
            # Intent không yêu cầu nguồn cụ thể (greet, general)
            # Cho điểm cao nếu ít nhất có 1 nguồn bất kỳ
            any_source = any([has_kb, has_calc, has_web, has_analysis])
            return 1.0 if normalized_intent == "greet" else (0.7 if any_source else 0.5)

        # Đếm số nguồn cần thiết đã có
        fulfilled = sum(1 for _, present in required_sources if present)
        total = len(required_sources)
        score = fulfilled / total if total > 0 else 0.5

        # Cảnh báo nếu thiếu nguồn quan trọng
        missing = [name for name, present in required_sources if not present]
        if missing:
            warnings.append(
                f"Thiếu dữ liệu cho intent '{intent}': {', '.join(missing)}."
            )

        return score

    # ================================================================== #
    #  Tính điểm tổng hợp và sinh báo cáo                                #
    # ================================================================== #

    def _compute_confidence(self, scores: dict[str, float], intent: str) -> float:
        """
        Tính điểm tin cậy tổng hợp dưới dạng trung bình có trọng số.

        Trọng số có thể được điều chỉnh tùy loại intent:
          - Intent 'calculate': tăng trọng số cho math_verify.
          - Intent 'rag'/'search': tăng trọng số cho source_coverage.

        Args:
            scores (dict[str, float]): Điểm của 4 bước kiểm tra.
            intent (str): Loại ý định (để điều chỉnh trọng số).

        Returns:
            float: Điểm tin cậy tổng hợp (0.0-1.0).
        """
        normalized_intent = intent.strip().lower()

        # Trọng số mặc định
        w_source = self._WEIGHT_SOURCE_COVERAGE
        w_math = self._WEIGHT_MATH_VERIFY
        w_consist = self._WEIGHT_CONSISTENCY
        w_complete = self._WEIGHT_COMPLETENESS

        # Điều chỉnh trọng số theo intent
        if normalized_intent == "calculate":
            w_math = 0.40
            w_source = 0.15
            w_consist = 0.15
            w_complete = 0.30
        elif normalized_intent in ("rag", "search"):
            w_source = 0.35
            w_math = 0.10
            w_consist = 0.25
            w_complete = 0.30
        elif normalized_intent in ("greet", "datetime"):
            # Chào hỏi và thời gian là tác vụ tất định nội bộ, luôn đáng tin cậy khi có kết quả
            return max(0.95, scores.get("completeness", 1.0))

        confidence = (
            w_source * scores.get("source_coverage", 0.0)
            + w_math * scores.get("math_verify", 1.0)
            + w_consist * scores.get("consistency", 0.8)
            + w_complete * scores.get("completeness", 0.0)
        )

        # Kẹp giá trị trong khoảng [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _build_report(
        self,
        scores: dict[str, float],
        confidence: float,
        warnings: list[str],
    ) -> str:
        """
        Sinh báo cáo kiểm tra dạng Markdown.

        Args:
            scores (dict[str, float]): Điểm từng bước.
            confidence (float): Điểm tổng hợp.
            warnings (list[str]): Danh sách cảnh báo.

        Returns:
            str: Báo cáo Markdown.
        """
        # Header với emoji trạng thái
        if confidence >= 0.8:
            status = "✅ Tin cậy cao"
        elif confidence >= 0.5:
            status = "⚠️ Tin cậy trung bình"
        else:
            status = "❌ Tin cậy thấp"

        lines = [
            f"**Báo cáo kiểm tra — {status}**",
            f"Điểm tin cậy tổng hợp: **{confidence:.2f}** / 1.00",
            "",
            "| Bước kiểm tra | Điểm |",
            "|---|---|",
        ]

        # Tên hiển thị tiếng Việt cho từng bước
        labels = {
            "source_coverage": "Độ phủ nguồn (Jaccard)",
            "math_verify": "Xác minh toán học",
            "consistency": "Tính nhất quán",
            "completeness": "Mức độ đầy đủ",
        }
        for key, label in labels.items():
            val = scores.get(key, 0.0)
            lines.append(f"| {label} | {val:.2f} |")

        if warnings:
            lines.append("")
            lines.append("**Cảnh báo:**")
            for w in warnings:
                lines.append(f"- {w}")

        return "\n".join(lines)
