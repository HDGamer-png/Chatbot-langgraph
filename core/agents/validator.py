import re
from .base import AgentBase

class ValidatorAgent(AgentBase):
    NAME = "Validator"
    _WEIGHT_SOURCE_COVERAGE = 0.30
    _WEIGHT_MATH_VERIFY = 0.25
    _WEIGHT_CONSISTENCY = 0.20
    _WEIGHT_COMPLETENESS = 0.25

    # Thẩm định chéo dữ liệu đa nguồn, kiểm tra tính toán và tính toán điểm tin cậy tổng thể.
    def run(self, query: str = "", intent: str = "general", kb_context: str = "", calc_result: str = "", datetime_result: str = "", web_results: str = "", analysis: str = "", planning: str = "", **kwargs) -> dict:
        warnings: list[str] = []
        scores: dict[str, float] = {}
        scores["source_coverage"] = self._check_source_coverage(query, kb_context, web_results, analysis, warnings)
        scores["math_verify"] = self._check_math(intent, calc_result, query, warnings)
        scores["consistency"] = self._check_consistency(kb_context, web_results, warnings)
        scores["completeness"] = self._check_completeness(intent, kb_context, calc_result, datetime_result, web_results, analysis, warnings)
        confidence_score = self._compute_confidence(scores, intent)
        if confidence_score < 0.5:
            warnings.append(f"⚠ Điểm tin cậy thấp ({confidence_score:.2f}). Câu trả lời có thể chưa đủ chính xác.")
        validation_report = self._build_report(scores, confidence_score, warnings)
        return {
            "validation": validation_report,
            "confidence_score": round(confidence_score, 4),
            "warnings": warnings,
            "message": self._log(f"Validation hoàn tất | confidence={confidence_score:.4f} | warnings={len(warnings)}"),
        }

    # Tách chuỗi thành tập hợp các từ khóa đã chuẩn hóa loại bỏ dấu câu.
    def _tokenize(self, text: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return {t for t in cleaned.split() if len(t) > 1}

    # Tính toán tỷ lệ bao phủ từ khóa giữa truy vấn và tập tài liệu thu thập theo hệ số Jaccard.
    def _check_source_coverage(self, query: str, kb_context: str, web_results: str, analysis: str, warnings: list[str]) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0
        combined = " ".join([kb_context, web_results, analysis])
        source_tokens = self._tokenize(combined)
        if not source_tokens:
            warnings.append("Không có dữ liệu nguồn nào được thu thập.")
            return 0.0
        intersection = query_tokens & source_tokens
        union = query_tokens | source_tokens
        score = len(intersection) / len(union) if union else 0.0
        if score < 0.1:
            warnings.append(f"Độ phủ nguồn rất thấp (Jaccard={score:.2f}). Dữ liệu thu thập có thể không liên quan đến truy vấn.")
        return score

    # Thẩm định lại kết quả tính toán số học để phòng ngừa sai lệch biểu thức.
    def _check_math(self, intent: str, calc_result: str, query: str, warnings: list[str]) -> float:
        if intent.strip().lower() != "calculate":
            return 1.0
        if not calc_result or not calc_result.strip():
            warnings.append("Intent là 'calculate' nhưng không có kết quả tính toán.")
            return 0.0
        match = re.search(r"(.+?)\s*=\s*(.+)", calc_result)
        if not match:
            return 0.8 if re.search(r"\d", calc_result) else 0.3
        expression_str = match.group(1).strip()
        result_str = match.group(2).strip()
        try:
            safe_pattern = re.compile(r'^[\d\s\+\-\*/\.\(\)\%\*\^]+$')
            eval_expr = expression_str.replace("^", "**")
            if not safe_pattern.match(eval_expr):
                return 0.7
            computed = eval(eval_expr, {"__builtins__": {}}, {})
            try:
                expected = float(result_str.replace(",", "").strip())
                if abs(computed - expected) < 1e-6:
                    return 1.0
                warnings.append(f"Kết quả tính toán không khớp: tính lại={computed}, báo cáo={expected}.")
                return 0.2
            except (ValueError, TypeError):
                return 0.5
        except Exception:
            return 0.6

    # So sánh mức độ tương đồng giữa nguồn tri thức nội bộ và kết quả tìm kiếm web.
    def _check_consistency(self, kb_context: str, web_results: str, warnings: list[str]) -> float:
        kb_tokens = self._tokenize(kb_context)
        web_tokens = self._tokenize(web_results)
        if not kb_tokens or not web_tokens:
            return 0.8
        intersection = kb_tokens & web_tokens
        smaller_set_size = min(len(kb_tokens), len(web_tokens))
        overlap_ratio = len(intersection) / smaller_set_size if smaller_set_size > 0 else 0.0
        if overlap_ratio < 0.05:
            warnings.append("Dữ liệu từ cơ sở tri thức và web search có rất ít trùng lặp. Có thể có thông tin mâu thuẫn.")
            return 0.3
        if overlap_ratio < 0.15:
            warnings.append("Mức độ nhất quán giữa các nguồn ở mức trung bình.")
            return 0.6
        return min(1.0, 0.5 + overlap_ratio)

    # Đánh giá mức độ hoàn thiện các trường dữ liệu bắt buộc theo từng nhóm ý định.
    def _check_completeness(self, intent: str, kb_context: str, calc_result: str, datetime_result: str, web_results: str, analysis: str, warnings: list[str]) -> float:
        norm_intent = intent.strip().lower()
        has_kb = bool(kb_context and kb_context.strip())
        has_calc = bool(calc_result and calc_result.strip())
        has_dt = bool(datetime_result and datetime_result.strip())
        has_web = bool(web_results and web_results.strip())
        has_analysis = bool(analysis and analysis.strip())
        requirements: dict[str, list[tuple[str, bool]]] = {
            "calculate": [("calc_result", has_calc)],
            "datetime": [("datetime_result", has_dt)],
            "rag": [("kb_context", has_kb)],
            "search": [("web_results", has_web)],
            "analyze": [("analysis", has_analysis), ("kb_context", has_kb)],
            "plan": [("analysis", has_analysis)],
            "greet": [],
            "general": [],
        }
        required_sources = requirements.get(norm_intent, [])
        if not required_sources:
            return 1.0 if norm_intent == "greet" else (0.7 if any([has_kb, has_calc, has_web, has_analysis]) else 0.5)
        fulfilled = sum(1 for _, present in required_sources if present)
        score = fulfilled / len(required_sources) if required_sources else 0.5
        missing = [name for name, present in required_sources if not present]
        if missing:
            warnings.append(f"Thiếu dữ liệu cho intent '{intent}': {', '.join(missing)}.")
        return score

    # Tính điểm tin cậy tổng hợp dựa trên trọng số điều chỉnh theo từng loại ý định.
    def _compute_confidence(self, scores: dict[str, float], intent: str) -> float:
        norm_intent = intent.strip().lower()
        w_source, w_math, w_consist, w_complete = self._WEIGHT_SOURCE_COVERAGE, self._WEIGHT_MATH_VERIFY, self._WEIGHT_CONSISTENCY, self._WEIGHT_COMPLETENESS
        if norm_intent == "calculate":
            w_math, w_source, w_consist, w_complete = 0.40, 0.15, 0.15, 0.30
        elif norm_intent in ("rag", "search"):
            w_source, w_math, w_consist, w_complete = 0.35, 0.10, 0.25, 0.30
        elif norm_intent in ("greet", "datetime"):
            return max(0.95, scores.get("completeness", 1.0))
        confidence = (
            w_source * scores.get("source_coverage", 0.0)
            + w_math * scores.get("math_verify", 1.0)
            + w_consist * scores.get("consistency", 0.8)
            + w_complete * scores.get("completeness", 0.0)
        )
        return max(0.0, min(1.0, confidence))

    # Xây dựng bảng báo cáo kết quả kiểm định chi tiết theo định dạng Markdown.
    def _build_report(self, scores: dict[str, float], confidence: float, warnings: list[str]) -> str:
        status = "✅ Tin cậy cao" if confidence >= 0.8 else ("⚠️ Tin cậy trung bình" if confidence >= 0.5 else "❌ Tin cậy thấp")
        lines = [
            f"**Báo cáo kiểm tra — {status}**",
            f"Điểm tin cậy tổng hợp: **{confidence:.2f}** / 1.00\n",
            "| Bước kiểm tra | Điểm |",
            "|---|---|",
        ]
        labels = {
            "source_coverage": "Độ phủ nguồn (Jaccard)",
            "math_verify": "Xác minh toán học",
            "consistency": "Tính nhất quán",
            "completeness": "Mức độ đầy đủ",
        }
        for key, label in labels.items():
            lines.append(f"| {label} | {scores.get(key, 0.0):.2f} |")
        if warnings:
            lines.append("\n**Cảnh báo:**")
            for w in warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)
