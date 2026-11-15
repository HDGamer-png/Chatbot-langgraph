import re
import math
from collections import Counter
from .base import AgentBase

VIETNAMESE_STOPWORDS: set[str] = {
    "là", "và", "của", "có", "được", "trong", "cho", "với", "các", "một",
    "này", "đó", "không", "cũng", "như", "những", "khi", "về", "đã", "sẽ",
    "để", "từ", "theo", "còn", "rất", "tại", "hay", "nếu", "vì", "do",
    "thì", "mà", "nhưng", "hoặc", "lại", "đang", "nên", "bị", "ra", "vào",
    "trên", "dưới", "sau", "trước", "giữa", "đến", "nào", "đây", "ấy", "thế",
    "bởi", "qua", "lên", "xuống", "nhiều", "ít", "mỗi", "tất", "cả", "rồi",
}

class AnalystAgent(AgentBase):
    NAME = "Analyst"

    # Phân tích nội dung văn bản bằng thuật toán TextRank, trích xuất từ khóa và thống kê số liệu.
    def run(self, topic: str = "", context: str = "", **kwargs) -> dict:
        text = context.strip() if context and context.strip() else topic.strip()
        if not text:
            return {
                "analysis": "Không có nội dung để phân tích.",
                "keywords": [],
                "summary": "",
                "stats": {"word_count": 0, "sentence_count": 0, "avg_sentence_length": 0.0},
                "message": self._log("Không nhận được văn bản đầu vào."),
            }
        stats = self._compute_stats(text)
        keywords = self._extract_keywords(text, top_n=5)
        sentences = self._sentence_split(text)
        if len(sentences) <= 2:
            summary = " ".join(sentences)
            analysis = self._build_short_analysis(topic, text, keywords, stats)
        else:
            top_n = min(3, len(sentences))
            ranked_sentences = self._textrank(sentences, top_n=top_n)
            summary = " ".join(ranked_sentences)
            analysis = self._build_full_analysis(topic, ranked_sentences, keywords, stats)
        return {
            "analysis": analysis,
            "keywords": keywords,
            "summary": summary,
            "stats": stats,
            "message": self._log(f"Phân tích hoàn tất: {stats['word_count']} từ, {stats['sentence_count']} câu, {len(keywords)} từ khóa."),
        }

    # Tách từ tiếng Việt đơn giản và lọc bỏ danh sách hư từ dừng.
    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+", text.lower())
        return [tok for tok in tokens if len(tok) > 1 and tok not in VIETNAMESE_STOPWORDS]

    # Phân tách đoạn văn bản thành danh sách các câu hoàn chỉnh dựa trên dấu câu.
    def _sentence_split(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]

    # Tính toán độ tương đồng giữa hai câu dựa trên kết hợp Jaccard và Cosine similarity.
    def _similarity(self, sent1_tokens: list[str], sent2_tokens: list[str]) -> float:
        if not sent1_tokens or not sent2_tokens:
            return 0.0
        s1, s2 = set(sent1_tokens), set(sent2_tokens)
        jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0
        c1, c2 = Counter(sent1_tokens), Counter(sent2_tokens)
        all_words = s1 | s2
        dot = sum(c1.get(w, 0) * c2.get(w, 0) for w in all_words)
        norm1 = math.sqrt(sum(v ** 2 for v in c1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in c2.values()))
        cosine = dot / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0
        return (jaccard + cosine) / 2.0

    # Xếp hạng và chọn lọc các câu mang giá trị thông tin cao nhất theo thuật toán TextRank.
    def _textrank(self, sentences: list[str], top_n: int = 3) -> list[str]:
        n = len(sentences)
        if n <= top_n:
            return list(sentences)
        tokenized = [self._tokenize(s) for s in sentences]
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                score = self._similarity(tokenized[i], tokenized[j])
                sim_matrix[i][j] = score
                sim_matrix[j][i] = score
        col_sums = [sum(sim_matrix[row][col] for row in range(n)) for col in range(n)]
        damping, scores = 0.85, [1.0] * n
        for _ in range(20):
            new_scores = [0.0] * n
            for i in range(n):
                rank_sum = sum((sim_matrix[i][j] / col_sums[j]) * scores[j] for j in range(n) if i != j and col_sums[j] > 0)
                new_scores[i] = (1 - damping) + damping * rank_sum
            scores = new_scores
        indexed_scores = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, _ in indexed_scores[:top_n]])
        return [sentences[i] for i in top_indices]

    # Trích xuất top các từ khóa có tần suất xuất hiện cao nhất trong văn bản.
    def _extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        tokens = self._tokenize(text)
        return [w for w, _ in Counter(tokens).most_common(top_n)] if tokens else []

    # Tính toán các chỉ số thống kê cơ bản gồm số từ, số câu và độ dài câu trung bình.
    def _compute_stats(self, text: str) -> dict:
        words = text.split()
        sentences = self._sentence_split(text)
        sc = len(sentences)
        return {
            "word_count": len(words),
            "sentence_count": sc,
            "avg_sentence_length": round(len(words) / sc, 2) if sc > 0 else 0.0,
        }

    # Định dạng kết quả phân tích chi tiết cho văn bản có dung lượng dài.
    def _build_full_analysis(self, topic: str, ranked_sentences: list[str], keywords: list[str], stats: dict) -> str:
        kw_str = ", ".join(keywords) if keywords else "(không có)"
        lines = [
            "📊 **Phân tích văn bản**",
            f"**Chủ đề:** {topic}" if topic else "",
            f"**Thống kê:** {stats['word_count']} từ, {stats['sentence_count']} câu, trung bình {stats['avg_sentence_length']} từ/câu.",
            f"**Từ khóa chính:** {kw_str}",
            "**Tóm tắt nội dung quan trọng:**",
            " ".join(ranked_sentences),
        ]
        return "\n".join(l for l in lines if l)

    # Định dạng kết quả phân tích tóm lược nhanh cho văn bản ngắn.
    def _build_short_analysis(self, topic: str, text: str, keywords: list[str], stats: dict) -> str:
        kw_str = ", ".join(keywords) if keywords else "(không có)"
        lines = [
            "📝 **Phân tích nhanh**",
            f"**Nội dung:** {text}",
            f"**Từ khóa:** {kw_str}",
            f"**Thống kê:** {stats['word_count']} từ, {stats['sentence_count']} câu.",
        ]
        return "\n".join(lines)
