"""
core/agents/analyst.py — AnalystAgent: Phân tích và tóm tắt văn bản.

Sử dụng thuật toán TextRank (biến thể đơn giản của PageRank) để thực hiện
extractive summarization — trích xuất các câu quan trọng nhất từ văn bản gốc.

Quy trình hoạt động:
  1. Tách văn bản thành các câu (sentence splitting).
  2. Tokenize từng câu, loại bỏ stopwords tiếng Việt.
  3. Xây dựng đồ thị tương đồng giữa các câu (similarity graph).
  4. Chạy thuật toán TextRank (PageRank iterative) 20 vòng lặp, damping = 0.85.
  5. Chọn top-N câu có điểm cao nhất làm tóm tắt.
  6. Trích xuất từ khóa dựa trên TF (Term Frequency).
  7. Tính toán thống kê cơ bản (số từ, số câu, độ dài câu trung bình).

Không sử dụng thư viện NLP bên ngoài — toàn bộ thuật toán được viết from scratch.
"""

import re
import math
from collections import Counter
from .base import AgentBase


# ---------------------------------------------------------------------------
# Danh sách Vietnamese stopwords (>= 30 từ phổ biến)
# ---------------------------------------------------------------------------
VIETNAMESE_STOPWORDS: set[str] = {
    "là", "và", "của", "có", "được", "trong", "cho", "với", "các", "một",
    "này", "đó", "không", "cũng", "như", "những", "khi", "về", "đã", "sẽ",
    "để", "từ", "theo", "còn", "rất", "tại", "hay", "nếu", "vì", "do",
    "thì", "mà", "nhưng", "hoặc", "lại", "đang", "nên", "bị", "ra", "vào",
    "trên", "dưới", "sau", "trước", "giữa", "đến", "nào", "đây", "ấy", "thế",
    "bởi", "qua", "lên", "xuống", "nhiều", "ít", "mỗi", "tất", "cả", "rồi",
}


class AnalystAgent(AgentBase):
    """
    Agent phân tích văn bản sử dụng thuật toán TextRank.

    Chức năng chính:
      - Tóm tắt văn bản (extractive summarization) bằng TextRank.
      - Trích xuất từ khóa quan trọng bằng TF scoring.
      - Tính thống kê cơ bản về văn bản (word count, sentence count, ...).

    Returns:
        dict với các key:
            - analysis  : str   — bài phân tích dạng văn bản tổng hợp
            - keywords  : list  — danh sách từ khóa chính
            - summary   : str   — tóm tắt ngắn (các câu quan trọng nhất)
            - stats     : dict  — thống kê (word_count, sentence_count, avg_sentence_length)
            - message   : str   — thông báo trạng thái từ Agent
    """

    NAME = "Analyst"

    # -----------------------------------------------------------------------
    # Phương thức chính — được gọi bởi safe_run() trong AgentBase
    # -----------------------------------------------------------------------

    def run(self, topic: str = "", context: str = "", **kwargs) -> dict:
        """
        Điểm vào chính của AnalystAgent.

        Args:
            topic   : Chủ đề hoặc câu hỏi mà người dùng muốn phân tích.
            context : Văn bản dài cần phân tích / tóm tắt. Nếu rỗng,
                      Agent sẽ phân tích chính câu hỏi (topic).

        Returns:
            dict chứa analysis, keywords, summary, stats, message.
        """
        # Xác định văn bản mục tiêu: ưu tiên context, fallback sang topic
        text = context.strip() if context and context.strip() else topic.strip()

        # Trường hợp không có dữ liệu đầu vào
        if not text:
            return {
                "analysis": "Không có nội dung để phân tích.",
                "keywords": [],
                "summary": "",
                "stats": {"word_count": 0, "sentence_count": 0, "avg_sentence_length": 0.0},
                "message": self._log("Không nhận được văn bản đầu vào."),
            }

        # --- Bước 1: Tính thống kê cơ bản ---
        stats = self._compute_stats(text)

        # --- Bước 2: Trích xuất từ khóa ---
        keywords = self._extract_keywords(text, top_n=5)

        # --- Bước 3: Tóm tắt bằng TextRank ---
        sentences = self._sentence_split(text)

        if len(sentences) <= 2:
            # Văn bản quá ngắn — không cần TextRank, trả nguyên bản
            summary = " ".join(sentences)
            analysis = self._build_short_analysis(topic, text, keywords, stats)
        else:
            # Áp dụng TextRank để chọn câu quan trọng nhất
            top_n = min(3, len(sentences))
            ranked_sentences = self._textrank(sentences, top_n=top_n)
            summary = " ".join(ranked_sentences)
            analysis = self._build_full_analysis(topic, ranked_sentences, keywords, stats)

        return {
            "analysis": analysis,
            "keywords": keywords,
            "summary": summary,
            "stats": stats,
            "message": self._log(
                f"Phân tích hoàn tất: {stats['word_count']} từ, "
                f"{stats['sentence_count']} câu, {len(keywords)} từ khóa."
            ),
        }

    # -----------------------------------------------------------------------
    # 1. Tokenize — Tách từ tiếng Việt đơn giản
    # -----------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """
        Tách văn bản thành danh sách token (từ đơn) cho tiếng Việt.

        Quy trình:
          - Chuyển thành chữ thường.
          - Tách theo khoảng trắng và dấu câu.
          - Loại bỏ token quá ngắn (<= 1 ký tự) và stopwords.

        Args:
            text: Chuỗi văn bản cần tokenize.

        Returns:
            Danh sách các token đã lọc.
        """
        # Chuyển lowercase
        text_lower = text.lower()

        # Tách theo khoảng trắng và dấu câu, giữ lại các từ có ký tự chữ cái
        # Pattern: bất cứ chuỗi ký tự liên tiếp nào KHÔNG phải khoảng trắng/dấu câu
        tokens = re.findall(r"[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
                            r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+",
                            text_lower)

        # Lọc bỏ stopwords và token quá ngắn
        filtered = [
            tok for tok in tokens
            if len(tok) > 1 and tok not in VIETNAMESE_STOPWORDS
        ]
        return filtered

    # -----------------------------------------------------------------------
    # 2. Tách câu — Sentence splitting
    # -----------------------------------------------------------------------

    def _sentence_split(self, text: str) -> list[str]:
        """
        Tách văn bản thành danh sách các câu.

        Quy tắc:
          - Tách tại các dấu: .  !  ?  xuống dòng (\\n)
          - Loại bỏ câu rỗng hoặc chỉ có khoảng trắng.
          - Giữ nguyên thứ tự xuất hiện trong văn bản gốc.

        Args:
            text: Văn bản cần tách câu.

        Returns:
            Danh sách các câu đã trim whitespace.
        """
        # Tách theo . ! ? hoặc newline, giữ lại phần nội dung
        raw_sentences = re.split(r'[.!?\n]+', text)

        # Lọc bỏ câu rỗng và trim whitespace
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    # -----------------------------------------------------------------------
    # 3. Tính tương đồng — Similarity giữa hai câu
    # -----------------------------------------------------------------------

    def _similarity(self, sent1_tokens: list[str], sent2_tokens: list[str]) -> float:
        """
        Tính độ tương đồng giữa hai câu đã được tokenize.

        Sử dụng kết hợp Jaccard similarity và Cosine similarity:
          - Jaccard = |A ∩ B| / |A ∪ B|
          - Cosine  = (A · B) / (|A| * |B|)
          - Kết quả cuối = trung bình cộng hai giá trị.

        Điều này cho kết quả ổn định hơn so với dùng riêng một phương pháp.

        Args:
            sent1_tokens: Danh sách token của câu thứ nhất.
            sent2_tokens: Danh sách token của câu thứ hai.

        Returns:
            Giá trị float trong khoảng [0.0, 1.0].
        """
        if not sent1_tokens or not sent2_tokens:
            return 0.0

        set1 = set(sent1_tokens)
        set2 = set(sent2_tokens)

        # --- Jaccard similarity ---
        intersection = set1 & set2
        union = set1 | set2
        jaccard = len(intersection) / len(union) if union else 0.0

        # --- Cosine similarity (dựa trên term frequency vectors) ---
        counter1 = Counter(sent1_tokens)
        counter2 = Counter(sent2_tokens)

        # Tập hợp tất cả các từ xuất hiện trong cả hai câu
        all_words = set1 | set2

        # Tính dot product và norms
        dot_product = sum(counter1.get(w, 0) * counter2.get(w, 0) for w in all_words)
        norm1 = math.sqrt(sum(v ** 2 for v in counter1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in counter2.values()))

        cosine = dot_product / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0

        # Trả về trung bình cộng hai giá trị
        return (jaccard + cosine) / 2.0

    # -----------------------------------------------------------------------
    # 4. TextRank — Thuật toán xếp hạng câu
    # -----------------------------------------------------------------------

    def _textrank(self, sentences: list[str], top_n: int = 3) -> list[str]:
        """
        Áp dụng thuật toán TextRank để xếp hạng các câu quan trọng nhất.

        Thuật toán (simplified PageRank):
          1. Tokenize mỗi câu.
          2. Xây dựng ma trận tương đồng (similarity matrix) giữa tất cả cặp câu.
          3. Lặp 20 vòng với công thức PageRank:
                score[i] = (1 - d) + d * Σ (sim[i][j] / Σ sim[k][j]) * score[j]
             trong đó d = 0.85 (damping factor).
          4. Sắp xếp câu theo điểm giảm dần, chọn top-N.
          5. Trả về các câu đã chọn theo thứ tự xuất hiện trong văn bản gốc.

        Args:
            sentences: Danh sách các câu (đã tách từ _sentence_split).
            top_n   : Số câu quan trọng cần trích xuất.

        Returns:
            Danh sách top-N câu theo thứ tự xuất hiện gốc.
        """
        n = len(sentences)
        if n == 0:
            return []
        if n <= top_n:
            return list(sentences)

        # Bước 1: Tokenize tất cả các câu
        tokenized = [self._tokenize(s) for s in sentences]

        # Bước 2: Xây dựng ma trận tương đồng n × n
        sim_matrix: list[list[float]] = [
            [0.0] * n for _ in range(n)
        ]
        for i in range(n):
            for j in range(i + 1, n):
                score = self._similarity(tokenized[i], tokenized[j])
                sim_matrix[i][j] = score
                sim_matrix[j][i] = score  # Ma trận đối xứng

        # Bước 3: Tính tổng cột (dùng cho chuẩn hóa trong công thức PageRank)
        col_sums = [
            sum(sim_matrix[row][col] for row in range(n))
            for col in range(n)
        ]

        # Bước 4: Chạy PageRank iterative — 20 vòng lặp, damping = 0.85
        damping = 0.85
        iterations = 20
        scores = [1.0] * n  # Khởi tạo điểm bằng nhau cho tất cả câu

        for _ in range(iterations):
            new_scores = [0.0] * n
            for i in range(n):
                rank_sum = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    # Chuẩn hóa: sim[i][j] / tổng cột j
                    if col_sums[j] > 0:
                        rank_sum += (sim_matrix[i][j] / col_sums[j]) * scores[j]
                new_scores[i] = (1 - damping) + damping * rank_sum
            scores = new_scores

        # Bước 5: Chọn top-N câu theo điểm cao nhất
        # Tạo danh sách (index, score) rồi sắp xếp giảm dần theo score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Lấy chỉ số top-N
        top_indices = sorted([idx for idx, _ in indexed_scores[:top_n]])

        # Trả về câu theo thứ tự xuất hiện gốc trong văn bản
        return [sentences[i] for i in top_indices]

    # -----------------------------------------------------------------------
    # 5. Trích xuất từ khóa — TF scoring
    # -----------------------------------------------------------------------

    def _extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """
        Trích xuất từ khóa quan trọng nhất từ văn bản bằng TF (Term Frequency).

        Quy trình:
          - Tokenize văn bản (loại bỏ stopwords trong quá trình tokenize).
          - Đếm tần suất xuất hiện của mỗi từ.
          - Sắp xếp giảm dần theo tần suất.
          - Trả về top-N từ.

        Args:
            text  : Văn bản nguồn.
            top_n : Số từ khóa cần trích xuất.

        Returns:
            Danh sách top-N từ khóa (string).
        """
        tokens = self._tokenize(text)
        if not tokens:
            return []

        # Đếm tần suất (Term Frequency)
        freq = Counter(tokens)

        # Lấy top-N từ phổ biến nhất
        return [word for word, _ in freq.most_common(top_n)]

    # -----------------------------------------------------------------------
    # 6. Thống kê văn bản
    # -----------------------------------------------------------------------

    def _compute_stats(self, text: str) -> dict:
        """
        Tính toán các chỉ số thống kê cơ bản của văn bản.

        Returns:
            dict gồm:
              - word_count           : Tổng số từ.
              - sentence_count       : Tổng số câu.
              - avg_sentence_length  : Độ dài trung bình mỗi câu (tính bằng số từ).
        """
        # Đếm từ: tách theo khoảng trắng (giữ nguyên tất cả, kể cả stopwords)
        words = text.split()
        word_count = len(words)

        # Đếm câu
        sentences = self._sentence_split(text)
        sentence_count = len(sentences)

        # Độ dài trung bình mỗi câu
        if sentence_count > 0:
            avg_sentence_length = round(word_count / sentence_count, 2)
        else:
            avg_sentence_length = 0.0

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
        }

    # -----------------------------------------------------------------------
    # Helpers — Xây dựng bài phân tích
    # -----------------------------------------------------------------------

    def _build_full_analysis(
        self,
        topic: str,
        ranked_sentences: list[str],
        keywords: list[str],
        stats: dict,
    ) -> str:
        """
        Xây dựng bài phân tích đầy đủ cho văn bản dài.

        Args:
            topic            : Chủ đề / câu hỏi gốc.
            ranked_sentences : Các câu được TextRank chọn ra.
            keywords         : Danh sách từ khóa.
            stats            : Thống kê văn bản.

        Returns:
            Chuỗi phân tích dạng văn bản.
        """
        kw_str = ", ".join(keywords) if keywords else "(không có)"
        summary_str = " ".join(ranked_sentences)

        lines = [
            f"📊 **Phân tích văn bản**",
            f"",
            f"**Chủ đề:** {topic}" if topic else "",
            f"**Thống kê:** {stats['word_count']} từ, "
            f"{stats['sentence_count']} câu, "
            f"trung bình {stats['avg_sentence_length']} từ/câu.",
            f"**Từ khóa chính:** {kw_str}",
            f"",
            f"**Tóm tắt nội dung quan trọng:**",
            summary_str,
        ]
        return "\n".join(line for line in lines if line is not None)

    def _build_short_analysis(
        self,
        topic: str,
        text: str,
        keywords: list[str],
        stats: dict,
    ) -> str:
        """
        Xây dựng phân tích ngắn gọn khi văn bản đầu vào quá ngắn.

        Args:
            topic    : Chủ đề / câu hỏi gốc.
            text     : Văn bản gốc (ngắn).
            keywords : Danh sách từ khóa.
            stats    : Thống kê văn bản.

        Returns:
            Chuỗi phân tích ngắn gọn.
        """
        kw_str = ", ".join(keywords) if keywords else "(không có)"

        lines = [
            f"📝 **Phân tích nhanh**",
            f"",
            f"**Nội dung:** {text}",
            f"**Từ khóa:** {kw_str}",
            f"**Thống kê:** {stats['word_count']} từ, "
            f"{stats['sentence_count']} câu.",
        ]
        return "\n".join(lines)
