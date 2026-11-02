"""
core/agents/intent.py — IntentClassifierAgent

Phân loại ý định (intent) của câu hỏi người dùng bằng Scikit-Learn.
Sử dụng pipeline TF-IDF + MultinomialNB để phân loại 8 loại ý định:
  1. calculate  — Tính toán, phép toán, phương trình
  2. datetime   — Hỏi ngày giờ, múi giờ, đếm ngược
  3. rag        — Hỏi kiến thức chuyên sâu (tìm trong Knowledge Base)
  4. search     — Tìm kiếm thông tin trên web
  5. analyze    — Phân tích dữ liệu, so sánh, đánh giá
  6. plan       — Lập kế hoạch, lộ trình, roadmap
  7. greet      — Chào hỏi, xã giao
  8. general    — Câu hỏi chung, không thuộc nhóm nào cụ thể

Agent này chạy song song trong ThreadPool cùng các Parallel Worker khác.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .base import AgentBase


class IntentClassifierAgent(AgentBase):
    """
    Agent phân loại ý định câu hỏi sử dụng TF-IDF + MultinomialNB.

    Khi được gọi lần đầu, tự động huấn luyện mô hình từ tập dữ liệu
    tĩnh được định nghĩa sẵn. Mô hình được cache để các lần gọi sau
    không cần huấn luyện lại.
    """

    NAME = "IntentClassifier"

    # ------------------------------------------------------------------ #
    #  Tập dữ liệu huấn luyện tĩnh: ánh xạ intent -> danh sách câu mẫu
    # ------------------------------------------------------------------ #
    _TRAINING_DATA: dict[str, list[str]] = {
        # ---- 1. calculate — tính toán -------------------------------- #
        "calculate": [
            "tính 25 cộng 30",
            "25 nhân 4 bằng bao nhiêu",
            "giải phương trình 2x + 5 = 15",
            "căn bậc hai của 144 là bao nhiêu",
            "tính diện tích hình tròn bán kính 5",
            "5 giai thừa bằng mấy",
            "tính tổng từ 1 đến 100",
            "phép chia 100 cho 7 dư bao nhiêu",
            "tính sin của 30 độ",
            "giải bất phương trình x^2 - 4 > 0",
            "10 lũy thừa 3 bằng bao nhiêu",
            "logarit cơ số 2 của 8",
            "tính đạo hàm của x^2 + 3x",
            "bao nhiêu phần trăm của 200 là 50",
            "quy đổi 100 usd ra vnd",
        ],

        # ---- 2. datetime — ngày giờ --------------------------------- #
        "datetime": [
            "hôm nay là ngày bao nhiêu",
            "bây giờ mấy giờ rồi",
            "ngày mai là thứ mấy",
            "còn bao nhiêu ngày nữa đến tết",
            "giờ ở Mỹ bây giờ là mấy giờ",
            "múi giờ của Tokyo là gì",
            "tuần sau là ngày mấy",
            "hôm qua là ngày bao nhiêu",
            "năm nay là năm con gì",
            "tháng này có bao nhiêu ngày",
            "giờ London bây giờ là mấy giờ",
            "thời gian hiện tại ở Hàn Quốc",
            "khi nào là quốc khánh Việt Nam",
            "đếm ngược đến giáng sinh",
            "tuần này bắt đầu từ ngày nào",
        ],

        # ---- 3. rag — kiến thức chuyên sâu -------------------------- #
        "rag": [
            "giải thích LangGraph là gì",
            "kiến trúc của hệ thống RAG hoạt động như thế nào",
            "so sánh CrewAI và AutoGen",
            "mô hình ReAct agent là gì",
            "cách tổ chức bộ nhớ cho chatbot",
            "tài liệu về multi-agent system",
            "hệ thống RAG có những thành phần nào",
            "vector database dùng để làm gì",
            "embedding là gì và hoạt động ra sao",
            "retrieval augmented generation nghĩa là gì",
            "cách xây dựng knowledge base cho chatbot",
            "LangChain khác LangGraph ở điểm nào",
            "agent memory có mấy loại",
            "cơ chế tool calling trong LLM hoạt động thế nào",
        ],

        # ---- 4. search — tìm kiếm web ------------------------------- #
        "search": [
            "python là gì",
            "machine learning là gì",
            "deep learning là gì",
            "trí tuệ nhân tạo là gì",
            "tìm kiếm thông tin về Python 3.12",
            "search Google về machine learning",
            "tìm trên mạng cách cài đặt Docker",
            "tra cứu tin tức mới nhất về AI",
            "tìm hiểu về framework FastAPI",
            "tìm bài viết về deep learning",
            "search cách sử dụng Git nâng cao",
            "tìm kiếm tutorial về React Native",
            "tra cứu thông tin về công ty OpenAI",
            "tìm trên web về thuật toán sắp xếp",
            "tìm xem thời tiết hôm nay thế nào",
            "search tin tức công nghệ hôm nay",
            "tìm thông tin về ngôn ngữ Rust",
        ],

        # ---- 5. analyze — phân tích dữ liệu ------------------------- #
        "analyze": [
            "phân tích ưu nhược điểm của Python",
            "so sánh hiệu năng giữa Java và C++",
            "đánh giá framework Django và Flask",
            "phân tích xu hướng công nghệ năm 2026",
            "đánh giá ưu nhược điểm của microservice",
            "so sánh SQL và NoSQL nên dùng cái nào",
            "phân tích điểm mạnh yếu của React và Vue",
            "review kiến trúc serverless",
            "đánh giá ngôn ngữ Go cho backend",
            "so sánh giữa REST API và GraphQL",
            "phân tích chi phí cloud AWS và Azure",
            "đánh giá mô hình Agile và Waterfall",
            "so sánh Docker và Kubernetes",
        ],

        # ---- 6. plan — lập kế hoạch --------------------------------- #
        "plan": [
            "lập kế hoạch học Python trong 3 tháng",
            "tạo roadmap trở thành fullstack developer",
            "lên lịch học máy learning cho người mới",
            "lập kế hoạch dự án web trong 2 tuần",
            "tạo timeline phát triển ứng dụng mobile",
            "lên kế hoạch ôn thi chứng chỉ AWS",
            "lập lộ trình học data science từ đầu",
            "planning sprint cho team 5 người",
            "lên kế hoạch triển khai hệ thống CI/CD",
            "tạo kế hoạch backup dữ liệu hàng tuần",
            "lập lộ trình chuyển đổi từ monolith sang microservice",
            "kế hoạch học tiếng Anh IT trong 6 tháng",
            "tạo roadmap học AI và machine learning",
        ],

        # ---- 7. greet — chào hỏi ------------------------------------ #
        "greet": [
            "xin chào",
            "chào bạn",
            "hello",
            "hi",
            "hey bạn ơi",
            "chào buổi sáng",
            "chào buổi tối",
            "bạn có khỏe không",
            "rất vui được gặp bạn",
            "bạn tên là gì",
            "bạn là ai vậy",
            "cảm ơn bạn nhiều",
            "tạm biệt nhé",
            "hẹn gặp lại",
            "bạn có thể giúp gì cho tôi",
        ],

        # ---- 8. general — chung chung ------------------------------- #
        "general": [
            "cho tôi hỏi một chút",
            "tôi muốn biết thêm",
            "bạn có thể giải thích rõ hơn không",
            "tôi không hiểu lắm",
            "nói thêm về vấn đề này đi",
            "có ý kiến gì không",
            "thử lại một lần nữa",
            "tiếp tục đi bạn",
            "tôi cần trợ giúp",
            "mình muốn thảo luận về một vấn đề",
            "bạn nghĩ sao về điều này",
            "có cách nào khác không",
            "tóm tắt lại cho tôi",
        ],
    }

    def __init__(self, called_by: str = "System"):
        """Khởi tạo agent, model sẽ được huấn luyện lazy khi gọi run() lần đầu."""
        super().__init__(called_by)
        # Pipeline sẽ được khởi tạo khi cần (lazy training)
        self._pipeline: Pipeline | None = None

    # ------------------------------------------------------------------ #
    #  Private methods
    # ------------------------------------------------------------------ #

    def _build_training_data(self) -> tuple[list[str], list[str]]:
        """
        Xây dựng cặp (texts, labels) từ tập dữ liệu tĩnh.

        Returns:
            tuple: (danh_sách_câu, danh_sách_nhãn) cùng kích thước
        """
        texts: list[str] = []
        labels: list[str] = []
        for intent_label, sentences in self._TRAINING_DATA.items():
            for sentence in sentences:
                texts.append(sentence)
                labels.append(intent_label)
        return texts, labels

    def _ensure_model(self) -> None:
        """
        Đảm bảo mô hình đã được huấn luyện.
        Nếu chưa, huấn luyện pipeline TF-IDF + MultinomialNB và cache lại.
        """
        if self._pipeline is not None:
            return  # Đã huấn luyện rồi, bỏ qua

        texts, labels = self._build_training_data()

        # Xây dựng pipeline: TF-IDF vectorizer -> Naive Bayes classifier
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",   # Dùng character n-gram cho tiếng Việt
                ngram_range=(2, 4),    # Bi-gram đến 4-gram ký tự
                max_features=8000,     # Giới hạn số feature
                sublinear_tf=True,     # Áp dụng log-scaling cho TF
            )),
            ("clf", MultinomialNB(alpha=0.1)),  # Smoothing nhỏ cho NB
        ])

        # Huấn luyện
        self._pipeline.fit(texts, labels)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def run(self, *, query: str = "", **kwargs) -> dict:
        """
        Phân loại ý định của câu truy vấn.

        Args:
            query: Câu hỏi của người dùng (tiếng Việt).

        Returns:
            dict: {
                "intent":     str   — nhãn ý định ("calculate", "greet", …),
                "confidence": float — độ tin cậy [0..1],
                "message":    str   — thông báo mô tả kết quả,
            }
        """
        if not query or not query.strip():
            return {
                "intent": "general",
                "confidence": 0.0,
                "message": self._log("Không có câu hỏi — mặc định intent=general"),
            }

        # Đảm bảo mô hình đã sẵn sàng
        self._ensure_model()

        # Dự đoán intent
        query_clean = query.strip().lower()
        predicted_label = self._pipeline.predict([query_clean])[0]

        # Lấy xác suất của tất cả các lớp, chọn max làm confidence
        probabilities = self._pipeline.predict_proba([query_clean])[0]
        confidence = float(max(probabilities))

        # Ánh xạ tên intent sang mô tả tiếng Việt cho thông báo
        intent_names_vi = {
            "calculate": "Tính toán",
            "datetime": "Ngày giờ",
            "rag": "Tra cứu kiến thức",
            "search": "Tìm kiếm web",
            "analyze": "Phân tích",
            "plan": "Lập kế hoạch",
            "greet": "Chào hỏi",
            "general": "Câu hỏi chung",
        }
        vi_name = intent_names_vi.get(predicted_label, predicted_label)

        return {
            "intent": predicted_label,
            "confidence": round(confidence, 4),
            "message": self._log(
                f"Ý định: {vi_name} ({predicted_label}) — "
                f"Độ tin cậy: {confidence:.2%}"
            ),
        }
