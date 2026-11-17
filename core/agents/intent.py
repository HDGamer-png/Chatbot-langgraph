import json
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from .base import AgentBase

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "intent_model.joblib"
DEFAULT_JSONL_PATH = BASE_DIR / "data" / "training" / "base_intents.jsonl"
DEFAULT_ACTIVE_DOMAIN_FILE = BASE_DIR / "data" / ".active_domain"


class IntentClassifierAgent(AgentBase):
    NAME = "IntentClassifier"

    _TRAINING_DATA: dict[str, list[str]] = {
        "calculate": [
            "tính 25 cộng 30", "25 nhân 4 bằng bao nhiêu", "giải phương trình 2x + 5 = 15",
            "căn bậc hai của 144 là bao nhiêu", "tính diện tích hình tròn bán kính 5",
            "5 giai thừa bằng mấy", "tính tổng từ 1 đến 100", "phép chia 100 cho 7 dư bao nhiêu",
            "tính sin của 30 độ", "giải bất phương trình x^2 - 4 > 0", "10 lũy thừa 3 bằng bao nhiêu",
            "logarit cơ số 2 của 8", "tính đạo hàm của x^2 + 3x", "bao nhiêu phần trăm của 200 là 50",
            "quy đổi 100 usd ra vnd",
        ],
        "datetime": [
            "hôm nay là ngày bao nhiêu", "bây giờ mấy giờ rồi", "ngày mai là thứ mấy",
            "còn bao nhiêu ngày nữa đến tết", "giờ ở Mỹ bây giờ là mấy giờ", "múi giờ của Tokyo là gì",
            "tuần sau là ngày mấy", "hôm qua là ngày bao nhiêu", "năm nay là năm con gì",
            "tháng này có bao nhiêu ngày", "giờ London bây giờ là mấy giờ",
            "thời gian hiện tại ở Hàn Quốc", "khi nào là quốc khánh Việt Nam",
            "đếm ngược đến giáng sinh", "tuần này bắt đầu từ ngày nào",
        ],
        "rag": [
            "giải thích LangGraph là gì", "kiến trúc của hệ thống RAG hoạt động như thế nào",
            "so sánh CrewAI và AutoGen", "mô hình ReAct agent là gì", "cách tổ chức bộ nhớ cho chatbot",
            "tài liệu về multi-agent system", "hệ thống RAG có những thành phần nào",
            "vector database dùng để làm gì", "embedding là gì và hoạt động ra sao",
            "retrieval augmented generation nghĩa là gì", "cách xây dựng knowledge base cho chatbot",
            "LangChain khác LangGraph ở điểm nào", "agent memory có mấy loại",
            "cơ chế tool calling trong LLM hoạt động thế nào",
        ],
        "search": [
            "python là gì", "machine learning là gì", "deep learning là gì", "trí tuệ nhân tạo là gì",
            "tìm kiếm thông tin về Python 3.12", "search Google về machine learning",
            "tìm trên mạng cách cài đặt Docker", "tra cứu tin tức mới nhất về AI",
            "tìm hiểu về framework FastAPI", "tìm bài viết về deep learning",
            "search cách sử dụng Git nâng cao", "tìm kiếm tutorial về React Native",
            "tra cứu thông tin về công ty OpenAI", "tìm trên web về thuật toán sắp xếp",
            "tìm xem thời tiết hôm nay thế nào", "search tin tức công nghệ hôm nay",
            "tìm thông tin về ngôn ngữ Rust",
        ],
        "analyze": [
            "phân tích ưu nhược điểm của Python", "so sánh hiệu năng giữa Java và C++",
            "đánh giá framework Django và Flask", "phân tích xu hướng công nghệ năm 2026",
            "đánh giá ưu nhược điểm của microservice", "so sánh SQL và NoSQL nên dùng cái nào",
            "phân tích điểm mạnh yếu của React và Vue", "review kiến trúc serverless",
            "đánh giá ngôn ngữ Go cho backend", "so sánh giữa REST API và GraphQL",
            "phân tích chi phí cloud AWS và Azure", "đánh giá mô hình Agile và Waterfall",
            "so sánh Docker và Kubernetes",
        ],
        "plan": [
            "lập kế hoạch học Python trong 3 tháng", "tạo roadmap trở thành fullstack developer",
            "lên lịch học máy learning cho người mới", "lập kế hoạch dự án web trong 2 tuần",
            "tạo timeline phát triển ứng dụng mobile", "lên kế hoạch ôn thi chứng chỉ AWS",
            "lập lộ trình học data science từ đầu", "planning sprint cho team 5 người",
            "lên kế hoạch triển khai hệ thống CI/CD", "tạo kế hoạch backup dữ liệu hàng tuần",
            "lập lộ trình chuyển đổi từ monolith sang microservice", "kế hoạch học tiếng Anh IT trong 6 tháng",
            "tạo roadmap học AI và machine learning",
        ],
        "greet": [
            "xin chào", "chào bạn", "hello", "hi", "hey bạn ơi", "chào buổi sáng", "chào buổi tối",
            "bạn có khỏe không", "rất vui được gặp bạn", "bạn tên là gì", "bạn là ai vậy",
            "cảm ơn bạn nhiều", "tạm biệt nhé", "hẹn gặp lại", "bạn có thể giúp gì cho tôi",
        ],
        "general": [
            "cho tôi hỏi một chút", "tôi muốn biết thêm", "bạn có thể giải thích rõ hơn không",
            "tôi không hiểu lắm", "nói thêm về vấn đề này đi", "có ý kiến gì không",
            "thử lại một lần nữa", "tiếp tục đi bạn", "tôi cần trợ giúp",
            "mình muốn thảo luận về một vấn đề", "bạn nghĩ sao về điều này",
            "có cách nào khác không", "tóm tắt lại cho tôi",
        ],
    }

    # Khởi tạo tác tử phân loại ý định với cơ chế nạp mô hình trễ.
    def __init__(self, called_by: str = "System"):
        super().__init__(called_by)
        self._pipeline: Pipeline | None = None

    # Tạo cặp danh sách văn bản và nhãn tương ứng từ tập dữ liệu mẫu.
    def _build_training_data(self) -> tuple[list[str], list[str]]:
        texts, labels = [], []
        for intent_label, sentences in self._TRAINING_DATA.items():
            for s in sentences:
                texts.append(s)
                labels.append(intent_label)
        return texts, labels

    # Đảm bảo mô hình phân loại ý định được nạp từ tệp nhị phân hoặc huấn luyện từ tập mẫu dự phòng.
    def _ensure_model(self, force: bool = False) -> None:
        if self._pipeline is not None and not force:
            return
        if DEFAULT_MODEL_PATH.exists():
            try:
                self._pipeline = joblib.load(DEFAULT_MODEL_PATH)
                return
            except Exception as exc:
                print(f"[IntentClassifier] Lỗi khi nạp mô hình đã huấn luyện từ {DEFAULT_MODEL_PATH}: {exc}")
        texts, labels = [], []
        if DEFAULT_JSONL_PATH.exists():
            try:
                with open(DEFAULT_JSONL_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        t = record.get("text", "").strip().lower()
                        l = record.get("intent", "").strip().lower()
                        if t and l:
                            texts.append(unicodedata.normalize("NFC", t))
                            labels.append(l)
            except Exception:
                pass
        if DEFAULT_ACTIVE_DOMAIN_FILE.exists():
            try:
                dom = DEFAULT_ACTIVE_DOMAIN_FILE.read_text(encoding="utf-8").strip()
                if dom and dom not in ("default", "base"):
                    dom_intents = BASE_DIR / "data" / "domains" / dom / "intents.jsonl"
                    if dom_intents.exists():
                        with open(dom_intents, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                record = json.loads(line)
                                t = record.get("text", "").strip().lower()
                                l = record.get("intent", "").strip().lower()
                                if t and l:
                                    texts.append(unicodedata.normalize("NFC", t))
                                    labels.append(l)
            except Exception:
                pass
        if not texts or not labels:
            texts, labels = self._build_training_data()
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=8000, sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.1)),
        ])
        self._pipeline.fit(texts, labels)

    # Tải lại mô hình phân loại ý định từ tệp đĩa sau khi có cập nhật chuyên ngành hoặc huấn luyện.
    def reload(self) -> None:
        self._pipeline = None
        self._ensure_model(force=True)

    # Phân loại ý định của câu truy vấn và trả về nhãn kèm độ tin cậy.
    def run(self, query: str = "", **kwargs) -> dict:
        if not query or not query.strip():
            return {
                "intent": "general",
                "confidence": 0.0,
                "message": self._log("Không có câu hỏi — mặc định intent=general"),
            }
        self._ensure_model()
        query_clean = unicodedata.normalize("NFC", query.strip().lower())
        predicted_label = self._pipeline.predict([query_clean])[0]
        if hasattr(self._pipeline, "predict_proba"):
            probs = self._pipeline.predict_proba([query_clean])[0]
            confidence = float(max(probs))
        else:
            confidence = 0.95
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
            "message": self._log(f"Ý định: {vi_name} ({predicted_label}) — Độ tin cậy: {confidence:.2%}"),
        }
