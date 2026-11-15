import math
import os
import re
import pickle
from pathlib import Path
from .base import AgentBase

class KnowledgeBaseAgent(AgentBase):
    NAME = "KnowledgeBase"
    _BM25_K1 = 1.5
    _BM25_B = 0.75
    _CHUNK_SIZE = 300
    _CHUNK_OVERLAP = 50
    _DOCS_DIR = os.path.join("data", "docs")
    _CACHE_FILE = Path("data/.kb_index_cache.pkl")

    _FALLBACK_KB: dict[str, str] = {
        "langgraph": (
            "LangGraph là framework mã nguồn mở của LangChain để xây dựng ứng dụng "
            "multi-agent dưới dạng đồ thị có trạng thái (stateful graph). Mỗi node trong "
            "graph là một bước xử lý (agent, tool, function), các cạnh (edge) định nghĩa "
            "luồng điều khiển. LangGraph hỗ trợ chu kỳ (cycles), điều kiện rẽ nhánh, "
            "và quản lý trạng thái phức tạp. Ưu điểm nổi bật: kiểm soát luồng chặt chẽ, "
            "hỗ trợ human-in-the-loop, persistence, và streaming. Thích hợp cho các ứng "
            "dụng agent phức tạp cần kiểm soát luồng xử lý rõ ràng."
        ),
        "crewai": (
            "CrewAI là framework đa tác tử (multi-agent) cho phép tạo nhóm AI agent "
            "có vai trò (role), mục tiêu (goal) và backstory riêng biệt. Các agent "
            "phối hợp thông qua cơ chế delegation (ủy quyền) và task assignment. "
            "CrewAI hỗ trợ sequential và hierarchical process, tích hợp tool, memory, "
            "và cho phép agent tự quyết định cách hoàn thành task. So với LangGraph, "
            "CrewAI trừu tượng hóa cao hơn, dễ dùng hơn nhưng ít linh hoạt hơn trong "
            "việc kiểm soát luồng xử lý chi tiết."
        ),
        "react": (
            "ReAct (Reasoning + Acting) là mô hình kết hợp suy luận và hành động "
            "cho LLM agent. Quy trình: Thought → Action → Observation → lặp lại. "
            "Agent suy nghĩ (Thought) về bước tiếp theo, thực hiện hành động (Action) "
            "như gọi tool hoặc tìm kiếm, quan sát kết quả (Observation), rồi tiếp tục "
            "suy luận. Mô hình này giúp agent giải quyết bài toán phức tạp một cách "
            "có hệ thống, minh bạch quá trình suy luận, và giảm hallucination so với "
            "việc chỉ dùng LLM thuần túy."
        ),
        "memory": (
            "Bộ nhớ (Memory) trong hệ thống chatbot/agent gồm 3 loại chính: "
            "1) Short-term Memory — lưu lịch sử hội thoại hiện tại, mất khi kết thúc phiên. "
            "2) Long-term Memory — lưu trữ vĩnh viễn (database, file), dùng cho cá nhân hóa. "
            "3) Working Memory — bộ nhớ tạm cho task đang xử lý, giống RAM. "
            "Các kỹ thuật quản lý: sliding window (giữ N tin nhắn gần nhất), "
            "summarization (tóm tắt lịch sử cũ), vector store (tìm kiếm ngữ nghĩa "
            "trong lịch sử). Memory giúp agent duy trì ngữ cảnh và nhất quán."
        ),
        "rag": (
            "RAG (Retrieval-Augmented Generation) là kiến trúc kết hợp truy xuất "
            "thông tin với sinh văn bản. Quy trình: 1) Nhận câu hỏi → 2) Truy xuất "
            "tài liệu liên quan từ knowledge base (dùng embedding + vector search hoặc "
            "BM25) → 3) Kết hợp tài liệu vào prompt → 4) LLM sinh câu trả lời dựa trên "
            "ngữ cảnh. Ưu điểm: giảm hallucination, cập nhật kiến thức không cần fine-tune, "
            "trích dẫn nguồn. Các thành phần: Document Loader, Text Splitter, Embedding Model, "
            "Vector Store, Retriever, và Generator (LLM)."
        ),
        "multiagent": (
            "Multi-Agent System (MAS) là kiến trúc gồm nhiều agent AI chuyên biệt "
            "phối hợp để giải quyết vấn đề phức tạp. Mỗi agent có vai trò riêng: "
            "phân tích, tìm kiếm, tính toán, lập kế hoạch, v.v. Các mô hình phối hợp: "
            "1) Sequential — agent chạy tuần tự, đầu ra agent trước là đầu vào agent sau. "
            "2) Parallel — các agent chạy đồng thời, kết quả được tổng hợp. "
            "3) Hierarchical — agent quản lý (supervisor) phân công task cho agent con. "
            "4) Collaborative — agent trao đổi thông tin qua message passing. "
            "Framework phổ biến: LangGraph, CrewAI, AutoGen, MetaGPT."
        ),
    }

    # Khởi tạo tác tử RAG nội bộ và nạp chỉ mục từ bộ nhớ đệm hoặc quét tài liệu.
    def __init__(self, called_by: str = "System"):
        super().__init__(called_by)
        self._chunks: list[dict[str, str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._doc_lengths: list[int] = []
        self._avg_doc_len: float = 0.0
        self._inverted_index: dict[str, list[tuple[int, int]]] = {}
        self._corpus_size: int = 0
        self._using_fallback: bool = False
        self._from_cache: bool = False
        current_sig = self._compute_files_signature()
        if self._try_load_cache(current_sig):
            self._from_cache = True
        else:
            self._load_documents()
            if self._chunks:
                self._build_index()
                self._save_cache(current_sig)

    # Tính toán chữ ký kích thước và thời gian sửa đổi của các tài liệu trong thư mục nguồn.
    def _compute_files_signature(self) -> dict[str, str]:
        docs_path = Path(self._DOCS_DIR)
        sig = {}
        if not docs_path.exists():
            return sig
        for p in sorted(docs_path.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                try:
                    st = p.stat()
                    sig[p.relative_to(docs_path).as_posix()] = f"{st.st_size}_{st.st_mtime_ns}"
                except OSError:
                    pass
        return sig

    # Nạp cấu trúc chỉ mục BM25 từ tệp đệm nhị phân nếu chữ ký tệp không thay đổi.
    def _try_load_cache(self, current_sig: dict[str, str]) -> bool:
        if not self._CACHE_FILE.exists() or not current_sig:
            return False
        try:
            with open(self._CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if data.get("signature") == current_sig:
                self._chunks = data.get("chunks", [])
                self._doc_freqs = data.get("doc_freqs", {})
                self._doc_lengths = data.get("doc_lengths", [])
                self._avg_doc_len = data.get("avg_doc_len", 1.0)
                self._inverted_index = data.get("inverted_index", {})
                self._corpus_size = data.get("corpus_size", 0)
                self._using_fallback = data.get("using_fallback", False)
                return True
        except Exception:
            pass
        return False

    # Lưu trữ chỉ mục BM25 vào tệp đệm nhị phân để tối ưu tốc độ khởi động cho các lần sau.
    def _save_cache(self, current_sig: dict[str, str]) -> None:
        if not current_sig or self._using_fallback:
            return
        try:
            data = {
                "signature": current_sig,
                "chunks": self._chunks,
                "doc_freqs": self._doc_freqs,
                "doc_lengths": self._doc_lengths,
                "avg_doc_len": self._avg_doc_len,
                "inverted_index": self._inverted_index,
                "corpus_size": self._corpus_size,
                "using_fallback": self._using_fallback,
            }
            self._CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self._CACHE_FILE, "wb") as f:
                pickle.dump(data, f)
        except Exception as exc:
            print(f"[KnowledgeBase] Lỗi khi ghi cache BM25: {exc}")

    # Xây dựng lại toàn bộ chỉ mục tìm kiếm khi có cập nhật tệp tài liệu mới.
    def rebuild_index(self, force: bool = False) -> None:
        current_sig = self._compute_files_signature()
        self._chunks = []
        self._load_documents()
        if self._chunks:
            self._build_index()
            self._save_cache(current_sig)

    # Tách từ tiếng Việt và lọc các từ quá ngắn khỏi chuỗi văn bản.
    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r'[\w]+', text.lower(), re.UNICODE)
        return [t for t in tokens if len(t) > 1 or t in ('à', 'ở', 'ý')]

    # Quét và đọc nội dung văn bản từ thư mục tài liệu hoặc nạp kho dự phòng.
    def _load_documents(self) -> None:
        try:
            from storage.file_reader import scan_documents
            doc_items = scan_documents(self._DOCS_DIR)
            for item in doc_items:
                content = item.get("text", "")
                rel_path = item.get("rel_path", item.get("filename", "unknown"))
                if content and content.strip():
                    for chunk_text in self._split_into_chunks(content):
                        self._chunks.append({"text": chunk_text, "source": rel_path})
        except Exception as exc:
            print(f"[KnowledgeBase] Lỗi khi nạp tài liệu từ {self._DOCS_DIR}: {exc}")
        if not self._chunks:
            self._using_fallback = True
            for topic, content in self._FALLBACK_KB.items():
                self._chunks.append({"text": content, "source": f"fallback:{topic}"})

    # Phân mảnh tài liệu dài thành các đoạn nhỏ có phần chồng lấp ngữ cảnh.
    def _split_into_chunks(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self._CHUNK_SIZE:
            return [text]
        chunks, start = [], 0
        while start < len(words):
            chunks.append(" ".join(words[start:start + self._CHUNK_SIZE]))
            start += self._CHUNK_SIZE - self._CHUNK_OVERLAP
        return chunks

    # Xây dựng chỉ mục ngược và tính toán tần suất từ phục vụ thuật toán BM25.
    def _build_index(self) -> None:
        self._corpus_size = len(self._chunks)
        self._doc_lengths, self._doc_freqs, self._inverted_index = [], {}, {}
        for doc_idx, chunk in enumerate(self._chunks):
            tokens = self._tokenize(chunk["text"])
            self._doc_lengths.append(len(tokens))
            tf_map: dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1
            for token, freq in tf_map.items():
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                if token not in self._inverted_index:
                    self._inverted_index[token] = []
                self._inverted_index[token].append((doc_idx, freq))
        self._avg_doc_len = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 1.0

    # Chấm điểm mức độ liên quan của các đoạn tài liệu theo công thức BM25Okapi.
    def _score(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        scores = [0.0] * self._corpus_size
        N = self._corpus_size
        for q_token in query_tokens:
            if q_token not in self._inverted_index:
                continue
            df = self._doc_freqs.get(q_token, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for doc_idx, tf in self._inverted_index[q_token]:
                doc_len = self._doc_lengths[doc_idx]
                num = tf * (self._BM25_K1 + 1)
                den = tf + self._BM25_K1 * (1 - self._BM25_B + self._BM25_B * doc_len / self._avg_doc_len)
                scores[doc_idx] += idf * (num / den)
        scored_docs = [(idx, s) for idx, s in enumerate(scores) if s > 0]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs

    # Tra cứu các đoạn tri thức liên quan nhất từ kho tài liệu theo câu truy vấn.
    def run(self, query: str = "", intent: str = "", **kwargs) -> dict:
        if not query or not query.strip():
            return {"kb_context": "", "sources": [], "message": self._log("Không có câu truy vấn.")}
        if not self._chunks:
            return {"kb_context": "", "sources": [], "message": self._log("Knowledge Base trống. Hãy thêm file .txt hoặc .md vào thư mục data/docs/")}
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {"kb_context": "", "sources": [], "message": self._log("Không thể trích xuất từ khóa từ câu truy vấn.")}
        scored_results = self._score(query_tokens)
        top_results = scored_results[:3]
        if not top_results:
            return {"kb_context": "", "sources": [], "message": self._log("Không tìm thấy tài liệu liên quan trong Knowledge Base.")}
        context_parts, sources = [], []
        for rank, (doc_idx, _) in enumerate(top_results, 1):
            chunk = self._chunks[doc_idx]
            context_parts.append(f"[{rank}] {chunk['text']}")
            src = chunk["source"]
            if src not in sources:
                sources.append(src)
        top_score = top_results[0][1] if top_results else 0
        fb_note = " (fallback KB)" if self._using_fallback else ""
        return {
            "kb_context": "\n\n".join(context_parts),
            "sources": sources,
            "message": self._log(f"Tìm thấy {len(top_results)} kết quả{fb_note} — BM25 score cao nhất: {top_score:.4f}"),
        }
