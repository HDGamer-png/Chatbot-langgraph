"""
core/agents/rag.py — KnowledgeBaseAgent

Local RAG Agent sử dụng thuật toán BM25Okapi tự cài đặt (pure Python).
Không phụ thuộc thư viện `rank_bm25` hay bất kỳ engine tìm kiếm nào.

Quy trình hoạt động:
  1. Quét thư mục `data/docs/` để tải các file .txt, .md
  2. Chia văn bản thành chunks (~300 từ, overlap 50 từ)
  3. Xây dựng BM25 inverted index
  4. Nhận query → tokenize → tính BM25 score → trả về top-k chunks

Nếu `data/docs/` trống hoặc không tồn tại, agent sẽ sử dụng
bộ kiến thức tĩnh (fallback) gồm 6 chủ đề công nghệ.
"""

import math
import os
import re
from pathlib import Path
from typing import Any

from .base import AgentBase


class KnowledgeBaseAgent(AgentBase):
    """
    Agent tra cứu kiến thức nội bộ (RAG) sử dụng thuật toán BM25Okapi.

    BM25 (Best Matching 25) là thuật toán xếp hạng dựa trên TF-IDF
    có hiệu chỉnh cho độ dài tài liệu. Được cài đặt hoàn toàn bằng
    Python thuần, không dùng thư viện ngoài.
    """

    NAME = "KnowledgeBase"

    # Tham số BM25Okapi chuẩn
    _BM25_K1 = 1.5   # Điều chỉnh mức bão hòa tần suất từ (term frequency saturation)
    _BM25_B = 0.75    # Điều chỉnh chuẩn hóa độ dài tài liệu (document length normalization)

    # Cấu hình chunking
    _CHUNK_SIZE = 300       # Số từ mỗi chunk
    _CHUNK_OVERLAP = 50     # Số từ chồng lấp giữa 2 chunk liên tiếp

    # Thư mục chứa tài liệu
    _DOCS_DIR = os.path.join("data", "docs")

    # ------------------------------------------------------------------ #
    #  Kiến thức tĩnh fallback (khi data/docs/ trống)
    # ------------------------------------------------------------------ #
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

    def __init__(self, called_by: str = "System"):
        """
        Khởi tạo agent và nạp tài liệu từ data/docs/.

        Nếu thư mục trống hoặc không tồn tại, sử dụng fallback KB.
        """
        super().__init__(called_by)

        # Danh sách chunk: mỗi phần tử là dict {"text": str, "source": str}
        self._chunks: list[dict[str, str]] = []

        # BM25 index structures
        self._doc_freqs: dict[str, int] = {}    # Document frequency: từ -> số doc chứa từ đó
        self._doc_lengths: list[int] = []        # Độ dài (số token) mỗi chunk
        self._avg_doc_len: float = 0.0           # Trung bình độ dài doc
        self._inverted_index: dict[str, list[tuple[int, int]]] = {}  # từ -> [(doc_idx, freq)]
        self._corpus_size: int = 0               # Tổng số chunk
        self._using_fallback: bool = False        # Đang dùng fallback KB?

        # Nạp dữ liệu
        self._load_documents()
        if self._chunks:
            self._build_index()

    # ================================================================== #
    #  TOKENIZATION
    # ================================================================== #

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize văn bản tiếng Việt: tách theo khoảng trắng và dấu câu,
        chuyển thường, loại bỏ token quá ngắn.

        Args:
            text: Chuỗi văn bản cần tokenize.

        Returns:
            Danh sách token đã chuẩn hóa.
        """
        # Chuyển thường
        text = text.lower()
        # Tách theo khoảng trắng và dấu câu (giữ lại chữ cái, số, dấu tiếng Việt)
        tokens = re.findall(r'[\w]+', text, re.UNICODE)
        # Lọc token quá ngắn (1 ký tự) trừ các từ tiếng Việt quan trọng
        tokens = [t for t in tokens if len(t) > 1 or t in ('à', 'ở', 'ý')]
        return tokens

    # ================================================================== #
    #  DOCUMENT LOADING & CHUNKING
    # ================================================================== #

    def _load_documents(self) -> None:
        """
        Quét đệ quy toàn bộ thư mục data/docs/ và mọi thư mục con bên trong.
        Hỗ trợ MỌI định dạng: Mã nguồn (mọi ngôn ngữ), PDF, Word (.docx),
        Excel (.xlsx), PowerPoint (.pptx), Ảnh, Văn bản (.txt, .md, .json, .yaml...).
        Nếu không tìm thấy tài liệu, chuyển sang fallback KB.
        """
        try:
            from storage.file_reader import scan_documents
            doc_items = scan_documents(self._DOCS_DIR)
            for item in doc_items:
                content = item.get("text", "")
                rel_path = item.get("rel_path", item.get("filename", "unknown"))
                if content and content.strip():
                    chunks = self._split_into_chunks(content)
                    for chunk_text in chunks:
                        self._chunks.append({
                            "text": chunk_text,
                            "source": rel_path,
                        })
        except Exception as exc:
            print(f"[KnowledgeBase] Lỗi khi nạp tài liệu từ {self._DOCS_DIR}: {exc}")

        # Nếu không có chunk nào từ file, dùng fallback
        if not self._chunks:
            self._using_fallback = True
            for topic, content in self._FALLBACK_KB.items():
                self._chunks.append({
                    "text": content,
                    "source": f"fallback:{topic}",
                })

    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Chia văn bản thành các chunk có kích thước ~CHUNK_SIZE từ,
        với CHUNK_OVERLAP từ chồng lấp giữa các chunk liên tiếp.

        Args:
            text: Văn bản cần chia.

        Returns:
            Danh sách các chunk dạng chuỗi.
        """
        words = text.split()
        if len(words) <= self._CHUNK_SIZE:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = start + self._CHUNK_SIZE
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            # Di chuyển vị trí bắt đầu: chunk_size - overlap
            start += self._CHUNK_SIZE - self._CHUNK_OVERLAP

        return chunks

    # ================================================================== #
    #  BM25 INDEX BUILDING
    # ================================================================== #

    def _build_index(self) -> None:
        """
        Xây dựng BM25 inverted index từ danh sách chunk đã nạp.

        Tính toán:
          - Document frequency (df): số chunk chứa mỗi từ
          - Term frequency (tf): tần suất từ trong mỗi chunk
          - Inverted index: từ -> [(chunk_idx, tf)]
          - Thống kê corpus: avg_doc_len, corpus_size
        """
        self._corpus_size = len(self._chunks)
        self._doc_lengths = []
        self._doc_freqs = {}
        self._inverted_index = {}

        for doc_idx, chunk in enumerate(self._chunks):
            tokens = self._tokenize(chunk["text"])
            self._doc_lengths.append(len(tokens))

            # Đếm tần suất từ trong chunk này
            tf_map: dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            # Cập nhật document frequency và inverted index
            for token, freq in tf_map.items():
                # Document frequency: đếm số doc chứa token
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

                # Inverted index: lưu (doc_idx, term_frequency)
                if token not in self._inverted_index:
                    self._inverted_index[token] = []
                self._inverted_index[token].append((doc_idx, freq))

        # Tính trung bình độ dài document
        if self._doc_lengths:
            self._avg_doc_len = sum(self._doc_lengths) / len(self._doc_lengths)
        else:
            self._avg_doc_len = 1.0

    # ================================================================== #
    #  BM25 SCORING
    # ================================================================== #

    def _score(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        """
        Tính điểm BM25Okapi cho tất cả chunk dựa trên query tokens.

        Công thức BM25Okapi:
          score(D, Q) = Σ IDF(qi) · (tf(qi, D) · (k1 + 1)) /
                         (tf(qi, D) + k1 · (1 - b + b · |D| / avgdl))

          IDF(qi) = ln((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)

        Args:
            query_tokens: Danh sách token từ câu truy vấn.

        Returns:
            Danh sách (chunk_index, score) đã sắp xếp giảm dần theo score.
        """
        # Khởi tạo mảng điểm cho mỗi chunk
        scores = [0.0] * self._corpus_size
        N = self._corpus_size

        for q_token in query_tokens:
            if q_token not in self._inverted_index:
                continue

            # Tính IDF cho token này
            df = self._doc_freqs.get(q_token, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            # Duyệt qua các chunk chứa token này
            for doc_idx, tf in self._inverted_index[q_token]:
                doc_len = self._doc_lengths[doc_idx]
                # Công thức BM25 chuẩn
                numerator = tf * (self._BM25_K1 + 1)
                denominator = tf + self._BM25_K1 * (
                    1 - self._BM25_B + self._BM25_B * doc_len / self._avg_doc_len
                )
                scores[doc_idx] += idf * (numerator / denominator)

        # Sắp xếp theo điểm giảm dần, chỉ giữ chunk có điểm > 0
        scored_docs = [
            (idx, score) for idx, score in enumerate(scores) if score > 0
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs

    # ================================================================== #
    #  PUBLIC API
    # ================================================================== #

    def run(self, *, query: str = "", intent: str = "", **kwargs) -> dict:
        """
        Tra cứu kiến thức từ Knowledge Base dựa trên câu truy vấn.

        Args:
            query:  Câu hỏi của người dùng.
            intent: Ý định đã phân loại (tùy chọn, dùng để lọc kết quả).

        Returns:
            dict: {
                "kb_context": str  — Nội dung ngữ cảnh từ KB (top chunks nối lại),
                "sources":    list — Danh sách nguồn tài liệu,
                "message":    str  — Thông báo trạng thái,
            }
        """
        if not query or not query.strip():
            return {
                "kb_context": "",
                "sources": [],
                "message": self._log("Không có câu truy vấn."),
            }

        if not self._chunks:
            return {
                "kb_context": "",
                "sources": [],
                "message": self._log(
                    "Knowledge Base trống. Hãy thêm file .txt hoặc .md vào thư mục data/docs/"
                ),
            }

        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {
                "kb_context": "",
                "sources": [],
                "message": self._log("Không thể trích xuất từ khóa từ câu truy vấn."),
            }

        # Tính điểm BM25 và lấy top 3 kết quả
        scored_results = self._score(query_tokens)
        top_k = 3
        top_results = scored_results[:top_k]

        if not top_results:
            return {
                "kb_context": "",
                "sources": [],
                "message": self._log("Không tìm thấy tài liệu liên quan trong Knowledge Base."),
            }

        # Tổng hợp kết quả
        context_parts: list[str] = []
        sources: list[str] = []

        for rank, (doc_idx, score) in enumerate(top_results, 1):
            chunk = self._chunks[doc_idx]
            context_parts.append(f"[{rank}] {chunk['text']}")
            source = chunk["source"]
            if source not in sources:
                sources.append(source)

        kb_context = "\n\n".join(context_parts)
        top_score = top_results[0][1] if top_results else 0

        # Xây dựng thông báo
        fb_note = " (fallback KB)" if self._using_fallback else ""
        message = self._log(
            f"Tìm thấy {len(top_results)} kết quả{fb_note} — "
            f"BM25 score cao nhất: {top_score:.4f}"
        )

        return {
            "kb_context": kb_context,
            "sources": sources,
            "message": message,
        }
