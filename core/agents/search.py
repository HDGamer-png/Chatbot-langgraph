"""
core/agents/search.py — WebSearchAgent
Tìm kiếm miễn phí qua DuckDuckGo HTTP + Fallback tri thức offline.

Hoạt động 100% KHÔNG CẦN API KEY.
- Primary: DuckDuckGo HTML Search (POST request, timeout 3s)
- Fallback: Từ điển tri thức công nghệ offline (~25 khái niệm)
"""

import re
from .base import AgentBase

# ── Fallback: Từ điển tri thức công nghệ offline ─────────────────────
_OFFLINE_KNOWLEDGE = {
    "python": "Python là ngôn ngữ lập trình bậc cao, đa mục đích, cú pháp rõ ràng. Được tạo bởi Guido van Rossum (1991). Phổ biến trong AI, Data Science, Web.",
    "javascript": "JavaScript là ngôn ngữ lập trình phía client (trình duyệt) và server (Node.js). Cốt lõi của web hiện đại.",
    "machine learning": "Machine Learning (Học máy) là nhánh của AI cho phép máy tính học từ dữ liệu mà không cần lập trình tường minh. Gồm: Supervised, Unsupervised, Reinforcement Learning.",
    "deep learning": "Deep Learning là nhánh con của Machine Learning sử dụng mạng nơ-ron nhiều lớp (Neural Networks). Ứng dụng: nhận dạng ảnh, xử lý ngôn ngữ, xe tự lái.",
    "artificial intelligence": "Trí tuệ nhân tạo (AI) là lĩnh vực khoa học máy tính mô phỏng trí thông minh con người. Gồm: ML, NLP, Computer Vision, Robotics.",
    "ai": "Trí tuệ nhân tạo (AI) là lĩnh vực khoa học máy tính mô phỏng trí thông minh con người. Gồm: ML, NLP, Computer Vision, Robotics.",
    "chatgpt": "ChatGPT là chatbot AI của OpenAI, dựa trên mô hình GPT (Generative Pre-trained Transformer). Ra mắt 11/2022, sử dụng kiến trúc Transformer.",
    "langgraph": "LangGraph là framework Python xây dựng hệ thống Multi-Agent dạng Directed Graph. Mỗi Node là Agent, Edge là luồng dữ liệu. Hỗ trợ StateGraph, parallel Fan-out/Fan-in.",
    "flask": "Flask là micro web framework cho Python. Nhẹ, linh hoạt, dễ mở rộng. Dùng Jinja2 template và Werkzeug WSGI toolkit.",
    "docker": "Docker là nền tảng container hóa ứng dụng. Đóng gói phần mềm + dependencies thành image, chạy nhất quán trên mọi môi trường.",
    "git": "Git là hệ thống quản lý phiên bản phân tán. Theo dõi thay đổi source code, hỗ trợ branching, merging. Tạo bởi Linus Torvalds (2005).",
    "react": "React là thư viện JavaScript của Meta (Facebook) cho xây dựng giao diện người dùng. Component-based, Virtual DOM, JSX syntax.",
    "api": "API (Application Programming Interface) là giao diện lập trình ứng dụng. Cho phép các phần mềm giao tiếp với nhau. REST API và GraphQL là 2 kiểu phổ biến.",
    "sql": "SQL (Structured Query Language) là ngôn ngữ truy vấn cơ sở dữ liệu quan hệ. Các lệnh chính: SELECT, INSERT, UPDATE, DELETE, JOIN.",
    "rag": "RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp tra cứu tài liệu (Retrieval) với sinh văn bản (Generation). Giải quyết LLM outdated và hallucination.",
    "transformer": "Transformer là kiến trúc mạng nơ-ron cốt lõi của GPT, BERT, T5. Dựa trên cơ chế Self-Attention. Bài báo gốc: 'Attention Is All You Need' (2017).",
    "numpy": "NumPy là thư viện Python cho tính toán khoa học. Cung cấp ndarray đa chiều, phép toán vector hóa, đại số tuyến tính.",
    "pandas": "Pandas là thư viện Python xử lý và phân tích dữ liệu dạng bảng (DataFrame). Hỗ trợ đọc CSV, Excel, SQL; lọc, nhóm, thống kê.",
    "scikit-learn": "Scikit-learn là thư viện Python cho Machine Learning cổ điển. Cung cấp: Classification, Regression, Clustering, Dimensionality Reduction.",
    "tensorflow": "TensorFlow là framework Deep Learning của Google. Hỗ trợ training và inference mô hình nơ-ron trên CPU, GPU, TPU.",
    "pytorch": "PyTorch là framework Deep Learning của Meta. Dynamic computation graph, dễ debug. Phổ biến trong nghiên cứu AI.",
    "linux": "Linux là hệ điều hành mã nguồn mở dựa trên nhân Linux. Tạo bởi Linus Torvalds (1991). Phổ biến trong server, cloud, IoT.",
    "blockchain": "Blockchain là công nghệ sổ cái phân tán. Dữ liệu được lưu thành chuỗi khối (blocks) liên kết bằng mã hóa. Ứng dụng: tiền mã hóa, smart contract.",
    "cloud computing": "Điện toán đám mây cung cấp tài nguyên máy tính (server, storage, database) qua Internet theo nhu cầu. 3 loại: IaaS, PaaS, SaaS.",
    "kubernetes": "Kubernetes (K8s) là hệ thống điều phối container mã nguồn mở. Tự động hóa triển khai, scaling, quản lý ứng dụng container.",
}


class WebSearchAgent(AgentBase):
    """
    Tác tử tìm kiếm — DuckDuckGo miễn phí + Fallback offline.
    Không cần API Key, token hay tài khoản.
    """

    NAME = "WebSearch"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    def run(self, query: str = "", **kwargs) -> dict:
        """
        Tìm kiếm thông tin. Ưu tiên DuckDuckGo, fallback offline.

        Returns:
            {"web_results": str, "sources": list, "message": str}
        """
        query = query.strip()
        if not query:
            return {"web_results": "", "sources": [], "message": "Không có từ khóa tìm kiếm."}

        # Thử tìm kiếm online trước
        online_result = self._search_duckduckgo(query)
        if online_result:
            return online_result

        # Fallback: Tra cứu offline
        return self._search_offline(query)

    def _search_duckduckgo(self, query: str) -> dict | None:
        """Tìm kiếm qua DuckDuckGo HTML (không cần API key)."""
        try:
            import requests
        except ImportError:
            return None

        try:
            url = "https://html.duckduckgo.com/html/"
            resp = requests.post(
                url,
                data={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                timeout=3,
            )
            if resp.status_code != 200:
                return None

            html = resp.text
            results = self._parse_ddg_html(html)
            if not results:
                return None

            # Định dạng kết quả
            lines = []
            sources = []
            for i, r in enumerate(results[:5], 1):
                lines.append(f"{i}. **{r['title']}**\n   {r['snippet']}")
                if r.get("url"):
                    sources.append(r["url"])

            web_text = "\n\n".join(lines)
            return {
                "web_results": web_text,
                "sources": sources,
                "message": f"Tìm thấy {len(results)} kết quả từ DuckDuckGo.",
            }
        except Exception:
            return None

    @staticmethod
    def _parse_ddg_html(html: str) -> list[dict]:
        """Phân tích HTML của DuckDuckGo để trích xuất kết quả."""
        results = []

        # Tìm các khối kết quả
        result_blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*>(.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )

        for title_raw, snippet_raw in result_blocks[:5]:
            # Xóa HTML tags
            title = re.sub(r'<[^>]+>', '', title_raw).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet_raw).strip()
            if title and snippet:
                results.append({"title": title, "snippet": snippet, "url": ""})

        # Phương pháp 2: Tìm riêng lẻ nếu phương pháp 1 không ra kết quả
        if not results:
            titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
            for t, s in zip(titles[:5], snippets[:5]):
                title = re.sub(r'<[^>]+>', '', t).strip()
                snippet = re.sub(r'<[^>]+>', '', s).strip()
                if title:
                    results.append({"title": title, "snippet": snippet, "url": ""})

        return results

    def _search_offline(self, query: str) -> dict:
        """Tra cứu từ điển tri thức offline khi không có mạng."""
        q_lower = query.lower()
        matches = []

        for key, info in _OFFLINE_KNOWLEDGE.items():
            if key in q_lower or any(word in q_lower for word in key.split()):
                matches.append({"term": key, "info": info})

        if matches:
            lines = []
            for m in matches[:3]:
                lines.append(f"📚 **{m['term'].title()}**: {m['info']}")
            return {
                "web_results": "\n\n".join(lines),
                "sources": [f"[Offline KB] {m['term']}" for m in matches[:3]],
                "message": f"Tra cứu offline: tìm thấy {len(matches)} kết quả.",
            }

        return {
            "web_results": "",
            "sources": [],
            "message": "Không tìm thấy kết quả (offline + online đều không khả dụng).",
        }

