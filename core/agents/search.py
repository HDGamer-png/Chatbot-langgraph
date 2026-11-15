import re
from .base import AgentBase

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
    NAME = "WebSearch"

    # Khởi tạo tác tử tìm kiếm thông tin web và tra cứu ngoại tuyến.
    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    # Thực thi tìm kiếm thông tin trực tuyến qua DuckDuckGo hoặc chuyển sang kho tri thức ngoại tuyến.
    def run(self, query: str = "", **kwargs) -> dict:
        query = query.strip()
        if not query:
            return {"web_results": "", "sources": [], "message": "Không có từ khóa tìm kiếm."}
        online = self._search_duckduckgo(query)
        return online if online else self._search_offline(query)

    # Truy vấn kết quả tìm kiếm từ DuckDuckGo qua giao thức HTTP không cần API key.
    def _search_duckduckgo(self, query: str) -> dict | None:
        try:
            import requests
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=3,
            )
            if resp.status_code != 200:
                return None
            results = self._parse_ddg_html(resp.text)
            if not results:
                return None
            lines = [f"{i}. **{r['title']}**\n   {r['snippet']}" for i, r in enumerate(results[:5], 1)]
            sources = [r["url"] for r in results[:5] if r.get("url")]
            return {"web_results": "\n\n".join(lines), "sources": sources, "message": f"Tìm thấy {len(results)} kết quả từ DuckDuckGo."}
        except Exception:
            return None

    # Phân tích cú pháp HTML của trang kết quả tìm kiếm để trích xuất tiêu đề và đoạn trích.
    @staticmethod
    def _parse_ddg_html(html: str) -> list[dict]:
        results = []
        blocks = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        for t_raw, s_raw in blocks[:5]:
            t = re.sub(r'<[^>]+>', '', t_raw).strip()
            s = re.sub(r'<[^>]+>', '', s_raw).strip()
            if t and s:
                results.append({"title": t, "snippet": s, "url": ""})
        if not results:
            titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
            for t, s in zip(titles[:5], snippets[:5]):
                t_clean = re.sub(r'<[^>]+>', '', t).strip()
                s_clean = re.sub(r'<[^>]+>', '', s).strip()
                if t_clean:
                    results.append({"title": t_clean, "snippet": s_clean, "url": ""})
        return results

    # Tra cứu định nghĩa từ điển kỹ thuật cục bộ khi máy chủ không có kết nối mạng.
    def _search_offline(self, query: str) -> dict:
        q_lower = query.lower()
        matches = [m for k, m in _OFFLINE_KNOWLEDGE.items() if k in q_lower or any(w in q_lower for w in k.split())]
        if matches:
            matched_items = [{"term": k, "info": v} for k, v in _OFFLINE_KNOWLEDGE.items() if k in q_lower or any(w in q_lower for w in k.split())]
            lines = [f"📚 **{m['term'].title()}**: {m['info']}" for m in matched_items[:3]]
            return {
                "web_results": "\n\n".join(lines),
                "sources": [f"[Offline KB] {m['term']}" for m in matched_items[:3]],
                "message": f"Tra cứu offline: tìm thấy {len(matched_items)} kết quả.",
            }
        return {"web_results": "", "sources": [], "message": "Không tìm thấy kết quả (offline + online đều không khả dụng)."}
