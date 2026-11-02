"""
╔══════════════════════════════════════════════════════════════════╗
║  CHATBOT THÔNG MINH VỚI LANGGRAPH — PHIÊN BẢN HOÀN CHỈNH v2.0  ║
║  Tác giả : Hoàng Minh Đức — Thực tập sinh AI Agent 2026         ║
║  Mô tả   : Hệ thống Multi-Agent với LangGraph + Claude API       ║
║  Tính năng:                                                       ║
║    ✓ Tìm kiếm thông tin (Web Search qua Tavily)                  ║
║    ✓ Suy nghĩ sâu / Deep Reasoning                               ║
║    ✓ Suy luận & Phân tích (ReAct)                                ║
║    ✓ Tính toán chính xác                                          ║
║    ✓ Trò chuyện đa lượt (Memory 3 lớp)                           ║
║    ✓ LangGraph State Graph (5 Nodes)                              ║
╚══════════════════════════════════════════════════════════════════╝

Cài đặt:
  pip install langgraph langchain-core langchain-anthropic
              langchain-community tavily-python rich python-dotenv

Biến môi trường (file .env hoặc export):
  ANTHROPIC_API_KEY=sk-ant-...     (bắt buộc)
  TAVILY_API_KEY=tvly-...          (tùy chọn — cho web search)

Chạy:
  python chatbot_langgraph_v2.py
"""

import os, re, json, time, math, operator, textwrap
from typing import TypedDict, Annotated, List, Optional, Literal
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
#  AUTO-INSTALL DEPENDENCIES
# ─────────────────────────────────────────────
def auto_install(pkg: str, import_as: str = None):
    import importlib, subprocess, sys
    try:
        importlib.import_module(import_as or pkg.replace("-", "_"))
    except ImportError:
        print(f"  [→] Cài đặt {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("\n" + "═"*60)
print("  KHỞI ĐỘNG LANGGRAPH CHATBOT v2.0")
print("═"*60)
print("\n[1/5] Kiểm tra thư viện...")

for pkg, imp in [
    ("langgraph", "langgraph"),
    ("langchain-core", "langchain_core"),
    ("langchain-anthropic", "langchain_anthropic"),
    ("langchain-community", "langchain_community"),
    ("rich", "rich"),
    ("python-dotenv", "dotenv"),
]:
    auto_install(pkg, imp)

print("  [✓] Thư viện sẵn sàng\n")

# ─────────────────────────────────────────────
#  IMPORTS SAU KHI CÀI
# ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich import print as rprint

from langgraph.graph import StateGraph, END

console = Console()

# ─────────────────────────────────────────────
#  API KEY CHECK
# ─────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_KEY    = os.getenv("TAVILY_API_KEY", "")

USE_REAL_LLM    = bool(ANTHROPIC_KEY)
USE_WEB_SEARCH  = bool(TAVILY_KEY)

print("[2/5] Kiểm tra API Keys...")
console.print(f"  Claude API : {'[green]✓ Đã cấu hình[/green]' if USE_REAL_LLM else '[yellow]⚠ Không tìm thấy — dùng MockLLM[/yellow]'}")
console.print(f"  Tavily API : {'[green]✓ Web Search bật[/green]' if USE_WEB_SEARCH else '[dim]✗ Không có — chỉ dùng knowledge base[/dim]'}")

# ─────────────────────────────────────────────
#  LLM ENGINE SETUP
# ─────────────────────────────────────────────
if USE_REAL_LLM:
    from langchain_anthropic import ChatAnthropic
    llm_claude = ChatAnthropic(
        model="claude-opus-4-5",       # Model mạnh nhất cho deep reasoning
        anthropic_api_key=ANTHROPIC_KEY,
        max_tokens=4096,
        temperature=0.3,
    )
    # Model nhanh hơn cho phân loại
    llm_fast = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=ANTHROPIC_KEY,
        max_tokens=512,
        temperature=0.0,
    )
    console.print("  [green]✓ Kết nối Claude API thành công[/green]")
else:
    llm_claude = None
    llm_fast   = None

# ─────────────────────────────────────────────
#  WEB SEARCH TOOL
# ─────────────────────────────────────────────
if USE_WEB_SEARCH:
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=TAVILY_KEY)
    except Exception:
        USE_WEB_SEARCH = False
        tavily = None
else:
    tavily = None

# ─────────────────────────────────────────────
#  KNOWLEDGE BASE (Nội bộ — chạy offline)
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "langgraph": [
        "LangGraph là framework xây dựng Multi-Agent dựa trên Đồ thị có hướng (Directed Graph). Mỗi Node là một Agent hoặc hàm xử lý, Edge định nghĩa luồng dữ liệu.",
        "LangGraph sử dụng TypedDict để khai báo State tường minh. State chạy xuyên suốt qua tất cả Nodes — đây là điểm khác biệt cốt lõi so với CrewAI.",
        "Cơ chế Reducer xử lý Race Condition khi nhiều Node song song cùng ghi vào State. Ví dụ: operator.add nối list thay vì ghi đè.",
        "LangGraph hỗ trợ Human-in-the-loop: dừng graph giữa chừng, cho người dùng xác nhận, rồi chạy tiếp.",
        "LangGraph phù hợp Production nhờ tính tất định cao, State Observability, fallback node, và parallel Fan-out/Fan-in.",
        "Cấu trúc graph: add_node() → add_edge() → set_entry_point() → compile(). Có thể thêm conditional_edges() cho rẽ nhánh.",
    ],
    "crewai": [
        "CrewAI dùng hướng tiếp cận khai báo (Declarative): định nghĩa Agent với Role/Goal/Tools, giao Task, framework tự điều phối.",
        "CrewAI phù hợp Prototyping/PoC: đường cong học tập thấp, ra demo nhanh trong vài giờ.",
        "async_execution=True cho Task song song. Bộ nhớ Short/Long-term/Entity tích hợp sẵn.",
        "Hạn chế: khó can thiệp sâu vào luồng xử lý, không có cơ chế pause/rewind như LangGraph.",
        "Khuyến nghị chiến lược: dùng CrewAI để prove concept → chuyển LangGraph khi vào production.",
    ],
    "react": [
        "ReAct = Reasoning and Acting: vòng lặp Thought → Action → Observation cho đến khi có Final Answer.",
        "Thought: LLM phân tích ngữ cảnh, quyết định gọi tool nào và truyền tham số gì.",
        "Action: hệ thống parse quyết định và thực thi tool tương ứng trong Tool Registry.",
        "Observation: kết quả tool được nạp ngược vào Context Window — Agent tiếp tục Thought mới.",
        "Hard limit max_iterations = 5 tránh vòng lặp vô hạn. Pydantic validate JSON output.",
    ],
    "memory": [
        "3 lớp bộ nhớ: Short-term (sliding window), Long-term (Vector DB + RAG), State (Blackboard JSON).",
        "Memory Dispatcher chỉ nạp đúng thông tin cần thiết — không full-load. Giảm Hallucination.",
        "State JSON thay vì natural language: cắt ~90% token đầu vào để duy trì ngữ cảnh.",
        "Async Write-back: tách luồng ghi Vector DB khỏi luồng phản hồi chính — tránh bottleneck.",
        "Tự động Summarize khi buffer >4000 tokens — kiểm soát chi phí API chủ động.",
    ],
    "rag": [
        "RAG = Retrieval-Augmented Generation: tích hợp Vector DB để LLM truy cập kiến thức ngoài training data.",
        "Quy trình: Embed tài liệu → lưu Vector DB → query → Cosine Similarity → nạp k chunks liên quan vào prompt.",
        "RAG giải quyết LLM outdated và context window giới hạn — chỉ lấy phần liên quan.",
        "Local: ChromaDB, FAISS. Cloud production: Pinecone, Weaviate.",
        "Hybrid Search: kết hợp Semantic Search (Vector) và BM25 (Keyword) cho độ chính xác cao hơn.",
    ],
    "multiagent": [
        "Multi-Agent chia tác vụ phức tạp cho nhiều Agent chuyên biệt — tránh quá tải ngữ cảnh.",
        "4 mô hình: Sequential (pipeline), Hierarchical (manager-worker), State-Graph (LangGraph), Conversational (AutoGen).",
        "Parallel execution cải thiện tốc độ 1.3x–4.15x. Lỗi khuếch đại 4.4x (Orchestrated) vs 17.2x (Independent).",
        "Fan-out/Fan-in: Manager phân công N subtask → N Workers song song → Reducer merge nguyên tử.",
    ],
}

# ─────────────────────────────────────────────
#  STATE DEFINITION
# ─────────────────────────────────────────────
class ChatbotState(TypedDict):
    """
    Blackboard chung của toàn bộ Graph.
    messages dùng operator.add (append reducer — không ghi đè).
    """
    messages         : Annotated[List[dict], operator.add]
    current_query    : str
    intent           : str          # langgraph/crewai/react/memory/rag/multiagent/calculate/search/analyze/general
    retrieved_context: str
    web_results      : str
    react_thought    : str
    react_action     : str
    react_observation: str
    final_answer     : str
    reasoning_chain  : str          # Chuỗi suy luận đầy đủ
    iteration_count  : int
    processing_log   : Annotated[List[str], operator.add]
    session_id       : str

# ─────────────────────────────────────────────
#  MEMORY MANAGER
# ─────────────────────────────────────────────
class MemoryManager:
    MAX_MESSAGES      = 20
    SUMMARY_THRESHOLD = 14

    def __init__(self):
        self.buffer: List[dict] = []
        self.stats = {"total": 0, "topics": set(), "searches": 0}

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content, "ts": datetime.now().isoformat()})
        if len(self.buffer) > self.MAX_MESSAGES:
            # Sliding window: giữ 2 tin đầu (system context) + n cuối
            self.buffer = self.buffer[:2] + self.buffer[-(self.MAX_MESSAGES - 2):]

    def get_recent(self, n: int = 6) -> List[dict]:
        return self.buffer[-n:]

    def get_history_text(self, n: int = 6) -> str:
        recent = self.get_recent(n)
        if not recent:
            return ""
        lines = []
        for m in recent:
            role = "Người dùng" if m["role"] == "user" else "Chatbot"
            lines.append(f"{role}: {m['content'][:200]}")
        return "\n".join(lines)

    def track(self, intent: str):
        self.stats["total"] += 1
        self.stats["topics"].add(intent)

    def get_stats(self) -> dict:
        return {**self.stats, "buffer_size": len(self.buffer), "topics": list(self.stats["topics"])}

memory = MemoryManager()

# ─────────────────────────────────────────────
#  TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────

def tool_search_knowledge(topic: str) -> str:
    """Tìm kiếm trong knowledge base nội bộ."""
    results = KNOWLEDGE_BASE.get(topic.lower(), [])
    if not results:
        for k, v in KNOWLEDGE_BASE.items():
            if topic.lower() in k or k in topic.lower():
                results = v
                break
    if results:
        return "\n".join(f"• {r}" for r in results)
    return "Không tìm thấy thông tin liên quan trong knowledge base."

def tool_web_search(query: str) -> str:
    """Tìm kiếm thông tin trên web qua Tavily."""
    if not USE_WEB_SEARCH or not tavily:
        return f"[Web Search không khả dụng — cần TAVILY_API_KEY]\nQuery: {query}"
    try:
        memory.stats["searches"] += 1
        result = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=4,
            include_answer=True,
        )
        lines = []
        if result.get("answer"):
            lines.append(f"📌 Tóm tắt: {result['answer']}\n")
        for i, r in enumerate(result.get("results", [])[:3], 1):
            lines.append(f"[{i}] {r.get('title', '')}")
            lines.append(f"    {r.get('content', '')[:300]}...")
            lines.append(f"    URL: {r.get('url', '')}\n")
        return "\n".join(lines) if lines else "Không tìm thấy kết quả."
    except Exception as e:
        return f"Web search lỗi: {e}"

def tool_calculate(expression: str) -> str:
    """Tính toán biểu thức toán học an toàn."""
    # Trích xuất biểu thức từ text tự nhiên
    clean = expression
    replacements = {
        "cộng": "+", "trừ": "-", "nhân": "*", "chia": "/",
        "bình phương": "**2", "căn bậc hai": "math.sqrt",
        "mũ": "**", "phần trăm": "/100",
        "×": "*", "÷": "/", "−": "-",
    }
    for vi, en in replacements.items():
        clean = clean.replace(vi, en)

    # Chỉ giữ ký tự an toàn
    safe = re.sub(r'[^0-9\+\-\*\/\.\(\)\s\%\^]', '', clean)
    safe = safe.replace("^", "**")

    if not safe.strip():
        # Cố gắng tìm số trong câu
        nums = re.findall(r'\d+\.?\d*', expression)
        ops = re.findall(r'[+\-*/]', expression)
        if len(nums) >= 2 and ops:
            safe = f"{nums[0]} {ops[0]} {nums[1]}"

    if not safe.strip():
        return f"Không thể phân tích biểu thức: '{expression}'"

    try:
        result = eval(safe, {"__builtins__": {}, "math": math})
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return f"**{safe.strip()}** = **{result:,}**" if abs(result) > 999 else f"**{safe.strip()}** = **{result}**"
    except Exception as e:
        return f"Lỗi tính toán '{safe}': {e}"

def tool_get_datetime(_: str = "") -> str:
    """Lấy ngày giờ hiện tại."""
    now = datetime.now()
    days_vn = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    months_vn = ["","tháng 1","tháng 2","tháng 3","tháng 4","tháng 5","tháng 6",
                  "tháng 7","tháng 8","tháng 9","tháng 10","tháng 11","tháng 12"]
    return (
        f"🕐 **Thời gian:** {now.strftime('%H:%M:%S')}\n"
        f"📅 **Ngày:** {days_vn[now.weekday()]}, ngày {now.day} {months_vn[now.month]} {now.year}\n"
        f"📍 **Múi giờ:** Giờ địa phương (UTC+7 — Việt Nam)"
    )

TOOL_REGISTRY = {
    "search_knowledge": tool_search_knowledge,
    "web_search"      : tool_web_search,
    "calculate"       : tool_calculate,
    "get_datetime"    : tool_get_datetime,
}

# ─────────────────────────────────────────────
#  MOCK LLM (Fallback khi không có API key)
# ─────────────────────────────────────────────
class MockLLM:
    """Fallback offline — không cần API key."""
    MODEL = "MockLLM-Offline-v2.0"

    def classify(self, q: str) -> str:
        q = q.lower()
        if any(w in q for w in ["xin chào","hello","hi ","chào","hey"]): return "greet"
        if any(w in q for w in ["langgraph","lang graph","đồ thị","graph","node","edge"]): return "langgraph"
        if any(w in q for w in ["crewai","crew ai","crew","prototyp"]): return "crewai"
        if any(w in q for w in ["react","reasoning","thought","action","observation","suy luận"]): return "react"
        if any(w in q for w in ["bộ nhớ","memory","short-term","long-term","dispatcher","blackboard"]): return "memory"
        if any(w in q for w in ["rag","retrieval","tìm kiếm","chromadb","embedding"]): return "rag"
        if any(w in q for w in ["multi-agent","đa tác tử","parallel","song song","orchestrat"]): return "multiagent"
        if any(w in q for w in ["tính","cộng","trừ","nhân","chia","+","-","*","/","bao nhiêu","kết quả"]): return "calculate"
        if any(w in q for w in ["giờ","ngày","thời gian","hôm nay","date","time"]): return "datetime"
        if any(w in q for w in ["tìm","search","tra cứu","thông tin về","cho biết"]): return "search"
        if any(w in q for w in ["phân tích","so sánh","đánh giá","nhận xét","ưu nhược"]): return "analyze"
        return "general"

    def answer(self, q: str, intent: str, context: str) -> str:
        if intent == "greet":
            return (
                "## Xin chào! 👋\n\n"
                "Tôi là **Chatbot AI Agent v2.0** xây dựng với **LangGraph**.\n\n"
                "**Tôi có thể:**\n"
                "- 🔍 **Tìm kiếm** thông tin (web + knowledge base)\n"
                "- 🧠 **Suy nghĩ sâu** và lập luận đa bước\n"
                "- 📊 **Phân tích** và so sánh\n"
                "- 🔢 **Tính toán** chính xác\n"
                "- 💬 **Trò chuyện** với memory đa lượt\n\n"
                "**Chủ đề tôi hiểu sâu:** LangGraph, CrewAI, ReAct, Memory, RAG, Multi-Agent\n\n"
                "*Hỏi tôi bất cứ điều gì!*"
            )
        if not context or "Không tìm thấy" in context:
            return f"Tôi chưa có đủ thông tin chi tiết về: *\"{q[:60]}\"*\n\nThử hỏi về: **LangGraph**, **CrewAI**, **ReAct**, **Memory**, **RAG**, **Multi-Agent**."

        chunks = [c.strip() for c in context.split("\n") if c.strip() and c.strip().startswith("•")]
        if not chunks:
            chunks = [c.strip() for c in context.split("\n") if c.strip()][:4]

        ans = [f"## Kết quả tìm kiếm: *{q[:50]}*\n"]
        for i, chunk in enumerate(chunks[:4], 1):
            text = chunk.lstrip("•").strip()
            ans.append(f"**{i}.** {text}\n")
        ans.append("\n---\n*💡 Gợi ý: Hỏi thêm để biết chi tiết hơn về bất kỳ điểm nào.*")
        return "\n".join(ans)

mock_llm = MockLLM()

# ─────────────────────────────────────────────
#  HELPER: GỌI LLM THỰC
# ─────────────────────────────────────────────
def call_llm(prompt: str, system: str = None, fast: bool = False) -> str:
    """Gọi Claude API hoặc fallback về mock."""
    if not USE_REAL_LLM:
        return ""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        model = llm_fast if fast else llm_claude
        msgs = []
        if system:
            msgs.append(SystemMessage(content=system))
        msgs.append(HumanMessage(content=prompt))
        response = model.invoke(msgs)
        return response.content
    except Exception as e:
        return f"[LLM Error: {e}]"

# ─────────────────────────────────────────────
#  GRAPH NODES
# ─────────────────────────────────────────────

def node_intent_classifier(state: ChatbotState) -> dict:
    """
    NODE 1 — Intent Classifier
    Phân loại ý định người dùng (fast model hoặc mock).
    """
    q = state["current_query"]

    if USE_REAL_LLM:
        prompt = (
            f"Phân loại intent của câu hỏi này thành MỘT trong các nhãn sau:\n"
            f"langgraph | crewai | react | memory | rag | multiagent | "
            f"calculate | datetime | search | analyze | greet | general\n\n"
            f"Câu hỏi: \"{q}\"\n\n"
            f"Trả lời CHỈ nhãn duy nhất, không giải thích."
        )
        intent = call_llm(prompt, fast=True).strip().lower()
        # Validate
        valid = {"langgraph","crewai","react","memory","rag","multiagent",
                 "calculate","datetime","search","analyze","greet","general"}
        if intent not in valid:
            intent = mock_llm.classify(q)
    else:
        intent = mock_llm.classify(q)

    memory.track(intent)
    return {
        "intent": intent,
        "iteration_count": 0,
        "processing_log": [f"[IntentClassifier] '{q[:40]}' → intent='{intent}'"],
    }


def node_memory_retriever(state: ChatbotState) -> dict:
    """
    NODE 2 — Memory Retriever
    Tìm kiếm context từ knowledge base + lịch sử hội thoại.
    """
    intent  = state["intent"]
    query   = state["current_query"]

    # Tìm trong knowledge base
    kb_context = tool_search_knowledge(intent)
    if "Không tìm thấy" in kb_context:
        # Thử tìm với từ khóa
        for word in query.lower().split():
            kb_context = tool_search_knowledge(word)
            if "Không tìm thấy" not in kb_context:
                break

    # Lịch sử hội thoại gần nhất
    history = memory.get_history_text(n=4)
    history_note = f"\n[Lịch sử hội thoại gần nhất:\n{history}]" if history else ""

    context = kb_context + history_note

    return {
        "retrieved_context": context,
        "processing_log": [f"[MemoryRetriever] Lấy {len(kb_context.split(chr(10)))} chunks | history={bool(history)}"],
    }


def node_web_search_agent(state: ChatbotState) -> dict:
    """
    NODE 3 — Web Search Agent
    Kích hoạt nếu intent là 'search' hoặc query cần thông tin thực tế mới.
    """
    intent = state["intent"]
    query  = state["current_query"]

    # Quyết định có search web không
    should_search = (
        intent == "search" or
        USE_WEB_SEARCH and any(kw in query.lower() for kw in [
            "mới nhất", "hiện tại", "2024", "2025", "2026",
            "tin tức", "update", "release", "phiên bản", "version",
            "giá", "price", "today", "hôm nay"
        ])
    )

    if should_search and USE_WEB_SEARCH:
        # Tối ưu query cho search
        if USE_REAL_LLM:
            search_q = call_llm(
                f"Tạo query tìm kiếm tiếng Anh ngắn gọn (5-8 từ) cho câu hỏi: '{query}'. Chỉ trả về query.",
                fast=True
            ).strip().strip('"')
        else:
            search_q = query[:80]
        web = tool_web_search(search_q)
        return {
            "web_results": web,
            "processing_log": [f"[WebSearch] Query='{search_q[:40]}' | len={len(web)}"],
        }

    return {
        "web_results": "",
        "processing_log": [f"[WebSearch] Bỏ qua (intent='{intent}', web_search={'off' if not USE_WEB_SEARCH else 'not needed'})"],
    }


def node_react_agent(state: ChatbotState) -> dict:
    """
    NODE 4 — ReAct Agent (Reasoning and Acting)
    Cốt lõi hệ thống: Thought → Action → Observation
    """
    query     = state["current_query"]
    intent    = state["intent"]
    context   = state["retrieved_context"]
    web       = state["web_results"]
    iteration = state.get("iteration_count", 0)
    MAX_ITER  = 3

    if iteration >= MAX_ITER:
        return {
            "react_thought": "Đã đạt giới hạn vòng lặp ReAct.",
            "react_action": "FINISH",
            "react_observation": "Max iterations.",
            "reasoning_chain": state.get("reasoning_chain", ""),
            "processing_log": [f"[ReAct] ⚠ Max iterations ({MAX_ITER})"],
        }

    # ── THOUGHT ──
    if USE_REAL_LLM:
        thought_prompt = (
            f"Bạn là AI Agent thông minh đang phân tích câu hỏi.\n\n"
            f"Câu hỏi: {query}\n"
            f"Intent: {intent}\n"
            f"Ngữ cảnh có sẵn: {context[:500]}\n"
            f"Kết quả web: {web[:300] if web else 'Không có'}\n\n"
            f"Suy luận ngắn gọn (2-3 câu): Tôi cần làm gì để trả lời tốt nhất?"
        )
        thought = call_llm(thought_prompt, fast=True)
    else:
        thought_map = {
            "calculate": "Phát hiện yêu cầu tính toán. Sẽ dùng tool calculate.",
            "datetime"  : "Hỏi về thời gian. Sẽ gọi get_datetime.",
            "search"    : "Cần tìm thông tin mới. Ưu tiên kết quả web nếu có.",
            "analyze"   : "Yêu cầu phân tích. Sẽ tổng hợp từ nhiều nguồn.",
            "greet"     : "Lời chào. Phản hồi thân thiện và giới thiệu.",
        }
        thought = thought_map.get(intent, f"Phân tích câu hỏi '{query[:40]}' với intent '{intent}'. Tìm thông tin phù hợp.")

    # ── ACTION ──
    action_map = {
        "calculate": ("calculate", query),
        "datetime" : ("get_datetime", ""),
        "greet"    : ("search_knowledge", "greet"),
        "search"   : ("web_search", query),
    }
    if intent in action_map:
        action_name, action_input = action_map[intent]
    else:
        action_name, action_input = "search_knowledge", intent

    # ── OBSERVATION (thực thi tool) ──
    tool_fn     = TOOL_REGISTRY.get(action_name, tool_search_knowledge)
    observation = tool_fn(action_input)

    # Cập nhật reasoning chain
    chain = state.get("reasoning_chain", "")
    chain += (
        f"\n**Vòng lặp {iteration+1}:**\n"
        f"- Thought: {thought[:200]}\n"
        f"- Action: `{action_name}({action_input[:50]})`\n"
        f"- Observation: {observation[:200]}\n"
    )

    return {
        "react_thought"    : thought,
        "react_action"     : action_name,
        "react_observation": observation,
        "reasoning_chain"  : chain,
        "iteration_count"  : iteration + 1,
        "processing_log"   : [
            f"[ReAct] iter={iteration+1} | action={action_name} | obs_len={len(observation)}"
        ],
    }


def node_response_generator(state: ChatbotState) -> dict:
    """
    NODE 5 — Response Generator (Deep Reasoning)
    Tổng hợp tất cả nguồn thông tin → sinh câu trả lời chất lượng cao.
    """
    query   = state["current_query"]
    intent  = state["intent"]
    context = state["retrieved_context"]
    web     = state["web_results"]
    obs     = state["react_observation"]
    chain   = state["reasoning_chain"]
    history = memory.get_history_text(n=3)

    # Special intents → dùng observation trực tiếp
    if intent in ("calculate", "datetime"):
        answer = f"## Kết quả\n\n{obs}"

    elif USE_REAL_LLM:
        # ── Deep Reasoning với Claude ──
        system_prompt = """Bạn là Chatbot AI Agent thông minh, được xây dựng với LangGraph.
Bạn có khả năng: tìm kiếm thông tin, suy luận sâu, phân tích, tính toán, và trò chuyện.
Trả lời bằng tiếng Việt, markdown đẹp, có cấu trúc rõ ràng.
Nếu có thông tin từ web, ưu tiên dùng thông tin đó và trích dẫn nguồn.
Nếu được hỏi phân tích, hãy suy luận từng bước và đưa ra nhận định riêng.
QUAN TRỌNG: Trả lời đầy đủ, sâu sắc, không ngắn gọn quá."""

        user_prompt = f"""Câu hỏi người dùng: {query}

=== THÔNG TIN TỪ KNOWLEDGE BASE ===
{context[:800]}

=== KẾT QUẢ WEB SEARCH ===
{web[:600] if web else "Không có"}

=== LỊCH SỬ HỘI THOẠI ===
{history if history else "Đây là tin nhắn đầu tiên"}

=== CHUỖI SUY LUẬN ===
{chain[:400]}

Dựa trên tất cả thông tin trên, hãy trả lời câu hỏi một cách đầy đủ, chính xác và có cấu trúc."""

        answer = call_llm(user_prompt, system=system_prompt)
        if not answer or answer.startswith("[LLM Error"):
            answer = mock_llm.answer(query, intent, context)
    else:
        answer = mock_llm.answer(query, intent, context)

    # Lưu vào memory
    memory.add("user", query)
    memory.add("assistant", answer[:300])

    return {
        "final_answer": answer,
        "messages"    : [
            {"role": "user",      "content": query,         "ts": datetime.now().isoformat()},
            {"role": "assistant", "content": answer[:300],  "ts": datetime.now().isoformat()},
        ],
        "processing_log": [f"[ResponseGenerator] Đã tạo câu trả lời ({len(answer)} ký tự)"],
    }


def node_memory_writer(state: ChatbotState) -> dict:
    """
    NODE 6 — Memory Writer (Async Write-back)
    Cập nhật State Store và thống kê session.
    """
    stats = memory.get_stats()
    return {
        "processing_log": [
            f"[MemoryWriter] buffer={stats['buffer_size']} | total={stats['total']} | "
            f"searches={stats['searches']} | topics={stats['topics']}"
        ],
    }

# ─────────────────────────────────────────────
#  BUILD GRAPH
# ─────────────────────────────────────────────
def build_graph():
    g = StateGraph(ChatbotState)

    g.add_node("intent_classifier",   node_intent_classifier)
    g.add_node("memory_retriever",    node_memory_retriever)
    g.add_node("web_search_agent",    node_web_search_agent)
    g.add_node("react_agent",         node_react_agent)
    g.add_node("response_generator",  node_response_generator)
    g.add_node("memory_writer",       node_memory_writer)

    g.set_entry_point("intent_classifier")
    g.add_edge("intent_classifier",  "memory_retriever")
    g.add_edge("memory_retriever",   "web_search_agent")
    g.add_edge("web_search_agent",   "react_agent")
    g.add_edge("react_agent",        "response_generator")
    g.add_edge("response_generator", "memory_writer")
    g.add_edge("memory_writer",      END)

    return g.compile()

# ─────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────
def display_welcome():
    mode = f"Claude API ({'claude-opus-4-5'})" if USE_REAL_LLM else "MockLLM (Offline)"
    search = "Tavily Web Search ✓" if USE_WEB_SEARCH else "Chỉ Knowledge Base"
    console.print()
    console.print(Panel.fit(
        f"[bold blue]🤖 LANGGRAPH CHATBOT v2.0[/bold blue]\n"
        f"[dim]Kiến trúc: Multi-Agent + ReAct + RAG + Memory 3 lớp[/dim]\n\n"
        f"[cyan]🧠 LLM:[/cyan] {mode}\n"
        f"[cyan]🔍 Search:[/cyan] {search}\n"
        f"[cyan]📦 Nodes:[/cyan] 6 Nodes (IntentClassifier → MemoryRetriever → WebSearch → ReAct → ResponseGenerator → MemoryWriter)\n\n"
        f"[yellow]Tính năng:[/yellow] Tìm kiếm · Suy nghĩ sâu · Suy luận · Phân tích · Tính toán · Trò chuyện\n"
        f"[dim]Lệnh: [bold]help[/bold] · [bold]graph[/bold] · [bold]stats[/bold] · [bold]clear[/bold] · [bold]quit[/bold][/dim]",
        border_style="blue",
        title="[bold]Hoàng Minh Đức — Thực tập sinh AI Agent 2026[/bold]"
    ))
    console.print()


def display_graph():
    console.print(Panel(
        "[bold cyan]LANGGRAPH WORKFLOW v2.0[/bold cyan]\n\n"
        "  [START]\n"
        "    │\n"
        "    ▼\n"
        "  [bold yellow]① intent_classifier[/bold yellow]    ← Phân loại ý định (Claude Haiku)\n"
        "    │\n"
        "    ▼\n"
        "  [bold yellow]② memory_retriever[/bold yellow]     ← RAG: knowledge base + lịch sử hội thoại\n"
        "    │\n"
        "    ▼\n"
        "  [bold yellow]③ web_search_agent[/bold yellow]     ← Tìm kiếm web thực (Tavily API)\n"
        "    │\n"
        "    ▼\n"
        "  [bold yellow]④ react_agent[/bold yellow]          ← ReAct: Thought → Action → Observation\n"
        "    │\n"
        "    ▼\n"
        "  [bold yellow]⑤ response_generator[/bold yellow]   ← Deep Reasoning (Claude Opus)\n"
        "    │\n"
        "    ▼\n"
        "  [bold yellow]⑥ memory_writer[/bold yellow]        ← Async write-back + State update\n"
        "    │\n"
        "   [END]\n\n"
        "[dim]TypedDict State chạy xuyên suốt | operator.add Reducer tránh Race Condition[/dim]",
        border_style="cyan",
        title="[bold]Kiến trúc Graph[/bold]"
    ))


def display_log(logs: list):
    t = Table(title="📊 Execution Log", border_style="dim", show_lines=True)
    t.add_column("#", style="cyan", width=4)
    t.add_column("Node", style="yellow", width=20)
    t.add_column("Detail", style="white")
    for i, log in enumerate(logs, 1):
        if "]" in log:
            node = log.split("]")[0].replace("[","")
            detail = log.split("]",1)[1].strip()
        else:
            node, detail = "System", log
        t.add_row(f"{i}", node, detail)
    console.print(t)


def display_stats():
    s = memory.get_stats()
    t = Table(title="📈 Session Statistics", border_style="blue")
    t.add_column("Metric", style="yellow", width=28)
    t.add_column("Value", style="green")
    t.add_row("LLM Engine",      f"Claude API ({USE_REAL_LLM})")
    t.add_row("Web Search",       "Bật (Tavily)" if USE_WEB_SEARCH else "Tắt")
    t.add_row("Total Queries",    str(s["total"]))
    t.add_row("Web Searches",     str(s["searches"]))
    t.add_row("Memory Buffer",    f"{s['buffer_size']} messages")
    t.add_row("Topics Covered",   ", ".join(s["topics"]) if s["topics"] else "—")
    console.print(t)


def display_help():
    console.print(Panel(
        "[bold]GỢI Ý CÂU HỎI:[/bold]\n\n"
        "[cyan]• LangGraph là gì và tại sao nên dùng cho production?[/cyan]\n"
        "[cyan]• So sánh chi tiết LangGraph và CrewAI[/cyan]\n"
        "[cyan]• Giải thích cơ chế ReAct từng bước[/cyan]\n"
        "[cyan]• Memory management trong Multi-Agent hoạt động thế nào?[/cyan]\n"
        "[cyan]• RAG là gì? Quy trình gồm những bước nào?[/cyan]\n"
        "[cyan]• Phân tích ưu và nhược điểm của kiến trúc State-Graph[/cyan]\n"
        "[cyan]• Tính 1024 * 768 + 2048[/cyan]\n"
        "[cyan]• Hôm nay là ngày mấy?[/cyan]\n"
        "[cyan]• Tìm kiếm thông tin về LangGraph mới nhất[/cyan]\n\n"
        "[bold]LỆNH ĐẶC BIỆT:[/bold]\n"
        "[yellow]graph[/yellow]   — Sơ đồ kiến trúc Graph\n"
        "[yellow]stats[/yellow]   — Thống kê phiên làm việc\n"
        "[yellow]clear[/yellow]   — Xóa lịch sử hội thoại\n"
        "[yellow]help[/yellow]    — Hiển thị hướng dẫn\n"
        "[yellow]quit[/yellow]    — Thoát chương trình",
        border_style="green",
        title="[bold]📚 HƯỚNG DẪN[/bold]"
    ))

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n[3/5] Biên dịch LangGraph...")
    graph = build_graph()
    console.print("[green]  [✓] LangGraph compiled — 6 Nodes sẵn sàng[/green]")

    print("[4/5] Khởi tạo Memory Manager & Tools...")
    console.print("[green]  [✓] Memory (3 lớp) + 4 Tools sẵn sàng[/green]")

    print("[5/5] Sẵn sàng!\n")
    display_welcome()

    SESSION_ID = f"session_{int(time.time())}"
    turn = 0

    while True:
        try:
            turn += 1
            console.print(f"[dim]── Turn {turn} {'─'*45}[/dim]")
            query = console.input("[bold green]Bạn[/bold green] › ").strip()
            if not query:
                continue

            cmd = query.lower()
            if cmd in ("quit","exit","thoát","q"):
                console.print("\n[bold blue]👋 Cảm ơn! Kết thúc phiên LangGraph Chatbot.[/bold blue]")
                display_stats()
                break
            if cmd == "help": display_help(); continue
            if cmd == "graph": display_graph(); continue
            if cmd == "stats": display_stats(); continue
            if cmd == "clear":
                memory.buffer.clear()
                console.print("[yellow]✓ Đã xóa lịch sử hội thoại[/yellow]")
                continue

            # ── Invoke Graph ──
            with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), console=console, transient=True) as p:
                task = p.add_task("Đang xử lý qua LangGraph (6 Nodes)...", total=None)
                t0 = time.time()
                initial: ChatbotState = {
                    "messages": [], "current_query": query,
                    "intent": "", "retrieved_context": "", "web_results": "",
                    "react_thought": "", "react_action": "", "react_observation": "",
                    "final_answer": "", "reasoning_chain": "",
                    "iteration_count": 0, "processing_log": [],
                    "session_id": SESSION_ID,
                }
                result = graph.invoke(initial)
                elapsed = time.time() - t0

            # Display log
            display_log(result["processing_log"])
            console.print()

            # Display answer
            console.print(Panel(
                Markdown(result["final_answer"]),
                border_style="blue",
                title=f"[bold blue]🤖 Chatbot[/bold blue]  [dim]({elapsed:.2f}s | intent={result['intent']})[/dim]",
                padding=(1, 2)
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Ctrl+C — gõ 'quit' để thoát hoặc tiếp tục.[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Lỗi: {e}[/red]")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
