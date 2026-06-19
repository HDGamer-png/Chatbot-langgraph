"""
╔══════════════════════════════════════════════════════════════════════╗
║   LANGGRAPH MULTI-AGENT CHATBOT v3.2                                 ║
║   Tác giả : Hoàng Minh Đức — Thực tập sinh AI Agent 2026            ║
║                                                                       ║
║   TÍNH NĂNG v3.2 — DUAL PROVIDER:                                    ║
║   ✓ Hỗ trợ Anthropic Claude (trả phí) + Google Gemini (miễn phí)   ║
║   ✓ Tự động chọn provider dựa trên key có trong .env                ║
║   ✓ Fallback: nếu Anthropic hết credit → tự dùng Gemini             ║
║   ✓ Tất cả fix từ v3.1 giữ nguyên (F1→F9)                          ║
║                                                                       ║
║   Cài đặt:                                                            ║
║     pip install langgraph langchain-anthropic langchain-core         ║
║                 langchain-google-genai tavily-python rich            ║
║                 python-dotenv                                         ║
║                                                                       ║
║   .env:                                                               ║
║     ANTHROPIC_API_KEY=sk-ant-...   (Claude — trả phí)               ║
║     GEMINI_API_KEY=AIza...         (Gemini — miễn phí)              ║
║     TAVILY_API_KEY=tvly-...        (Search — tuỳ chọn)              ║
║                                                                       ║
║   Chạy:  python multi_agent_chatbot_v3.py                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════
import os, re, math, json, time, operator, traceback
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.tree import Tree
from rich import box

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

console = Console()

# ══════════════════════════════════════════════════════════════════
#  CONFIG & PROVIDER SELECTION
# ══════════════════════════════════════════════════════════════════
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GROQ_KEY      = os.getenv("GROQ_API_KEY", "").strip()
TAVILY_KEY    = os.getenv("TAVILY_API_KEY", "").strip()
CHAT_DIR      = Path("chat_history")
CHAT_DIR.mkdir(exist_ok=True)

# ── Chọn provider ──────────────────────────────────────────────────
# Ưu tiên: Anthropic → GROQ → hỏi user
PROVIDER = "none"
if ANTHROPIC_KEY and ANTHROPIC_KEY != "sk-ant-api03-ĐẶT_KEY_CỦA_BẠN_VÀO_ĐÂY":
    PROVIDER = "anthropic"
elif GROQ_KEY:
    PROVIDER = "groq"

# Model names theo provider
MODELS = {
    "anthropic": {
        "fast"   : "claude-haiku-4-5-20251001",
        "main"   : "claude-sonnet-4-20250514",
        "analyst": "claude-sonnet-4-20250514",
    },
    "groq": {
        "fast"   : "compound-beta-mini",
        "main"   : "compound-beta",
        "analyst": "compound-beta",
    },
}

def get_model(role: str) -> str:
    """Lấy tên model theo provider và role (fast/main/analyst)."""
    return MODELS.get(PROVIDER, MODELS["groq"])[role]

def _gemini_token_arg(max_tokens: int) -> dict:
    try:
        import langchain_google_genai
        version = getattr(langchain_google_genai, "__version__", "")
        if version.startswith("1."):
            return {"max_tokens": max_tokens}
    except Exception:
        pass
    return {"max_output_tokens": max_tokens}
def _check_gemini_packages():
    try:
        import langchain_google_genai
        import google.generativeai
        console.print(
            f"[dim]langchain-google-genai={getattr(langchain_google_genai, '__version__', 'unknown')}, "
            f"google-generativeai={getattr(google.generativeai, '__version__', 'unknown')}[/dim]"
        )
    except Exception as e:
        console.print(f"[yellow]Warning: không thể kiểm tra Gemini packages: {e}[/yellow]")

def _probe_gemini_connectivity(key: str):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            api_key=key,
            google_api_key=key,
            **_gemini_token_arg(8),
        )
        resp = llm.invoke([HumanMessage(content="ping")])
        console.print(f"[green]✓ Gemini probe OK[/green]")
    except Exception as e:
        console.print(f"[yellow]Gemini probe failed (non-fatal): {e}[/yellow]")
class GeminiLLMWrapper:
    def __init__(self, google_api_key: str, max_tokens: int, temperature: float):
        self.google_api_key = google_api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._llm = None

    def _build_llm(self, model_name: str):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self._llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.google_api_key,
            google_api_key=self.google_api_key,
            temperature=self.temperature,
            **_gemini_token_arg(self.max_tokens),
        )

    def _is_quota_error(self, err: Exception) -> bool:
        s = str(err)
        return "RESOURCE_EXHAUSTED" in s or "quota" in s or "exceeded your current quota" in s or "429" in s

    def _is_not_found(self, err: Exception) -> bool:
        s = str(err)
        return "NOT_FOUND" in s or "is not found" in s or "not supported for generateContent" in s or "404" in s

    def invoke(self, messages):
        last_err = None
        saw_quota = False
        for model_name in GEMINI_FALLBACK_MODELS:
            try:
                self._build_llm(model_name)
                resp = self._llm.invoke(messages)
                MODELS["gemini"]["fast"] = MODELS["gemini"]["main"] = MODELS["gemini"]["analyst"] = model_name
                return resp
            except Exception as e:
                last_err = e
                if self._is_quota_error(e):
                    saw_quota = True
                    console.print(f"  [dim yellow]⚠ {model_name} failed (quota): {e}. Thử model tiếp...[/dim yellow]")
                    # if response includes retryDelay in message, you could parse and sleep here (optional)
                    continue
                if self._is_not_found(e):
                    console.print(f"  [dim yellow]⚠ Model không hỗ trợ / không tồn tại: {model_name} → bỏ qua.[/dim yellow]")
                    continue
                # unknown error: raise immediately
                raise
        # after loop
        if saw_quota:
            raise RuntimeError(
                "Tất cả model Gemini 2.0 đã thử nhưng gặp giới hạn quota (RESOURCE_EXHAUSTED). "
                "Hãy kiểm tra billing/quota cho project, hoặc sử dụng key khác / chờ retryDelay. "
                f"Last error: {last_err}"
            )
        raise RuntimeError(
            "Không có model Gemini khả dụng cho phương thức generateContent (tất cả model thử trả NOT_FOUND). "
            "Hãy gọi ModelService.ListModels trong Google API hoặc kiểm tra danh sách model được hỗ trợ."
        )

    def __getattr__(self, name):
        return getattr(self._llm, name)

class GroqLLMWrapper:
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from groq import Client
            self._client = Client(api_key=self.api_key)

    def invoke(self, messages):
        if self._client is None:
            self._ensure_client()
        normalized = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role") or m.get("type") or "user"
                content = m.get("content")
            else:
                role = getattr(m, "role", None) or getattr(m, "type", None) or "user"
                content = getattr(m, "content", None)
                if content is None:
                    content = getattr(m, "text", str(m))

            role = str(role).lower()
            if role == "human":
                role = "user"
            if role not in {"system", "user", "assistant"}:
                role = "user"

            normalized.append({"role": role, "content": content})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=normalized,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        text = ""
        if hasattr(response, "choices") and response.choices:
            first_choice = response.choices[0]
            message = getattr(first_choice, "message", None)
            if message is not None:
                text = getattr(message, "content", "") or ""
            elif isinstance(first_choice, dict):
                text = first_choice.get("message", {}).get("content", "") or ""
        else:
            text = str(response)

        class ResponseProxy:
            pass
        resp = ResponseProxy()
        resp.content = text
        return resp


def make_llm(role: str = "main", max_tokens: int = 1500, temperature: float = 0.4):
    """
    Factory tạo LLM theo PROVIDER.
    """
    if PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model             =get_model(role),
            anthropic_api_key =ANTHROPIC_KEY,
            max_tokens        =max_tokens,
            temperature       =temperature,
        )

    elif PROVIDER == "groq":
        return GroqLLMWrapper(
            api_key=GROQ_KEY,
            model=get_model(role),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    else:
        raise RuntimeError("Chưa cấu hình API Key. Kiểm tra file .env!")
        

# ══════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════
class AgentTiming(TypedDict):
    agent    : str
    start_ts : float
    end_ts   : float
    duration : float
    called_by: str
    calls    : List[str]

class ChatState(TypedDict):
    session_id           : str
    turn_id              : int
    user_query           : str
    intent               : str
    kb_context           : str
    web_results          : str
    calc_result          : str
    datetime_result      : str
    analysis             : str
    final_answer         : str
    conversation_history : Annotated[List[dict], operator.add]
    agent_timings        : Annotated[List[AgentTiming], operator.add]
    call_graph           : Annotated[List[dict], operator.add]
    log                  : Annotated[List[str], operator.add]

# ══════════════════════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════════════════════
class ChatPersistence:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.filepath   = CHAT_DIR / f"session_{session_id}.json"
        self._data      = self._load()

    def _load(self) -> dict:
        if self.filepath.exists():
            with open(self.filepath, encoding="utf-8") as f:
                return json.load(f)
        return {
            "session_id" : self.session_id,
            "created_at" : datetime.now().isoformat(),
            "provider"   : PROVIDER,
            "model_fast" : get_model("fast"),
            "model_main" : get_model("main"),
            "turns"      : [],
            "agent_stats": {},
        }

    def save_turn(self, turn: dict):
        self._data["turns"].append(turn)
        self._data["updated_at"] = datetime.now().isoformat()
        self._update_agent_stats(turn.get("agent_timings", []))
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2, default=str)

    def _update_agent_stats(self, timings: List[dict]):
        stats = self._data["agent_stats"]
        for t in timings:
            name = t["agent"]
            if name not in stats:
                stats[name] = {"calls": 0, "total_time": 0.0, "avg_time": 0.0, "max_time": 0.0}
            s = stats[name]
            s["calls"]      += 1
            s["total_time"] += t["duration"]
            s["avg_time"]    = s["total_time"] / s["calls"]
            s["max_time"]    = max(s["max_time"], t["duration"])

    def get_history_text(self, n: int = 6) -> str:
        turns = self._data["turns"][-n:]
        lines = []
        for t in turns:
            lines.append(f"Người dùng: {t['user_query']}")
            lines.append(f"Chatbot: {t['final_answer'][:200]}")
        return "\n".join(lines)

    def get_all_stats(self) -> dict:
        return self._data.get("agent_stats", {})

# ══════════════════════════════════════════════════════════════════
#  AGENT BASE
# ══════════════════════════════════════════════════════════════════
class AgentBase:
    NAME = "base"

    def __init__(self, called_by: str = "orchestrator"):
        self.called_by = called_by
        self._calls: List[str] = []

    def _record_call(self, target: str):
        self._calls.append(target)

    def run(self, **kwargs) -> dict:
        raise NotImplementedError

    def execute(self, **kwargs) -> Tuple[dict, AgentTiming]:
        t0       = time.perf_counter()
        ts_start = time.time()
        result   = self.run(**kwargs)
        duration = time.perf_counter() - t0
        timing   = AgentTiming(
            agent    =self.NAME,
            start_ts =ts_start,
            end_ts   =ts_start + duration,
            duration =round(duration, 4),
            called_by=self.called_by,
            calls    =list(self._calls),
        )
        return result, timing

# ══════════════════════════════════════════════════════════════════
#  AGENT 1 — Intent Classifier
# ══════════════════════════════════════════════════════════════════
class IntentClassifierAgent(AgentBase):
    NAME   = "IntentClassifier"
    LABELS = ["langgraph","crewai","react","memory","rag","multiagent",
              "calculate","datetime","search","analyze","greet","general"]

    def __init__(self):
        super().__init__(called_by="GraphEntryPoint")
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = make_llm(role="fast", max_tokens=30, temperature=0.0)
        return self._llm

    def run(self, query: str) -> dict:
        prompt = (
            f"Phân loại intent của câu hỏi sau thành MỘT nhãn:\n"
            f"{' | '.join(self.LABELS)}\n\n"
            f"Câu hỏi: \"{query}\"\n\n"
            f"Trả lời CHỈ nhãn, không thêm gì khác."
        )
        try:
            resp   = self._get_llm().invoke([HumanMessage(content=prompt)])
            intent = resp.content.strip().lower().split()[0]
        except Exception:
            intent = ""

        if intent not in self.LABELS:
            q = query.lower()
            if any(w in q for w in ["tính","cộng","trừ","nhân","chia","+","-","*","/","sqrt","^","bình phương"]): intent = "calculate"
            elif any(w in q for w in ["ngày","giờ","hôm nay","thời gian","mấy giờ","thứ mấy"]): intent = "datetime"
            elif any(w in q for w in ["tìm","search","tra cứu","tìm kiếm","tra"]): intent = "search"
            elif any(w in q for w in ["phân tích","so sánh","đánh giá","ưu","nhược","pros","cons"]): intent = "analyze"
            elif any(w in q for w in ["langgraph","graph","node","edge","state","stategraph"]): intent = "langgraph"
            elif any(w in q for w in ["crewai","crew"]): intent = "crewai"
            elif any(w in q for w in ["react","reasoning","acting"]): intent = "react"
            elif any(w in q for w in ["memory","bộ nhớ","nhớ"]): intent = "memory"
            elif any(w in q for w in ["rag","retrieval","vector"]): intent = "rag"
            elif any(w in q for w in ["multi-agent","multiagent","đa agent"]): intent = "multiagent"
            elif any(w in q for w in ["xin chào","hello","hi","chào","hey"]): intent = "greet"
            else: intent = "general"
        return {"intent": intent}

# ══════════════════════════════════════════════════════════════════
#  AGENT 2 — Knowledge Base
# ══════════════════════════════════════════════════════════════════
class KnowledgeBaseAgent(AgentBase):
    NAME = "KnowledgeBase"
    KB = {
        "langgraph": [
            "LangGraph xây dựng Multi-Agent dưới dạng Directed Graph. Node=Agent, Edge=luồng dữ liệu. State TypedDict chạy xuyên suốt.",
            "LangGraph hỗ trợ Human-in-the-loop: dừng graph giữa chừng, cho phép xác nhận, rồi chạy tiếp.",
            "Cơ chế Reducer (operator.add) xử lý Race Condition khi nhiều Node song song cùng ghi vào State.",
            "LangGraph phù hợp Production: tính tất định cao, State Observability, fallback node, parallel Fan-out/Fan-in.",
        ],
        "crewai": [
            "CrewAI dùng khai báo declarative: Agent+Task+Crew. Phù hợp PoC nhanh.",
            "CrewAI vs LangGraph: CrewAI dễ bắt đầu, LangGraph mạnh hơn khi production.",
            "async_execution=True cho Task song song trong CrewAI.",
        ],
        "react": [
            "ReAct=Reasoning+Acting: vòng Thought→Action→Observation cho đến Final Answer.",
            "Thought: LLM phân tích ngữ cảnh, quyết định gọi tool nào. Action: thực thi tool.",
            "Hard limit max_iterations=5 tránh vòng lặp vô hạn.",
        ],
        "memory": [
            "3 lớp bộ nhớ: Short-term (sliding window), Long-term (VectorDB+RAG), State (Blackboard JSON).",
            "Memory Dispatcher chỉ nạp thông tin cần thiết — không full-load. Giảm Hallucination.",
            "Async Write-back: tách luồng ghi Vector DB khỏi luồng phản hồi chính.",
        ],
        "rag": [
            "RAG=Retrieval-Augmented Generation: Embed→VectorDB→CosineSimilarity→top-k chunks→prompt.",
            "RAG giải quyết LLM outdated và context window giới hạn.",
            "Local: ChromaDB, FAISS. Cloud production: Pinecone, Weaviate.",
        ],
        "multiagent": [
            "Multi-Agent chia tác vụ phức tạp cho nhiều Agent chuyên biệt.",
            "4 mô hình: Sequential, Hierarchical, State-Graph (LangGraph), Conversational (AutoGen).",
            "Parallel execution cải thiện tốc độ 1.3x–4.15x theo benchmark.",
        ],
    }

    def __init__(self, called_by: str = "IntentClassifier"):
        super().__init__(called_by=called_by)

    def run(self, query: str, intent: str) -> dict:
        results = self.KB.get(intent, [])
        if not results:
            for k, v in self.KB.items():
                if k in query.lower():
                    results = v
                    break
        context = "\n".join(f"• {r}" for r in results) if results else ""
        return {"kb_context": context}

# ══════════════════════════════════════════════════════════════════
#  Calculator worker — top-level để pickle được trên Windows
# ══════════════════════════════════════════════════════════════════
def _calc_worker(expression: str) -> str:
    import math as _math, re as _re
    expr = expression
    for a, b in [("^","**"),("×","*"),("÷","/"),("√","_math.sqrt"),
                 ("π","_math.pi"),("sin","_math.sin"),("cos","_math.cos"),
                 ("tan","_math.tan"),("log","_math.log10"),("ln","_math.log"),
                 ("sqrt","_math.sqrt")]:
        expr = expr.replace(a, b)
    safe = _re.sub(r'[^0-9\+\-\*\/\.\(\)\s\_a-zA-Z]', '', expr)
    try:
        ns     = {"__builtins__": {}, "_math": _math}
        result = eval(safe, ns)
        if isinstance(result, float):
            return f"{expression.strip()} = {result:,.10g}"
        return f"{expression.strip()} = {result:,}"
    except Exception as e:
        return f"Lỗi tính toán: {e}"

# ══════════════════════════════════════════════════════════════════
#  AGENT 3 — Calculator
# ══════════════════════════════════════════════════════════════════
class CalculatorAgent(AgentBase):
    NAME = "Calculator"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    def run(self, query: str) -> dict:
        m    = re.search(r'[\d][\d\s\+\-\*\/\^\(\)\.×÷√πsincostan]+[\d\)]', query)
        expr = m.group(0).strip() if m else query
        try:
            with ProcessPoolExecutor(max_workers=1) as pool:
                result = pool.submit(_calc_worker, expr).result(timeout=8)
        except Exception:
            result = _calc_worker(expr)
        return {"calc_result": result}

# ══════════════════════════════════════════════════════════════════
#  AGENT 4 — DateTime
# ══════════════════════════════════════════════════════════════════
class DateTimeAgent(AgentBase):
    NAME = "DateTime"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    def run(self, **kwargs) -> dict:
        now    = datetime.now()
        days   = ["Thứ Hai","Thứ Ba","Thứ Tư","Thứ Năm","Thứ Sáu","Thứ Bảy","Chủ Nhật"]
        months = ["","tháng 1","tháng 2","tháng 3","tháng 4","tháng 5","tháng 6",
                  "tháng 7","tháng 8","tháng 9","tháng 10","tháng 11","tháng 12"]
        result = (
            f"**{days[now.weekday()]}**, ngày {now.day} {months[now.month]} {now.year}\n"
            f"Giờ: **{now.strftime('%H:%M:%S')}** (UTC+7 — Việt Nam)\n"
            f"Tuần thứ: {now.isocalendar()[1]}"
        )
        return {"datetime_result": result}

# ══════════════════════════════════════════════════════════════════
#  AGENT 5 — Web Search
# ══════════════════════════════════════════════════════════════════
class WebSearchAgent(AgentBase):
    NAME = "WebSearch"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    def run(self, query: str) -> dict:
        if not TAVILY_KEY:
            return {"web_results": "[Tavily chưa cấu hình — thêm TAVILY_API_KEY vào .env để tìm kiếm web]"}
        try:
            from tavily import TavilyClient
            r     = TavilyClient(api_key=TAVILY_KEY).search(
                        query=query, max_results=4,
                        include_answer=True, search_depth="advanced")
            lines = []
            if r.get("answer"):
                lines.append(f"📌 **Tóm tắt:** {r['answer']}\n")
            for i, item in enumerate(r.get("results", [])[:3], 1):
                lines.append(f"[{i}] **{item.get('title','')}**")
                lines.append(f"    {item.get('content','')[:300]}")
                lines.append(f"    *{item.get('url','')}*\n")
            return {"web_results": "\n".join(lines) or "Không có kết quả."}
        except Exception as e:
            return {"web_results": f"Web search lỗi: {e}"}

# ══════════════════════════════════════════════════════════════════
#  AGENT 6 — Analyst
# ══════════════════════════════════════════════════════════════════
class AnalystAgent(AgentBase):
    NAME = "Analyst"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)
        self._record_call(f"{PROVIDER.title()}-LLM (sub-call)")
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = make_llm(role="analyst", max_tokens=600, temperature=0.3)
        return self._llm

    def run(self, topic: str, context: str = "") -> dict:
        system = "Bạn là chuyên gia phân tích AI Agent. Phân tích ngắn gọn, có cấu trúc, tiếng Việt."
        prompt = f"Phân tích chủ đề: {topic}"
        if context:
            prompt += f"\n\nNgữ cảnh:\n{context}"
        resp = self._get_llm().invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        return {"analysis": resp.content}

# ══════════════════════════════════════════════════════════════════
#  AGENT 7 — Router / Dispatcher
# ══════════════════════════════════════════════════════════════════
class RouterAgent(AgentBase):
    NAME = "Router"

    def __init__(self):
        super().__init__(called_by="GraphNode")

    def run(self, query: str, intent: str, kb_context: str = "") -> dict:
        self._calls = []  # reset mỗi turn
        results     = {}

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}

            if intent == "calculate":
                self._record_call("Calculator")
                agent = CalculatorAgent(called_by=self.NAME)
                futures["calc"] = (pool.submit(agent.run, query=query), agent)

            elif intent == "datetime":
                self._record_call("DateTime")
                agent = DateTimeAgent(called_by=self.NAME)
                futures["dt"] = (pool.submit(agent.run), agent)

            elif intent == "search":
                self._record_call("WebSearch")
                agent = WebSearchAgent(called_by=self.NAME)
                futures["web"] = (pool.submit(agent.run, query=query), agent)

            elif intent == "analyze":
                self._record_call("WebSearch")
                self._record_call("Analyst")
                ws = WebSearchAgent(called_by=self.NAME)
                an = AnalystAgent(called_by=self.NAME)
                futures["web"]  = (pool.submit(ws.run, query=query), ws)
                futures["anal"] = (pool.submit(an.run, topic=query, context=kb_context), an)

            elif intent in ("langgraph","crewai","react","memory","rag","multiagent"):
                self._record_call("KnowledgeBase")
                if TAVILY_KEY:
                    self._record_call("WebSearch")
                    ws = WebSearchAgent(called_by=self.NAME)
                    futures["web"] = (pool.submit(ws.run, query=query), ws)

            # greet / general: không cần agent thêm

            sub_timings: List[AgentTiming] = []
            for key, (fut, agt) in futures.items():
                try:
                    ts_s = time.time()
                    t0   = time.perf_counter()
                    r    = fut.result(timeout=30)
                    dur  = time.perf_counter() - t0
                    results.update(r)
                    sub_timings.append(AgentTiming(
                        agent    =agt.NAME,
                        start_ts =ts_s,
                        end_ts   =ts_s + dur,
                        duration =round(dur, 4),
                        called_by=self.NAME,
                        calls    =agt._calls,
                    ))
                except Exception as e:
                    results[key] = f"Lỗi agent: {e}"

        results["_sub_timings"] = sub_timings
        return results

# ══════════════════════════════════════════════════════════════════
#  AGENT 8 — Response Generator
# ══════════════════════════════════════════════════════════════════
class ResponseGeneratorAgent(AgentBase):
    NAME = "ResponseGenerator"

    def __init__(self):
        super().__init__(called_by="GraphNode")
        self._record_call(f"{PROVIDER.title()}-LLM (main)")
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = make_llm(role="main", max_tokens=1500, temperature=0.4)
        return self._llm

    def run(self, query: str, intent: str = "", kb_context: str = "",
            web_results: str = "", calc_result: str = "",
            datetime_result: str = "", analysis: str = "",
            history: str = "") -> dict:

        system = (
            "Bạn là trợ lý AI thông minh, trả lời tiếng Việt rõ ràng và có cấu trúc. "
            "Sử dụng Markdown cho định dạng. Dựa vào ngữ cảnh được cung cấp."
        )
        parts = []
        if kb_context:      parts.append(f"**Knowledge Base:**\n{kb_context}")
        if web_results:     parts.append(f"**Kết quả tìm kiếm web:**\n{web_results}")
        if calc_result:     parts.append(f"**Kết quả tính toán:**\n{calc_result}")
        if datetime_result: parts.append(f"**Thời gian:**\n{datetime_result}")
        if analysis:        parts.append(f"**Phân tích chuyên sâu:**\n{analysis}")
        if history:         parts.append(f"**Lịch sử hội thoại gần đây:**\n{history}")

        context_block = "\n\n---\n\n".join(parts)
        prompt = (
            f"Câu hỏi: {query}\n\n"
            + (f"Ngữ cảnh:\n{context_block}\n\n" if context_block else "")
            + "Hãy trả lời đầy đủ và rõ ràng."
        )
        resp = self._get_llm().invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        return {"final_answer": resp.content}

# ══════════════════════════════════════════════════════════════════
#  GLOBAL INSTANCES (lazy LLM bên trong — an toàn)
# ══════════════════════════════════════════════════════════════════
_intent_agent   = IntentClassifierAgent()
_kb_agent       = KnowledgeBaseAgent()
_router_agent   = RouterAgent()
_response_agent = ResponseGeneratorAgent()
db: Optional[ChatPersistence] = None

# ══════════════════════════════════════════════════════════════════
#  LANGGRAPH NODES
# ══════════════════════════════════════════════════════════════════
def node_intent_classifier(state: ChatState) -> dict:
    result, timing = _intent_agent.execute(query=state["user_query"])
    return {
        **result,
        "agent_timings": [timing],
        "call_graph"   : [{"from": "GraphEntry", "to": "IntentClassifier", "ts": timing["start_ts"]}],
        "log"          : [f"[IntentClassifier] intent='{result['intent']}' ({timing['duration']:.3f}s)"],
    }

def node_kb_retriever(state: ChatState) -> dict:
    result, timing = _kb_agent.execute(query=state["user_query"], intent=state["intent"])
    return {
        **result,
        "agent_timings": [timing],
        "call_graph"   : [{"from": "IntentClassifier", "to": "KnowledgeBase", "ts": timing["start_ts"]}],
        "log"          : [f"[KnowledgeBase] {'found' if result['kb_context'] else 'empty'} ({timing['duration']:.3f}s)"],
    }

def node_router(state: ChatState) -> dict:
    ts_start = time.time()
    t0       = time.perf_counter()
    raw      = _router_agent.run(
        query     =state["user_query"],
        intent    =state["intent"],
        kb_context=state.get("kb_context", ""),
    )
    dur         = time.perf_counter() - t0
    sub_timings = raw.pop("_sub_timings", [])
    router_timing = AgentTiming(
        agent="Router", start_ts=ts_start, end_ts=ts_start+dur,
        duration=round(dur,4), called_by="KnowledgeBase", calls=list(_router_agent._calls),
    )
    cg   = [{"from": "KnowledgeBase", "to": "Router", "ts": ts_start}]
    cg  += [{"from": "Router", "to": t["agent"], "ts": t["start_ts"]} for t in sub_timings]
    logs = [f"[Router] → {[t['agent'] for t in sub_timings] or ['(none)'] } ({dur:.3f}s)"]
    for t in sub_timings:
        logs.append(f"  ↳ [{t['agent']}] {t['duration']:.3f}s")
    return {**raw, "agent_timings": [router_timing]+sub_timings, "call_graph": cg, "log": logs}

def node_response_generator(state: ChatState) -> dict:
    history = db.get_history_text(4) if db is not None else ""
    result, timing = _response_agent.execute(
        query          =state["user_query"],
        intent         =state.get("intent",""),
        kb_context     =state.get("kb_context",""),
        web_results    =state.get("web_results",""),
        calc_result    =state.get("calc_result",""),
        datetime_result=state.get("datetime_result",""),
        analysis       =state.get("analysis",""),
        history        =history,
    )
    return {
        **result,
        "agent_timings": [timing],
        "call_graph"   : [{"from": "Router", "to": "ResponseGenerator", "ts": timing["start_ts"]}],
        "log"          : [f"[ResponseGenerator] ({timing['duration']:.3f}s)"],
    }

def node_persistence(state: ChatState) -> dict:
    if db is not None:
        db.save_turn({
            "turn_id"      : state["turn_id"],
            "timestamp"    : datetime.now().isoformat(),
            "user_query"   : state["user_query"],
            "intent"       : state["intent"],
            "final_answer" : state["final_answer"],
            "agent_timings": state["agent_timings"],
            "call_graph"   : state["call_graph"],
        })
    return {"log": [f"[Persistence] saved turn {state['turn_id']}"]}

# ══════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ══════════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(ChatState)
    g.add_node("intent_classifier",  node_intent_classifier)
    g.add_node("kb_retriever",       node_kb_retriever)
    g.add_node("router",             node_router)
    g.add_node("response_generator", node_response_generator)
    g.add_node("persistence",        node_persistence)
    g.set_entry_point("intent_classifier")
    g.add_edge("intent_classifier",  "kb_retriever")
    g.add_edge("kb_retriever",       "router")
    g.add_edge("router",             "response_generator")
    g.add_edge("response_generator", "persistence")
    g.add_edge("persistence",        END)
    return g.compile()

# ══════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════
def _provider_badge() -> str:
    if PROVIDER == "anthropic":
        return "[bold white on blue] 🤖 Anthropic Claude [/bold white on blue]"
    elif PROVIDER == "gemini":
        return "[bold white on dark_green] 🟢 Google Gemini [/bold white on dark_green]"
    return "[bold white on red] ⚠ No Provider [/bold white on red]"

def display_timing_analysis(timings: List[AgentTiming], total_elapsed: float):
    table = Table(title="⏱  Agent Timing Analysis", box=box.ROUNDED,
                  border_style="cyan", show_lines=True)
    table.add_column("#",         style="dim",     width=3)
    table.add_column("Agent",     style="yellow",  width=22)
    table.add_column("Called By", style="blue",    width=22)
    table.add_column("Calls",     style="green",   width=28)
    table.add_column("Duration",  style="magenta", width=10, justify="right")
    table.add_column("% Total",   style="cyan",    width=11, justify="right")
    for i, t in enumerate(timings, 1):
        pct       = (t["duration"] / total_elapsed * 100) if total_elapsed > 0 else 0
        bar       = "█" * int(pct/10) + "░" * (10-int(pct/10))
        calls_str = ", ".join(t.get("calls",[])) or "—"
        table.add_row(str(i), t["agent"], t["called_by"],
                      calls_str[:28], f"{t['duration']:.3f}s", f"{bar} {pct:.0f}%")
    console.print(table)
    console.print(f"  [dim]Tổng: [bold]{total_elapsed:.3f}s[/bold][/dim]\n")

def display_call_graph(call_graph: List[dict], timings: List[AgentTiming]):
    edges: dict = {}
    for e in call_graph:
        edges.setdefault(e["from"], []).append(e["to"])
    timing_map  = {t["agent"]: t["duration"] for t in timings}
    tree        = Tree("[bold cyan]📊 Agent Call Graph[/bold cyan]", guide_style="dim cyan")
    all_targets = {e["to"] for e in call_graph}
    roots       = list(dict.fromkeys(e["from"] for e in call_graph if e["from"] not in all_targets))
    if not roots: roots = ["GraphEntry"]

    def add_children(node, agent: str, visited: set):
        for child in edges.get(agent, []):
            if child in visited: continue
            visited.add(child)
            dur = timing_map.get(child, 0)
            branch = node.add(f"[yellow]{child}[/yellow] [dim]({dur:.3f}s)[/dim]")
            add_children(branch, child, visited)

    visited_g: set = set()
    for root in roots:
        visited_g.add(root)
        rb = tree.add(f"[green]{root}[/green]")
        add_children(rb, root, visited_g)
    console.print(tree)

def display_log(logs: List[str]):
    table = Table(box=box.SIMPLE, border_style="dim", title="📋 Execution Log")
    table.add_column("#",    style="dim cyan", width=4)
    table.add_column("Step", style="white")
    for i, log in enumerate(logs, 1):
        table.add_row(str(i), log)
    console.print(table)

def display_cumulative_stats():
    if db is None: return
    stats = db.get_all_stats()
    if not stats: return
    table = Table(title=f"📈 Agent Stats — {db.filepath.name}",
                  box=box.ROUNDED, border_style="blue")
    table.add_column("Agent",      style="yellow", width=22)
    table.add_column("Calls",      style="cyan",   width=8,  justify="right")
    table.add_column("Total",      style="green",  width=10, justify="right")
    table.add_column("Avg",        style="magenta",width=10, justify="right")
    table.add_column("Max",        style="red",    width=10, justify="right")
    for agent, s in sorted(stats.items(), key=lambda x: -x[1]["total_time"]):
        table.add_row(agent, str(s["calls"]),
                      f"{s['total_time']:.3f}s", f"{s['avg_time']:.3f}s", f"{s['max_time']:.3f}s")
    console.print(table)
    console.print(f"  [dim]File: {db.filepath}[/dim]\n")

def display_welcome():
    model_info = (
        f"[cyan]Model Fast:[/cyan]  {get_model('fast')}\n"
        f"[cyan]Model Main:[/cyan]  {get_model('main')}"
    )
    console.print()
    console.print(Panel.fit(
        f"[bold blue]🤖 LANGGRAPH MULTI-AGENT CHATBOT v3.2[/bold blue]\n"
        f"[dim]Dual Provider: Anthropic Claude + Google Gemini[/dim]\n\n"
        f"Provider: {_provider_badge()}\n"
        f"{model_info}\n"
        f"[cyan]Web Search:[/cyan]  {'Tavily ✓' if TAVILY_KEY else 'Tắt — thêm TAVILY_API_KEY vào .env'}\n"
        f"[cyan]Lưu chat:[/cyan]   {CHAT_DIR}/session_*.json\n\n"
        "[dim]Lệnh: [bold]graph[/bold] · [bold]timing[/bold] · [bold]stats[/bold] · [bold]help[/bold] · [bold]quit[/bold][/dim]",
        border_style="blue",
        title="[bold]Hoàng Minh Đức — Thực tập sinh AI Agent 2026[/bold]",
    ))
    console.print()

def display_graph_diagram():
    console.print(Panel(
        "[bold cyan]LANGGRAPH WORKFLOW v3.2[/bold cyan]\n\n"
        "  [START]\n    │\n    ▼\n"
        "  [yellow]① IntentClassifier[/yellow]  ← LLM fast (Haiku / Gemini Flash)\n    │\n    ▼\n"
        "  [yellow]② KnowledgeBase[/yellow]     ← Pure Python RAG\n    │\n    ▼\n"
        "  [yellow]③ Router[/yellow]            ← ThreadPool Dispatcher\n"
        "    │    ╔══════════════════════════════════╗\n"
        "    ├───►║ WebSearch  (Thread, Tavily API)  ║\n"
        "    ├───►║ Calculator (Process, eval)       ║ ← Song song\n"
        "    ├───►║ DateTime   (Python inline)       ║\n"
        "    └───►║ Analyst    (Thread, LLM sub-call)║\n"
        "         ╚══════════════════════════════════╝\n    ▼\n"
        "  [yellow]④ ResponseGenerator[/yellow] ← LLM main (Sonnet / Gemini Flash)\n    │\n    ▼\n"
        "  [yellow]⑤ Persistence[/yellow]       ← JSON file\n   [END]",
        border_style="cyan", title="[bold]Kiến trúc Graph[/bold]",
    ))

def display_help():
    console.print(Panel(
        "[bold]GỢI Ý CÂU HỎI:[/bold]\n\n"
        "[cyan]• LangGraph là gì? So sánh với CrewAI[/cyan]\n"
        "[cyan]• Giải thích cơ chế ReAct từng bước[/cyan]\n"
        "[cyan]• Tính 1024 * 768 + sqrt(2048)[/cyan]\n"
        "[cyan]• Hôm nay là thứ mấy, ngày mấy?[/cyan]\n"
        "[cyan]• Tìm kiếm thông tin về LangGraph 2025[/cyan]\n"
        "[cyan]• Phân tích ưu nhược điểm Multi-Agent[/cyan]\n\n"
        "[bold]LỆNH:[/bold]\n"
        "[yellow]graph[/yellow]   — Sơ đồ kiến trúc Nodes\n"
        "[yellow]timing[/yellow]  — Timing turn vừa rồi\n"
        "[yellow]stats[/yellow]   — Thống kê toàn phiên\n"
        "[yellow]help[/yellow]    — Hướng dẫn này\n"
        "[yellow]quit[/yellow]    — Thoát\n",
        border_style="green", title="[bold]📚 HƯỚNG DẪN[/bold]",
    ))

# ══════════════════════════════════════════════════════════════════
#  PROVIDER SETUP (chạy khi khởi động nếu chưa có key)
# ══════════════════════════════════════════════════════════════════
def setup_provider():
    """Hỏi user chọn provider nếu .env chưa có key nào."""
    global PROVIDER, ANTHROPIC_KEY, GEMINI_KEY

    if PROVIDER != "none":
        return  # đã có key từ .env

    console.print(Panel(
        "[bold yellow]⚠  Chưa tìm thấy API Key trong file .env[/bold yellow]\n\n"
        "Chọn provider để tiếp tục:\n"
        "  [cyan]1[/cyan] — Anthropic Claude (trả phí, chất lượng cao)\n"
        "  [cyan]2[/cyan] — Google Gemini   (miễn phí, 15 req/phút)\n",
        border_style="yellow", title="[bold]Chọn Provider[/bold]",
    ))

    choice = console.input("[yellow]Nhập 1 hoặc 2: [/yellow]").strip()

    if choice == "1":
        ANTHROPIC_KEY = console.input("[cyan]Nhập Anthropic API Key (sk-ant-...): [/cyan]").strip()
        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_KEY
        PROVIDER = "anthropic"
        # Ghi vào .env
        env_path = Path(".env")
        content  = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        if "ANTHROPIC_API_KEY" not in content:
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(f"\nANTHROPIC_API_KEY={ANTHROPIC_KEY}\n")
        console.print("[green]✓ Đã lưu Anthropic Key vào .env[/green]")

    else:
        GEMINI_KEY = console.input("[cyan]Nhập Google Gemini API Key (AIza...): [/cyan]").strip()
        os.environ["GEMINI_API_KEY"] = GEMINI_KEY
        PROVIDER = "gemini"
        env_path = Path(".env")
        content  = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        if "GEMINI_API_KEY" not in content:
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(f"\nGEMINI_API_KEY={GEMINI_KEY}\n")
        console.print("[green]✓ Đã lưu Gemini Key vào .env[/green]")
        console.print("[dim]Cài thêm nếu chưa có: pip install langchain-google-genai[/dim]")

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    global db

    # Bước 1: Đảm bảo có provider
    setup_provider()

    if PROVIDER == "none":
        console.print("[red]❌ Không có API Key. Thoát.[/red]")
        return

    _check_gemini_packages()
    if PROVIDER == "gemini" and GEMINI_KEY:
        _probe_gemini_connectivity(GEMINI_KEY)

    # Bước 2: Khởi tạo DB và Graph
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    db         = ChatPersistence(session_id)

    console.print(f"\n[dim]Biên dịch LangGraph (provider={PROVIDER})...[/dim]")
    graph = build_graph()
    console.print(f"[green]✓ Graph sẵn sàng — 5 Nodes | {db.filepath}[/green]\n")

    display_welcome()

    last_timings:    List[AgentTiming] = []
    last_call_graph: List[dict]        = []
    last_total:      float             = 0.0
    turn = 0

    while True:
        try:
            console.rule(f"[dim]Turn {turn+1}[/dim]", style="dim")
            query = console.input("[bold green]Bạn[/bold green] › ").strip()
            if not query: continue

            cmd = query.lower()
            if cmd in ("quit","exit","thoát","q"):
                console.print(f"\n[bold blue]👋 Kết thúc phiên.[/bold blue]")
                console.print(f"  File: [underline]{db.filepath}[/underline]\n")
                display_cumulative_stats()
                break
            if cmd == "graph":  display_graph_diagram(); continue
            if cmd == "timing":
                if last_timings:
                    display_timing_analysis(last_timings, last_total)
                    display_call_graph(last_call_graph, last_timings)
                else:
                    console.print("[dim]Chưa có dữ liệu. Hãy hỏi câu đầu tiên.[/dim]")
                continue
            if cmd == "stats": display_cumulative_stats(); continue
            if cmd == "help":  display_help(); continue

            turn += 1
            initial: ChatState = {
                "session_id": session_id, "turn_id": turn,
                "user_query": query, "intent": "",
                "kb_context": "", "web_results": "", "calc_result": "",
                "datetime_result": "", "analysis": "", "final_answer": "",
                "conversation_history": [], "agent_timings": [],
                "call_graph": [], "log": [],
            }

            with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                          TimeElapsedColumn(), console=console, transient=True) as prog:
                prog.add_task(f"Đang xử lý [{PROVIDER}]...", total=None)
                t0      = time.perf_counter()
                result  = graph.invoke(initial)
                elapsed = time.perf_counter() - t0

            last_timings    = result["agent_timings"]
            last_call_graph = result["call_graph"]
            last_total      = elapsed

            display_log(result["log"])
            display_timing_analysis(last_timings, elapsed)
            display_call_graph(last_call_graph, last_timings)

            console.print(Panel(
                Markdown(result["final_answer"]),
                border_style="blue",
                title=(f"[bold blue]🤖 Chatbot[/bold blue]  "
                       f"[dim]{_provider_badge()} intent={result['intent']} | {elapsed:.2f}s[/dim]"),
                padding=(1, 2),
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Ctrl+C — gõ 'quit' để thoát.[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Lỗi: {e}[/red]")
            traceback.print_exc()
# ══════════════════════════════════════════════════════════════════
#  EXPORTED API FOR WEB SERVER
# ══════════════════════════════════════════════════════════════════
# Ensure graph and db are accessible for web app import

def get_chatbot_graph():
    """Return compiled LangGraph instance (lazy init)."""
    global db
    if db is None:
        db = ChatPersistence(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return build_graph()

# Make setup accessible
__all__ = [
    'build_graph',
    'ChatState',
    'ChatPersistence',
    'PROVIDER',
    'GROQ_KEY',
    'ANTHROPIC_KEY',
    '_check_gemini_packages',
    'get_chatbot_graph',
]

# Guard main() to avoid running when imported
# BẮT BUỘC trên Windows — ProcessPoolExecutor cần guard này
if __name__ == "__main__":
    main()