"""
╔══════════════════════════════════════════════════════════════════════╗
║   LANGGRAPH MULTI-AGENT CHATBOT v3.1 — FIXED                        ║
║   Tác giả : Hoàng Minh Đức — Thực tập sinh AI Agent 2026            ║
║                                                                       ║
║   SỬA LỖI v3.1:                                                      ║
║   [F1] tuple[dict,AgentTiming] → Tuple[dict,AgentTiming] (Py<3.9)  ║
║   [F2] ProcessPoolExecutor Windows cần if __name__=="__main__"      ║
║   [F3] CalculatorAgent pool tạo mới mỗi lần (tránh spawn lỗi)      ║
║   [F4] _router_agent._calls tích luỹ qua turns → reset mỗi turn    ║
║   [F5] node_response_generator gọi db trước khi db khởi tạo        ║
║   [F6] Agent khởi tạo ở module-level gọi API khi import            ║
║   [F7] Manager/Queue import thừa gây warning                        ║
║   [F8] ChatState thiếu default cho các trường optional              ║
║   [F9] display_call_graph không render nếu GraphEntry không có edge ║
║                                                                       ║
║   Cài đặt:                                                            ║
║     pip install langgraph langchain-anthropic langchain-core         ║
║                 tavily-python rich python-dotenv                      ║
║                                                                       ║
║   Chạy:  python multi_agent_chatbot_v3.py                            ║
║   File lưu: ./chat_history/session_<timestamp>.json                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════
import os, re, math, json, time, random, operator, traceback
from datetime import datetime
from pathlib import Path
# [F1] FIX: Dùng typing.Tuple thay vì tuple[...] để tương thích Python 3.8/3.9
from typing import TypedDict, Annotated, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# [F7] FIX: Bỏ Manager, Queue, Process — không dùng trực tiếp
import threading

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
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

console = Console()

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GROQ_KEY      = os.getenv("GROQ_API_KEY", "").strip()
TAVILY_KEY    = os.getenv("TAVILY_API_KEY", "").strip()

# Tự chọn provider còn credit
USE_GROQ = bool(GROQ_KEY and not ANTHROPIC_KEY)

MODEL_FAST    = "claude-haiku-4-5-20251001" if not USE_GROQ else "compound-beta-mini"
MODEL_MAIN    = "claude-sonnet-4-20250514"  if not USE_GROQ else "compound-beta"
MODEL_ANALYST = MODEL_MAIN
GROQ_MAX_TOKENS = 600
GROQ_PROMPT_SECTION_LIMIT = 1000
GROQ_PROMPT_TOTAL_LIMIT = 3000
GROQ_FALLBACK_MODEL = "compound-beta-mini"
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
GROQ_BACKOFF_BASE = float(os.getenv("GROQ_BACKOFF_BASE", "1.0"))

CHAT_DIR = Path(os.path.dirname(__file__)) / "chat_history"
CHAT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_chat_messages(messages):
    normalized = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role") or m.get("type") or "user"
            content = m.get("content")
        else:
            role = getattr(m, "role", None) or getattr(m, "type", None) or "user"
            content = getattr(m, "content", None)
            if content is None:
                if hasattr(m, "text"):
                    content = m.text
                else:
                    content = str(m)

        role = str(role).lower()
        if role == "human":
            role = "user"
        if role not in {"system", "user", "assistant"}:
            role = "user"

        normalized.append({"role": role, "content": content})
    return normalized


def _truncate_text(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text
    return text[: max_chars - 36].rstrip() + "\n\n...[Nội dung đã cắt để giảm kích thước prompt]"


class GroqLLMWrapper:
    _last_instance = None

    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._debug_state = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "initial_model": model,
            "initial_max_tokens": max_tokens,
            "attempts": [],
            "last_model": model,
            "started_at": None,
            "ended_at": None,
            "completed": False,
            "success": False,
            "final_model": None,
            "error": None,
        }
        GroqLLMWrapper._last_instance = self

    @classmethod
    def get_last_debug_state(cls) -> dict:
        if cls._last_instance is None:
            return {}
        return cls._last_instance._debug_state

    def _ensure_client(self):
        if self._client is None:
            from groq import Client
            self._client = Client(api_key=self.api_key)

    def _handle_error(self, err):
        # Return a normalized error string for classification
        return str(err).lower()

    def _fallback_to_anthropic(self, messages):
        if not ANTHROPIC_KEY:
            return None
        from langchain_anthropic import ChatAnthropic

        console.print("[yellow]Groq quota exceeded và Anthropic key đã được cấu hình. Chuyển sang Anthropic để tiếp tục...[/yellow]")
        anthropic = ChatAnthropic(
            model=MODEL_FAST,
            anthropic_api_key=ANTHROPIC_KEY,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return anthropic.invoke(messages)

    def _classify_groq_error(self, err_text: str) -> Tuple[str, str]:
        kind = "unknown"
        details = err_text
        if "rate_limit_exceeded" in err_text or "rate limit" in err_text:
            if "requests per minute" in err_text or "rpm" in err_text:
                kind = "requests_per_minute"
            elif "requests per day" in err_text or "rpd" in err_text or ("requests" in err_text and "per day" in err_text):
                kind = "requests_per_day"
            elif "tokens per day" in err_text or "tpd" in err_text or ("tokens" in err_text and "per day" in err_text):
                kind = "tokens_per_day"
            else:
                kind = "rate_limit"
        elif "request entity too large" in err_text or "request_too_large" in err_text or "413" in err_text:
            kind = "request_too_large"
        return kind, details

    def _raise_quota_error(self, kind: str, details: str):
        self._debug_state["error"] = details
        self._debug_state["completed"] = True
        self._debug_state["final_model"] = self._debug_state.get("last_model")
        self._debug_state["ended_at"] = datetime.utcnow().isoformat() + "Z"
        raise GroqQuotaExceededError(kind, details)

    def invoke(self, messages):
        self._ensure_client()
        payload = _normalize_chat_messages(messages)

        model = self.model
        max_tokens = self.max_tokens
        last_err_text = None
        attempt = 0
        self._debug_state["started_at"] = datetime.utcnow().isoformat() + "Z"
        self._debug_state["last_model"] = model

        def record_attempt(action: str, kind: Optional[str] = None, details: Optional[str] = None):
            self._debug_state["attempts"].append({
                "attempt": attempt,
                "model": model,
                "max_tokens": max_tokens,
                "action": action,
                "error_kind": kind,
                "error_details": details,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
            self._debug_state["last_model"] = model

        while attempt <= GROQ_MAX_RETRIES:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=payload,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
            except Exception as e:
                err_text = self._handle_error(e)
                last_err_text = err_text
                if "rate_limit_exceeded" in err_text or "rate limit" in err_text:
                    kind, details = self._classify_groq_error(err_text)
                    record_attempt("rate_limit_error", kind, details)
                    console.print(
                        f"[yellow]Groq rate limit detected ({kind}) for model={model}. attempt={attempt}. error={details}[/yellow]"
                    )
                    if kind in {"tokens_per_day", "requests_per_day"}:
                        if ANTHROPIC_KEY:
                            return self._fallback_to_anthropic(payload)
                        self._raise_quota_error(kind, details)
                    if kind == "requests_per_minute" and attempt < GROQ_MAX_RETRIES:
                        backoff = GROQ_BACKOFF_BASE * (2 ** attempt) + random.random()
                        console.print(f"[yellow]Retrying after backoff {backoff:.2f}s (RPM limit)...[/yellow]")
                        time.sleep(backoff)
                        attempt += 1
                        continue
                    if model != GROQ_FALLBACK_MODEL:
                        console.print(
                            f"[yellow]Switching from {model} to fallback {GROQ_FALLBACK_MODEL} and retrying...[/yellow]"
                        )
                        model = GROQ_FALLBACK_MODEL
                        max_tokens = min(max_tokens, GROQ_MAX_TOKENS)
                        attempt += 1
                        continue
                    if attempt < GROQ_MAX_RETRIES:
                        backoff = GROQ_BACKOFF_BASE * (2 ** attempt) + random.random()
                        console.print(f"[yellow]Retrying after backoff {backoff:.2f}s...[/yellow]")
                        time.sleep(backoff)
                        attempt += 1
                        continue
                    raise RuntimeError(
                        f"Groq rate limit exceeded ({kind}). {details}"
                    )
                if "request entity too large" in err_text or "request_too_large" in err_text or "413" in err_text:
                    record_attempt("request_too_large", "request_too_large", err_text)
                    if model != GROQ_FALLBACK_MODEL:
                        console.print(
                            f"[yellow]Groq request too large detected for {model}. Retrying with smaller model and prompt...[/yellow]"
                        )
                        model = GROQ_FALLBACK_MODEL
                        max_tokens = min(max_tokens, GROQ_MAX_TOKENS)
                        attempt += 1
                        continue
                    self._debug_state["error"] = err_text
                    self._debug_state["completed"] = True
                    self._debug_state["ended_at"] = datetime.utcnow().isoformat() + "Z"
                    raise RuntimeError(
                        "Groq request quá lớn. Hệ thống đã cố gắng giảm kích thước yêu cầu nhưng vẫn bị giới hạn."
                    )
                raise RuntimeError(str(e))

            if getattr(response, "error", None):
                err_text = self._handle_error(response.error)
                last_err_text = err_text
                if "rate_limit_exceeded" in err_text or "rate limit" in err_text:
                    kind, details = self._classify_groq_error(err_text)
                    record_attempt("response_rate_limit_error", kind, details)
                    console.print(
                        f"[yellow]Groq rate limit detected ({kind}) for model={model}. attempt={attempt}. error={details}[/yellow]"
                    )
                    if kind in {"tokens_per_day", "requests_per_day"}:
                        if ANTHROPIC_KEY:
                            return self._fallback_to_anthropic(payload)
                        self._raise_quota_error(kind, details)
                    if kind == "requests_per_minute" and attempt < GROQ_MAX_RETRIES:
                        backoff = GROQ_BACKOFF_BASE * (2 ** attempt) + random.random()
                        console.print(f"[yellow]Retrying after backoff {backoff:.2f}s (RPM limit)...[/yellow]")
                        time.sleep(backoff)
                        attempt += 1
                        continue
                    if model != GROQ_FALLBACK_MODEL:
                        console.print(
                            f"[yellow]Switching from {model} to fallback {GROQ_FALLBACK_MODEL} and retrying...[/yellow]"
                        )
                        model = GROQ_FALLBACK_MODEL
                        max_tokens = min(max_tokens, GROQ_MAX_TOKENS)
                        attempt += 1
                        continue
                    if attempt < GROQ_MAX_RETRIES:
                        backoff = GROQ_BACKOFF_BASE * (2 ** attempt) + random.random()
                        console.print(f"[yellow]Retrying after backoff {backoff:.2f}s...[/yellow]")
                        time.sleep(backoff)
                        attempt += 1
                        continue
                    raise RuntimeError(
                        f"Groq rate limit exceeded ({kind}). {details}"
                    )
                if isinstance(response, dict) and response.get("error"):
                    self._debug_state["error"] = str(response.get("error"))
                    self._debug_state["completed"] = True
                    self._debug_state["ended_at"] = datetime.utcnow().isoformat() + "Z"
                    raise RuntimeError(str(response.get("error")))

            text = ""
            if hasattr(response, "choices") and response.choices:
                first = response.choices[0]
                if hasattr(first, "message") and getattr(first.message, "content", None) is not None:
                    text = first.message.content
                elif isinstance(first, dict):
                    text = first.get("message", {}).get("content", "") or ""
            else:
                text = str(response)
                if "error" in text.lower() or "request entity too large" in text.lower() or "request_too_large" in text.lower() or "413" in text:
                    last_err_text = text.lower()
                    record_attempt("response_text_error", "response_error", last_err_text)
                    attempt += 1
                    continue

            self._debug_state["success"] = True
            self._debug_state["completed"] = True
            self._debug_state["final_model"] = model
            self._debug_state["ended_at"] = datetime.utcnow().isoformat() + "Z"

            class ResponseProxy:
                pass

            resp = ResponseProxy()
            resp.content = text
            return resp

        self._debug_state["error"] = last_err_text
        self._debug_state["completed"] = True
        self._debug_state["ended_at"] = datetime.utcnow().isoformat() + "Z"
        raise RuntimeError(last_err_text or "Groq unknown error")


class GroqQuotaExceededError(RuntimeError):
    def __init__(self, quota_type: str, details: str):
        super().__init__(f"Groq quota exceeded ({quota_type}): {details}")
        self.quota_type = quota_type
        self.details = details

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
    user_id              : str
    intent               : str
    goal                 : str
    kb_context           : str
    workspace_messages   : Annotated[List[str], operator.add]
    coordination_summary : str
    planning             : str
    validation           : str
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
        self._lock      = threading.Lock()
        self._data      = self._load()

    def _load(self) -> dict:
        if self.filepath.exists():
            with open(self.filepath, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
        else:
            data = {}

        if not isinstance(data, dict):
            data = {}

        data.setdefault("session_id", self.session_id)
        data.setdefault("created_at", datetime.now().isoformat())
        data.setdefault("model_fast", MODEL_FAST)
        data.setdefault("model_main", MODEL_MAIN)
        data.setdefault("turns", [])
        data.setdefault("agent_stats", {})
        return data

    def save_turn(self, turn: dict):
        with self._lock:
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

    # [F1] FIX: Tuple[dict, AgentTiming] thay vì tuple[dict, AgentTiming]
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
        # [F6] FIX: Khởi tạo LLM lazy và cache để giảm overhead mỗi lượt
        self._llm = None

    def _get_llm(self, max_tokens=600, temperature=0.2):
        if self._llm is not None:
            return self._llm

        if USE_GROQ:
            self._llm = GroqLLMWrapper(
                api_key=GROQ_KEY,
                model=MODEL_FAST,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            self._llm = ChatAnthropic(
                model=MODEL_FAST,
                anthropic_api_key=ANTHROPIC_KEY,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return self._llm

    def _quick_intent(self, query: str) -> Optional[str]:
        q = query.lower()
        if any(w in q for w in ["langgraph","graph","node","edge","state","multiagent","multi-agent","multi agent"]):
            return "langgraph"
        if (
            (re.search(r'\d', q) and re.search(r'[\+\-\*\/\^×÷]', q))
            or (
                any(w in q for w in ["cộng","trừ","nhân","chia","sqrt","căn","lũy","bình phương","tổng","hiệu","thương","tích"])
                and re.search(r'\d', q)
            )
            or (
                re.search(r'\b(?:tính|tính toán)\b', q)
                and re.search(r'[\+\-\*\/\^×÷]', q)
            )
        ):
            return "calculate"
        if any(w in q for w in ["ngày","giờ","hôm nay","thời gian","mấy giờ"]):
            return "datetime"
        if any(w in q for w in ["tìm","search","tra cứu","tìm kiếm"]):
            return "search"
        if any(w in q for w in ["phân tích","so sánh","đánh giá","ưu","nhược"]):
            return "analyze"
        if any(w in q for w in ["xin chào","hello","hi","chào"]):
            return "greet"
        return None

    def run(self, query: str) -> dict:
        intent = self._quick_intent(query)
        if intent is None:
            prompt = (
                f"Phân loại intent của câu hỏi sau thành MỘT nhãn:\n"
                f"{' | '.join(self.LABELS)}\n\n"
                f"Câu hỏi: \"{query}\"\n\n"
                f"Trả lời CHỈ nhãn, không thêm gì khác."
            )
            try:
                resp = self._get_llm().invoke([HumanMessage(content=prompt)])
                intent = resp.content.strip().lower()
            except Exception:
                intent = ""

        if intent not in self.LABELS:
            intent = self._quick_intent(query) or "general"

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
        message = (
            f"KnowledgeBase đã cung cấp ngữ cảnh cho intent '{intent}'."
            if context else "KnowledgeBase không tìm thấy ngữ cảnh đặc thù."
        )
        return {"kb_context": context, "message": message}

# ══════════════════════════════════════════════════════════════════
#  Calculator worker — PHẢI ở module-level để pickle được (Windows)
# ══════════════════════════════════════════════════════════════════
def _calc_worker(expression: str) -> str:
    """[F2] FIX: Hàm top-level để ProcessPoolExecutor pickle được trên Windows."""
    import math as _math
    import re as _re
    expr = expression
    replacements = [
        ("^","**"), ("×","*"), ("÷","/"),
        ("√","_math.sqrt"), ("π","_math.pi"),
        ("sin","_math.sin"), ("cos","_math.cos"), ("tan","_math.tan"),
        ("log","_math.log10"), ("ln","_math.log"), ("sqrt","_math.sqrt"),
    ]
    for a, b in replacements:
        expr = expr.replace(a, b)
    # Chỉ giữ ký tự an toàn
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
        # Trích biểu thức số học từ câu hỏi
        m = re.search(r'[\d][\d\s\+\-\*\/\^\(\)\.×÷√πsincostan]+[\d\)]', query)
        expr = m.group(0).strip() if m else query

        # [F3] FIX: Tạo pool mới mỗi lần, tránh lỗi spawn trên Windows
        try:
            with ProcessPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_calc_worker, expr)
                result = future.result(timeout=8)
        except Exception as e:
            # Fallback: chạy trực tiếp nếu ProcessPool lỗi (thường gặp trên Windows debug)
            result = _calc_worker(expr)
        message = f"Calculator đã xử lý yêu cầu và cho kết quả: {result}"
        return {"calc_result": result, "message": message}

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
        message = "DateTime đã cung cấp thông tin thời gian hiện tại cho câu trả lời."
        return {"datetime_result": result, "message": message}

# ══════════════════════════════════════════════════════════════════
#  AGENT 5 — Web Search
# ══════════════════════════════════════════════════════════════════
class WebSearchAgent(AgentBase):
    NAME = "WebSearch"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    def run(self, query: str) -> dict:
        if not TAVILY_KEY:
            return {"web_results": "[Tavily chưa cấu hình — thêm TAVILY_API_KEY=tvly-... vào file .env]"}
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=TAVILY_KEY)
            r      = client.search(query=query, max_results=2, include_answer=True, search_depth="advanced")
            lines  = []
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
        self._record_call("Claude-Sonnet (sub-LLM)")
        self._llm = None  # [F6] lazy init

    def _get_llm(self):
        if self._llm is None:
            if USE_GROQ:
                self._llm = GroqLLMWrapper(
                    api_key=GROQ_KEY,
                    model=MODEL_ANALYST,
                    max_tokens=600,
                    temperature=0.3,
                )
            else:
                self._llm = ChatAnthropic(
                    model=MODEL_ANALYST,
                    anthropic_api_key=ANTHROPIC_KEY,
                    max_tokens=600,
                    temperature=0.3,
                )
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
        message = f"Analyst đã phân tích chủ đề '{topic}' và cung cấp luận giải chuyên sâu."
        return {"analysis": resp.content, "message": message}


class PlannerAgent(AgentBase):
    NAME = "Planner"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    def run(self, query: str, context: str = "") -> dict:
        plan = (
            f"1. Phân tích yêu cầu: {query}\n"
            "2. Xác định nhiệm vụ phụ cần thực hiện theo từng bước.\n"
            "3. Lập kế hoạch thu thập thông tin và mô tả cách các agent phối hợp.\n"
        )
        if context:
            plan += f"\n\nNgữ cảnh bổ sung:\n{context}"
        message = "Planner đã xác định cấu trúc nhiệm vụ và cách các agent hợp tác."
        return {"planning": plan, "message": message}


class ValidatorAgent(AgentBase):
    NAME = "Validator"

    def __init__(self, called_by: str = "Router"):
        super().__init__(called_by=called_by)

    def run(self, query: str, context: str = "") -> dict:
        report = (
            f"- Kiểm tra tính phù hợp: câu hỏi '{query}' có thể giải quyết bằng hệ thống multi-agent.\n"
            "- Tài nguyên hiện tại: KnowledgeBase, WebSearch, Analyst, Planner, Validator.\n"
            "- Gợi ý: tối ưu hoá thông tin trả lời bằng cách kết hợp kết quả từ các agent.\n"
        )
        if context:
            report += f"\n\nNgữ cảnh kiểm tra:\n{context}"
        message = "Validator đã xác nhận tính khả thi và gợi ý phối hợp giữa các agent."
        return {"validation": report, "message": message}

# ══════════════════════════════════════════════════════════════════
#  AGENT 7 — Router / Dispatcher
# ══════════════════════════════════════════════════════════════════
class RouterAgent(AgentBase):
    NAME = "Router"

    def __init__(self):
        super().__init__(called_by="GraphNode")

    def run(self, query: str, intent: str, kb_context: str = "") -> dict:
        # [F4] FIX: Reset _calls mỗi lần run để không tích luỹ qua nhiều turns
        self._calls = []
        results     = {"workspace_messages": []}

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
                self._record_call("Planner")
                self._record_call("Validator")
                ws_agent = WebSearchAgent(called_by=self.NAME)
                an_agent = AnalystAgent(called_by=self.NAME)
                pl_agent = PlannerAgent(called_by=self.NAME)
                val_agent = ValidatorAgent(called_by=self.NAME)
                futures["web"]  = (pool.submit(ws_agent.run, query=query), ws_agent)
                futures["anal"] = (pool.submit(an_agent.run, topic=query, context=kb_context), an_agent)
                futures["plan"] = (pool.submit(pl_agent.run, query=query, context=kb_context), pl_agent)
                futures["val"]  = (pool.submit(val_agent.run, query=query, context=kb_context), val_agent)

            elif intent in ("langgraph","crewai","react","memory","rag","multiagent"):
                self._record_call("KnowledgeBase")
                self._record_call("Analyst")
                self._record_call("Planner")
                self._record_call("Validator")
                self._record_call("WebSearch")
                ws_agent = WebSearchAgent(called_by=self.NAME)
                futures["web"] = (pool.submit(ws_agent.run, query=query), ws_agent)
                kb_agent  = KnowledgeBaseAgent(called_by=self.NAME)
                an_agent  = AnalystAgent(called_by=self.NAME)
                pl_agent  = PlannerAgent(called_by=self.NAME)
                val_agent = ValidatorAgent(called_by=self.NAME)
                futures["kb"]   = (pool.submit(kb_agent.run, query=query, intent=intent), kb_agent)
                futures["anal"] = (pool.submit(an_agent.run, topic=query, context=kb_context), an_agent)
                futures["plan"] = (pool.submit(pl_agent.run, query=query, context=kb_context), pl_agent)
                futures["val"]  = (pool.submit(val_agent.run, query=query, context=kb_context), val_agent)

            elif intent in ("greet", "general"):
                # Không cần agent nào thêm — ResponseGenerator tự trả lời
                pass

            # Thu kết quả
            sub_timings: List[AgentTiming] = []
            for key, (fut, agt) in futures.items():
                try:
                    ts_start = time.time()
                    t0       = time.perf_counter()
                    r        = fut.result(timeout=30)
                    dur      = time.perf_counter() - t0
                    message = r.pop("message", None)
                    if message:
                        results["workspace_messages"].append(message)
                    results.update(r)
                    sub_timings.append(AgentTiming(
                        agent    =agt.NAME,
                        start_ts =ts_start,
                        end_ts   =ts_start + dur,
                        duration =round(dur, 4),
                        called_by=self.NAME,
                        calls    =agt._calls,
                    ))
                except Exception as e:
                    error_msg = f"Lỗi agent: {e}"
                    results["workspace_messages"].append(error_msg)
                    results[key] = error_msg

        if not results["workspace_messages"]:
            results["workspace_messages"].append("Router đã không cần gọi thêm agent phụ.")

        results["_sub_timings"] = sub_timings
        return results

# ══════════════════════════════════════════════════════════════════
#  AGENT 8 — Response Generator
# ══════════════════════════════════════════════════════════════════
class ResponseGeneratorAgent(AgentBase):
    NAME = "ResponseGenerator"

    def __init__(self):
        super().__init__(called_by="GraphNode")
        self._record_call("Claude-Sonnet (main)")
        self._llm = None  # [F6] lazy init

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        if USE_GROQ:
            self._llm = GroqLLMWrapper(
                api_key=GROQ_KEY,
                model=MODEL_MAIN,
                max_tokens=GROQ_MAX_TOKENS,
                temperature=0.35,
            )
        else:
            # Use the fast Anthropic model by default to improve response latency.
            self._llm = ChatAnthropic(
                model=MODEL_FAST,
                anthropic_api_key=ANTHROPIC_KEY,
                max_tokens=900,
                temperature=0.35,
            )
        return self._llm

    def run(self, query: str, intent: str = "", kb_context: str = "",
            web_results: str = "", calc_result: str = "",
            datetime_result: str = "", analysis: str = "",
            planning: str = "", validation: str = "",
            coordination_summary: str = "", history: str = "") -> dict:

        system = (
            "Bạn là một hệ thống multi-agent AI hợp tác, gồm nhiều agent độc lập nhưng làm việc cùng nhau để thực hiện mục tiêu chung. \n\n"
            "Mỗi agent có vai trò riêng và bạn tổng hợp kết quả của chúng thành câu trả lời cuối cùng. \n\n"
            "**Nguyên tắc trả lời:**\n"
            "1. Luôn trả lời tiếng Việt chuẩn mực, rõ ràng và có cấu trúc logic.\n"
            "2. Sử dụng Markdown để định dạng: tiêu đề (##, ###), danh sách (-), bảng, code block.\n"
            "3. Chỉ sử dụng in đậm (**text**) cho: tiêu đề, keywords quan trọng, cảnh báo, hoặc nội dung đáng chú ý. Không in đậm văn bản thông thường.\n"
            "4. Trả lời đầy đủ: mở đầu rõ ràng → giải thích chi tiết → kết luận hoặc gợi ý.\n"
            "5. Nếu có bảng dữ liệu, dùng định dạng bảng Markdown để dễ đọc.\n"
            "6. Luôn dựa trên ngữ cảnh, lịch sử hội thoại và thông tin được cung cấp.\n"
            "7. Nếu thiếu thông tin, hãy nói rõ và gợi ý thêm câu hỏi.\n\n"
            "8. Nếu cần làm nổi bật một từ hoặc cụm từ, hãy dùng marker dạng `!!important: <short phrase>` — KHÔNG sử dụng thẻ HTML (`<strong>`, `<b>`) hoặc ký hiệu `**` để in đậm. Front-end sẽ chỉ in đậm những đoạn được đánh dấu bằng marker này.\n\n"
            "**Tông độ:** Thân thiện, chuyên nghiệp, hỗ trợ tích cực."
        )

        context_parts = []
        if kb_context:
            context_parts.append(f"**Knowledge Base:**\n{_truncate_text(kb_context, GROQ_PROMPT_SECTION_LIMIT)}")
        if web_results:
            context_parts.append(f"**Kết quả tìm kiếm web:**\n{_truncate_text(web_results, GROQ_PROMPT_SECTION_LIMIT)}")
        if calc_result:
            context_parts.append(f"**Kết quả tính toán:**\n{_truncate_text(calc_result, GROQ_PROMPT_SECTION_LIMIT)}")
        if datetime_result:
            context_parts.append(f"**Thời gian:**\n{_truncate_text(datetime_result, GROQ_PROMPT_SECTION_LIMIT)}")
        if analysis:
            context_parts.append(f"**Phân tích chuyên sâu:**\n{_truncate_text(analysis, GROQ_PROMPT_SECTION_LIMIT)}")
        if planning:
            context_parts.append(f"**Kế hoạch agent:**\n{_truncate_text(planning, GROQ_PROMPT_SECTION_LIMIT)}")
        if validation:
            context_parts.append(f"**Kiểm định agent:**\n{_truncate_text(validation, GROQ_PROMPT_SECTION_LIMIT)}")
        if coordination_summary:
            context_parts.append(f"**Tóm tắt phối hợp agent:**\n{_truncate_text(coordination_summary, GROQ_PROMPT_SECTION_LIMIT)}")
        if history:
            context_parts.append(f"**Lịch sử hội thoại gần đây:**\n{_truncate_text(history, GROQ_PROMPT_SECTION_LIMIT)}")

        context_block = "\n\n---\n\n".join(context_parts)
        prompt = (
            f"Câu hỏi: {query}\n\n"
            + (f"Ngữ cảnh:\n{context_block}\n\n" if context_block else "")
            + "Hãy trả lời câu hỏi trên một cách đầy đủ và rõ ràng."
        )
        prompt = _truncate_text(prompt, GROQ_PROMPT_TOTAL_LIMIT)

        resp = self._get_llm().invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])

        # Server-side post-processing: strip raw HTML bold tags and ** markers
        answer = (resp.content or "")
        # Remove raw <strong> / <b> tags
        answer = re.sub(r'</?(?:strong|b)[^>]*>', '', answer, flags=re.IGNORECASE)
        # Remove escaped forms like &lt;strong&gt;
        answer = re.sub(r'&lt;/?(?:strong|b)[^&]*&gt;', '', answer, flags=re.IGNORECASE)
        # Remove markdown bold markers but keep marker-based highlights (!!important:)
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer, flags=re.DOTALL)
        # As a final cleanup remove any leftover bare '**'
        answer = answer.replace('**', '')

        return {"final_answer": answer}

# ══════════════════════════════════════════════════════════════════
#  GLOBAL AGENT INSTANCES & DB
#  [F6] FIX: Khởi tạo ở đây nhưng LLM bên trong lazy — an toàn khi import
# ══════════════════════════════════════════════════════════════════
_intent_agent   = IntentClassifierAgent()
_kb_agent       = KnowledgeBaseAgent()
_router_agent   = RouterAgent()
_response_agent = ResponseGeneratorAgent()

# [F5] FIX: db khai báo None ở module-level, khởi tạo trong main()
db: Optional[ChatPersistence] = None

# ══════════════════════════════════════════════════════════════════
#  LANGGRAPH NODES
# ══════════════════════════════════════════════════════════════════
def node_intent_classifier(state: ChatState) -> dict:
    result, timing = _intent_agent.execute(query=state["user_query"])
    goal = f"Giải quyết: {state['user_query']}"
    workspace_msg = (
        f"IntentClassifier đã nhận diện intent='{result['intent']}' và xác định mục tiêu chung."
    )
    return {
        **result,
        "goal": goal,
        "workspace_messages": [workspace_msg],
        "agent_timings": [timing],
        "call_graph"   : [{"from": "GraphEntry", "to": "IntentClassifier", "ts": timing["start_ts"]}],
        "log"          : [f"[IntentClassifier] '{state['user_query'][:40]}' → intent='{result['intent']}' ({timing['duration']:.3f}s)"],
    }


def node_kb_retriever(state: ChatState) -> dict:
    result, timing = _kb_agent.execute(query=state["user_query"], intent=state["intent"])
    workspace_msg = (
        f"KnowledgeBase đã cung cấp thông tin bối cảnh cho intent='{state['intent']}'."
        if result.get("kb_context") else
        "KnowledgeBase chưa có bối cảnh phù hợp, tiếp tục thu thập thông tin."
    )
    return {
        **result,
        "workspace_messages": [workspace_msg],
        "agent_timings": [timing],
        "call_graph"   : [{"from": "IntentClassifier", "to": "KnowledgeBase", "ts": timing["start_ts"]}],
        "log"          : [f"[KnowledgeBase] context={'found' if result['kb_context'] else 'empty'} ({timing['duration']:.3f}s)"],
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
        agent    ="Router",
        start_ts =ts_start,
        end_ts   =ts_start + dur,
        duration =round(dur, 4),
        called_by="KnowledgeBase",
        calls    =list(_router_agent._calls),
    )
    cg   = [{"from": "KnowledgeBase", "to": "Router", "ts": ts_start}]
    cg  += [{"from": "Router", "to": t["agent"], "ts": t["start_ts"]} for t in sub_timings]
    logs = [f"[Router] dispatched → {[t['agent'] for t in sub_timings]} ({dur:.3f}s)"]
    for t in sub_timings:
        logs.append(f"  ↳ [{t['agent']}] {t['duration']:.3f}s (called_by=Router)")

    return {
        **raw,
        "agent_timings": [router_timing] + sub_timings,
        "call_graph"   : cg,
        "log"          : logs,
    }


def node_coordinator(state: ChatState) -> dict:
    messages = state.get("workspace_messages", []) or []
    goal = state.get("goal", "")
    summary_lines = [f"Mục tiêu chung: {goal}"] if goal else []
    if messages:
        summary_lines.append("Thông tin hợp tác từ các agent:")
        summary_lines += [f"- {m}" for m in messages]
    else:
        summary_lines.append("Không có thông tin phụ trợ từ các agent. Tiếp tục dùng bối cảnh hiện tại.")

    summary = "\n".join(summary_lines)
    timing = AgentTiming(
        agent    ="Coordinator",
        start_ts =time.time(),
        end_ts   =time.time(),
        duration =0.0,
        called_by="Router",
        calls    =[],
    )
    return {
        "coordination_summary": summary,
        "agent_timings"      : [timing],
        "call_graph"         : [{"from": "Router", "to": "Coordinator", "ts": timing["start_ts"]}],
        "log"                : [f"[Coordinator] Tổng hợp tin nhắn workspace và mục tiêu chung."],
    }


def node_response_generator(state: ChatState) -> dict:
    # [F5] FIX: kiểm tra db trước khi gọi get_history_text
    history = db.get_history_text(4) if db is not None else ""
    result, timing = _response_agent.execute(
        query               =state["user_query"],
        intent              =state.get("intent", ""),
        kb_context          =state.get("kb_context", ""),
        web_results         =state.get("web_results", ""),
        calc_result         =state.get("calc_result", ""),
        datetime_result     =state.get("datetime_result", ""),
        analysis            =state.get("analysis", ""),
        planning            =state.get("planning", ""),
        validation          =state.get("validation", ""),
        coordination_summary=state.get("coordination_summary", ""),
        history             =history,
    )
    return {
        **result,
        "agent_timings": [timing],
        "call_graph"   : [{"from": "Coordinator", "to": "ResponseGenerator", "ts": timing["start_ts"]}],
        "log"          : [f"[ResponseGenerator] answer generated ({timing['duration']:.3f}s)"],
    }


def node_persistence(state: ChatState) -> dict:
    if db is not None:
        turn_data = {
            "turn_id"      : state["turn_id"],
            "timestamp"    : datetime.now().isoformat(),
            "user_query"   : state["user_query"],
            "user_id"      : state.get("user_id", ""),
            "intent"       : state["intent"],
            "final_answer" : state["final_answer"],
            "agent_timings": state["agent_timings"],
            "call_graph"   : state["call_graph"],
        }
        threading.Thread(target=db.save_turn, args=(turn_data,), daemon=True).start()
    filepath = db.filepath if db else "N/A"
    return {"log": [f"[Persistence] Đã lưu lượt {state['turn_id']} → {filepath} (async)"]}

# ══════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ══════════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(ChatState)
    g.add_node("intent_classifier",  node_intent_classifier)
    g.add_node("kb_retriever",       node_kb_retriever)
    g.add_node("router",             node_router)
    g.add_node("coordinator",        node_coordinator)
    g.add_node("response_generator", node_response_generator)
    g.add_node("persistence",        node_persistence)

    g.set_entry_point("intent_classifier")
    g.add_edge("intent_classifier",  "kb_retriever")
    g.add_edge("kb_retriever",       "router")
    g.add_edge("router",             "coordinator")
    g.add_edge("coordinator",        "response_generator")
    g.add_edge("response_generator", "persistence")
    return g.compile()

# ══════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════
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
        bar       = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        calls_str = ", ".join(t.get("calls", [])) or "—"
        table.add_row(
            str(i), t["agent"], t["called_by"],
            calls_str[:28], f"{t['duration']:.3f}s",
            f"{bar} {pct:.0f}%",
        )
    console.print(table)
    console.print(f"  [dim]Tổng: [bold]{total_elapsed:.3f}s[/bold][/dim]\n")


def display_call_graph(call_graph: List[dict], timings: List[AgentTiming]):
    # [F9] FIX: Xây edges dict đầy đủ rồi mới render tree
    edges: dict = {}
    for e in call_graph:
        edges.setdefault(e["from"], []).append(e["to"])

    timing_map = {t["agent"]: t["duration"] for t in timings}

    tree = Tree("[bold cyan]📊 Agent Call Graph[/bold cyan]", guide_style="dim cyan")

    def add_children(node, agent: str, visited: set):
        for child in edges.get(agent, []):
            if child in visited:
                continue
            visited.add(child)
            dur    = timing_map.get(child, 0)
            branch = node.add(f"[yellow]{child}[/yellow] [dim]({dur:.3f}s)[/dim]")
            add_children(branch, child, visited)

    # [F9] FIX: Tìm root thực sự (node không là target của bất kỳ edge nào)
    all_targets = {e["to"] for e in call_graph}
    roots = [e["from"] for e in call_graph if e["from"] not in all_targets]
    if not roots:
        roots = ["GraphEntry"]

    visited_global: set = set()
    for root in dict.fromkeys(roots):  # deduplicate, giữ thứ tự
        visited_global.add(root)
        root_branch = tree.add(f"[green]{root}[/green]")
        add_children(root_branch, root, visited_global)

    console.print(tree)


def display_log(logs: List[str]):
    table = Table(box=box.SIMPLE, border_style="dim", title="📋 Execution Log")
    table.add_column("#",    style="dim cyan", width=4)
    table.add_column("Step", style="white")
    for i, log in enumerate(logs, 1):
        table.add_row(str(i), log)
    console.print(table)


def display_cumulative_stats():
    if db is None:
        return
    stats = db.get_all_stats()
    if not stats:
        return
    table = Table(
        title=f"📈 Cumulative Agent Stats — {db.filepath.name}",
        box=box.ROUNDED, border_style="blue",
    )
    table.add_column("Agent",      style="yellow", width=22)
    table.add_column("Calls",      style="cyan",   width=8,  justify="right")
    table.add_column("Total Time", style="green",  width=12, justify="right")
    table.add_column("Avg Time",   style="magenta",width=12, justify="right")
    table.add_column("Max Time",   style="red",    width=12, justify="right")
    for agent, s in sorted(stats.items(), key=lambda x: -x[1]["total_time"]):
        table.add_row(agent, str(s["calls"]),
                      f"{s['total_time']:.3f}s",
                      f"{s['avg_time']:.3f}s",
                      f"{s['max_time']:.3f}s")
    console.print(table)
    console.print(f"  [dim]File: {db.filepath}[/dim]\n")


def display_welcome():
    console.print()
    console.print(Panel.fit(
        "[bold blue]🤖 LANGGRAPH MULTI-AGENT CHATBOT v3.1[/bold blue]\n"
        "[dim]Kiến trúc: Real Processes · Claude API · JSON Persistence · Timing Analysis[/dim]\n\n"
        f"[cyan]🧠 Fast Model:[/cyan]  {MODEL_FAST}\n"
        f"[cyan]🧠 Main Model:[/cyan]  {MODEL_MAIN}\n"
        f"[cyan]🔍 Web Search:[/cyan]  {'Tavily ✓' if TAVILY_KEY else 'Tắt — thêm TAVILY_API_KEY vào .env'}\n"
        f"[cyan]💾 Lưu chat:[/cyan]   {CHAT_DIR}/session_*.json\n\n"
        "[yellow]Flow:[/yellow] IntentClassifier → KB → Router ⟶ [WebSearch|Calculator|DateTime|Analyst] → ResponseGenerator → Persistence\n\n"
        "[dim]Lệnh: [bold]graph[/bold] · [bold]timing[/bold] · [bold]stats[/bold] · [bold]help[/bold] · [bold]quit[/bold][/dim]",
        border_style="blue",
        title="[bold]Hoàng Minh Đức — Thực tập sinh AI Agent 2026[/bold]",
    ))
    console.print()


def display_graph_diagram():
    console.print(Panel(
        "[bold cyan]LANGGRAPH WORKFLOW v3.1[/bold cyan]\n\n"
        "  [START]\n    │\n    ▼\n"
        "  [yellow]① IntentClassifier[/yellow]  ← Claude Haiku (fast)\n"
        "    │\n    ▼\n"
        "  [yellow]② KnowledgeBase[/yellow]     ← Pure Python RAG\n"
        "    │\n    ▼\n"
        "  [yellow]③ Router[/yellow]            ← ThreadPool Dispatcher\n"
        "    │    ╔══════════════════════════════╗\n"
        "    ├───►║ WebSearch  (Thread, Tavily)  ║\n"
        "    ├───►║ Calculator (Process, eval)   ║ ← Song song\n"
        "    ├───►║ DateTime   (Python inline)   ║\n"
        "    └───►║ Analyst    (Thread, Sonnet)  ║\n"
        "         ╚══════════════════════════════╝\n"
        "    ▼\n"
        "  [yellow]④ ResponseGenerator[/yellow] ← Claude Sonnet\n"
        "    │\n    ▼\n"
        "  [yellow]⑤ Persistence[/yellow]       ← JSON file\n"
        "   [END]\n\n"
        "[dim]State TypedDict (operator.add) chạy xuyên suốt tất cả Nodes[/dim]",
        border_style="cyan",
        title="[bold]Kiến trúc Graph[/bold]",
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
        "[yellow]graph[/yellow]   — Sơ đồ kiến trúc\n"
        "[yellow]timing[/yellow]  — Timing turn vừa rồi\n"
        "[yellow]stats[/yellow]   — Thống kê toàn phiên\n"
        "[yellow]help[/yellow]    — Hướng dẫn này\n"
        "[yellow]quit[/yellow]    — Thoát\n",
        border_style="green",
        title="[bold]📚 HƯỚNG DẪN[/bold]",
    ))

# ══════════════════════════════════════════════════════════════════
#  MAIN — [F2] Bắt buộc có if __name__ == "__main__" trên Windows
#          để ProcessPoolExecutor không spawn vô hạn
# ══════════════════════════════════════════════════════════════════
def main():
    global db

    # ── Kiểm tra API Key ──────────────────────────────────────────
    global ANTHROPIC_KEY
    if not ANTHROPIC_KEY:
        console.print("[red]❌ Thiếu ANTHROPIC_API_KEY trong file .env![/red]")
        console.print("[dim]Tạo file .env trong cùng thư mục với nội dung:[/dim]")
        console.print("[yellow]ANTHROPIC_API_KEY=sk-ant-...[/yellow]")
        ANTHROPIC_KEY = console.input("[yellow]Hoặc nhập thẳng Key ngay bây giờ: [/yellow]").strip()
        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_KEY

    # ── Khởi tạo DB ───────────────────────────────────────────────
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    db         = ChatPersistence(session_id)

    console.print("\n[dim]Biên dịch LangGraph...[/dim]")
    graph = build_graph()
    console.print(f"[green]✓ Graph sẵn sàng — 5 Nodes | Lưu tại: {db.filepath}[/green]\n")

    display_welcome()

    _last_timings:    List[AgentTiming] = []
    _last_call_graph: List[dict]        = []
    _last_total:      float             = 0.0
    turn = 0

    while True:
        try:
            console.rule(f"[dim]Turn {turn + 1}[/dim]", style="dim")
            query = console.input("[bold green]Bạn[/bold green] › ").strip()
            if not query:
                continue

            cmd = query.lower()
            if cmd in ("quit","exit","thoát","q"):
                console.print("\n[bold blue]👋 Kết thúc phiên.[/bold blue]")
                console.print(f"  File: [underline]{db.filepath}[/underline]\n")
                display_cumulative_stats()
                break
            if cmd == "graph":
                display_graph_diagram(); continue
            if cmd == "timing":
                if _last_timings:
                    display_timing_analysis(_last_timings, _last_total)
                    display_call_graph(_last_call_graph, _last_timings)
                else:
                    console.print("[dim]Chưa có dữ liệu. Hãy hỏi câu đầu tiên.[/dim]")
                continue
            if cmd == "stats":
                display_cumulative_stats(); continue
            if cmd == "help":
                display_help(); continue

            turn += 1
            # [F8] FIX: Cung cấp giá trị mặc định rõ ràng cho tất cả trường
            initial: ChatState = {
                "session_id"          : session_id,
                "turn_id"             : turn,
                "user_query"          : query,
                "intent"              : "",
                "goal"                : "",
                "kb_context"          : "",
                "workspace_messages"  : [],
                "coordination_summary": "",
                "planning"            : "",
                "validation"          : "",
                "web_results"         : "",
                "calc_result"         : "",
                "datetime_result"     : "",
                "analysis"            : "",
                "final_answer"        : "",
                "conversation_history": [],
                "agent_timings"       : [],
                "call_graph"          : [],
                "log"                 : [],
            }

            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                TimeElapsedColumn(),
                console=console, transient=True
            ) as prog:
                prog.add_task("Đang xử lý qua LangGraph...", total=None)
                t0      = time.perf_counter()
                result  = graph.invoke(initial)
                elapsed = time.perf_counter() - t0

            _last_timings    = result["agent_timings"]
            _last_call_graph = result["call_graph"]
            _last_total      = elapsed

            display_log(result["log"])
            display_timing_analysis(_last_timings, elapsed)
            display_call_graph(_last_call_graph, _last_timings)

            console.print(Panel(
                Markdown(result["final_answer"]),
                border_style="blue",
                title=f"[bold blue]🤖 Chatbot[/bold blue]  [dim]intent={result['intent']} | {elapsed:.2f}s[/dim]",
                padding=(1, 2),
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Ctrl+C — gõ 'quit' để thoát.[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Lỗi: {e}[/red]")
            traceback.print_exc()


# [F2] BẮT BUỘC trên Windows — ProcessPoolExecutor spawn cần guard này
if __name__ == "__main__":
    main()
