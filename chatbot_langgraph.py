"""
========================================================
  CHATBOT THÔNG MINH VỚI LANGGRAPH + RAG + MEMORY
  Tác giả : Hoàng Minh Đức — Thực tập sinh AI Agent
  Ngày    : 2026
  Mô tả   : Ứng dụng minh họa kiến trúc LangGraph với:
             - Multi-Agent orchestration (Graph & State)
             - Cơ chế ReAct (Reasoning & Acting)
             - Memory 3 lớp: Short-term / Long-term / State
             - RAG (Retrieval-Augmented Generation) mock
             - Mock LLM — chạy OFFLINE, không cần API key
========================================================
  Chạy: python chatbot_langgraph.py
  Thoát: gõ 'quit' hoặc 'exit'
========================================================
"""

import os
import re
import json
import time
import random
import textwrap
from typing import TypedDict, Annotated, List, Optional
from datetime import datetime
import operator

# ──────────────────────────────────────────────────────
#  KIỂM TRA & CÀI THƯ VIỆN TỰ ĐỘNG
# ──────────────────────────────────────────────────────
def check_and_install(package: str, import_name: str = None):
    import importlib, subprocess, sys
    name = import_name or package
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"  [→] Đang cài {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("\n" + "="*60)
print("  KHỞI ĐỘNG HỆ THỐNG LANGGRAPH CHATBOT")
print("="*60)
print("\n[1/4] Kiểm tra thư viện...")
check_and_install("langgraph")
check_and_install("langchain-core", "langchain_core")
check_and_install("rich")
print("  [✓] Tất cả thư viện sẵn sàng\n")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import print as rprint

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("  [!] LangGraph chưa cài — dùng simulation mode")

console = Console()


# ──────────────────────────────────────────────────────
#  KNOWLEDGE BASE — DỮ LIỆU NỘI BỘ (Giả lập Vector DB)
# ──────────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "langgraph": [
        "LangGraph là framework xây dựng Multi-Agent dựa trên Đồ thị có hướng (Directed Graph). Mỗi Node là một Agent hoặc hàm xử lý, Edge định nghĩa luồng dữ liệu.",
        "LangGraph sử dụng TypedDict để khai báo State tường minh. State này chạy xuyên suốt qua tất cả các Node trong graph.",
        "Cơ chế Reducer trong LangGraph giải quyết Race Condition khi nhiều Node song song cùng ghi vào State. Ví dụ: operator.add để nối danh sách thay vì ghi đè.",
        "LangGraph hỗ trợ Human-in-the-loop: dừng graph giữa chừng, cho người dùng xác nhận, rồi chạy tiếp — tính năng sống còn cho enterprise.",
        "LangGraph phù hợp cho Production vì tính tất định cao, State Observability, và khả năng xử lý lỗi (fallback node) tốt hơn CrewAI.",
    ],
    "crewai": [
        "CrewAI dùng hướng tiếp cận khai báo (Declarative): định nghĩa Agent với Role, Goal, Tools rồi giao Task. Framework tự điều phối.",
        "CrewAI phù hợp cho Prototyping/PoC nhờ đường cong học tập thấp và khả năng ra demo nhanh.",
        "CrewAI hỗ trợ async_execution=True để chạy Task song song, nhưng khó tùy biến logic merge kết quả.",
        "Bộ nhớ CrewAI tích hợp sẵn Short-term, Long-term, Entity memory — dễ bật bằng cấu hình nhưng khó can thiệp sâu.",
    ],
    "react": [
        "ReAct (Reasoning and Acting) là cơ chế cốt lõi của Agent: Thought → Action → Observation lặp đến khi có Final Answer.",
        "Thought: LLM phân tích ngữ cảnh, quyết định gọi tool nào. Action: thực thi tool. Observation: nhận kết quả, tiếp tục suy luận.",
        "ReAct giảm thiểu lỗi logic bằng cách ép LLM 'nghĩ lớn tiếng' trước khi hành động — không cho phép LLM trả lời ngay lập tức.",
        "Rủi ro ReAct: LLM có thể rơi vào vòng lặp vô hạn nếu tool liên tục lỗi. Giải pháp: hard limit max_iterations = 5.",
    ],
    "memory": [
        "Hệ thống bộ nhớ 3 lớp: Short-term (In-Context, sliding window), Long-term (Vector DB, RAG), State (Blackboard, JSON).",
        "Memory Dispatcher là 'hệ điều hành bộ nhớ': chỉ nạp đúng thông tin cần thiết thay vì full-load — giảm Hallucination.",
        "Async Write-back: tách luồng ghi Vector DB khỏi luồng phản hồi chính — tránh bottleneck, tăng tốc độ.",
        "State Store lưu dưới dạng JSON thay vì natural language — cắt giảm ~90% token đầu vào cần để duy trì ngữ cảnh.",
        "Khi buffer hội thoại >4000 tokens, tự động Summarize và nén — kiểm soát chi phí API chủ động.",
    ],
    "multiagent": [
        "Multi-Agent System chia nhỏ tác vụ phức tạp cho nhiều Agent chuyên biệt — tránh quá tải ngữ cảnh của một Agent duy nhất.",
        "4 mô hình điều phối: Sequential (pipeline), Hierarchical (manager-worker), State-Graph (LangGraph), Conversational (AutoGen).",
        "Parallel execution cải thiện tốc độ 1.3x đến 4.15x nhưng cần xử lý Race Condition và tăng chi phí token ~2.3x.",
        "Fan-out/Fan-in: Manager phân công N task con cho N Worker song song, thu thập và merge kết quả qua Reducer.",
    ],
    "rag": [
        "RAG (Retrieval-Augmented Generation) tích hợp Vector DB để LLM truy cập kiến thức bên ngoài training data.",
        "Quy trình RAG: Embed tài liệu → lưu Vector DB → khi có query, tính Cosine Similarity → nạp k chunks liên quan nhất vào prompt.",
        "RAG giải quyết vấn đề LLM bị outdated và context window giới hạn — thay vì nhồi toàn bộ tài liệu, chỉ lấy phần liên quan.",
        "ChromaDB và FAISS là 2 Vector DB phổ biến cho local development. Pinecone cho production cloud.",
    ],
    "greet": [
        "Xin chào! Tôi là Chatbot AI Agent được xây dựng với LangGraph.",
        "Tôi có thể giải thích về LangGraph, CrewAI, ReAct, Memory Management, RAG, và Multi-Agent Systems.",
    ]
}

TOOL_DESCRIPTIONS = {
    "search_knowledge": "Tìm kiếm trong knowledge base nội bộ",
    "calculate": "Thực hiện tính toán cơ bản",
    "get_datetime": "Lấy ngày giờ hiện tại",
    "summarize_topic": "Tóm tắt một chủ đề",
}


# ──────────────────────────────────────────────────────
#  LANGGRAPH STATE DEFINITION
# ──────────────────────────────────────────────────────
class ChatbotState(TypedDict):
    """
    State tường minh của toàn bộ Graph.
    Đây là 'Blackboard' chung — tất cả Nodes đọc/ghi vào đây.
    messages dùng operator.add làm Reducer (append, không ghi đè).
    """
    messages: Annotated[List[dict], operator.add]   # Lịch sử hội thoại
    current_query: str                               # Câu hỏi hiện tại
    intent: str                                      # Ý định đã phân loại
    retrieved_context: str                           # Ngữ cảnh từ "Vector DB"
    react_thought: str                               # Chuỗi suy luận ReAct
    react_action: str                                # Hành động quyết định
    react_observation: str                           # Kết quả quan sát
    final_answer: str                                # Câu trả lời cuối
    iteration_count: int                             # Đếm vòng lặp ReAct
    processing_log: Annotated[List[str], operator.add]  # Log xử lý


# ──────────────────────────────────────────────────────
#  MOCK LLM ENGINE (Không cần API key)
# ──────────────────────────────────────────────────────
class MockLLM:
    """
    LLM giả lập — hoạt động hoàn toàn offline.
    Trong production: thay bằng ChatOpenAI, ChatAnthropic, v.v.
    """
    MODEL_NAME = "MockLLM-Demo-v1.0 (Offline Mode)"

    def classify_intent(self, query: str) -> str:
        query_lower = query.lower()
        if any(w in query_lower for w in ["xin chào", "hello", "hi ", "chào", "hey"]):
            return "greet"
        if any(w in query_lower for w in ["langgraph", "lang graph", "đồ thị", "graph", "node", "edge", "state"]):
            return "langgraph"
        if any(w in query_lower for w in ["crewai", "crew ai", "crew", "prototyp"]):
            return "crewai"
        if any(w in query_lower for w in ["react", "reasoning", "thought", "action", "observation", "suy luận"]):
            return "react"
        if any(w in query_lower for w in ["bộ nhớ", "memory", "short-term", "long-term", "vector", "dispatcher", "blackboard"]):
            return "memory"
        if any(w in query_lower for w in ["rag", "retrieval", "tìm kiếm", "chromadb", "embedding", "chunk"]):
            return "rag"
        if any(w in query_lower for w in ["multi-agent", "đa tác tử", "agent", "parallel", "song song", "orchestrat"]):
            return "multiagent"
        if any(w in query_lower for w in ["tính", "cộng", "trừ", "nhân", "chia", "+", "-", "*", "/"]):
            return "calculate"
        if any(w in query_lower for w in ["giờ", "ngày", "thời gian", "hôm nay", "date", "time"]):
            return "datetime"
        return "general"

    def generate_thought(self, query: str, intent: str) -> str:
        thoughts = {
            "langgraph": f"Câu hỏi về LangGraph. Tôi cần tìm kiếm kiến thức về framework này trong knowledge base.",
            "crewai": f"Câu hỏi về CrewAI. Cần truy xuất thông tin so sánh CrewAI và ưu điểm của từng framework.",
            "react": f"Câu hỏi về cơ chế ReAct. Cần giải thích vòng lặp Thought→Action→Observation.",
            "memory": f"Câu hỏi về Memory Management. Cần trình bày kiến trúc 3 lớp bộ nhớ.",
            "rag": f"Câu hỏi về RAG. Cần giải thích quy trình Retrieval-Augmented Generation.",
            "multiagent": f"Câu hỏi về Multi-Agent. Cần trình bày các mô hình điều phối và so sánh.",
            "calculate": f"Phát hiện yêu cầu tính toán trong query. Sẽ gọi tool 'calculate'.",
            "datetime": f"Người dùng hỏi về thời gian. Sẽ gọi tool 'get_datetime'.",
            "greet": f"Người dùng chào hỏi. Phản hồi thân thiện và giới thiệu khả năng.",
            "general": f"Câu hỏi chung. Tìm kiếm trong knowledge base để tìm thông tin liên quan nhất.",
        }
        return thoughts.get(intent, f"Phân tích query: '{query[:50]}'. Tìm kiếm thông tin phù hợp.")

    def decide_action(self, intent: str) -> tuple:
        action_map = {
            "calculate": ("calculate", "expression từ query"),
            "datetime": ("get_datetime", ""),
            "greet": ("search_knowledge", "greet"),
        }
        if intent in action_map:
            return action_map[intent]
        return ("search_knowledge", intent)

    def generate_answer(self, query: str, context: str, observation: str, intent: str) -> str:
        intro_templates = [
            "Dựa trên kiến thức trong hệ thống,",
            "Theo tài liệu nghiên cứu,",
            "Từ knowledge base của hệ thống,",
        ]
        intro = random.choice(intro_templates)
        if intent == "greet":
            return (
                f"Xin chào! 👋 Tôi là **Chatbot AI Agent** được xây dựng với **LangGraph**.\n\n"
                f"Tôi có thể giải thích về:\n"
                f"• **LangGraph** — kiến trúc Graph & State\n"
                f"• **CrewAI** — framework khai báo đa tác tử\n"
                f"• **ReAct** — cơ chế suy luận của Agent\n"
                f"• **Memory Management** — 3 lớp bộ nhớ\n"
                f"• **RAG** — Retrieval-Augmented Generation\n"
                f"• **Multi-Agent** — các mô hình điều phối\n\n"
                f"Bạn muốn tìm hiểu về chủ đề nào?"
            )
        if not context or context == "Không tìm thấy thông tin liên quan.":
            return (
                f"Tôi chưa có đủ thông tin chi tiết về câu hỏi này trong knowledge base.\n\n"
                f"Bạn có thể hỏi về: **LangGraph**, **CrewAI**, **ReAct**, **Memory**, **RAG**, hoặc **Multi-Agent**."
            )
        lines = [l.strip() for l in context.split('\n') if l.strip()]
        answer_parts = [f"{intro} đây là thông tin về **{query[:40]}...**:\n"]
        for i, line in enumerate(lines[:3], 1):
            answer_parts.append(f"\n**{i}.** {line}")
        if len(lines) > 3:
            answer_parts.append(f"\n\n*...và {len(lines)-3} thông tin bổ sung khác trong knowledge base.*")
        answer_parts.append(
            f"\n\n---\n*💡 Gợi ý: Hỏi thêm về bất kỳ khía cạnh nào để biết chi tiết hơn.*"
        )
        return "".join(answer_parts)


# ──────────────────────────────────────────────────────
#  TOOL REGISTRY (Function Calling)
# ──────────────────────────────────────────────────────
def tool_search_knowledge(topic: str) -> str:
    """Tìm kiếm trong knowledge base theo chủ đề."""
    results = KNOWLEDGE_BASE.get(topic, [])
    if not results:
        # Tìm fuzzy
        for key, values in KNOWLEDGE_BASE.items():
            if topic.lower() in key.lower() or key.lower() in topic.lower():
                results = values
                break
    if results:
        return "\n".join(f"• {r}" for r in results[:4])
    return "Không tìm thấy thông tin liên quan."

def tool_calculate(expression: str) -> str:
    """Tính toán biểu thức toán học đơn giản."""
    # Trích xuất số và phép tính từ câu
    numbers = re.findall(r'\d+\.?\d*', expression)
    if len(numbers) >= 2:
        a, b = float(numbers[0]), float(numbers[1])
        if "+" in expression or "cộng" in expression:
            return f"{a} + {b} = {a + b}"
        elif "-" in expression or "trừ" in expression:
            return f"{a} - {b} = {a - b}"
        elif "*" in expression or "nhân" in expression or "x" in expression.lower():
            return f"{a} × {b} = {a * b}"
        elif "/" in expression or "chia" in expression:
            if b != 0:
                return f"{a} ÷ {b} = {a / b:.4f}"
            return "Lỗi: Không thể chia cho 0"
    return f"Không thể tính toán từ: '{expression}'"

def tool_get_datetime(_: str = "") -> str:
    """Lấy ngày giờ hiện tại."""
    now = datetime.now()
    return f"Ngày: {now.strftime('%d/%m/%Y')} | Giờ: {now.strftime('%H:%M:%S')} | Thứ: {['Hai','Ba','Tư','Năm','Sáu','Bảy','Chủ nhật'][now.weekday()]}"

TOOL_REGISTRY = {
    "search_knowledge": tool_search_knowledge,
    "calculate": tool_calculate,
    "get_datetime": tool_get_datetime,
}


# ──────────────────────────────────────────────────────
#  MEMORY MANAGER (Short-term + State)
# ──────────────────────────────────────────────────────
class MemoryManager:
    MAX_MESSAGES = 10   # Sliding window
    SUMMARY_THRESHOLD = 6

    def __init__(self):
        self.conversation_buffer: List[dict] = []
        self.session_stats = {"total_queries": 0, "topics_covered": set()}

    def add_message(self, role: str, content: str):
        self.conversation_buffer.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.conversation_buffer) > self.MAX_MESSAGES:
            # Sliding window: giữ 2 đầu + n cuối
            self.conversation_buffer = (
                self.conversation_buffer[:2] +
                self.conversation_buffer[-(self.MAX_MESSAGES - 2):]
            )

    def get_context_window(self) -> List[dict]:
        return self.conversation_buffer[-self.SUMMARY_THRESHOLD:]

    def get_stats(self) -> dict:
        return {
            "buffer_size": len(self.conversation_buffer),
            "total_queries": self.session_stats["total_queries"],
            "topics": list(self.session_stats["topics_covered"])
        }

    def track_topic(self, topic: str):
        self.session_stats["topics_covered"].add(topic)
        self.session_stats["total_queries"] += 1


# ──────────────────────────────────────────────────────
#  LANGGRAPH NODES (Các Agent trong hệ thống)
# ──────────────────────────────────────────────────────
llm = MockLLM()
memory_manager = MemoryManager()

def node_intent_classifier(state: ChatbotState) -> dict:
    """
    NODE 1: Intent Classifier
    Phân loại ý định của người dùng.
    Output: cập nhật trường 'intent' trong State.
    """
    query = state["current_query"]
    intent = llm.classify_intent(query)
    memory_manager.track_topic(intent)
    return {
        "intent": intent,
        "processing_log": [f"[IntentClassifier] Query: '{query[:40]}' → Intent: '{intent}'"],
        "iteration_count": 0,
    }

def node_memory_retriever(state: ChatbotState) -> dict:
    """
    NODE 2: Memory Retriever
    Truy vấn Short-term Memory và Long-term Memory (RAG mock).
    Output: cập nhật 'retrieved_context' trong State.
    """
    intent = state["intent"]
    query = state["current_query"]
    # Giả lập Semantic Search trong Vector DB
    context = tool_search_knowledge(intent)
    # Thêm conversation history context nếu có
    recent = memory_manager.get_context_window()
    history_context = ""
    if len(recent) > 2:
        history_context = f"\n[Ngữ cảnh hội thoại gần nhất: {len(recent)} tin nhắn]"
    return {
        "retrieved_context": context + history_context,
        "processing_log": [f"[MemoryRetriever] Retrieved {len(context.split(chr(10)))} chunks for intent '{intent}'"],
    }

def node_react_agent(state: ChatbotState) -> dict:
    """
    NODE 3: ReAct Agent (Reasoning and Acting)
    Vòng lặp: Thought → Action → Observation
    Đây là node trung tâm thể hiện cơ chế ReAct.
    """
    query = state["current_query"]
    intent = state["intent"]
    iteration = state.get("iteration_count", 0)
    MAX_ITER = 5  # Hard limit tránh infinite loop

    if iteration >= MAX_ITER:
        return {
            "react_thought": "Đã đạt giới hạn vòng lặp. Dừng ReAct.",
            "react_action": "FINISH",
            "react_observation": "Max iterations reached.",
            "processing_log": [f"[ReActAgent] ⚠ Max iterations ({MAX_ITER}) reached — stopping"],
        }

    # THOUGHT
    thought = llm.generate_thought(query, intent)
    # ACTION
    action_name, action_input = llm.decide_action(intent)
    # Nếu calculate, dùng query làm input
    if action_name == "calculate":
        action_input = query
    # OBSERVATION — thực thi tool
    tool_fn = TOOL_REGISTRY.get(action_name, tool_search_knowledge)
    observation = tool_fn(action_input)

    log_entry = (
        f"[ReActAgent] Iter {iteration+1} | "
        f"Thought: '{thought[:50]}...' | "
        f"Action: {action_name}({action_input[:20]}) | "
        f"Obs: '{observation[:40]}...'"
    )
    return {
        "react_thought": thought,
        "react_action": action_name,
        "react_observation": observation,
        "iteration_count": iteration + 1,
        "processing_log": [log_entry],
    }

def node_response_generator(state: ChatbotState) -> dict:
    """
    NODE 4: Response Generator
    Tổng hợp tất cả ngữ cảnh → sinh câu trả lời cuối cùng.
    """
    query = state["current_query"]
    intent = state["intent"]
    context = state["retrieved_context"]
    observation = state["react_observation"]
    # Nếu là datetime hoặc calculate, dùng trực tiếp observation
    if intent in ("datetime", "calculate") or state["react_action"] in ("get_datetime", "calculate"):
        answer = f"**Kết quả:**\n\n{observation}"
    else:
        answer = llm.generate_answer(query, context, observation, intent)
    # Lưu vào memory
    memory_manager.add_message("user", query)
    memory_manager.add_message("assistant", answer)
    return {
        "final_answer": answer,
        "messages": [
            {"role": "user", "content": query, "ts": datetime.now().isoformat()},
            {"role": "assistant", "content": answer[:200], "ts": datetime.now().isoformat()},
        ],
        "processing_log": [f"[ResponseGenerator] Answer generated ({len(answer)} chars)"],
    }

def node_memory_writer(state: ChatbotState) -> dict:
    """
    NODE 5: Memory Writer (Async Write-back simulation)
    Cập nhật State Store và ghi vào Long-term Memory.
    Trong production: đây sẽ là async task.
    """
    stats = memory_manager.get_stats()
    return {
        "processing_log": [
            f"[MemoryWriter] Buffer: {stats['buffer_size']} msgs | "
            f"Total queries: {stats['total_queries']} | "
            f"Topics: {stats['topics']}"
        ],
    }


# ──────────────────────────────────────────────────────
#  XÂY DỰNG LANGGRAPH
# ──────────────────────────────────────────────────────
def build_graph():
    """Khởi tạo và compile LangGraph workflow."""
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(ChatbotState)

    # Thêm các Nodes
    graph.add_node("intent_classifier", node_intent_classifier)
    graph.add_node("memory_retriever", node_memory_retriever)
    graph.add_node("react_agent", node_react_agent)
    graph.add_node("response_generator", node_response_generator)
    graph.add_node("memory_writer", node_memory_writer)

    # Định nghĩa Edges (luồng dữ liệu)
    graph.set_entry_point("intent_classifier")
    graph.add_edge("intent_classifier", "memory_retriever")
    graph.add_edge("memory_retriever", "react_agent")
    graph.add_edge("react_agent", "response_generator")
    graph.add_edge("response_generator", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile()


# ──────────────────────────────────────────────────────
#  FALLBACK: Simulation mode (nếu không có LangGraph)
# ──────────────────────────────────────────────────────
def simulate_graph(query: str) -> tuple:
    """Giả lập luồng Graph khi LangGraph không có."""
    logs = []
    state = {
        "current_query": query, "messages": [], "intent": "",
        "retrieved_context": "", "react_thought": "", "react_action": "",
        "react_observation": "", "final_answer": "", "iteration_count": 0,
        "processing_log": []
    }
    for node_fn in [node_intent_classifier, node_memory_retriever,
                    node_react_agent, node_response_generator, node_memory_writer]:
        updates = node_fn(state)
        state.update(updates)
        logs.extend(updates.get("processing_log", []))
    return state["final_answer"], logs


# ──────────────────────────────────────────────────────
#  RICH UI — GIAO DIỆN ĐẸP TRONG TERMINAL
# ──────────────────────────────────────────────────────
def display_welcome():
    console.print()
    console.print(Panel.fit(
        "[bold blue]🤖 CHATBOT AI AGENT — LANGGRAPH[/bold blue]\n"
        "[dim]Kiến trúc: Multi-Agent + RAG + Memory 3 lớp[/dim]\n"
        "[dim]Mode: MockLLM (Offline — không cần API key)[/dim]\n\n"
        "[yellow]Chủ đề:[/yellow] LangGraph, CrewAI, ReAct, Memory, RAG, Multi-Agent\n"
        "[dim]Gõ [bold]'help'[/bold] để xem hướng dẫn | [bold]'stats'[/bold] xem thống kê | [bold]'quit'[/bold] để thoát[/dim]",
        border_style="blue",
        title="[bold]THỰC TẬP SINH — HOÀNG MINH ĐỨC[/bold]"
    ))
    console.print()

def display_graph_flow():
    """Hiển thị sơ đồ Graph trong terminal."""
    console.print(Panel(
        "[bold cyan]LANGGRAPH WORKFLOW:[/bold cyan]\n\n"
        "  [START]\n"
        "    ↓\n"
        "  [bold yellow]① intent_classifier[/bold yellow]   ← Phân loại ý định\n"
        "    ↓\n"
        "  [bold yellow]② memory_retriever[/bold yellow]    ← RAG: tìm ngữ cảnh liên quan\n"
        "    ↓\n"
        "  [bold yellow]③ react_agent[/bold yellow]         ← ReAct: Thought→Action→Observation\n"
        "    ↓\n"
        "  [bold yellow]④ response_generator[/bold yellow]  ← Tổng hợp → sinh câu trả lời\n"
        "    ↓\n"
        "  [bold yellow]⑤ memory_writer[/bold yellow]       ← Async write-back vào State Store\n"
        "    ↓\n"
        "  [END]\n\n"
        "[dim]State (TypedDict) chạy xuyên suốt qua tất cả Nodes[/dim]",
        border_style="cyan",
        title="[bold]Kiến trúc Graph[/bold]"
    ))

def display_processing_log(logs: list):
    """Hiển thị log xử lý của từng Node."""
    table = Table(
        title="📊 Processing Log — LangGraph Execution",
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Step", style="cyan", width=6)
    table.add_column("Log", style="white")
    for i, log in enumerate(logs, 1):
        node_name = log.split("]")[0].replace("[", "") if "]" in log else "System"
        detail = log.split("]", 1)[1].strip() if "]" in log else log
        table.add_row(f"[{i}]", f"[bold cyan]{node_name}[/bold cyan]: {detail}")
    console.print(table)

def display_stats():
    stats = memory_manager.get_stats()
    table = Table(title="📈 Session Statistics", border_style="blue")
    table.add_column("Metric", style="yellow", width=25)
    table.add_column("Value", style="green")
    table.add_row("Total Queries", str(stats["total_queries"]))
    table.add_row("Memory Buffer Size", f"{stats['buffer_size']} messages")
    table.add_row("Topics Covered", ", ".join(stats["topics"]) if stats["topics"] else "—")
    table.add_row("LangGraph Available", "✓ Yes" if LANGGRAPH_AVAILABLE else "⚠ Simulation mode")
    table.add_row("LLM Mode", MockLLM.MODEL_NAME)
    console.print(table)

def display_help():
    console.print(Panel(
        "[bold]CÁC CÂU HỎI GỢI Ý:[/bold]\n\n"
        "[cyan]• LangGraph là gì và tại sao nên dùng?[/cyan]\n"
        "[cyan]• So sánh LangGraph và CrewAI[/cyan]\n"
        "[cyan]• Giải thích cơ chế ReAct cho tôi[/cyan]\n"
        "[cyan]• Memory management trong Multi-Agent hoạt động thế nào?[/cyan]\n"
        "[cyan]• RAG là gì? Quy trình gồm những bước nào?[/cyan]\n"
        "[cyan]• Multi-Agent system có những mô hình điều phối nào?[/cyan]\n"
        "[cyan]• Tính 123 + 456[/cyan]\n"
        "[cyan]• Hôm nay là ngày mấy?[/cyan]\n\n"
        "[bold]LỆNH ĐẶC BIỆT:[/bold]\n"
        "[yellow]• graph[/yellow]  — Xem sơ đồ kiến trúc Graph\n"
        "[yellow]• stats[/yellow]  — Xem thống kê phiên làm việc\n"
        "[yellow]• help[/yellow]   — Hiển thị trợ giúp này\n"
        "[yellow]• quit[/yellow]   — Thoát chương trình",
        border_style="green",
        title="[bold]📚 HƯỚNG DẪN SỬ DỤNG[/bold]"
    ))


# ──────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────
def main():
    print("\n[2/4] Khởi tạo LangGraph State Graph...")
    graph = build_graph()
    if graph:
        console.print("[green]  [✓] LangGraph compiled thành công[/green]")
    else:
        console.print("[yellow]  [⚠] Dùng simulation mode[/yellow]")

    print("[3/4] Khởi tạo Memory Manager...")
    console.print("[green]  [✓] Memory Manager sẵn sàng[/green]")

    print("[4/4] Khởi tạo MockLLM Engine...")
    console.print(f"[green]  [✓] {MockLLM.MODEL_NAME}[/green]")
    console.print()

    display_welcome()

    turn = 0
    while True:
        try:
            # Prompt đầu vào
            turn += 1
            console.print(f"[dim]─── Turn {turn} ───────────────────────────────────[/dim]")
            user_input = console.input("[bold green]Bạn[/bold green] › ").strip()

            if not user_input:
                continue

            # Lệnh đặc biệt
            if user_input.lower() in ("quit", "exit", "thoát"):
                console.print("\n[bold blue]👋 Cảm ơn bạn đã sử dụng LangGraph Chatbot![/bold blue]")
                display_stats()
                break
            if user_input.lower() == "help":
                display_help()
                continue
            if user_input.lower() == "stats":
                display_stats()
                continue
            if user_input.lower() == "graph":
                display_graph_flow()
                continue

            # Xử lý câu hỏi qua LangGraph
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console, transient=True
            ) as progress:
                task = progress.add_task("[cyan]Đang xử lý qua LangGraph...", total=None)

                initial_state: ChatbotState = {
                    "messages": [], "current_query": user_input,
                    "intent": "", "retrieved_context": "",
                    "react_thought": "", "react_action": "",
                    "react_observation": "", "final_answer": "",
                    "iteration_count": 0, "processing_log": [],
                }
                start_time = time.time()
                if graph:
                    result = graph.invoke(initial_state)
                    answer = result["final_answer"]
                    logs = result["processing_log"]
                else:
                    answer, logs = simulate_graph(user_input)
                elapsed = time.time() - start_time

            # Hiển thị log (có thể tắt nếu muốn gọn hơn)
            display_processing_log(logs)
            console.print()

            # Hiển thị câu trả lời
            console.print(Panel(
                Markdown(answer),
                border_style="blue",
                title=f"[bold blue]🤖 Chatbot[/bold blue]  [dim]({elapsed:.2f}s)[/dim]",
                padding=(1, 2)
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Ctrl+C được nhấn. Gõ 'quit' để thoát hoặc tiếp tục.[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Lỗi: {e}[/red]")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
