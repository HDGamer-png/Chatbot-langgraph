"""
core/graph.py — Đồ thị LangGraph StateGraph kết nối 10 Agent.

Luồng thực thi 2 pha:
  START → IntentClassifier → Router (Fan-out Song song) → BARRIER
        → Planner (Tuần tự) → Validator → ResponseGenerator → END

Thiết kế linh hoạt:
  - Nếu LangGraph được cài đặt: sử dụng StateGraph chính thức.
  - Nếu không có LangGraph: sử dụng SimpleGraph nội bộ (fallback tương thích).
"""

import time
from datetime import datetime
from typing import Any

from .state import AgentState, create_initial_state
from .agents.intent import IntentClassifierAgent
from .agents.planner import PlannerAgent
from .agents.validator import ValidatorAgent
from .agents.response import ResponseGeneratorAgent
from .router import RouterAgent


# ══════════════════════════════════════════════════════════════════
#  CÁC NODE FUNCTION (Mỗi hàm nhận State, xử lý, trả về dict cập nhật)
# ══════════════════════════════════════════════════════════════════

def node_classify_intent(state: dict) -> dict:
    """Node 0: Phân loại ý định người dùng."""
    agent = IntentClassifierAgent(called_by="GraphStart")
    result = agent.safe_run(query=state["user_query"])
    return {
        "intent": result.get("intent", "general"),
        "intent_confidence": result.get("confidence", 0.0),
        "agent_timings": state.get("agent_timings", []) + [agent.get_timing()],
        "call_graph": state.get("call_graph", []) + agent.get_call_edges(),
        "log": state.get("log", []) + [
            agent._log(f"Phân loại: intent='{result.get('intent')}' "
                        f"(confidence={result.get('confidence', 0):.2f})")
        ],
    }


def node_router_parallel(state: dict) -> dict:
    """Node 1: Điều phối song song — Fan-out các Worker Agent."""
    router = RouterAgent()
    result = router.safe_run(
        query=state["user_query"],
        intent=state["intent"],
        kb_context=state.get("kb_context", ""),
    )
    return {
        "kb_context": result.get("kb_context", state.get("kb_context", "")),
        "calc_result": result.get("calc_result", ""),
        "datetime_result": result.get("datetime_result", ""),
        "web_results": result.get("web_results", ""),
        "analysis": result.get("analysis", ""),
        "agent_timings": state.get("agent_timings", []) + [router.get_timing()] + result.get("agent_timings", []),
        "call_graph": state.get("call_graph", []) + router.get_call_edges() + result.get("call_graph", []),
        "log": state.get("log", []) + result.get("log", []),
    }


def node_planner(state: dict) -> dict:
    """Node 2 (Tuần tự): Lập kế hoạch hành động."""
    agent = PlannerAgent(called_by="Router")
    result = agent.safe_run(
        query=state["user_query"],
        intent=state["intent"],
        context=state.get("kb_context", ""),
    )
    return {
        "planning": result.get("planning", ""),
        "agent_timings": state.get("agent_timings", []) + [agent.get_timing()],
        "call_graph": state.get("call_graph", []) + agent.get_call_edges(),
        "log": state.get("log", []) + [agent._log("Kế hoạch hành động đã được lập.")],
    }


def node_validator(state: dict) -> dict:
    """Node 3 (Tuần tự): Kiểm định chéo kết quả."""
    agent = ValidatorAgent(called_by="Planner")
    result = agent.safe_run(
        query=state["user_query"],
        intent=state["intent"],
        kb_context=state.get("kb_context", ""),
        calc_result=state.get("calc_result", ""),
        web_results=state.get("web_results", ""),
        analysis=state.get("analysis", ""),
        planning=state.get("planning", ""),
    )
    return {
        "validation": result.get("validation", ""),
        "confidence_score": result.get("confidence_score", 0.0),
        "agent_timings": state.get("agent_timings", []) + [agent.get_timing()],
        "call_graph": state.get("call_graph", []) + agent.get_call_edges(),
        "log": state.get("log", []) + [
            agent._log(f"Confidence Score = {result.get('confidence_score', 0):.0%}")
        ],
    }


def node_response_generator(state: dict) -> dict:
    """Node 4 (Tuần tự cuối): Tổng hợp câu trả lời Markdown."""
    agent = ResponseGeneratorAgent(called_by="Validator")
    result = agent.safe_run(
        query=state["user_query"],
        intent=state["intent"],
        kb_context=state.get("kb_context", ""),
        calc_result=state.get("calc_result", ""),
        datetime_result=state.get("datetime_result", ""),
        web_results=state.get("web_results", ""),
        analysis=state.get("analysis", ""),
        planning=state.get("planning", ""),
        validation=state.get("validation", ""),
        confidence_score=state.get("confidence_score", 0.0),
        agent_timings=state.get("agent_timings", []),
    )
    return {
        "final_answer": result.get("final_answer", "Xin lỗi, tôi không thể tạo câu trả lời."),
        "agent_timings": state.get("agent_timings", []) + [agent.get_timing()],
        "call_graph": state.get("call_graph", []) + agent.get_call_edges(),
        "log": state.get("log", []) + [agent._log("Câu trả lời cuối cùng đã được tổng hợp.")],
    }


# ══════════════════════════════════════════════════════════════════
#  SIMPLE GRAPH — Fallback khi không có LangGraph
# ══════════════════════════════════════════════════════════════════

class SimpleGraph:
    """
    Đồ thị thực thi tuần tự đơn giản, mô phỏng giao diện của LangGraph StateGraph.
    Dùng làm fallback khi langgraph chưa được cài đặt.
    """

    def __init__(self):
        self._nodes: list[tuple[str, callable]] = []

    def add_node(self, name: str, fn: callable):
        self._nodes.append((name, fn))

    def invoke(self, state: dict) -> dict:
        """Chạy tuần tự qua tất cả các node, cập nhật state sau mỗi bước."""
        for node_name, fn in self._nodes:
            try:
                updates = fn(state)
                if isinstance(updates, dict):
                    state.update(updates)
            except Exception as exc:
                state.setdefault("log", []).append(
                    f"[ERROR] Node '{node_name}' gặp lỗi: {exc}"
                )
        return state


# ══════════════════════════════════════════════════════════════════
#  BUILD GRAPH — Hàm chính xây dựng đồ thị
# ══════════════════════════════════════════════════════════════════

def build_graph():
    """
    Xây dựng đồ thị thực thi Multi-Agent.

    Thử sử dụng LangGraph StateGraph trước.
    Nếu không có, dùng SimpleGraph fallback.

    Returns:
        Đối tượng graph có phương thức .invoke(state).
    """
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)

        # Thêm các Node
        graph.add_node("classify_intent", node_classify_intent)
        graph.add_node("router_parallel", node_router_parallel)
        graph.add_node("planner", node_planner)
        graph.add_node("validator", node_validator)
        graph.add_node("response_generator", node_response_generator)

        # Thiết lập luồng:
        # START → classify → router(parallel) → planner → validator → response → END
        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "router_parallel")
        graph.add_edge("router_parallel", "planner")
        graph.add_edge("planner", "validator")
        graph.add_edge("validator", "response_generator")
        graph.add_edge("response_generator", END)

        compiled = graph.compile()
        print("[Graph] [OK] LangGraph StateGraph da duoc bien dich thanh cong.")
        return compiled

    except ImportError:
        print("[Graph] [WARN] LangGraph chua duoc cai dat. Su dung SimpleGraph fallback.")
        graph = SimpleGraph()
        graph.add_node("classify_intent", node_classify_intent)
        graph.add_node("router_parallel", node_router_parallel)
        graph.add_node("planner", node_planner)
        graph.add_node("validator", node_validator)
        graph.add_node("response_generator", node_response_generator)
        return graph


# ══════════════════════════════════════════════════════════════════
#  HANDLE QUERY — Hàm tiện ích xử lý một câu hỏi hoàn chỉnh
# ══════════════════════════════════════════════════════════════════

# Cache graph toàn cục để không phải build lại mỗi lượt
_graph = None


def handle_query(
    session_id: str,
    user_query: str,
    turn_id: int = 0,
    user_id: str = "",
    history: list[dict] | None = None,
) -> dict:
    """
    Xử lý một câu hỏi từ người dùng qua toàn bộ pipeline Multi-Agent.

    Args:
        session_id: ID phiên hội thoại.
        user_query: Câu hỏi của người dùng.
        turn_id: Số thứ tự lượt hỏi.
        user_id: ID người dùng (tùy chọn).
        history: Lịch sử hội thoại trước đó.

    Returns:
        dict chứa final_answer, intent, agent_timings, call_graph, log.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()

    # Tạo state khởi tạo
    initial_state = create_initial_state(
        session_id=session_id,
        user_query=user_query,
        turn_id=turn_id,
        user_id=user_id,
        history=history,
    )

    # Chạy đồ thị
    start_time = time.time()
    final_state = _graph.invoke(initial_state)
    total_time = time.time() - start_time

    # Bổ sung tổng thời gian
    final_state["log"] = final_state.get("log", []) + [
        f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [Pipeline] "
        f"Tổng thời gian xử lý: {total_time:.3f}s"
    ]

    return dict(final_state)

