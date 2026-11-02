"""
core/state.py — Định nghĩa AgentState (Blackboard) xuyên suốt đồ thị LangGraph.

Mọi dữ liệu truyền qua các Node (Agent) đều nằm trong đối tượng State này.
Chia làm 4 vùng rõ ràng:
  1. Input:           Dữ liệu đầu vào từ người dùng
  2. Parallel Results: Kết quả từ pha chạy song song (Fan-out)
  3. Sequential Results: Kết quả từ pha chạy tuần tự (Pipeline)
  4. Tracing:          Dữ liệu giám sát hiệu năng & cây gọi hàm
"""

from typing import TypedDict, List, Any


class AgentState(TypedDict):
    # ── Vùng 1: Đầu vào (Input) ──────────────────────────────────────────
    session_id: str
    turn_id: int
    user_query: str
    user_id: str

    # ── Vùng 2: Phân loại ý định (Classification) ────────────────────────
    intent: str                    # Nhãn ý định: calculate, datetime, rag, search, analyze, plan, greet, general
    intent_confidence: float       # Độ tin cậy phân loại (0.0 - 1.0)

    # ── Vùng 3: Kết quả Pha Song song (Parallel Workers) ─────────────────
    kb_context: str                # Trích dẫn tài liệu từ KnowledgeBaseAgent (RAG)
    calc_result: str               # Kết quả tính toán từ CalculatorAgent
    datetime_result: str           # Kết quả thời gian từ DateTimeAgent
    web_results: str               # Kết quả tìm kiếm từ WebSearchAgent
    analysis: str                  # Kết quả phân tích từ AnalystAgent

    # ── Vùng 4: Kết quả Pha Tuần tự (Sequential Pipeline) ────────────────
    planning: str                  # Kế hoạch hành động từ PlannerAgent
    validation: str                # Báo cáo kiểm định từ ValidatorAgent
    confidence_score: float        # Điểm tin cậy tổng thể (0.0 - 1.0)
    final_answer: str              # Câu trả lời cuối cùng từ ResponseGenerator

    # ── Vùng 5: Giám sát & Truy vết (Tracing) ────────────────────────────
    agent_timings: List[dict]      # Danh sách {agent, called_by, duration, calls}
    call_graph: List[dict]         # Cây gọi hàm {from, to, ts}
    log: List[str]                 # Nhật ký xử lý từng bước
    conversation_history: List[dict]  # Lịch sử hội thoại (user_query, final_answer)


def create_initial_state(
    session_id: str,
    user_query: str,
    turn_id: int = 0,
    user_id: str = "",
    history: List[dict] | None = None,
) -> AgentState:
    """Tạo state khởi tạo cho mỗi lượt hội thoại."""
    return AgentState(
        session_id=session_id,
        turn_id=turn_id,
        user_query=user_query,
        user_id=user_id,
        intent="",
        intent_confidence=0.0,
        kb_context="",
        calc_result="",
        datetime_result="",
        web_results="",
        analysis="",
        planning="",
        validation="",
        confidence_score=0.0,
        final_answer="",
        agent_timings=[],
        call_graph=[],
        log=[],
        conversation_history=history or [],
    )

