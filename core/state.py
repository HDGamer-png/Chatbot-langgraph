from typing import TypedDict, List

class AgentState(TypedDict):
    session_id: str
    turn_id: int
    user_query: str
    user_id: str
    intent: str
    intent_confidence: float
    kb_context: str
    calc_result: str
    datetime_result: str
    web_results: str
    analysis: str
    planning: str
    validation: str
    confidence_score: float
    final_answer: str
    agent_timings: List[dict]
    call_graph: List[dict]
    log: List[str]
    conversation_history: List[dict]
    history_summary: str
    recent_context: str

# Khởi tạo đối tượng trạng thái ban đầu cho một lượt hội thoại.
def create_initial_state(
    session_id: str,
    user_query: str,
    turn_id: int = 0,
    user_id: str = "",
    history: List[dict] | None = None,
    history_summary: str = "",
    recent_context: str = "",
) -> AgentState:
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
        history_summary=history_summary,
        recent_context=recent_context,
    )
