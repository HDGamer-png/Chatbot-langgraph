"""
LangGraph Multi-Agent Chatbot
Orchestrator → goi cac Agent chuyen biet qua Tool Calls
Chay: python multi_agent_chatbot.py
.env: ANTHROPIC_API_KEY=sk-ant-...  TAVILY_API_KEY=tvly-... (tuy chon)
"""
import os, re, math, json, operator
from datetime import datetime
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
load_dotenv()

# ── Packages ─────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# ── Config ───────────────────────────────────────────────────────────
ANT_KEY    = os.getenv("ANTHROPIC_API_KEY","")
TAV_KEY    = os.getenv("TAVILY_API_KEY","")
MODEL      = "claude-sonnet-4-20250514"
MAX_ITER   = 5   # ReAct loop limit

if not ANT_KEY:
    ANT_KEY = input("Anthropic API Key: ").strip()
    os.environ["ANTHROPIC_API_KEY"] = ANT_KEY

# ═══════════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════════
class State(TypedDict):
    messages : Annotated[List, operator.add]   # toàn bộ lịch sử LC messages
    iteration: int
    final    : str

# ═══════════════════════════════════════════════════════════════════
#  CÁC AI AGENT CHUYÊN BIỆT (mỗi agent là 1 hàm Python)
# ═══════════════════════════════════════════════════════════════════

def agent_search(query: str) -> str:
    """Search Agent — tìm kiếm web (Tavily) hoặc KB nội bộ."""
    if TAV_KEY:
        from tavily import TavilyClient
        try:
            r = TavilyClient(api_key=TAV_KEY).search(
                query=query, max_results=4, include_answer=True)
            ans = r.get("answer","")
            snippets = " | ".join(x.get("content","")[:200] for x in r.get("results",[])[:3])
            return ans or snippets or "Khong tim thay ket qua."
        except Exception as e:
            return f"[Search error] {e}"
    # KB fallback
    KB = {
        "langgraph": "LangGraph xay dung Multi-Agent duoi dang Directed Graph. Node=Agent, Edge=luong du lieu. State TypedDict chay xuyen suot.",
        "crewai":    "CrewAI dung khai bao declarative: Agent+Task+Crew. Phu hop PoC nhanh, khong co human-in-the-loop.",
        "react":     "ReAct=Reasoning+Acting: Thought->Action->Observation. Ep LLM suy nghi truoc hanh dong.",
        "rag":       "RAG: Embed->VectorDB->CosineSimilarity->top-k chunks->prompt. Giai quyet LLM outdated.",
        "memory":    "3 lop: Short-term(sliding window), Long-term(VectorDB), State(Blackboard JSON).",
    }
    q = query.lower()
    for k,v in KB.items():
        if k in q: return v
    return "Khong co thong tin trong KB. Can Tavily key de tim web."


def agent_calculator(expression: str) -> str:
    """Calculator Agent — tính toán toán học an toàn."""
    try:
        expr = expression
        for a,b in [("^","**"),("×","*"),("÷","/"),("√","sqrt"),
                    ("π","pi"),("sin","sin"),("cos","cos"),("tan","tan"),
                    ("log","log10"),("ln","log"),("sqrt","sqrt")]:
            expr = re.sub(rf'\b{re.escape(a)}\b', b, expr, flags=re.IGNORECASE)
        ns = {"__builtins__":{}, "sqrt":math.sqrt, "log":math.log,
              "log10":math.log10, "sin":math.sin, "cos":math.cos,
              "tan":math.tan, "pi":math.pi, "e":math.e, "abs":abs}
        result = eval(expr, ns)
        fmt = f"{result:,.10g}" if isinstance(result,float) else f"{result:,}"
        return f"{expression} = {fmt}"
    except Exception as ex:
        return f"Loi tinh toan: {ex}"


def agent_datetime(_: str = "") -> str:
    """DateTime Agent — lấy ngày giờ hiện tại."""
    now = datetime.now()
    days = ["Thu Hai","Thu Ba","Thu Tu","Thu Nam","Thu Sau","Thu Bay","Chu Nhat"]
    return (f"{days[now.weekday()]}, {now.day}/{now.month}/{now.year}, "
            f"{now.strftime('%H:%M:%S')}, Tuan {now.isocalendar()[1]}")


def agent_analyst(topic: str) -> str:
    """Analyst Agent — phân tích sâu bằng LLM riêng (sub-call)."""
    llm = ChatAnthropic(model=MODEL, max_tokens=400, temperature=0.3)
    resp = llm.invoke([
        SystemMessage(content="Chuyen gia phan tich AI Agent. Tra loi ngan gon, co cau truc, tieng Viet."),
        HumanMessage(content=f"Phan tich ngan gon: {topic}")
    ])
    return resp.content


def agent_coder(task: str) -> str:
    """Coder Agent — sinh code Python mẫu."""
    llm = ChatAnthropic(model=MODEL, max_tokens=500, temperature=0.2)
    resp = llm.invoke([
        SystemMessage(content="Senior Python dev. Chi tra code + comment ngan. Khong giai thich dai."),
        HumanMessage(content=task)
    ])
    return resp.content

# ═══════════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS (JSON Schema cho Orchestrator LLM biết gọi)
# ═══════════════════════════════════════════════════════════════════
TOOLS = [
    {
        "name": "search_agent",
        "description": "Tim kiem thong tin web (Tavily) hoac knowledge base noi bo. Dung khi can du lieu thuc te, su kien, thong tin moi.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Tu khoa tim kiem"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculator_agent",
        "description": "Tinh toan toan hoc: so hoc, luy thua(^), can(sqrt), sin/cos/tan, log, ln, pi, e.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Bieu thuc can tinh"}},
            "required": ["expression"]
        }
    },
    {
        "name": "datetime_agent",
        "description": "Lay ngay gio hien tai, thu trong tuan, so tuan.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": []
        }
    },
    {
        "name": "analyst_agent",
        "description": "Phan tich sau mot chu de: uu nhuoc diem, so sanh, danh gia ky thuat. Goi LLM rieng.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "Chu de can phan tich"}},
            "required": ["topic"]
        }
    },
    {
        "name": "coder_agent",
        "description": "Sinh code Python mau cho mot nhiem vu lap trinh cu the.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "Mo ta nhiem vu can code"}},
            "required": ["task"]
        }
    },
]

# Router: tool name -> hàm agent
AGENT_FN = {
    "search_agent":    lambda args: agent_search(args.get("query","")),
    "calculator_agent":lambda args: agent_calculator(args.get("expression","")),
    "datetime_agent":  lambda args: agent_datetime(),
    "analyst_agent":   lambda args: agent_analyst(args.get("topic","")),
    "coder_agent":     lambda args: agent_coder(args.get("task","")),
}

# ═══════════════════════════════════════════════════════════════════
#  ORCHESTRATOR LLM
# ═══════════════════════════════════════════════════════════════════
orchestrator = ChatAnthropic(
    model=MODEL,
    max_tokens=1024,
    temperature=0.5,
).bind_tools(TOOLS)   # bind tools → LLM tự sinh tool_use blocks

SYSTEM = SystemMessage(content=
    "Ban la Orchestrator cua he thong Multi-Agent. "
    "Phan tich yeu cau, goi dung Agent chuyen biet qua tools, "
    "roi tong hop ket qua thanh cau tra loi hoan chinh. "
    "Tra loi tieng Viet, ngan gon, ro rang."
)

# ═══════════════════════════════════════════════════════════════════
#  LANGGRAPH NODES
# ═══════════════════════════════════════════════════════════════════

def node_orchestrator(state: State) -> dict:
    """Orchestrator gọi LLM → có thể sinh tool_use hoặc text cuối."""
    msgs = [SYSTEM] + state["messages"]
    resp = orchestrator.invoke(msgs)
    return {
        "messages":  [resp],
        "iteration": state["iteration"] + 1,
        "final":     resp.content if not resp.tool_calls else "",
    }


def node_tool_router(state: State) -> dict:
    """Đọc tool_calls từ message cuối, dispatch đến đúng Agent, trả ToolMessage."""
    last = state["messages"][-1]
    tool_msgs = []
    for tc in last.tool_calls:
        name   = tc["name"]
        args   = tc["args"]
        fn     = AGENT_FN.get(name)
        result = fn(args) if fn else f"[Loi] Khong tim thay agent: {name}"
        print(f"  → [{name}] {json.dumps(args, ensure_ascii=False)[:60]}")
        print(f"    ← {result[:120]}{'...' if len(result)>120 else ''}")
        tool_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": tool_msgs}


# ── Conditional edge ─────────────────────────────────────────────
def should_continue(state: State) -> str:
    last = state["messages"][-1]
    # Nếu LLM sinh tool_calls VÀ chưa đạt max → chạy tool
    if getattr(last, "tool_calls", None) and state["iteration"] < MAX_ITER:
        return "tools"
    return "end"

# ═══════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════
g = StateGraph(State)
g.add_node("orchestrator", node_orchestrator)
g.add_node("tools",        node_tool_router)

g.set_entry_point("orchestrator")
g.add_conditional_edges("orchestrator", should_continue, {
    "tools": "tools",
    "end":   END,
})
g.add_edge("tools", "orchestrator")   # sau tools → quay lại orchestrator

graph = g.compile()

# ═══════════════════════════════════════════════════════════════════
#  CONVERSATION MEMORY
# ═══════════════════════════════════════════════════════════════════
history: List = []   # lưu HumanMessage + AIMessage xuyên phiên

def chat(user_input: str) -> str:
    global history
    history.append(HumanMessage(content=user_input))

    result = graph.invoke({
        "messages":  history,
        "iteration": 0,
        "final":     "",
    })

    # Lấy câu trả lời text cuối cùng từ AIMessage
    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content
            break

    # Cập nhật history (chỉ giữ HumanMessage + AIMessage cuối)
    history.append(AIMessage(content=answer))
    if len(history) > 20:   # sliding window 10 turns
        history = history[-20:]

    return answer

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
AGENTS_INFO = """
Agents co san:
  search_agent     - Tim kiem web (Tavily) / KB noi bo
  calculator_agent - Tinh toan toan hoc
  datetime_agent   - Ngay gio hien tai
  analyst_agent    - Phan tich sau (sub LLM call)
  coder_agent      - Sinh Python code
"""

def main():
    print("\n" + "="*55)
    print("  LANGGRAPH MULTI-AGENT CHATBOT")
    print(f"  Model: {MODEL}")
    print(f"  Search: {'Tavily (that)' if TAV_KEY else 'KB noi bo (mock)'}")
    print("="*55)
    print(AGENTS_INFO)
    print("Go 'quit' de thoat | 'agents' xem danh sach\n")

    while True:
        try:
            user = input("Ban > ").strip()
            if not user: continue
            if user.lower() in ("quit","exit","q"):
                print("Tam biet!")
                break
            if user.lower() == "agents":
                print(AGENTS_INFO)
                continue

            print("Dang xu ly", end="", flush=True)
            answer = chat(user)
            print(f"\rChatbot > {answer}\n")

        except KeyboardInterrupt:
            print("\nCtrl+C - go 'quit' de thoat")
        except Exception as ex:
            print(f"\n[Loi] {ex}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
