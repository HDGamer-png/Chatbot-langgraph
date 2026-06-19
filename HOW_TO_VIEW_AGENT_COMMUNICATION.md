# 🏗️ Multi-Agent Communication Flow - Visual Guide

## 🎯 Quick Overview

**Ứng dụng này CÓ kiến trúc Multi-Agent**  
Có **2 version chính**:
- ✅ **v1**: `multi_agent_chatbot.py` — Đơn giản, dễ hiểu
- ✅ **v3**: `multi_agent_chatbot_v3.py` — Advanced, nhanh (khuyến khích)

---

## 📍 Xem Code Chi Tiết

### ✅ Version 1 (Simple & Clear)

**File: `multi_agent_chatbot.py`**

#### 1️⃣ **Xem các Agents chuyên biệt** (Dòng 40-110)
```python
def agent_search(query: str) -> str:
    """Tìm kiếm web (Tavily) hoặc KB"""
    ...

def agent_calculator(expression: str) -> str:
    """Tính toán toán học"""
    ...

def agent_datetime(_: str = "") -> str:
    """Lấy ngày giờ hiện tại"""
    ...

def agent_analyst(topic: str) -> str:
    """Phân tích sâu bằng LLM riêng"""
    ...

def agent_coder(task: str) -> str:
    """Sinh code Python mẫu"""
    ...
```

👉 **Xem ở:** Dòng 40-110 trong `multi_agent_chatbot.py`

---

#### 2️⃣ **Xem cách Orchestrator gọi Agents** (Dòng 112-170)

```python
# TOOLS = định nghĩa schema cho Orchestrator LLM biết
TOOLS = [
    {
        "name": "search_agent",
        "description": "Tìm kiếm thông tin...",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }
    },
    # ... 5 agents khác
]

# AGENT_FN = router mapping
AGENT_FN = {
    "search_agent":     lambda args: agent_search(args.get("query","")),
    "calculator_agent": lambda args: agent_calculator(args.get("expression","")),
    # ... etc
}
```

👉 **Xem ở:** Dòng 112-170 trong `multi_agent_chatbot.py`

---

#### 3️⃣ **Xem Orchestrator Setup** (Dòng 172-186)

```python
# Orchestrator LLM + bind tools
orchestrator = ChatAnthropic(
    model=MODEL,
    max_tokens=1024,
    temperature=0.5,
).bind_tools(TOOLS)  # ◄─── LLM giờ biết gọi tools

SYSTEM = SystemMessage(content=
    "Bạn là Orchestrator của hệ thống Multi-Agent. "
    "Phân tích yêu cầu, gọi đúng Agent chuyên biệt qua tools, "
    "rồi tổng hợp kết quả thành câu trả lời hoàn chỉnh."
)
```

👉 **Xem ở:** Dòng 172-186 trong `multi_agent_chatbot.py`

---

#### 4️⃣ **Xem Node Orchestrator gọi LLM** (Dòng 191-200)

```python
def node_orchestrator(state: State) -> dict:
    """Orchestrator gọi LLM → có thể sinh tool_use hoặc text cuối"""
    msgs = [SYSTEM] + state["messages"]
    
    # LLM tự quyết định: gọi tool hay trả lời?
    resp = orchestrator.invoke(msgs)
    
    return {
        "messages":  [resp],
        "iteration": state["iteration"] + 1,
        "final":     resp.content if not resp.tool_calls else "",
    }
```

👉 **Xem ở:** Dòng 191-200 trong `multi_agent_chatbot.py`

---

#### 5️⃣ **Xem Tool Router dispatch Agents** (Dòng 203-217)

```python
def node_tool_router(state: State) -> dict:
    """Đọc tool_calls từ message cuối, dispatch đến đúng Agent"""
    last = state["messages"][-1]
    tool_msgs = []
    
    # Parse tool_calls từ Orchestrator
    for tc in last.tool_calls:  # ◄─── tool_calls = {"name": "...", "args": {...}}
        name   = tc["name"]           # ◄─── "search_agent"
        args   = tc["args"]           # ◄─── {"query": "..."}
        
        fn     = AGENT_FN.get(name)   # ◄─── Tìm hàm đúng
        result = fn(args) if fn else f"[Lỗi]"
        
        # Trace output
        print(f"  → [{name}] {json.dumps(args)[:60]}")
        print(f"    ← {result[:120]}")
        
        # Trả ToolMessage
        tool_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    
    return {"messages": tool_msgs}
```

👉 **Xem ở:** Dòng 203-217 trong `multi_agent_chatbot.py`

---

#### 6️⃣ **Xem LangGraph Setup** (Dòng 229-238)

```python
# Build graph
g = StateGraph(State)
g.add_node("orchestrator", node_orchestrator)
g.add_node("tools",        node_tool_router)

# Set entry point
g.set_entry_point("orchestrator")

# Conditional routing
def should_continue(state: State) -> str:
    last = state["messages"][-1]
    # Nếu có tool_calls + chưa max iteration → chạy tools
    if getattr(last, "tool_calls", None) and state["iteration"] < MAX_ITER:
        return "tools"
    return "end"

g.add_conditional_edges("orchestrator", should_continue, {
    "tools": "tools",
    "end":   END,
})

g.add_edge("tools", "orchestrator")  # ◄─── Loop: tools → orchestrator

graph = g.compile()
```

👉 **Xem ở:** Dòng 229-238 trong `multi_agent_chatbot.py`

---

## 🚀 Version 3 (Advanced & Fast)

**File: `multi_agent_chatbot_v3.py`** — **Khuyến khích dùng**

#### 🏗️ **Agent Architecture**

| Agent | Mục Đích | Dòng |
|-------|---------|------|
| **IntentClassifier** | Phân loại ý định user (search/calc/qa) | 482-560 |
| **KnowledgeBase** | Lấy chunk từ vector DB | 561-643 |
| **Calculator** | Tính toán toán học | 644-668 |
| **DateTime** | Lấy ngày giờ | 669-690 |
| **WebSearch** | Tìm kiếm web (Tavily) | 691-717 |
| **Analyst** | Phân tích sâu (LLM call) | 718-756 |
| **Planner** | Lập kế hoạch | 757-774 |
| **Validator** | Kiểm tra kết quả | 775-792 |
| **Router** (Orchestrator v2) | Dispatch song song tất cả agents | **795-892** |
| **ResponseGenerator** | Tổng hợp kết quả → response | 893-992 |

👉 **Xem tất cả agents:** Dòng 450-1000 trong `multi_agent_chatbot_v3.py`

---

#### 🔑 **Router Agent — Trái tim giao tiếp** (Dòng 795-892)

```python
class RouterAgent(AgentBase):
    """Dispatcher — gọi 5-8 agents song song"""
    
    def invoke(self, intent, entities, kb_chunk, prev_answer):
        # Step 1: Xác định agents cần gọi
        agents_to_call = self._determine_agents(intent)
        
        # Step 2: ThreadPool - gọi song song
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            
            # Submit tất cả agents cùng lúc
            for agent_name in agents_to_call:
                future = executor.submit(
                    self._call_agent,
                    agent_name,
                    entities
                )
                futures[future] = agent_name
            
            # Collect kết quả khi xong
            results = {}
            for future in as_completed(futures):
                agent_name = futures[future]
                result = future.result()  # ◄─── Nhận result từ agent
                results[agent_name] = result
                
                # Trace timing
                timing = {
                    "agent": agent_name,
                    "duration": result.get("timing", 0)
                }
                self._calls.append(timing)
        
        return results  # ◄─── Gửi tất cả results tới Coordinator
```

👉 **Xem ở:** Dòng 795-892 trong `multi_agent_chatbot_v3.py`

---

#### 🔗 **Coordinator Node — Merge Results** (Dòng 1069-1114)

```python
def node_coordinator(state: ChatState) -> dict:
    """Coordinator merge tất cả agent results thành workspace chung"""
    
    # Nhận router_output (tất cả agents đã chạy)
    router_output = state.get("router_output", {})
    
    # Merge theo priority
    workspace = {
        "kb_context": router_output.get("kb", ""),
        "search_results": router_output.get("search", ""),
        "calculations": router_output.get("calc", ""),
        "analysis": router_output.get("analyst", ""),
        "plan": router_output.get("planner", ""),
    }
    
    # Build call_graph (trace tất cả gọi)
    call_graph = [
        {"from": "IntentClassifier", "to": "Router"},
        {"from": "Router", "to": ["KB", "Search", "Calc", "Analyst"]},
        {"from": "Coordinator", "to": "ResponseGenerator"},
    ]
    
    return {
        "coordinator_output": workspace,
        "call_graph": call_graph,
        "log": [f"[Coordinator] Merged {len(router_output)} agent results"],
    }
```

👉 **Xem ở:** Dòng 1069-1114 trong `multi_agent_chatbot_v3.py`

---

#### 📊 **LangGraph Build v3** (Dòng 1143-1160)

```python
# Build graph
g = StateGraph(ChatState)
g.add_node("intent_classifier", node_intent_classifier)
g.add_node("router",            node_router)
g.add_node("coordinator",       node_coordinator)
g.add_node("response_generator",node_response_generator)

# Flow
g.set_entry_point("intent_classifier")
g.add_edge("intent_classifier", "router")
g.add_edge("router",            "coordinator")
g.add_edge("coordinator",       "response_generator")

graph = g.compile()
```

👉 **Xem ở:** Dòng 1143-1160 trong `multi_agent_chatbot_v3.py`

---

## 🎬 Live Trace Example

### Chạy v1:

```bash
$ python multi_agent_chatbot.py

Ban > Tính căn bậc hai của 144, rồi kể ngày hôm nay
Dang xu ly|

  → [calculator_agent] {"expression": "sqrt(144)"}
    ← sqrt(144) = 12

  → [datetime_agent] {}
    ← Thu Hai, 19/6/2026, 22:35:14, Tuan 25

Chatbot > Căn bậc hai của 144 là 12. Hôm nay là thứ Hai, ngày 19 tháng 6 năm 2026, tuần 25.
```

**Flow chi tiết:**
```
1. Orchestrator nhận user input
2. LLM quyết định: cần 2 tools (calculator_agent + datetime_agent)
3. Sinh tool_calls:
   {
     "type": "tool_use",
     "name": "calculator_agent",
     "input": {"expression": "sqrt(144)"}
   }
   {
     "type": "tool_use",
     "name": "datetime_agent",
     "input": {}
   }
4. Tool Router parse → gọi từng agent
5. Agent trả result → ToolMessage
6. Back to Orchestrator → LLM tổng hợp → Final answer
```

---

### Chạy v3:

```bash
$ python multi_agent_chatbot_v3.py

Ban > So sánh LangGraph vs CrewAI

┌─────────────────────────────────────────────────────────┐
│ LANGGRAPH MULTI-AGENT CHATBOT v3.1                      │
│                                                         │
│ [i] Processing: "So sánh LangGraph vs CrewAI"          │
├─────────────────────────────────────────────────────────┤
│ [1] IntentClassifier → intent: comparison              │
│ [2] Router → 6 agents parallel                         │
│                                                         │
│     ┌─────────────────────────────────────┐             │
│     │ KB Agent............ 245ms ✓        │             │
│     │ Search Agent........ 380ms ✓        │             │
│     │ Analyst Agent....... 520ms ✓        │             │
│     │ Planner Agent....... 180ms ✓        │             │
│     │ Validator Agent..... 150ms ✓        │             │
│     │ DateTime Agent...... 45ms  ✓        │             │
│     └─────────────────────────────────────┘             │
│                                                         │
│ [3] Coordinator → Merged 6 results                     │
│ [4] ResponseGenerator → Synthesized output             │
│                                                         │
│ Total: 1.47s (parallel vs 1.5s sequential)            │
├─────────────────────────────────────────────────────────┤

[CALL GRAPH]
IntentClassifier
  └─> Router (6 agents)
       ├─> KB Agent
       ├─> Search Agent
       ├─> Analyst Agent
       ├─> Planner Agent
       ├─> Validator Agent
       └─> DateTime Agent
  └─> Coordinator (merge)
       └─> ResponseGenerator

Chatbot > LangGraph là một directed-graph orchestrator framework từ LangChain. CrewAI là một no-code framework cho AI agents... [so sánh chi tiết]
```

**Flow chi tiết:**
```
1. IntentClassifier → "comparison" intent
2. Router → ThreadPoolExecutor submit 6 agents (KỈ CÙNG LÚC)
3. Agents chạy song song:
   - KB: 245ms (vector search)
   - Search: 380ms (Tavily API)
   - Analyst: 520ms (LLM synthesis)
   - etc...
4. Coordinator nhận tất cả results → merge theo priority
5. ResponseGenerator → LLM synthesize → final response
```

**Khác biệt chính:**
- ✅ v1: Gọi agent một cái một cái (tuần tự) = ~2-3s
- ✅ v3: Gọi agents song song = ~1-1.5s (nhanh 2x)

---

## 📚 How to Find Specific Communication

### ❓ "Làm sao Router gọi SearchAgent?"

**File:** `multi_agent_chatbot_v3.py`  
**Tìm:** Dòng 820-860 (hàm `_call_agent`)

```python
def _call_agent(self, agent_name: str, args: dict) -> dict:
    """Execute a specific agent"""
    
    if agent_name == "search":
        result = web_search_agent.invoke(args)
    elif agent_name == "kb":
        result = kb_agent.invoke(args)
    # ... etc
    
    return {
        "result": result,
        "timing": elapsed,
    }
```

### ❓ "Làm sao Coordinator nhận kết quả?"

**File:** `multi_agent_chatbot_v3.py`  
**Tìm:** Dòng 1090-1100

```python
def node_coordinator(state: ChatState) -> dict:
    # Lấy router_output
    router_output = state.get("router_output", {})  # ◄─── TỪ ROUTER
    
    # Merge
    workspace = {...}
    
    return {...}
```

### ❓ "Làm sao ResponseGenerator dùng Coordinator output?"

**File:** `multi_agent_chatbot_v3.py`  
**Tìm:** Dòng 1117-1130

```python
def node_response_generator(state: ChatState) -> dict:
    # Lấy coordinator workspace
    workspace = state.get("coordinator_output", {})  # ◄─── TỪ COORDINATOR
    
    # Dùng để sinh response
    llm_input = f"""
    User: {state["user_input"]}
    KB Context: {workspace['kb_context']}
    Search Results: {workspace['search_results']}
    ...
    """
    
    response = response_gen_agent.invoke(llm_input)
```

---

## 🎯 Summary

| Câu Hỏi | Câu Trả Lời | Xem Ở Đâu |
|---------|-----------|---------|
| **Có Multi-Agent không?** | ✅ CÓ — 2 versions | Phần đầu file này |
| **Agents hợp tác như thế nào?** | Tool calls → Dispatch → Results | v1: Dòng 112-217 / v3: Dòng 795-1114 |
| **Xem agents hợp tác trực tiếp?** | Chạy script + xem output console | `python multi_agent_chatbot.py` |
| **Xem call graph?** | Xem state["call_graph"] | `multi_agent_chatbot_v3.py` |
| **Thêm agent mới?** | Copy Agent class + register ở Router | v1: Dòng 40-110 / v3: Dòng 450+ |

---

**🚀 Start here:**
```bash
# Simple version - dễ hiểu
python multi_agent_chatbot.py

# Advanced version - nhanh + chi tiết
python multi_agent_chatbot_v3.py
```

---

**Tác giả:** Hoàng Minh Đức  
**Ngày:** 2026-06-19
