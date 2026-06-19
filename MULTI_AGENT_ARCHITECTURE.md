# 🤖 Multi-Agent Architecture Document

## 📊 Trạng Thái Hiện Tại

**✅ CÓ - Ứng dụng này là kiến trúc Multi-Agent**

Có **3 phiên bản** với độ phức tạp tăng dần:
1. **`multi_agent_chatbot.py`** — v1: Simple Orchestrator
2. **`multi_agent_chatbot_v3.py`** — v3.1: Advanced Router + Coordinator (khuyến khích sử dụng)
3. **`chatbot_langgraph_v2.py`** — Biến thể LangGraph khác

---

## 🏗️ Kiến Trúc Multi-Agent (v1 - Simple)

### Sơ Đồ Luồng

```
┌─────────────┐
│ User Input  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│    ORCHESTRATOR LLM (Main Brain)     │ ◄─── Phân tích yêu cầu
│    - Nhận input từ user              │      Quyết định gọi agent nào
│    - Quyết định gọi agent nào        │      Tổng hợp kết quả
└──────┬──────────────────────────────┘
       │
       │ Sinh tool_call blocks (JSON)
       ▼
┌──────────────────────────────────────┐
│      TOOL ROUTER (Dispatcher)        │ ◄─── Parse tool_calls
│    - Đọc tool_calls từ LLM           │      Dispatch → agents
│    - Gọi agent đúng                  │      Collect results
│    - Trả ToolMessage                 │
└──────┬──────────────────────────────┘
       │
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
   ┌────────┐   ┌──────────┐  ┌───────────┐  ┌─────────┐
   │ Search │   │Calculator│  │  DateTime │  │Analyst  │
   │ Agent  │   │  Agent   │  │   Agent   │  │ Agent   │
   └────────┘   └──────────┘  └───────────┘  └─────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
              │
              ▼
      ┌───────────────┐
      │ ToolMessages  │ ◄─── Kết quả từ các agents
      └───────┬───────┘
              │
              ▼
    ┌──────────────────────────────────┐
    │  Orchestrator (next iteration)   │ ◄─── Loop hoặc End
    │  - Có tool_calls & iter<MAX → loại │
    │  - Nếu không → Return final answer │
    └──────────────────────────────────┘
              │
              ▼
      ┌──────────────┐
      │ Final Answer │ ◄─── Text response cho user
      └──────────────┘
```

### 🔗 Cách Agents Giao Tiếp

1. **Tool Definitions (JSON Schema)**
   ```python
   TOOLS = [
       {
           "name": "search_agent",
           "description": "Tìm kiếm web...",
           "input_schema": {...}
       },
       # ...
   ]
   ```

2. **Orchestrator Binding Tools**
   ```python
   orchestrator = ChatAnthropic(...).bind_tools(TOOLS)
   # LLM tự sinh tool_use blocks khi cần
   ```

3. **LangGraph Routing**
   ```python
   def node_tool_router(state):
       last = state["messages"][-1]
       for tool_call in last.tool_calls:  # ◄─── Parse tool calls
           agent_fn = AGENT_FN[tool_call["name"]]
           result = agent_fn(tool_call["args"])
           tool_msgs.append(ToolMessage(...))
   ```

### 📍 Xem Code Ở Đâu?

| Thành Phần | File | Dòng |
|-----------|------|------|
| 5 Agents (Search, Calc, DateTime, Analyst, Coder) | `multi_agent_chatbot.py` | 40-110 |
| Tool Definitions | `multi_agent_chatbot.py` | 112-170 |
| Orchestrator Setup | `multi_agent_chatbot.py` | 172-181 |
| Orchestrator Node | `multi_agent_chatbot.py` | 191-198 |
| Tool Router Node | `multi_agent_chatbot.py` | 203-214 |
| LangGraph Build | `multi_agent_chatbot.py` | 229-238 |
| State Management | `multi_agent_chatbot.py` | 31-36 |

---

## 🚀 Kiến Trúc Multi-Agent (v3 - Advanced)

### Sơ Đồ Luồng Nâng Cao

```
┌──────────────────────────────────────┐
│ User Input + Chat History            │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│    INTENT CLASSIFIER AGENT           │ ◄─── Step 1: Phân loại ý định
│    - Nhận diện loại yêu cầu          │      (search, calc, qa, etc)
│    - Trích xuất entities             │
└──────┬───────────────────────────────┘
       │ intent + entities
       ▼
┌──────────────────────────────────────┐
│    ROUTER AGENT (ThreadPoolExecutor) │ ◄─── Step 2: Dispatch song song
│    - Dispatch agents song song       │      (5-8 agents cùng lúc)
│    - Collect results nhanh           │
└──────┬───────────────────────────────┘
       │
   ┌───┼────┬────┬────┬────┐
   │   │    │    │    │    │
   ▼   ▼    ▼    ▼    ▼    ▼
┌─────┐ ┌──────┐ ┌───────┐ ┌──────┐ ┌────────┐ ┌─────────┐
│ KB  │ │Search│ │Calc   │ │Analyst│ │Planner│ │Validator│
│Agent│ │Agent │ │Agent  │ │Agent  │ │Agent  │ │ Agent   │
└─────┘ └──────┘ └───────┘ └──────┘ └────────┘ └─────────┘
   │       │        │        │        │         │
   └───┬───┴────┬───┴───┬────┴────┬───┴─────┬───┘
       │        │       │         │         │
       ▼        ▼       ▼         ▼         ▼
    ┌─────────────────────────────────┐
    │  COORDINATOR AGENT              │ ◄─── Step 3: Merge + Priority
    │  - Merge tất cả results         │      Ưu tiên results
    │  - Detect conflicts             │
    │  - Create workspace             │
    └──────┬──────────────────────────┘
           │ merged_context
           ▼
    ┌─────────────────────────────────┐
    │ RESPONSE GENERATOR AGENT        │ ◄─── Step 4: Generate answer
    │  - LLM synthesize final output  │      Tối ưu hóa response
    │  - Format + style               │
    └─────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │ Final Response + Call Graph      │ ◄─── Trả về user
    │ (Hiển thị tất cả agents đã gọi) │
    └──────────────────────────────────┘
```

### 🤝 Giao Tiếp Giữa Agents

#### 1. **Sequential Communication** (v1)
```
Orchestrator → Agent1 → ToolMessage → Orchestrator → Agent2 → ...
```
❌ Slow: Tuần tự  
✅ Safe: Xử lý tuần tự

#### 2. **Parallel Communication** (v3 - ThreadPoolExecutor)
```python
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {
        executor.submit(kb_agent.invoke, ...): "kb",
        executor.submit(search_agent.invoke, ...): "search",
        executor.submit(calc_agent.invoke, ...): "calc",
        # ... tất cả agents chạy cùng lúc
    }
    for future in as_completed(futures):
        result = future.result()
```
✅ Fast: Song song (5-8 agents cùng lúc)  
✅ Efficient: Tận dụng CPU threads

### 📍 Xem Code Ở Đâu?

| Thành Phần | File | Dòng |
|-----------|------|------|
| **Agent Definitions** | `multi_agent_chatbot_v3.py` | 450-1000 |
| - IntentClassifierAgent | `multi_agent_chatbot_v3.py` | 482-560 |
| - KnowledgeBaseAgent | `multi_agent_chatbot_v3.py` | 561-643 |
| - CalculatorAgent | `multi_agent_chatbot_v3.py` | 644-668 |
| - DateTimeAgent | `multi_agent_chatbot_v3.py` | 669-690 |
| - WebSearchAgent | `multi_agent_chatbot_v3.py` | 691-717 |
| - AnalystAgent | `multi_agent_chatbot_v3.py` | 718-756 |
| - PlannerAgent | `multi_agent_chatbot_v3.py` | 757-774 |
| - ValidatorAgent | `multi_agent_chatbot_v3.py` | 775-792 |
| - **RouterAgent (Dispatcher)** | `multi_agent_chatbot_v3.py` | 795-892 |
| - ResponseGeneratorAgent | `multi_agent_chatbot_v3.py` | 893-992 |
| **LangGraph Nodes** | `multi_agent_chatbot_v3.py` | 1069-1140 |
| - node_coordinator | `multi_agent_chatbot_v3.py` | 1069-1114 |
| - node_response_generator | `multi_agent_chatbot_v3.py` | 1117-1140 |
| **Graph Build** | `multi_agent_chatbot_v3.py` | 1143-1160 |

---

## 🔍 Chi Tiết Giao Tiếp Giữa Agents

### **Router Agent (v3)** — Trái Tim Hệ Thống

```python
class RouterAgent(AgentBase):
    """Dispatcher — gọi 5-8 agents song song dựa vào intent"""
    
    def invoke(self, intent, entities, kb_chunk, prev_ans):
        # Bước 1: Xác định agents cần gọi
        agents_to_call = self._determine_agents(intent)
        
        # Bước 2: Gọi song song (ThreadPoolExecutor)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            
            # Gửi request đến tất cả agents
            for agent_name in agents_to_call:
                future = executor.submit(
                    self._call_agent,
                    agent_name,
                    entities
                )
                futures[future] = agent_name
            
            # Collect results
            results = {}
            for future in as_completed(futures):
                agent_name = futures[future]
                results[agent_name] = future.result()  # ◄─── Nhận kết quả
        
        # Bước 3: Trả về cho Coordinator
        return results  # ◄─── Gửi đến agent tiếp theo
```

**Dòng mã key:**
- **L820-835**: Xác định agents
- **L840-865**: ThreadPool dispatch
- **L870-880**: Collect results

### **Coordinator Agent (v3)** — Quản Lý Trung Ương

```python
def node_coordinator(state):
    """Merge tất cả agent results thành workspace chung"""
    
    # Nhận input từ Router
    router_results = state["router_output"]  # ◄─── Từ RouterAgent
    
    # Merge theo priority
    workspace = {
        "kb": router_results.get("kb", ""),
        "search": router_results.get("search", ""),
        "calc": router_results.get("calc", ""),
        "analysis": router_results.get("analyst", ""),
    }
    
    # Tính call_graph (tTrace quá trình)
    call_graph = [
        {"from": "Router", "to": "Coordinator"},
        {"from": "Coordinator", "to": "ResponseGenerator"},
    ]
    
    # Trả về state tiếp theo
    return {
        "coordinator_output": workspace,
        "call_graph": call_graph,
    }
```

**Dòng mã key:**
- **L1069-1114**: Toàn bộ Coordinator node

---

## 📊 Call Graph — Tracing Giao Tiếp

### **Hiển Thị Tất Cả Agents Được Gọi**

```python
def display_call_graph(call_graph: List[dict]) -> str:
    """
    Input: [
        {"from": "IntentClassifier", "to": "Router", ...},
        {"from": "Router", "to": ["KB", "Search", "Calc"], ...},
        {"from": "Coordinator", "to": "ResponseGenerator", ...},
    ]
    
    Output:
    ┌─────────────────────────────────┐
    │  CALL GRAPH TRACE               │
    ├─────────────────────────────────┤
    │ IntentClassifier                │
    │   └─> Router (5 agents parallel)│
    │        ├─> KB Agent             │
    │        ├─> Search Agent         │
    │        ├─> Calculator Agent     │
    │        ├─> Analyst Agent        │
    │        └─> Planner Agent        │
    │   └─> Coordinator (merge)       │
    │        └─> ResponseGenerator    │
    └─────────────────────────────────┘
    """
```

**Code xem ở:**
- File: `multi_agent_chatbot_v3.py`
- Function: `display_call_graph()` khoảng dòng 1170-1220
- Hay chạy để xem output live

---

## 🎯 So Sánh v1 vs v3

| Tiêu Chí | v1 (Simple) | v3 (Advanced) |
|---------|-----------|--------------|
| **Agents** | 5 agents | 8+ agents |
| **Communication** | Sequential (tuần tự) | Parallel (song song) |
| **Speed** | Slow (~3-5s/query) | Fast (~1-2s/query) |
| **Architecture** | Orchestrator only | Orchestrator + Router + Coordinator |
| **Threading** | None | ThreadPoolExecutor (6 workers) |
| **Scalability** | Low | High |
| **Tracing** | Basic | Advanced call_graph |
| **Error Handling** | Basic try-except | Retry + Backoff |
| **Memory** | Simple state | Workspace + context window |

---

## ▶️ Cách Chạy & Theo Dõi

### **Run v1 (Simple)**
```bash
python multi_agent_chatbot.py
```
```
Ban > Tính căn bậc hai của 144
  → [calculator_agent] {"expression": "sqrt(144)"}
    ← sqrt(144) = 12
Chatbot > căn bậc hai của 144 bằng 12
```

### **Run v3 (Advanced)**
```bash
python multi_agent_chatbot_v3.py
```
```
Ban > So sánh LangGraph vs CrewAI
┌─────────────────────────────────────┐
│ LANGGRAPH MULTI-AGENT CHATBOT v3.1  │
│ [✓] Intent: comparison              │
│ [✓] Routing to 6 agents...          │
├─────────────────────────────────────┤
│ [=] KB Agent        → 245ms         │
│ [=] Search Agent    → 380ms         │
│ [=] Analyst Agent   → 520ms         │
│ [=] Planner Agent   → 180ms         │
│ [=] Validator Agent → 150ms         │
├─────────────────────────────────────┤
│ [→] Coordinator merging...          │
│ [→] ResponseGenerator synthesizing..│
│ Total: 1.47s                        │
└─────────────────────────────────────┘

Chatbot > [So sánh LangGraph vs CrewAI với đầy đủ thông tin...]
```

---

## 🔗 Agent Dependencies Diagram

```
User Input
  │
  ├─→ IntentClassifier (Phân loại)
  │        │
  │        └─→ Router (Dispatcher)
  │             │
  │             ├─→ KB Agent
  │             ├─→ Search Agent
  │             ├─→ Calculator Agent
  │             ├─→ Analyst Agent
  │             ├─→ Planner Agent
  │             └─→ Validator Agent
  │             │
  │             └─→ Coordinator (Merge)
  │                  │
  │                  └─→ ResponseGenerator
  │                       │
  └─→ Final Answer + Call Graph
```

---

## 🎓 Key Concepts

### **ReAct Pattern** (Reasoning + Acting)
1. **Thinking**: Agent suy nghĩ → gọi tool
2. **Acting**: Thực thi tool → nhận result
3. **Observing**: Phân tích result → quyết định tiếp theo

### **Tool Use Protocol**
```python
# Orchestrator LLM sinh:
{
  "type": "tool_use",
  "name": "search_agent",
  "input": {"query": "..."}
}
# Router parse → gọi agent → trả ToolMessage
```

### **State Machine** (LangGraph)
```
Entry → node_coordinator
        │
        └─→ node_response_generator
            │
            └─→ End
```

---

## 📚 Files Liên Quan

| File | Mục Đích |
|------|---------|
| `multi_agent_chatbot.py` | v1 - Orchestrator basics |
| `multi_agent_chatbot_v3.py` | **v3 - Recommended** |
| `chatbot_langgraph_v2.py` | Alternative LangGraph |
| `chatbot_langgraph.py` | Original LangGraph version |
| `app.py` | Flask web server |
| `langgraph_chatbot_demo.html` | Demo UI |
| `langgraph_demo_v2.html` | Advanced UI |

---

## 🚀 Next Steps

1. **Chạy v3 để xem call graph**
   ```bash
   python multi_agent_chatbot_v3.py
   ```

2. **Xem lịch sử giao tiếp**
   ```bash
   cat chat_history/session_*.json | jq '.call_graph'
   ```

3. **Thêm Agent mới**
   - Copy `AnalystAgent` class
   - Định nghĩa `invoke()` method
   - Register ở `_determine_agents()` 
   - Add vào `ThreadPoolExecutor` loop

---

**Tác giả**: Hoàng Minh Đức (Thực tập sinh AI Agent 2026)  
**Cập nhật**: 2026-06-19
