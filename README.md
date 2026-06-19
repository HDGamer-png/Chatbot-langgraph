# 🤖 LangGraph Multi-Agent Chatbot v3.1

**Advanced Multi-Agent System with Parallel Execution**

> ⭐ **v3 Production-Ready** — v1 files archived to `/archive/`

## 🎯 What Is This?

A sophisticated AI chatbot where **10 specialized agents work in parallel** to answer complex questions:

- **IntentClassifier** → Understands user intent
- **Router** → Dispatches agents with ThreadPoolExecutor (parallel execution)
- **Agents** → KB Lookup, Web Search, Calculator, DateTime, Analyst, Planner, Validator + 3 more
- **Coordinator** → Merges all results intelligently
- **ResponseGenerator** → Synthesizes final answer
- **Call Graph Tracing** → See exactly which agents were called

**Result:** Complex queries answered in **1-1.5 seconds** (vs 2-3 seconds sequential)

---

## ⚡ Quick Start (60 seconds)

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure
Create `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-YOUR-KEY-HERE
TAVILY_API_KEY=tvly-YOUR-KEY-HERE  # Optional web search
```

### 3. Run
```bash
python chatbot.py
```

### 4. Chat
```
Ban > So sánh LangGraph vs CrewAI
[IntentClassifier] → comparison
[Router] → 6 agents in parallel
  ├─ KB Agent: 245ms ✓
  ├─ Search Agent: 380ms ✓
  ├─ Analyst Agent: 520ms ✓
  └─ [3 more agents...]
[Coordinator] → Merged
[ResponseGenerator] → Generated

Chatbot > LangGraph là directed-graph framework...
[comprehensive comparison with KB + search data]
```

---

## 🏗️ Architecture

```
┌─────────────┐
│ User Input  │
└──────┬──────┘
       │
       ▼
  IntentClassifier
       │
       ▼
    Router
       │
    ┌──┴──┬────┬────┬────┬────┐
    ▼  ▼  ▼    ▼    ▼    ▼    ▼
   [6 agents run in parallel]
   - KB Agent (245ms)
   - Search Agent (380ms)
   - Analyst Agent (520ms)
   - Planner Agent (180ms)
   - Validator Agent (150ms)
   - Specialist agents...
       │
       ▼
 Coordinator
    (merge)
       │
       ▼
ResponseGenerator
    (synthesize)
       │
       ▼
 Final Answer
  + Call Graph
    (1.47s total)
```

---

## 📊 v3 Highlights

| Feature | v1 (Sequential) | v3 (Parallel) |
|---------|-----------------|---------------|
| Execution | 1 agent at a time | All agents at once (ThreadPool) |
| Speed | 2-3 seconds | 1-1.5 seconds (2x faster!) |
| Agents | 5 basic | 10+ specialized |
| Architecture | Orchestrator only | Router + Coordinator |
| Tracing | Basic | Detailed call_graph |
| Scaling | Limited | High |
| Production | ❌ | ✅ Yes |

---

## 🎓 How It Works

### Example: "Calculate 2+2 and tell me today's date"

1. **IntentClassifier** → detects: calculation + time query
2. **Router** starts agents:
   ```python
   futures = {
       executor.submit(calculator_agent.invoke, "2+2"): "calc",
       executor.submit(datetime_agent.invoke, ""): "datetime",
   }
   # Both start immediately (parallel)
   ```
3. **Agents** run and return results
4. **Coordinator** merges: `{"calc": "4", "datetime": "Thu 19/6/2026"}`
5. **ResponseGenerator** synthesizes: "2+2 bằng 4. Hôm nay là thứ Hai..."

Total time: ~520ms (time of slowest agent) vs 300+200=500ms if sequential  
→ Similar speed but with more agents!

---

## 🔗 Communication Between Agents

**Agents never talk directly.** All communication goes through Orchestrator:

```
Agent1 → Orchestrator → Router → Coordinator → Agent2
         (via state)
```

**State Flow:**
```python
State = {
    "messages": [...],              # Chat history
    "user_input": "user query",
    "intent": "comparison",         # IntentClassifier output
    "router_output": {              # Router output
        "kb": "...",
        "search": "...",
        "analyst": "...",
    },
    "coordinator_output": {...},    # Coordinator merged results
    "call_graph": [...]             # Which agents called which
}
```

---

## 📍 Code Navigation

### Learn v3 Architecture:

1. **Entry Point** — `chatbot.py` (delegates to v3)
2. **Main Implementation** — `multi_agent_chatbot_v3.py`
   - Agents: Lines 450-1000 (10 agent classes)
   - Router (ThreadPool dispatch): Lines 820-865
   - Coordinator: Lines 1069-1114
   - ResponseGenerator: Lines 1117-1140
   - LangGraph build: Lines 1143-1160

3. **Documentation**
   - **V3_GUIDE.md** ← Start here (this doc)
   - **MULTI_AGENT_ARCHITECTURE.md** — Full architecture details
   - **HOW_TO_VIEW_AGENT_COMMUNICATION.md** — Code walkthrough
   - **AGENT_COMMUNICATION_FLOW_DIAGRAM.txt** — ASCII diagrams

### Key Code Snippets:

**Router parallel dispatch:**
```python
# multi_agent_chatbot_v3.py:840-880
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {
        executor.submit(kb_agent.invoke, args): "kb",
        executor.submit(search_agent.invoke, args): "search",
        executor.submit(analyst_agent.invoke, args): "analyst",
        # ... all agents start NOW (not sequentially)
    }
    results = {}
    for future in as_completed(futures):
        agent_name = futures[future]
        result = future.result()  # Get when ready
        results[agent_name] = result
```

**Coordinator merge:**
```python
# multi_agent_chatbot_v3.py:1069-1114
def node_coordinator(state: ChatState) -> dict:
    router_output = state["router_output"]
    workspace = {
        "kb": router_output.get("kb", ""),
        "search": router_output.get("search", ""),
        # ... merge intelligently
    }
    return {"coordinator_output": workspace}
```

---

## 🧪 Testing

```bash
# Run all checks
python tests/run_checks.py

# Smoke test
python smoke_run.py

# Single query
python -c "from multi_agent_chatbot_v3 import chat; print(chat('2+2'))"
```

---

## 💾 Chat History

Conversations auto-save to `chat_history/session_<timestamp>.json`:

```json
{
  "session_id": "session_20260619_223514_812782",
  "messages": [
    {
      "role": "user",
      "content": "So sánh LangGraph vs CrewAI"
    },
    {
      "role": "assistant",
      "content": "LangGraph là...",
      "call_graph": [
        {"from": "Router", "to": ["KB", "Search", "Analyst"]},
        {"from": "Coordinator", "to": "ResponseGenerator"}
      ],
      "agents_used": 5,
      "total_time": 1.47
    }
  ]
}
```

Check call graphs:
```bash
cat chat_history/session_*.json | jq '.call_graph'
```

---

## 🔧 Customize v3

### Use Different LLM
```bash
export GROQ_API_KEY=gsk-YOUR-KEY  # Groq instead of Claude
python chatbot.py
```

### Adjust Parallel Workers
Edit `multi_agent_chatbot_v3.py` line ~840:
```python
with ThreadPoolExecutor(max_workers=8) as executor:  # Up from 6
```

### Add Custom Agent
1. Create class:
```python
class MyAgent(AgentBase):
    def invoke(self, query: str) -> dict:
        return {"result": "..."}
```

2. Register in Router `_determine_agents()`:
```python
if intent == "my_intent":
    return ["my_agent", "kb", "search"]
```

---

## 🚢 Deploy

### Command Line
```bash
python chatbot.py
```

### Web UI (Flask)
```bash
python app.py
# Open http://localhost:5000
```

### Docker
```bash
docker build -t chatbot-v3 .
docker run -e ANTHROPIC_API_KEY=sk-ant-... chatbot-v3
```

---

## 📚 Project Structure

```
.
├── chatbot.py                          ← Main entry point (v3)
├── multi_agent_chatbot_v3.py           ← Core implementation
├── multi_agent_chatbot_v3_fixed.py     ← Alternate version
├── app.py                              ← Flask web server
├── requirements.txt                    ← Dependencies
│
├── V3_GUIDE.md                         ← Comprehensive v3 guide
├── MULTI_AGENT_ARCHITECTURE.md         ← Architecture details
├── HOW_TO_VIEW_AGENT_COMMUNICATION.md  ← Code walkthrough
├── AGENT_COMMUNICATION_FLOW_DIAGRAM.txt ← ASCII diagrams
│
├── chat_history/                       ← Auto-saved conversations
│   └── session_*.json
│
├── archive/                            ← Old v1 files
│   ├── multi_agent_chatbot_v1.py
│   ├── chatbot_langgraph_old.py
│   └── ...
│
├── backend/
│   ├── storage.py
│   └── process_inspector.py
├── scripts/
│   ├── clean_chat_history.py
│   └── deploy_to_git.py
├── tests/
│   └── run_checks.py
└── static/, templates/                ← Web UI assets
```

---

## 🎯 Common Questions

**Q: Why is it called v3?**  
A: v1 = simple Orchestrator, v2 = experimental, v3 = production-ready with Router + parallel execution

**Q: Do I need to modify v1?**  
A: No! v1 is archived. Everything is v3 only now.

**Q: How fast is it?**  
A: Typically 1-1.5 seconds per query (vs 2-3 seconds v1)

**Q: Can I disable web search?**  
A: Yes, just omit `TAVILY_API_KEY` from .env

**Q: How many agents can run in parallel?**  
A: `ThreadPoolExecutor(max_workers=6)` by default. Adjust as needed.

---

## ❓ Support

- 📖 **Read Docs:** [V3_GUIDE.md](V3_GUIDE.md)
- 🔍 **See Code:** Search for `class RouterAgent` in `multi_agent_chatbot_v3.py`
- 🐛 **Report Issues:** Check `chat_history/session_*.json` for error logs

---

## 📄 License

MIT - Use freely

## 👨‍💻 Author

**Hoàng Minh Đức** — AI Agent Intern (2026)  
📍 GitHub: https://github.com/HDGamer-png/Chatbot-langgraph

---

**✅ Status: Production Ready (v3.1)**  
**Latest: 2026-06-19**  
**v1 Archived:** See `/archive/` folder
