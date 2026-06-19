# Smoke test for multi_agent_chatbot_v3 with mocked LLMs
import os
from types import SimpleNamespace

# Ensure we import project module from cwd
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Minimal dummy LLM that matches interface used in code
class DummyResp:
    def __init__(self, text):
        self.content = text

class DummyLLM:
    def __init__(self, reply="[MOCK_REPLY] Đây là phản hồi giả."):
        self.reply = reply
    def invoke(self, messages):
        return DummyResp(self.reply)

# Set env keys empty to avoid selecting remote providers
os.environ.pop('GROQ_API_KEY', None)
os.environ.pop('ANTHROPIC_API_KEY', None)

import multi_agent_chatbot_v3 as m

# Patch global agent LLM getters to return dummy
m._intent_agent._get_llm = lambda *a, **k: DummyLLM("intent:langgraph")
# For analysts and response generator
m._response_agent._get_llm = lambda *a, **k: DummyLLM("Đây là câu trả lời cuối (mock).")
# Analyst agent class instances create _get_llm lazily; monkeypatch class method as well
from multi_agent_chatbot_v3 import AnalystAgent, IntentClassifierAgent
AnalystAgent._get_llm = lambda self, *a, **k: DummyLLM("Analyst mock")
IntentClassifierAgent._get_llm = lambda self, *a, **k: DummyLLM("langgraph")


def run():
    graph = m.build_graph()
    initial = {
        'session_id': 'smoke',
        'turn_id': 1,
        'user_query': 'cậu có phải là multi-agent không?',
        'user_id': 'tester',
        'intent': '',
        'goal': '',
        'kb_context': '',
        'workspace_messages': [],
        'coordination_summary': '',
        'planning': '',
        'validation': '',
        'web_results': '',
        'calc_result': '',
        'datetime_result': '',
        'analysis': '',
        'final_answer': '',
        'conversation_history': [],
        'agent_timings': [],
        'call_graph': [],
        'log': [],
    }
    out = graph.invoke(initial)
    print('Result keys:', sorted(out.keys()))
    print('\nIntent:', out.get('intent'))
    print('\nCoordination summary:\n', out.get('coordination_summary'))
    print('\nFinal answer preview:\n', out.get('final_answer'))

if __name__ == '__main__':
    run()
