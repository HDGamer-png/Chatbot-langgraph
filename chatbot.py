#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   LANGGRAPH MULTI-AGENT CHATBOT v3.1                                ║
║   Main Entry Point — Production Ready                               ║
║                                                                      ║
║   Tác giả: Hoàng Minh Đức — Thực tập sinh AI Agent 2026            ║
║                                                                      ║
║   Architecture:                                                      ║
║   • IntentClassifier → Router (ThreadPool) → Coordinator            ║
║   • ResponseGenerator → Call Graph Trace                             ║
║   • 10 Specialized Agents (KB, Search, Calc, Analyst, etc)         ║
║                                                                      ║
║   Cài đặt:                                                           ║
║     pip install -r requirements.txt                                  ║
║                                                                      ║
║   Chạy:  python chatbot.py                                          ║
║   File lưu: ./chat_history/session_<timestamp>.json                ║
║                                                                      ║
║   Env vars (optional):                                              ║
║     ANTHROPIC_API_KEY=sk-ant-...  (Claude - khuyến khích)         ║
║     GROQ_API_KEY=gsk-...          (Groq alternative)               ║
║     TAVILY_API_KEY=tvly-...       (Web search)                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
from pathlib import Path

# Ensure v3 is available
try:
    from multi_agent_chatbot_v3 import main as run_v3_chatbot
except ImportError as e:
    print(f"❌ Error: Cannot import v3 chatbot. {e}")
    print("Please ensure multi_agent_chatbot_v3.py is in the same directory.")
    sys.exit(1)

def main():
    """Main entry point - delegates to v3"""
    try:
        run_v3_chatbot()
    except KeyboardInterrupt:
        print("\n\nTam biệt!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
