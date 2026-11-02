#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   LANGGRAPH MULTI-AGENT CHATBOT v4.0                                ║
║   Pure Python Local Architecture — No API Key Required               ║
║                                                                      ║
║   Tác giả: Hoàng Minh Đức — Thực tập sinh AI Agent 2026            ║
║                                                                      ║
║   Architecture:                                                      ║
║   • PHA 1 (Song song): 6 Worker Agents (RAG, Math, Search, DateTime, ║
║                        Analyst, Intent) qua ThreadPool              ║
║   • PHA 2 (Tuần tự):   Planner → Validator → ResponseGenerator      ║
║   • Storage: SQLite (chatbot.db)                                     ║
║   • Frontend: Web UI Dark Theme + SVG Call Graph Dashboard           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Cấu hình UTF-8 cho Windows Terminal tránh lỗi mã hóa ký tự tiếng Việt
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục hiện tại vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.graph import handle_query
from storage.database import ChatDatabase


def cli_interactive():
    """Chế độ trò chuyện tương tác trong Terminal."""
    db = ChatDatabase("chatbot.db")
    session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.create_session(session_id, user_id="cli_user")

    print("\n" + "="*65)
    print("🤖 CHATBOT AI MULTI-AGENT v4.0 (100% Pure Python Local)")
    print(f"📁 Phiên làm việc: {session_id}")
    print("👉 Nhập câu hỏi của bạn (hoặc gõ 'quit' / 'exit' để thoát):")
    print("="*65 + "\n")

    turn_counter = 1
    while True:
        try:
            user_input = input("\n👤 Bạn: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "thoat"):
                print("\n👋 Tạm biệt! Hẹn gặp lại bạn.")
                break

            print("\n🔄 [Multi-Agent đang phối hợp xử lý...]")
            res = handle_query(
                session_id=session_id,
                user_query=user_input,
                turn_id=turn_counter,
                user_id="cli_user",
            )

            final_answer = res.get("final_answer", "")
            intent = res.get("intent", "general")
            confidence = res.get("confidence_score", 0.0)
            timings = res.get("agent_timings", [])

            print(f"\n🎯 [Ý định: {intent.upper()} | Độ tin cậy: {confidence*100:.0f}%]")
            print(f"🤖 Bot:\n{final_answer}")

            # Lưu vào SQLite
            db.append_turn(session_id, {
                "turn_id": turn_counter,
                "timestamp": datetime.now().isoformat(),
                "user_query": user_input,
                "final_answer": final_answer,
                "intent": intent,
                "confidence_score": confidence,
                "agent_timings": timings,
                "call_graph": res.get("call_graph", []),
                "log": res.get("log", []),
                "attachments": [],
            })
            turn_counter += 1

        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as exc:
            print(f"\n❌ Lỗi: {exc}")


def main():
    parser = argparse.ArgumentParser(description="LangGraph Multi-Agent Chatbot v4.0")
    parser.add_argument("--web", action="store_true", help="Khởi động Web server Flask")
    parser.add_argument("--port", type=int, default=5000, help="Cổng chạy Web server (mặc định: 5000)")
    args = parser.parse_args()

    if args.web:
        from app import app
        print(f"🚀 Khởi động Web Server tại port {args.port}...")
        app.run(host="0.0.0.0", port=args.port, debug=False)
    else:
        cli_interactive()


if __name__ == "__main__":
    main()
