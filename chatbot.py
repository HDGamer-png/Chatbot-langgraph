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
║   • Data Layer: Hiểu mọi định dạng (mã nguồn mọi ngôn ngữ,          ║
║                 PDF, Word, Excel, PPTX, Ảnh, ZIP, Thư mục).          ║
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
from storage.file_reader import extract_text


def cli_interactive(initial_attach: str = None):
    """Chế độ trò chuyện tương tác trong Terminal với khả năng nạp file & folder."""
    db = ChatDatabase("chatbot.db")
    session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.create_session(session_id, user_id="cli_user")

    print("\n" + "=" * 68)
    print("🤖 CHATBOT AI MULTI-AGENT v4.0 (100% Pure Python Local)")
    print(f"📁 Phiên làm việc: {session_id}")
    print("💡 Hỗ trợ: Mã nguồn mọi ngôn ngữ, PDF, Word, Excel, PPTX, Ảnh, Thư mục")
    print("📌 Lệnh hỗ trợ:")
    print("   • /attach <đường_dẫn> : Đính kèm file hoặc cả thư mục dự án")
    print("   • /detach             : Hủy tệp đính kèm hiện tại")
    print("   • /info               : Xem tệp/thư mục đang được đính kèm")
    print("   • /help               : Xem danh sách lệnh")
    print("   • quit / exit         : Thoát chương trình")
    print("=" * 68 + "\n")

    current_attachment_text = ""
    current_attachment_name = ""

    if initial_attach:
        p = Path(initial_attach)
        if p.exists():
            txt = extract_text(p)
            if txt:
                current_attachment_text = txt
                current_attachment_name = p.name
                print(f"📎 [Đã đính kèm ban đầu]: {p.name} ({len(txt):,} ký tự)")
            else:
                print(f"⚠️ [Cảnh báo]: Không thể trích xuất nội dung từ '{initial_attach}'")
        else:
            print(f"⚠️ [Cảnh báo]: Đường dẫn không tồn tại: '{initial_attach}'")

    turn_counter = 1
    while True:
        try:
            prompt_str = f"👤 Bạn [{current_attachment_name}]: " if current_attachment_name else "👤 Bạn: "
            user_input = input(f"\n{prompt_str}").strip()
            if not user_input:
                continue

            # Xử lý các lệnh đặc biệt
            cmd_lower = user_input.lower()
            if cmd_lower in ("quit", "exit", "thoat"):
                print("\n👋 Tạm biệt! Hẹn gặp lại bạn.")
                break

            if cmd_lower.startswith("/attach "):
                target_path_str = user_input[8:].strip().strip('"\'')
                target_path = Path(target_path_str)
                if not target_path.exists():
                    print(f"❌ Đường dẫn không tồn tại: {target_path_str}")
                    continue

                print(f"⏳ Đang đọc và phân tích '{target_path.name}'...")
                extracted = extract_text(target_path)
                if extracted:
                    current_attachment_text = extracted
                    current_attachment_name = target_path.name
                    print(f"✅ Đã nạp thành công: {target_path.name} ({len(extracted):,} ký tự).")
                    print("👉 Bạn có thể đặt câu hỏi về tệp/thư mục này ngay bây giờ.")
                else:
                    print(f"❌ Không thể trích xuất nội dung từ '{target_path.name}'.")
                continue

            if cmd_lower == "/detach":
                if current_attachment_name:
                    print(f"🗑️ Đã hủy đính kèm: {current_attachment_name}")
                    current_attachment_text = ""
                    current_attachment_name = ""
                else:
                    print("ℹ️ Hiện không có tệp/thư mục nào đang được đính kèm.")
                continue

            if cmd_lower == "/info":
                if current_attachment_name:
                    print(f"📎 Đang đính kèm: {current_attachment_name} ({len(current_attachment_text):,} ký tự)")
                else:
                    print("ℹ️ Hiện không có tệp nào được đính kèm. Dùng lệnh /attach <đường dẫn> để thêm.")
                continue

            if cmd_lower == "/help":
                print("\n📌 HƯỚNG DẪN LỆNH:")
                print("  • /attach <path> : Nạp file mã nguồn (.py, .js, .java...), PDF, DOCX, XLSX, hoặc cả thư mục.")
                print("  • /detach        : Gỡ bỏ tệp đính kèm hiện tại.")
                print("  • /info          : Xem thông tin tệp đang nạp.")
                print("  • quit / exit    : Thoát ứng dụng.")
                continue

            # Chuẩn bị payload câu hỏi kèm nội dung file/thư mục
            message_payload = user_input
            if current_attachment_text:
                message_payload = f"{user_input}\n\nDưới đây là nội dung đính kèm [{current_attachment_name}]:\n{current_attachment_text}"

            print("\n🔄 [Multi-Agent đang phối hợp xử lý...]")
            res = handle_query(
                session_id=session_id,
                user_query=message_payload,
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
                "attachments": [{"filename": current_attachment_name}] if current_attachment_name else [],
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
    parser.add_argument("--attach", type=str, default=None, help="Đường dẫn tệp hoặc thư mục đính kèm ngay khi khởi động CLI")
    args = parser.parse_args()

    if args.web:
        from app import app
        print(f"🚀 Khởi động Web Server tại port {args.port}...")
        app.run(host="0.0.0.0", port=args.port, debug=False)
    else:
        cli_interactive(initial_attach=args.attach)


if __name__ == "__main__":
    main()
