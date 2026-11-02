"""
app.py — Máy chủ Web Flask v4.0 (Pure Python Multi-Agent Engine)

Điểm nâng cấp nổi bật trong v4.0:
  1. Sử dụng 100% Logic Python nội bộ & Multi-Agent LangGraph (No-API-Key).
  2. Mở khóa và tích hợp triệt để xử lý Tệp đính kèm (PDF, DOCX, XLSX, TXT, Ảnh OCR).
  3. Cơ sở dữ liệu SQLite (chatbot.db) thay thế hoàn toàn việc ghi file JSON.
  4. Tích hợp trực tiếp Dashboard đồ họa SVG (agent_dashboard.html) vào route /process/<session_id>.
  5. Hỗ trợ Server-Sent Events (SSE) streaming tại /api/chat/stream.
"""

import os
import sys
import time
import uuid
import json
from datetime import datetime
from pathlib import Path

# Cấu hình UTF-8 cho Windows Terminal tránh lỗi mã hóa ký tự tiếng Việt
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import Core Multi-Agent & Storage
from core.graph import handle_query
from storage.database import ChatDatabase
from storage.file_reader import extract_text

# ── Khởi tạo Flask App ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATE_DIR),
)
CORS(app)

# Khởi tạo SQLite Database
db = ChatDatabase(str(BASE_DIR / "chatbot.db"))


# ── Hàm trợ giúp trích xuất tệp đính kèm ──────────────────────────────
def _process_attachments(attachments: list[dict]) -> tuple[list[dict], str]:
    """
    Trích xuất văn bản từ các tệp đính kèm được tải lên.
    Hỗ trợ: PDF, Word (.docx), Excel (.xlsx), Text (.txt, .md, .py...), Ảnh.
    """
    contexts = []
    lines = []

    for att in attachments:
        filename = att.get("filename") or "unknown"
        content_type = att.get("type") or ""
        url = att.get("url") or ""

        # Tìm file trên đĩa
        file_path = None
        if url.startswith("/static/uploads/"):
            safe_name = url.replace("/static/uploads/", "").strip()
            potential_path = UPLOAD_DIR / safe_name
            if potential_path.exists():
                file_path = potential_path

        extracted = None
        if file_path:
            extracted = extract_text(file_path, content_type)

        note = extracted if extracted else "[Không thể trích xuất nội dung tự động]"
        contexts.append({
            "filename": filename,
            "url": url,
            "type": content_type,
            "text": extracted,
            "path": str(file_path) if file_path else None,
        })
        lines.append(f"📄 Tệp đính kèm: {filename}\n{note}")

    summary = "\n\n".join(lines)
    if summary:
        summary = "Dưới đây là nội dung trích xuất từ tệp đính kèm:\n\n" + summary
    return contexts, summary


# ══════════════════════════════════════════════════════════════════
#  ROUTES CHÍNH
# ══════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Giao diện chính của Chatbot."""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Endpoint tải lên tệp đính kèm."""
    try:
        uploaded_file = request.files.get("file")
        if not uploaded_file or uploaded_file.filename == "":
            return jsonify({"error": "Không có tệp nào được gửi."}), 400

        original_filename = secure_filename(uploaded_file.filename)
        extension = Path(original_filename).suffix or ""
        safe_name = f"{uuid.uuid4().hex}{extension}"
        filepath = UPLOAD_DIR / safe_name
        uploaded_file.save(filepath)

        return jsonify({
            "filename": original_filename,
            "url": f"/static/uploads/{safe_name}",
            "type": uploaded_file.content_type or "application/octet-stream",
            "size": filepath.stat().st_size,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint chat chính thức — Điều phối qua LangGraph Multi-Agent.
    """
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        session_id = data.get("session_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = data.get("user_id") or ""
        attachments = data.get("attachments", [])

        if not message and not attachments:
            return jsonify({"error": "Nội dung tin nhắn hoặc tệp đính kèm không được rỗng."}), 400

        # Đảm bảo session tồn tại trong SQLite
        if db.load_session(session_id) is None:
            db.create_session(session_id, user_id=user_id)

        # Trích xuất nội dung từ tệp đính kèm (nếu có)
        attachment_contexts, attachment_summary = _process_attachments(attachments)
        message_payload = message
        if attachment_summary:
            if message_payload:
                message_payload = f"{message_payload}\n\n{attachment_summary}"
            else:
                message_payload = attachment_summary

        # Lấy lịch sử hội thoại gần nhất
        history_text = db.get_history_text(session_id, n=6)

        # Chạy pipeline Multi-Agent qua LangGraph StateGraph
        start_ts = time.time()
        turn_id = int(start_ts * 1000)

        result = handle_query(
            session_id=session_id,
            user_query=message_payload,
            turn_id=turn_id,
            user_id=user_id,
        )
        elapsed = round(time.time() - start_ts, 4)

        final_answer = result.get("final_answer", "")
        intent = result.get("intent", "general")
        agent_timings = result.get("agent_timings", [])
        call_graph = result.get("call_graph", [])
        log = result.get("log", [])
        confidence = result.get("confidence_score", 0.0)

        # Lưu lượt hội thoại vào SQLite
        db.append_turn(session_id, {
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "user_query": message,
            "final_answer": final_answer,
            "intent": intent,
            "confidence_score": confidence,
            "agent_timings": agent_timings,
            "call_graph": call_graph,
            "log": log,
            "attachments": attachment_contexts,
        })

        return jsonify({
            "session_id": session_id,
            "reply": final_answer,
            "intent": intent,
            "elapsed": elapsed,
            "provider": "local_python_v4",
            "confidence_score": confidence,
            "turn_id": turn_id,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """
    Endpoint phản hồi Streaming (Server-Sent Events - SSE).
    Tạo hiệu ứng gõ chữ từng từ mượt mà cho người dùng.
    """
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        session_id = data.get("session_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = data.get("user_id") or ""
        attachments = data.get("attachments", [])

        if not message and not attachments:
            return jsonify({"error": "Tin nhắn rỗng"}), 400

        # Đảm bảo session tồn tại
        if db.load_session(session_id) is None:
            db.create_session(session_id, user_id=user_id)

        # Xử lý tệp đính kèm
        attachment_contexts, attachment_summary = _process_attachments(attachments)
        message_payload = message
        if attachment_summary:
            message_payload = f"{message_payload}\n\n{attachment_summary}" if message_payload else attachment_summary

        start_ts = time.time()
        turn_id = int(start_ts * 1000)

        # Chạy pipeline Multi-Agent
        result = handle_query(session_id=session_id, user_query=message_payload, turn_id=turn_id, user_id=user_id)
        final_answer = result.get("final_answer", "")
        intent = result.get("intent", "general")
        elapsed = round(time.time() - start_ts, 4)

        # Lưu SQLite
        db.append_turn(session_id, {
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "user_query": message,
            "final_answer": final_answer,
            "intent": intent,
            "confidence_score": result.get("confidence_score", 0.0),
            "agent_timings": result.get("agent_timings", []),
            "call_graph": result.get("call_graph", []),
            "log": result.get("log", []),
            "attachments": attachment_contexts,
        })

        def event_stream():
            # Tách nhỏ văn bản thành các chunk từ để tạo hiệu ứng typing
            words = final_answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                data_payload = json.dumps({"token": chunk, "intent": intent})
                yield f"data: {data_payload}\n\n"
                time.sleep(0.015)  # 15ms per token
            # Gửi tín hiệu hoàn tất
            yield f"data: {json.dumps({'done': True, 'elapsed': elapsed, 'intent': intent})}\n\n"

        return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Kiểm tra tình trạng máy chủ."""
    return jsonify({
        "status": "ok",
        "engine": "LangGraph Multi-Agent v4.0",
        "mode": "100% Pure Python Local (No-API)",
        "database": "SQLite (chatbot.db)",
        "agents_count": 10,
    })


@app.route("/api/sessions", methods=["GET"])
def sessions():
    """Lấy danh sách các phiên hội thoại."""
    try:
        return jsonify({"sessions": db.list_sessions()})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/session/new", methods=["POST"])
def new_session():
    """Tạo một phiên hội thoại mới."""
    try:
        data = request.get_json() or {}
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        user_id = data.get("user_id") or ""
        created = db.create_session(session_id, user_id=user_id)
        return jsonify(created)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/history/<session_id>", methods=["GET"])
def history(session_id: str):
    """Lấy toàn bộ lịch sử các turns của một phiên hội thoại."""
    try:
        sess = db.load_session(session_id)
        if not sess:
            return jsonify({"error": "Phiên không tồn tại."}), 404
        return jsonify(sess)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/process/<session_id>", methods=["GET"])
def process(session_id: str):
    """Trả về dữ liệu giám sát Multi-Agent của một phiên."""
    try:
        turn_id = request.args.get("turn_id", type=int)
        report = db.get_process(session_id, turn_id)
        if not report:
            return jsonify({"error": "Không tìm thấy dữ liệu process."}), 404
        return jsonify(report)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/process/<session_id>", methods=["GET"])
def process_page(session_id: str):
    """
    Hiển thị Dashboard đồ họa SVG Call Graph tương tác của chính phiên chat đó.
    """
    return render_template("agent_dashboard.html", session_id=session_id)


# Chạy trực tiếp qua lệnh python app.py
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n{'='*65}")
    print(f"🤖 LangGraph Multi-Agent Chatbot v4.0 (Pure Python Local Engine)")
    print(f"🚀 Server chạy tại: http://127.0.0.1:{port}")
    print(f"📊 Dashboard tiến trình: http://127.0.0.1:{port}/process/<session_id>")
    print(f"{'='*65}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
