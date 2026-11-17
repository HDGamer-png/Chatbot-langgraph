import os
import sys
import time
import uuid
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.graph import handle_query
from storage.database import ChatDatabase
from storage.file_reader import extract_text

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

db = ChatDatabase(str(BASE_DIR / "chatbot.db"))


# Trích xuất và tổng hợp nội dung văn bản từ các tệp và thư mục đính kèm
def _process_attachments(attachments: list[dict]) -> tuple[list[dict], str]:
    contexts = []
    lines = []
    for att in attachments:
        filename = att.get("filename") or "unknown"
        content_type = att.get("type") or ""
        url = att.get("url") or ""
        file_path = None
        if url.startswith("/static/uploads/"):
            safe_name = url.replace("/static/uploads/", "").strip()
            potential_path = UPLOAD_DIR / safe_name
            if potential_path.exists():
                file_path = potential_path

        extracted = extract_text(file_path, content_type) if file_path else None
        note = extracted if extracted else "[Không thể trích xuất nội dung tự động]"
        contexts.append({
            "filename": filename,
            "url": url,
            "type": content_type,
            "text": extracted,
            "path": str(file_path) if file_path else None,
        })
        prefix = "📁 Thư mục đính kèm" if (content_type == "folder" or (file_path and file_path.is_dir())) else "📄 Tệp đính kèm"
        lines.append(f"{prefix}: {filename}\n{note}")

    summary = "\n\n".join(lines)
    if summary:
        summary = "Dưới đây là nội dung trích xuất từ tệp/thư mục đính kèm:\n\n" + summary
    return contexts, summary


# Trả về giao diện web chính của ứng dụng Chatbot
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Nhận tải lên tệp tin đơn lẻ và lưu vào thư mục static uploads
@app.route("/api/upload", methods=["POST"])
def upload_file():
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


# Nhận tải lên nguyên thư mục dự án và lưu giữ cấu trúc cây thư mục con
@app.route("/api/upload-folder", methods=["POST"])
def upload_folder():
    try:
        files = request.files.getlist("files")
        paths = request.form.getlist("paths")
        folder_name = request.form.get("folder_name", "folder").strip() or "folder"

        if not files:
            return jsonify({"error": "Không có tệp nào trong thư mục được chọn."}), 400

        folder_token = f"folder_{uuid.uuid4().hex[:8]}_{secure_filename(folder_name) or 'dir'}"
        target_dir = UPLOAD_DIR / folder_token
        target_dir.mkdir(parents=True, exist_ok=True)

        total_size = 0
        saved_count = 0
        for idx, f in enumerate(files):
            if not f or f.filename == "":
                continue
            rel_str = paths[idx] if idx < len(paths) else f.filename
            safe_parts = [secure_filename(p) for p in Path(rel_str).parts if p and p != ".."]
            if not safe_parts:
                continue
            save_file_path = target_dir.joinpath(*safe_parts)
            save_file_path.parent.mkdir(parents=True, exist_ok=True)
            f.save(save_file_path)
            total_size += save_file_path.stat().st_size
            saved_count += 1

        return jsonify({
            "filename": f"📁 {folder_name} ({saved_count} tệp)",
            "url": f"/static/uploads/{folder_token}",
            "type": "folder",
            "size": total_size,
            "count": saved_count,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# Xử lý truy vấn chat đồng bộ qua pipeline Multi-Agent và lưu trữ lượt tương tác
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        session_id = data.get("session_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = data.get("user_id") or ""
        attachments = data.get("attachments", [])

        if not message and not attachments:
            return jsonify({"error": "Nội dung tin nhắn hoặc tệp đính kèm không được rỗng."}), 400

        if db.load_session(session_id) is None:
            db.create_session(session_id, user_id=user_id)

        attachment_contexts, attachment_summary = _process_attachments(attachments)
        message_payload = message
        if attachment_summary:
            message_payload = f"{message_payload}\n\n{attachment_summary}" if message_payload else attachment_summary

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
        confidence = result.get("confidence_score", 0.0)

        db.append_turn(session_id, {
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "user_query": message,
            "final_answer": final_answer,
            "intent": intent,
            "confidence_score": confidence,
            "agent_timings": result.get("agent_timings", []),
            "call_graph": result.get("call_graph", []),
            "log": result.get("log", []),
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
        return jsonify({"error": str(exc)}), 500


# Xử lý phản hồi dạng luồng Server-Sent Events tạo hiệu ứng gõ phím mượt mà
@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        session_id = data.get("session_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = data.get("user_id") or ""
        attachments = data.get("attachments", [])

        if not message and not attachments:
            return jsonify({"error": "Tin nhắn rỗng"}), 400

        if db.load_session(session_id) is None:
            db.create_session(session_id, user_id=user_id)

        attachment_contexts, attachment_summary = _process_attachments(attachments)
        message_payload = message
        if attachment_summary:
            message_payload = f"{message_payload}\n\n{attachment_summary}" if message_payload else attachment_summary

        start_ts = time.time()
        turn_id = int(start_ts * 1000)

        result = handle_query(session_id=session_id, user_query=message_payload, turn_id=turn_id, user_id=user_id)
        final_answer = result.get("final_answer", "")
        intent = result.get("intent", "general")
        elapsed = round(time.time() - start_ts, 4)

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

        # Tạo luồng dữ liệu SSE truyền từng token từ phản hồi
        def event_stream():
            words = final_answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                data_payload = json.dumps({"token": chunk, "intent": intent})
                yield f"data: {data_payload}\n\n"
                time.sleep(0.015)
            yield f"data: {json.dumps({'done': True, 'elapsed': elapsed, 'intent': intent})}\n\n"

        return Response(stream_with_context(event_stream()), mimetype="text/event-stream")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# Kiểm tra trạng thái hoạt động và thông tin cấu hình của máy chủ
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "engine": "LangGraph Multi-Agent v4.0",
        "mode": "100% Pure Python Local (No-API)",
        "database": "SQLite (chatbot.db)",
        "agents_count": 10,
    })


# Trả về thông tin lĩnh vực chuyên ngành đang kích hoạt và phiên bản mô hình
@app.route("/api/domain/current", methods=["GET"])
def current_domain():
    try:
        active_domain_file = BASE_DIR / "data" / ".active_domain"
        active_domain = "default"
        if active_domain_file.exists():
            active_domain = active_domain_file.read_text(encoding="utf-8").strip() or "default"
        meta_file = BASE_DIR / "models" / "intent_meta.json"
        meta = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        return jsonify({
            "active_domain": active_domain,
            "engine": meta.get("engine", "fast"),
            "model": meta.get("selected_model", "Unknown"),
            "trained_at": meta.get("trained_at", ""),
            "classes": meta.get("classes", []),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# Trả về danh sách tất cả các phiên hội thoại hiện có trong cơ sở dữ liệu
@app.route("/api/sessions", methods=["GET"])
def sessions():
    try:
        return jsonify({"sessions": db.list_sessions()})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# Khởi tạo một phiên trò chuyện mới với mã định danh ngẫu nhiên
@app.route("/api/session/new", methods=["POST"])
def new_session():
    try:
        data = request.get_json() or {}
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        user_id = data.get("user_id") or ""
        created = db.create_session(session_id, user_id=user_id)
        return jsonify(created)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# Trả về chi tiết toàn bộ lịch sử các lượt tương tác của một phiên trò chuyện
@app.route("/api/history/<session_id>", methods=["GET"])
def history(session_id: str):
    try:
        sess = db.load_session(session_id)
        if not sess:
            return jsonify({"error": "Phiên không tồn tại."}), 404
        return jsonify(sess)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# Cung cấp dữ liệu hiệu năng chi tiết và chuỗi thực thi của các Agent theo phiên
@app.route("/api/process/<session_id>", methods=["GET"])
def process(session_id: str):
    try:
        turn_id = request.args.get("turn_id", type=int)
        report = db.get_process(session_id, turn_id)
        if not report:
            return jsonify({"error": "Không tìm thấy dữ liệu process."}), 404
        return jsonify(report)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# Hiển thị trang giao diện dashboard trực quan hóa đồ thị thực thi của các Agent
@app.route("/process/<session_id>", methods=["GET"])
def process_page(session_id: str):
    return render_template("agent_dashboard.html", session_id=session_id)


# Lấy thông tin hồ sơ người dùng theo user_id
@app.route("/api/users/<user_id>", methods=["GET"])
def get_user(user_id: str):
    try:
        user_info = db.get_user_profile(user_id)
        if not user_info:
            return jsonify({"user_id": user_id, "profile": {"name": "", "email": ""}})
        return jsonify(user_info)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# Lưu hoặc cập nhật thông tin hồ sơ người dùng
@app.route("/api/users/<user_id>", methods=["POST"])
def save_user(user_id: str):
    try:
        data = request.get_json() or {}
        saved = db.save_user_profile(user_id, data)
        return jsonify(saved)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n{'='*65}")
    print(f"🤖 LangGraph Multi-Agent Chatbot v4.0 (Pure Python Local Engine)")
    print(f"🚀 Server chạy tại: http://127.0.0.1:{port}")
    print(f"📊 Dashboard tiến trình: http://127.0.0.1:{port}/process/<session_id>")
    print(f"{'='*65}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
