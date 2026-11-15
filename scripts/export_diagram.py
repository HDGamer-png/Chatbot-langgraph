# -*- coding: utf-8 -*-
"""
scripts/export_diagram.py — Xuất sơ đồ Multi-Agent thành ảnh PNG, JPG và SVG độ nét cao.
Phục vụ chèn trực tiếp vào báo cáo, luận văn, và trình chiếu PowerPoint.
"""

import io
import sys
import shutil
import base64
import requests
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent

# 1. Đường dẫn ảnh screenshot người dùng upload
USER_UPLOADED_IMG = Path(r"C:\Users\DUC\.gemini\antigravity\brain\31b7a173-d5f7-45d6-a5e2-0dfb3b3f0890\.user_uploaded\media_1794212906188_af1ccb85.png")

# Lưu bản copy screenshot trực tiếp ra root
if USER_UPLOADED_IMG.exists():
    shutil.copy2(USER_UPLOADED_IMG, REPO_ROOT / "so_do_multi_agent_chup.png")
    # Xuất thêm bản JPG chất lượng cao
    im_user = Image.open(USER_UPLOADED_IMG).convert("RGB")
    im_user.save(REPO_ROOT / "so_do_multi_agent_chup.jpg", quality=95)
    print("✓ Đã lưu bản chụp màn hình gốc: so_do_multi_agent_chup.png & .jpg")


# 2. Định nghĩa Mermaid chuẩn v4.1
MERMAID_CODE = """flowchart TD
    classDef startEnd fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef phase0 fill:#7c3aed,stroke:#6d28d9,stroke-width:2px,color:#fff;
    classDef router fill:#d97706,stroke:#b45309,stroke-width:2px,color:#fff;
    classDef worker fill:#059669,stroke:#047857,stroke-width:2px,color:#fff;
    classDef barrier fill:#475569,stroke:#334155,stroke-width:2px,color:#fff;
    classDef pipeline fill:#0284c7,stroke:#0369a1,stroke-width:2px,color:#fff;
    classDef storage fill:#374151,stroke:#1f2937,stroke-width:1px,stroke-dasharray: 5 5,color:#cbd5e1;

    User(["👤 Người dùng (Web UI / Terminal CLI)"]):::startEnd --> CheckCache{"⚡ LRU Cache Check"}
    CheckCache -- "Trùng lặp (< 1ms)" --> QuickReturn["Trả lời ngay lập tức"]:::startEnd
    CheckCache -- "Truy vấn mới" --> START(["▶ START: Khởi tạo AgentState"]):::startEnd

    START --> Node0["🧠 1. IntentClassifierAgent<br/>(Phân loại ý định & Xử lý ngữ cảnh tiếp nối)"]:::phase0
    Node0 --> Router["🔀 2. RouterAgent<br/>(Bộ điều phối ThreadPoolExecutor)"]:::router

    subgraph Phase1 ["⚡ PHA 1: THỰC THI SONG SONG (ThreadPool max_workers=6)"]
        direction TB
        Router -.->|"Fan-Out"| W1["📚 3. KnowledgeBaseAgent (RAG)<br/>BM25 Cục bộ + Tài liệu dự án"]:::worker
        Router -.->|"Fan-Out"| W2["🧮 4. CalculatorAgent<br/>SymPy & Safe AST Engine"]:::worker
        Router -.->|"Fan-Out"| W3["⏰ 5. DateTimeAgent<br/>Thời gian thực & Âm lịch"]:::worker
        Router -.->|"Fan-Out"| W4["🌐 6. WebSearchAgent<br/>DuckDuckGo / Local Fallback"]:::worker
        Router -.->|"Fan-Out"| W5["💻 7. AnalystAgent<br/>Phân tích AST & Thuật toán"]:::worker
    end

    W1 --> Barrier["🛑 SYNCHRONIZATION BARRIER<br/>(Fan-in: Thu thập kết quả & Timing)"]:::barrier
    W2 --> Barrier
    W3 --> Barrier
    W4 --> Barrier
    W5 --> Barrier

    subgraph Phase2 ["🔄 PHA 2: CHUỖI XỬ LÝ TUẦN TỰ (Sequential Pipeline)"]
        direction TB
        Barrier --> NodePlanner["📋 8. PlannerAgent<br/>(Lập kế hoạch & Lộ trình kỹ thuật)"]:::pipeline
        NodePlanner --> NodeValidator["🛡️ 9. ValidatorAgent<br/>(Kiểm định chéo & Chấm điểm Confidence)"]:::pipeline
        NodeValidator --> NodeResponse["📝 10. ResponseGeneratorAgent<br/>(Tổng hợp báo cáo kỹ thuật Tech Lead)"]:::pipeline
    end

    NodeResponse --> SaveDB["💾 Lưu trữ SQLite (chatbot.db)<br/>Sessions, Turns, Timings, Users"]:::storage
    SaveDB --> Output(["🏁 END: Stream SSE Web / In ra CLI"]):::startEnd
"""

# Render sang PNG & SVG nét căng từ Kroki (Server render Mermaid chính xác 100%)
try:
    print("Đang kết xuất vector SVG từ Kroki...")
    r_svg = requests.post(
        "https://kroki.io/mermaid/svg",
        data=MERMAID_CODE.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        timeout=25,
    )
    print("SVG Status:", r_svg.status_code)
    if r_svg.status_code == 200:
        svg_path = REPO_ROOT / "so_do_multi_agent.svg"
        svg_path.write_bytes(r_svg.content)
        print("✓ Đã lưu file vector sắc nét: so_do_multi_agent.svg")
    else:
        print("SVG Error:", r_svg.text[:300])

    print("Đang kết xuất ảnh PNG HD từ Kroki...")
    r_png = requests.post(
        "https://kroki.io/mermaid/png",
        data=MERMAID_CODE.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        timeout=25,
    )
    print("PNG Status:", r_png.status_code)
    if r_png.status_code == 200:
        hd_png_path = REPO_ROOT / "so_do_multi_agent_hd.png"
        hd_png_path.write_bytes(r_png.content)
        print("✓ Đã lưu ảnh nét chuẩn PowerPoint: so_do_multi_agent_hd.png")

        # Tạo thêm bản JPG chất lượng cao (RGB, 95% quality)
        im_hd = Image.open(io.BytesIO(r_png.content)).convert("RGB")
        im_hd.save(REPO_ROOT / "so_do_multi_agent_hd.jpg", quality=95)
        print("✓ Đã lưu ảnh JPG độ nét cao: so_do_multi_agent_hd.jpg")
    else:
        print("PNG Error:", r_png.text[:300])

    # 3. Tạo thêm bản Sơ đồ Ngang (Landscape - Flowchart LR) tối ưu chuẩn khung hình slide 16:9
    MERMAID_LR = """flowchart LR
    classDef startEnd fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef phase0 fill:#7c3aed,stroke:#6d28d9,stroke-width:2px,color:#fff;
    classDef router fill:#d97706,stroke:#b45309,stroke-width:2px,color:#fff;
    classDef worker fill:#059669,stroke:#047857,stroke-width:2px,color:#fff;
    classDef barrier fill:#475569,stroke:#334155,stroke-width:2px,color:#fff;
    classDef pipeline fill:#0284c7,stroke:#0369a1,stroke-width:2px,color:#fff;
    classDef storage fill:#374151,stroke:#1f2937,stroke-width:1px,stroke-dasharray: 5 5,color:#cbd5e1;

    User(["👤 Người dùng"]):::startEnd --> Cache{"⚡ Cache"}
    Cache --> Node0["🧠 1. IntentClassifier"]:::phase0
    Node0 --> Router["🔀 2. RouterAgent"]:::router

    subgraph Phase1 ["⚡ PHA 1: SONG SONG"]
        direction TB
        W1["📚 3. KnowledgeBase (RAG)"]:::worker
        W2["🧮 4. Calculator"]:::worker
        W3["⏰ 5. DateTime"]:::worker
        W4["🌐 6. WebSearch"]:::worker
        W5["💻 7. Analyst"]:::worker
    end
    Router -.-> W1 & W2 & W3 & W4 & W5

    Barrier["🛑 BARRIER (Fan-in)"]:::barrier
    W1 & W2 & W3 & W4 & W5 --> Barrier

    subgraph Phase2 ["🔄 PHA 2: TUẦN TỰ"]
        direction LR
        P1["📋 8. Planner"]:::pipeline --> P2["🛡️ 9. Validator"]:::pipeline --> P3["📝 10. Response"]:::pipeline
    end
    Barrier --> P1

    P3 --> SaveDB[("💾 SQLite")]:::storage --> Output(["🏁 END: Tra loi"]):::startEnd
"""
    print("Đang kết xuất sơ đồ Ngang (Slide 16:9) từ Kroki...")
    r_lr_png = requests.post(
        "https://kroki.io/mermaid/png",
        data=MERMAID_LR.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        timeout=25,
    )
    if r_lr_png.status_code == 200:
        lr_png_path = REPO_ROOT / "so_do_multi_agent_ngang.png"
        lr_png_path.write_bytes(r_lr_png.content)
        im_lr = Image.open(io.BytesIO(r_lr_png.content)).convert("RGB")
        im_lr.save(REPO_ROOT / "so_do_multi_agent_ngang.jpg", quality=95)
        print("✓ Đã lưu sơ đồ ngang tỷ lệ slide: so_do_multi_agent_ngang.png & .jpg")

        r_lr_svg = requests.post(
            "https://kroki.io/mermaid/svg",
            data=MERMAID_LR.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=25,
        )
        if r_lr_svg.status_code == 200:
            (REPO_ROOT / "so_do_multi_agent_ngang.svg").write_bytes(r_lr_svg.content)
            print("✓ Đã lưu sơ đồ ngang vector: so_do_multi_agent_ngang.svg")

except Exception as err:
    print(f"[!] Lỗi kết xuất: {err}")

print("\nHoàn tất tạo tất cả các định dạng ảnh cho Slide PowerPoint!")
