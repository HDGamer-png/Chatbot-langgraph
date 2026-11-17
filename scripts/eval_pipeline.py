"""
scripts/eval_pipeline.py — Thẩm định tự động độ chính xác, độ phủ tri thức và độ trễ toàn hệ thống.

Đo lường 4 chỉ số cốt lõi:
  1. Intent Accuracy: Tỷ lệ phân loại đúng ý định người dùng.
  2. RAG Hit Rate / Recall@3: Tỷ lệ câu hỏi tìm được đúng đoạn văn bản/từ khóa chuyên môn.
  3. End-to-End Latency: Thời gian đáp ứng của toàn bộ pipeline và từng tác tử thành phần.
  4. Confidence Score Calibration: Điểm tin cậy trung bình của ValidatorAgent.

Xuất kết quả:
  - Báo cáo định dạng Markdown trực quan.
  - Tệp dữ liệu JSON phục vụ đối sánh và lưu trữ khoa học.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.graph import handle_query, clear_query_cache


# Nạp danh sách các cặp câu hỏi - câu trả lời mẫu từ tệp dữ liệu kiểm định JSONL.
def load_qa_pairs(qa_path: Path) -> list[dict]:
    pairs = []
    if not qa_path.exists():
        return pairs
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except Exception:
                continue
    return pairs


# Đánh giá toàn bộ tập câu hỏi qua pipeline Multi-Agent và tính toán các chỉ số thống kê.
def run_evaluation(qa_pairs: list[dict], domain_name: str = "default") -> dict:
    clear_query_cache()
    total = len(qa_pairs)
    if total == 0:
        raise ValueError("Tập dữ liệu câu hỏi thẩm định qa_pairs trống.")

    intent_correct = 0
    keyword_hits = 0
    latencies: list[float] = []
    confidence_scores: list[float] = []
    agent_timing_accum: dict[str, list[float]] = {}
    details: list[dict] = []

    print(f"\n{'='*75}")
    print(f"📊 BẮT ĐẦU THẨM ĐỊNH HIỆU NĂNG MULTI-AGENT RUNTIME")
    print(f"• Domain: {domain_name}")
    print(f"• Số lượng câu hỏi kiểm thử: {total}")
    print(f"{'='*75}\n")

    for idx, item in enumerate(qa_pairs, 1):
        question = item.get("question", "")
        expected_intent = item.get("expected_intent")
        expected_keywords = item.get("expected_keywords", [])
        expected_ans = item.get("answer", "")

        t0 = time.time()
        res = handle_query(session_id=f"eval_{idx}", user_query=question, turn_id=idx)
        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)

        pred_intent = res.get("intent", "general")
        final_answer = res.get("final_answer", "")
        kb_context = res.get("kb_context", "")
        confidence = float(res.get("confidence_score", 0.0))
        confidence_scores.append(confidence)

        # Kiểm tra Intent Accuracy
        is_intent_ok = (pred_intent == expected_intent) if expected_intent else True
        if is_intent_ok:
            intent_correct += 1

        # Kiểm tra Recall / Keyword hit bao gồm cả câu trả lời, context trích xuất và log
        check_text = (final_answer + " " + kb_context + " " + " ".join(res.get("log", []))).lower()
        targets = [k.lower() for k in expected_keywords]
        if expected_ans:
            targets.append(expected_ans.lower())
        hit = any(t in check_text for t in targets) if targets else True
        if hit:
            keyword_hits += 1

        # Ghi nhận thời gian từng agent
        for timing in res.get("agent_timings", []):
            agent_name = timing.get("agent", "Unknown")
            dur = timing.get("duration", 0.0) * 1000
            if agent_name not in agent_timing_accum:
                agent_timing_accum[agent_name] = []
            agent_timing_accum[agent_name].append(dur)

        status_icon = "✅" if (is_intent_ok and hit) else ("🟡" if (is_intent_ok or hit) else "❌")
        print(f"  [{idx}/{total}] {status_icon} '{question}' => Intent={pred_intent} ({'✓' if is_intent_ok else '✗'}) | Hit={'✓' if hit else '✗'} | {elapsed_ms:.1f}ms")

        details.append({
            "index": idx,
            "question": question,
            "expected_intent": expected_intent,
            "predicted_intent": pred_intent,
            "intent_match": is_intent_ok,
            "keyword_hit": hit,
            "confidence": confidence,
            "latency_ms": round(elapsed_ms, 2),
        })

    # Thống kê tổng hợp
    intent_acc = (intent_correct / total) * 100
    hit_rate = (keyword_hits / total) * 100
    avg_latency = sum(latencies) / len(latencies)
    sorted_latencies = sorted(latencies)
    p50_latency = sorted_latencies[int(len(sorted_latencies) * 0.5)]
    p95_latency = sorted_latencies[min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)]
    avg_confidence = sum(confidence_scores) / len(confidence_scores)

    agent_summary = {
        agent: {
            "avg_ms": round(sum(durs) / len(durs), 2),
            "max_ms": round(max(durs), 2),
        }
        for agent, durs in agent_timing_accum.items()
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "domain": domain_name,
        "total_queries": total,
        "intent_accuracy_percent": round(intent_acc, 2),
        "recall_hit_rate_percent": round(hit_rate, 2),
        "latency_ms": {
            "avg": round(avg_latency, 2),
            "p50": round(p50_latency, 2),
            "p95": round(p95_latency, 2),
        },
        "average_confidence": round(avg_confidence, 4),
        "agent_performance": agent_summary,
        "details": details,
    }

    print(f"\n{'='*75}")
    print(f"📈 KẾT QUẢ THẨM ĐỊNH TỔNG HỢP:")
    print(f"  • Intent Accuracy       : {intent_acc:.2f}% ({intent_correct}/{total})")
    print(f"  • RAG Recall / Hit Rate : {hit_rate:.2f}% ({keyword_hits}/{total})")
    print(f"  • End-to-End Latency    : Trung bình={avg_latency:.1f}ms | P50={p50_latency:.1f}ms | P95={p95_latency:.1f}ms")
    print(f"  • Điểm tin cậy TB       : {avg_confidence:.2%}")
    print(f"{'='*75}\n")

    return report


# Xuất báo cáo thẩm định ra tệp định dạng Markdown và JSON theo chuẩn báo cáo khoa học.
def export_reports(report: dict, md_out: Path, json_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [
        f"# Báo Cáo Thẩm Định Hiệu Năng Hệ Thống Multi-Agent ({report.get('domain', 'Default').upper()})",
        f"",
        f"- **Thời điểm kiểm định:** `{report.get('timestamp')}`",
        f"- **Lĩnh vực hoạt động:** `{report.get('domain')}`",
        f"- **Số lượng mẫu thẩm định:** `{report.get('total_queries')}` câu hỏi",
        f"",
        f"## 1. Bảng Chỉ Số Cốt Lõi (Key Performance Indicators)",
        f"",
        f"| Chỉ số đo lường | Kết quả đạt được | Mục tiêu tiêu chuẩn | Trạng thái |",
        f"|---|---|---|---|",
        f"| **Độ chính xác Ý định (Intent Accuracy)** | **{report['intent_accuracy_percent']:.2f}%** | $\\ge 90\\%$ | {'✅ Đạt' if report['intent_accuracy_percent'] >= 90 else '⚠️ Cần cải thiện'} |",
        f"| **Độ phủ Tri thức (Recall / Hit Rate)** | **{report['recall_hit_rate_percent']:.2f}%** | $\\ge 85\\%$ | {'✅ Đạt' if report['recall_hit_rate_percent'] >= 85 else '⚠️ Cần cải thiện'} |",
        f"| **Độ trễ trung bình (Avg Latency)** | **{report['latency_ms']['avg']:.1f} ms** | $< 350\\text{{ ms}}$ | {'✅ Đạt' if report['latency_ms']['avg'] < 350 else '⚠️ Cần cải thiện'} |",
        f"| **Độ trễ P95 (Tail Latency)** | **{report['latency_ms']['p95']:.1f} ms** | $< 600\\text{{ ms}}$ | {'✅ Đạt' if report['latency_ms']['p95'] < 600 else '⚠️ Cần cải thiện'} |",
        f"| **Điểm tin cậy trung bình** | **{report['average_confidence']:.2%}** | $\\ge 75\\%$ | {'✅ Đạt' if report['average_confidence'] >= 0.75 else '⚠️ Cần cải thiện'} |",
        f"",
        f"## 2. Chi Tiết Độ Trễ Từng Tác Tử (Agent Latency Breakdown)",
        f"",
        f"| Tác tử (AI Agent) | Thời gian xử lý trung bình | Thời gian đỉnh (Max) |",
        f"|---|---|---|",
    ]
    for agent, perf in report.get("agent_performance", {}).items():
        md_lines.append(f"| `{agent}` | {perf['avg_ms']:.2f} ms | {perf['max_ms']:.2f} ms |")

    md_lines.extend([
        f"",
        f"## 3. Bảng Kiểm Thử Chi Tiết Từng Câu Hỏi",
        f"",
        f"| # | Câu hỏi thẩm định | Ý định kỳ vọng | Dự đoán | Khớp ý định | Trúng tri thức | Độ tin cậy | Độ trễ |",
        f"|---|---|---|---|---|---|---|---|",
    ])
    for d in report.get("details", []):
        md_lines.append(
            f"| {d['index']} | {d['question']} | `{d['expected_intent']}` | `{d['predicted_intent']}` | "
            f"{'✅' if d['intent_match'] else '❌'} | {'✅' if d['keyword_hit'] else '❌'} | "
            f"{d['confidence']:.2f} | {d['latency_ms']:.1f} ms |"
        )

    md_out.parent.mkdir(parents=True, exist_ok=True)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"📝 Đã xuất báo cáo thẩm định Markdown: {md_out}")
    print(f"📦 Đã xuất báo cáo thẩm định JSON    : {json_out}\n")


# Xử lý tham số dòng lệnh phục vụ thực thi quy trình thẩm định độc lập.
def main():
    parser = argparse.ArgumentParser(description="Đánh giá hiệu năng toàn diện của hệ thống Multi-Agent.")
    parser.add_argument("--qa-file", default=None, help="Đường dẫn file qa_pairs.jsonl")
    parser.add_argument("--domain", default=None, help="Tên lĩnh vực chuyên ngành")
    parser.add_argument("--out-md", default="eval_report.md", help="Tệp báo cáo Markdown đầu ra")
    parser.add_argument("--out-json", default="eval_results.json", help="Tệp báo cáo JSON đầu ra")
    args = parser.parse_args()

    qa_file = Path(args.qa_file) if args.qa_file else None
    if qa_file is None:
        if args.domain:
            qa_file = BASE_DIR / "data" / "domains" / args.domain / "qa_pairs.jsonl"
        else:
            qa_file = BASE_DIR / "data" / "domains" / "_template" / "qa_pairs.jsonl"

    qa_pairs = load_qa_pairs(qa_file)
    if not qa_pairs:
        print(f"❌ Không tìm thấy dữ liệu QA tại {qa_file}")
        sys.exit(1)

    report = run_evaluation(qa_pairs, domain_name=args.domain or "_template")
    export_reports(report, Path(args.out_md), Path(args.out_json))


if __name__ == "__main__":
    main()
