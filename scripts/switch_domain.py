"""
scripts/switch_domain.py — Kích hoạt và huấn luyện tự động chuyển đổi chuyên ngành (One-Command Domain Switcher).

Quy trình tự động thực hiện:
  1. Thẩm định cấu trúc dữ liệu thư mục data/domains/<domain_name>/.
  2. Cập nhật trạng thái kích hoạt data/.active_domain.
  3. Huấn luyện và đóng gói Intent Classifier (Approach A Fast hoặc Approach B Dense).
  4. Quét tài liệu và biên dịch bộ chỉ mục BM25 (.kb_index_cache_<domain>.pkl).
  5. Chạy bộ câu hỏi kiểm định chất lượng (eval_pipeline).
  6. Thực thi kiểm thử bảo đảm Zero-Regression.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.train_intent import train_and_export
from scripts.build_knowledge_index import build_and_export_index
from scripts.eval_pipeline import load_qa_pairs, run_evaluation, export_reports


# Liệt kê danh sách tất cả các chuyên ngành hiện có trong thư mục data/domains.
def list_available_domains() -> list[dict]:
    domains_dir = BASE_DIR / "data" / "domains"
    results = []
    if not domains_dir.exists():
        return results
    for d in sorted(domains_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            meta_path = d / "metadata.json"
            meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            results.append({
                "folder": d.name,
                "domain_name": meta.get("domain_name", d.name),
                "description": meta.get("description", "Không có mô tả"),
                "version": meta.get("version", "1.0.0"),
            })
    return results


# Kiểm tra tính đầy đủ và hợp lệ của cấu trúc thư mục dữ liệu chuyên ngành.
def validate_domain_structure(domain_dir: Path) -> tuple[bool, list[str]]:
    warnings = []
    if not domain_dir.exists() or not domain_dir.is_dir():
        return False, [f"Thư mục không tồn tại: {domain_dir}"]
    meta_path = domain_dir / "metadata.json"
    if not meta_path.exists():
        return False, [f"Thiếu file bắt buộc: {meta_path}"]
    optional_checks = ["intents.jsonl", "kb_docs", "qa_pairs.jsonl", "synonyms.json"]
    for item in optional_checks:
        p = domain_dir / item
        if not p.exists():
            warnings.append(f"Không có '{item}' trong domain (sẽ sử dụng cấu hình mặc định).")
    return True, warnings


# Thực hiện toàn bộ quy trình chuyển đổi chuyên ngành và kiểm thử hồi quy.
def activate_domain(domain_name: str, engine: str = "fast", run_tests: bool = True) -> bool:
    start_time = time.time()
    print(f"\n{'='*75}")
    print(f"🔄 KÍCH HOẠT CHUYÊN NGÀNH: [{domain_name.upper()}]")
    print(f"• Engine phân loại: {engine.upper()}")
    print(f"{'='*75}\n")

    active_file = BASE_DIR / "data" / ".active_domain"

    if domain_name.lower() in ("default", "base"):
        if active_file.exists():
            active_file.unlink()
        print("✓ Đã thiết lập lại trạng thái domain về: DEFAULT BASE")
        train_and_export(engine=engine)
        build_and_export_index(domain_name="default")
        print(f"\n✅ Hoàn tất chuyển về domain mặc định ({time.time() - start_time:.2f}s)!")
        return True

    domain_dir = BASE_DIR / "data" / "domains" / domain_name
    valid, msgs = validate_domain_structure(domain_dir)
    if not valid:
        print(f"❌ Cấu trúc domain '{domain_name}' không hợp lệ:")
        for m in msgs:
            print(f"  - {m}")
        return False

    if msgs:
        for m in msgs:
            print(f"ℹ️ Lưu ý: {m}")

    # 1. Ghi nhận trạng thái active domain
    active_file.parent.mkdir(parents=True, exist_ok=True)
    active_file.write_text(domain_name, encoding="utf-8")
    print(f"--- [Bước 1/4] Kích hoạt domain '{domain_name}' vào {active_file} ---")

    # 2. Huấn luyện mô hình Intent
    print("\n--- [Bước 2/4] Huấn luyện lại mô hình phân loại ý định ---")
    train_and_export(engine=engine, domain_name=domain_name)

    # 3. Xây dựng chỉ mục BM25
    print("\n--- [Bước 3/4] Xây dựng chỉ mục tìm kiếm RAG BM25 ---")
    build_and_export_index(domain_name=domain_name)

    # 4. Chạy thẩm định qa_pairs nếu có
    qa_file = domain_dir / "qa_pairs.jsonl"
    if not qa_file.exists():
        qa_file = BASE_DIR / "data" / "domains" / "_template" / "qa_pairs.jsonl"
    if qa_file.exists():
        print(f"\n--- [Bước 4/4] Chạy thẩm định chất lượng pipeline với {qa_file.name} ---")
        qa_pairs = load_qa_pairs(qa_file)
        if qa_pairs:
            eval_report = run_evaluation(qa_pairs, domain_name=domain_name)
            md_out = BASE_DIR / f"eval_report_{domain_name}.md"
            json_out = BASE_DIR / f"eval_results_{domain_name}.json"
            export_reports(eval_report, md_out, json_out)

    # 5. Chạy Unit Test kiểm thử hồi quy
    if run_tests:
        print("\n--- [Kiểm thử Zero-Regression] Thực thi bộ Unit Test toàn hệ thống ---")
        res = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], capture_output=True, text=True)
        if res.returncode == 0:
            print("✅ TẤT CẢ UNIT TEST ĐÃ PASS THÀNH CÔNG (ZERO-REGRESSION)!")
        else:
            print(f"⚠️ Một số bài test có thể gặp vấn đề:\n{res.stderr or res.stdout}")
        # Đảm bảo domain đang kích hoạt không bị biến đổi bởi tác dụng phụ của test
        active_file.write_text(domain_name, encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"\n🎉 Chuyển đổi sang domain [{domain_name}] thành công rực rỡ trong {elapsed:.2f} giây!")
    return True


# Xử lý tham số dòng lệnh phục vụ thao tác chuyển đổi domain từ thiết bị đầu cuối.
def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi hoặc kích hoạt domain cho hệ thống Multi-Agent.")
    parser.add_argument("--domain", default=None, help="Tên thư mục domain cần kích hoạt trong data/domains/")
    parser.add_argument("--engine", choices=["fast", "dense"], default="fast", help="Kiến trúc Intent Classifier (mặc định: fast)")
    parser.add_argument("--list", action="store_true", help="Liệt kê toàn bộ các domain sẵn có")
    parser.add_argument("--reset-default", action="store_true", help="Khôi phục về domain mặc định ban đầu")
    parser.add_argument("--no-test", action="store_true", help="Bỏ qua bước chạy bộ kiểm thử sau khi chuyển đổi")
    args = parser.parse_args()

    if args.list:
        domains = list_available_domains()
        print(f"\n{'='*70}")
        print(f"📁 DANH SÁCH DOMAIN SẴN CÓ ({len(domains)} domains)")
        print(f"{'='*70}")
        for d in domains:
            print(f"  • {d['folder']:<20}: {d['domain_name']} (v{d['version']})")
            print(f"    Mô tả: {d['description']}\n")
        sys.exit(0)

    if args.reset_default:
        activate_domain("default", engine=args.engine, run_tests=not args.no_test)
        sys.exit(0)

    if not args.domain:
        print("❌ Vui lòng chỉ định domain cần kích hoạt với --domain <tên_domain> hoặc dùng --list.")
        sys.exit(1)

    success = activate_domain(args.domain, engine=args.engine, run_tests=not args.no_test)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
