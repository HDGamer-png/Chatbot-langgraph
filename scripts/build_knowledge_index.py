"""
scripts/build_knowledge_index.py — Biên dịch chỉ mục tìm kiếm BM25 và tạo persistent cache.

Quy trình:
  1. Quét tài liệu văn bản (.md, .txt, .docx, .xlsx, mã nguồn) từ thư mục tài liệu.
  2. Chia nhỏ văn bản thành các đoạn ngữ cảnh (chunking) có độ chồng lấp.
  3. Tính toán chữ ký tệp và mã băm toàn vẹn SHA-256 của toàn bộ kho tri thức.
  4. Xây dựng bảng chỉ mục ngược (Inverted Index) và tần suất tài liệu BM25Okapi.
  5. Đóng gói bộ chỉ mục vào tệp nhị phân (.kb_index_cache.pkl) cho runtime nạp tức thì (< 20ms).
"""

import os
import sys
import time
import pickle
import hashlib
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.agents.rag import KnowledgeBaseAgent


# Tính toán chữ ký băm SHA-256 tổng thể của toàn bộ các tệp tài liệu trong danh mục.
def compute_corpus_sha256(file_paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for fp in sorted(file_paths):
        if fp.is_file():
            try:
                hasher.update(fp.read_bytes())
            except Exception:
                pass
    return hasher.hexdigest()


# Quét và tổng hợp toàn bộ các đường dẫn tệp tài liệu cần lập chỉ mục tìm kiếm.
def collect_document_files(docs_dir: Path, domain_name: str | None = None) -> list[Path]:
    files = []
    if docs_dir.exists():
        for p in sorted(docs_dir.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                files.append(p)
    if domain_name and domain_name not in ("default", "base"):
        domain_docs = BASE_DIR / "data" / "domains" / domain_name / "kb_docs"
        if domain_docs.exists():
            for p in sorted(domain_docs.rglob("*")):
                if p.is_file() and not p.name.startswith("."):
                    files.append(p)
    return files


# Xây dựng chỉ mục tìm kiếm BM25 và lưu trữ bộ nhớ đệm nhị phân ra đĩa.
def build_and_export_index(domain_name: str | None = None, output_file: Path | None = None) -> dict:
    start_time = time.time()
    docs_dir = BASE_DIR / "data" / "docs"

    target_domain = domain_name or "default"
    target_cache = output_file
    if target_cache is None:
        if target_domain not in ("default", "base"):
            target_cache = BASE_DIR / "data" / f".kb_index_cache_{target_domain}.pkl"
        else:
            target_cache = BASE_DIR / "data" / ".kb_index_cache.pkl"

    files = collect_document_files(docs_dir, target_domain)
    corpus_hash = compute_corpus_sha256(files)

    print(f"\n{'='*70}")
    print(f"📚 BIÊN DỊCH CHỈ MỤC TRI THỨC RAG (BM25 INDEX BUILDER)")
    print(f"• Domain: {target_domain}")
    print(f"• Thư mục tài liệu gốc: {docs_dir}")
    print(f"• Tổng số tệp tài liệu: {len(files)} tệp")
    print(f"• Chữ ký toàn vẹn SHA-256: {corpus_hash[:16]}...{corpus_hash[-8:]}")
    print(f"• Tệp đệm đầu ra (Cache): {target_cache}")
    print(f"{'='*70}\n")

    # Khởi tạo KnowledgeBaseAgent với domain và cache_file cụ thể
    agent = KnowledgeBaseAgent(domain=target_domain, cache_file=target_cache)
    agent.rebuild_index(force=True)

    build_duration = time.time() - start_time

    # Đo tốc độ nạp nóng từ cache
    hot_start = time.time()
    agent_cached = KnowledgeBaseAgent(domain=target_domain, cache_file=target_cache)
    hot_load_ms = (time.time() - hot_start) * 1000

    cache_size_kb = target_cache.stat().st_size / 1024 if target_cache.exists() else 0.0

    print(f"✅ Biên dịch chỉ mục BM25 hoàn tất:")
    print(f"  • Tổng số đoạn văn bản (chunks): {len(agent._chunks)}")
    print(f"  • Số lượng từ khóa trong index : {len(agent._inverted_index)}")
    print(f"  • Kích thước tệp đệm cache     : {cache_size_kb:.2f} KB")
    print(f"  • Thời gian biên dịch ban đầu  : {build_duration:.3f}s")
    print(f"  • Tốc độ nạp nóng từ Cache     : {hot_load_ms:.2f}ms (Mục tiêu < 25ms)")

    return {
        "domain": target_domain,
        "total_files": len(files),
        "total_chunks": len(agent._chunks),
        "vocab_size": len(agent._inverted_index),
        "sha256": corpus_hash,
        "cache_file": str(target_cache),
        "cache_size_kb": cache_size_kb,
        "build_duration_sec": build_duration,
        "hot_load_ms": hot_load_ms,
    }


# Xử lý tham số dòng lệnh phục vụ thực thi kịch bản xây dựng chỉ mục độc lập.
def main():
    parser = argparse.ArgumentParser(description="Biên dịch chỉ mục BM25 cho KnowledgeBaseAgent.")
    parser.add_argument("--domain", default=None, help="Tên lĩnh vực chuyên ngành (mặc định: default)")
    parser.add_argument("--output", default=None, help="Đường dẫn file cache đầu ra (mặc định: data/.kb_index_cache.pkl)")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    build_and_export_index(domain_name=args.domain, output_file=out_path)


if __name__ == "__main__":
    main()
