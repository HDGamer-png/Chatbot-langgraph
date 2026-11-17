"""
scripts/train_intent.py — Huấn luyện và đóng gói bộ phân loại ý định (Intent Classifier).

Hỗ trợ 2 kiến trúc thuần Python/Scikit-Learn (No-Cloud-API):
  1. Approach A (fast): TF-IDF + Tinh chỉnh siêu tham số (LinearSVC / LogisticRegression / MultinomialNB).
  2. Approach B (dense): TF-IDF + TruncatedSVD Dense Embeddings + Normalizer + LogisticRegression.

Xuất file:
  - models/intent_model.joblib: Pipeline nhị phân phục vụ suy luận thời gian thực.
  - models/intent_meta.json: Siêu dữ liệu, nhãn ý định, chỉ số Cross-Validation và báo cáo chi tiết.
"""

import os
import sys
import json
import time
import argparse
import warnings
import unicodedata
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "training" / "base_intents.jsonl"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "intent_model.joblib"
DEFAULT_META_PATH = BASE_DIR / "models" / "intent_meta.json"


# Chuẩn hóa chuỗi văn bản tiếng Việt sang dạng Unicode dựng sẵn NFC và loại bỏ ký tự rác.
def normalize_vietnamese_text(text: str) -> str:
    if not text:
        return ""
    text_nfc = unicodedata.normalize("NFC", text.strip().lower())
    return " ".join(text_nfc.split())


# Nạp tập dữ liệu huấn luyện mẫu từ base_intents.jsonl và thư mục domain được chỉ định.
def load_dataset(base_path: Path, domain_name: str | None = None) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    seen: set[tuple[str, str]] = set()

    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    t = normalize_vietnamese_text(record.get("text", ""))
                    l = record.get("intent", "").strip().lower()
                    if t and l and (l, t) not in seen:
                        seen.add((l, t))
                        texts.append(t)
                        labels.append(l)
                except Exception:
                    continue

    if domain_name and domain_name not in ("default", "base"):
        domain_intents_file = BASE_DIR / "data" / "domains" / domain_name / "intents.jsonl"
        if domain_intents_file.exists():
            with open(domain_intents_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        t = normalize_vietnamese_text(record.get("text", ""))
                        l = record.get("intent", "").strip().lower()
                        if t and l and (l, t) not in seen:
                            seen.add((l, t))
                            texts.append(t)
                            labels.append(l)
                    except Exception:
                        continue

    return texts, labels


# Xây dựng danh sách ứng viên mô hình cho Approach A phục vụ so sánh và tinh chỉnh siêu tham số.
def build_fast_candidates() -> dict[str, Pipeline]:
    # Khởi tạo pipeline trích xuất đặc trưng kết hợp từ vựng và n-gram ký tự độc lập.
    def make_feature_union():
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)),
            ("char", TfidfVectorizer(ngram_range=(2, 4), analyzer="char_wb", sublinear_tf=True)),
        ])

    candidates = {
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=8000, sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", make_feature_union()),
            ("clf", LogisticRegression(C=5.0, max_iter=1000, random_state=42)),
        ]),
        "LinearSVC_Calibrated": Pipeline([
            ("tfidf", make_feature_union()),
            ("clf", CalibratedClassifierCV(estimator=LinearSVC(C=1.0, dual="auto", max_iter=2000, random_state=42), cv=3)),
        ]),
    }
    return candidates


# Xây dựng pipeline xử lý đặc trưng biểu diễn dầy Dense Lexical Embeddings cho Approach B.
def build_dense_pipeline(n_components: int = 48) -> Pipeline:
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)),
            ("char", TfidfVectorizer(ngram_range=(2, 4), analyzer="char_wb", sublinear_tf=True)),
        ])),
        ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
        ("norm", Normalizer(norm="l2")),
        ("clf", LogisticRegression(C=5.0, max_iter=1000, random_state=42)),
    ])


# Thực hiện kiểm định chéo K-Fold và tổng hợp các chỉ số đánh giá độ chính xác của mô hình.
def evaluate_cv(pipeline: Pipeline, texts: list[str], labels: list[str], n_splits: int = 5) -> dict:
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    min_count = min(counts.values()) if counts else 0
    actual_splits = max(2, min(n_splits, min_count)) if min_count >= 2 else 0

    if actual_splits >= 2:
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
        preds = cross_val_predict(pipeline, texts, labels, cv=cv)
    else:
        pipeline.fit(texts, labels)
        preds = pipeline.predict(texts)

    acc = float(accuracy_score(labels, preds))
    f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    f1_weighted = float(f1_score(labels, preds, average="weighted", zero_division=0))
    prec_macro = float(precision_score(labels, preds, average="macro", zero_division=0))
    rec_macro = float(recall_score(labels, preds, average="macro", zero_division=0))
    report_dict = classification_report(labels, preds, output_dict=True, zero_division=0)
    report_text = classification_report(labels, preds, zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "n_splits": actual_splits,
    }


# Điểm điều phối chính huấn luyện, đánh giá K-Fold và xuất mô hình nhị phân ra đĩa.
def train_and_export(engine: str = "fast", domain_name: str | None = None, n_splits: int = 5,
                     output_model: Path = DEFAULT_MODEL_PATH, output_meta: Path = DEFAULT_META_PATH) -> dict:
    start_time = time.time()
    texts, labels = load_dataset(DEFAULT_DATA_PATH, domain_name)
    if not texts:
        raise ValueError(f"Không tìm thấy dữ liệu huấn luyện tại {DEFAULT_DATA_PATH}")

    classes = sorted(list(set(labels)))
    print(f"\n{'='*70}")
    print(f"🚀 HUẤN LUYỆN Ý ĐỊNH MULTI-AGENT (INTENT CLASSIFIER)")
    print(f"• Phương pháp (Engine): {engine.upper()}")
    print(f"• Domain: {domain_name or 'Default Base'}")
    print(f"• Tổng số mẫu: {len(texts)} mẫu | {len(classes)} nhãn: {classes}")
    print(f"{'='*70}")

    best_pipeline = None
    best_name = ""
    best_metrics = {}

    if engine == "fast":
        print("\n[Approach A] Đang so sánh và tinh chỉnh các bộ phân loại Scikit-Learn...")
        candidates = build_fast_candidates()
        best_f1 = -1.0
        for name, pipe in candidates.items():
            metrics = evaluate_cv(pipe, texts, labels, n_splits=n_splits)
            print(f"  - {name:<22}: Acc={metrics['accuracy']:.2%} | F1-Macro={metrics['f1_macro']:.4f} | F1-Weighted={metrics['f1_weighted']:.4f}")
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                best_pipeline = pipe
                best_name = name
                best_metrics = metrics
        print(f"\n=> Chọn mô hình tối ưu: [{best_name}] với F1-Macro = {best_f1:.4f}")
    elif engine == "dense":
        print("\n[Approach B] Đang huấn luyện Dense Lexical Embeddings (TruncatedSVD + Normalizer + LogisticRegression)...")
        n_comp = max(2, min(48, len(texts) - 1))
        best_pipeline = build_dense_pipeline(n_components=n_comp)
        best_name = "TruncatedSVD_Dense_Embeddings"
        best_metrics = evaluate_cv(best_pipeline, texts, labels, n_splits=n_splits)
        print(f"  - Acc={best_metrics['accuracy']:.2%} | F1-Macro={best_metrics['f1_macro']:.4f} | F1-Weighted={best_metrics['f1_weighted']:.4f}")
    else:
        raise ValueError(f"Engine không hợp lệ: '{engine}'. Chọn 'fast' hoặc 'dense'.")

    # Huấn luyện lại trên 100% dữ liệu
    print("\n⏳ Đang huấn luyện lại toàn bộ dữ liệu (100% fit)...")
    best_pipeline.fit(texts, labels)

    # Đóng gói và lưu file nhị phân
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, output_model)
    model_size_kb = output_model.stat().st_size / 1024

    elapsed = time.time() - start_time
    meta_data = {
        "engine": engine,
        "selected_model": best_name,
        "domain": domain_name or "default",
        "trained_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "total_samples": len(texts),
        "classes": classes,
        "model_file": str(output_model.name),
        "model_size_kb": round(model_size_kb, 2),
        "metrics": {
            "accuracy": best_metrics.get("accuracy", 0.0),
            "f1_macro": best_metrics.get("f1_macro", 0.0),
            "f1_weighted": best_metrics.get("f1_weighted", 0.0),
            "precision_macro": best_metrics.get("precision_macro", 0.0),
            "recall_macro": best_metrics.get("recall_macro", 0.0),
            "n_splits": best_metrics.get("n_splits", n_splits),
        },
        "classification_report": best_metrics.get("classification_report_dict", {}),
    }

    output_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Xuất mô hình thành công:")
    print(f"  • Model artifact : {output_model} ({model_size_kb:.1f} KB)")
    print(f"  • Model metadata : {output_meta}")
    print(f"  • Thời gian chạy : {elapsed:.2f}s")
    print(f"\n📊 Báo cáo phân loại chi tiết (Cross-Validation):\n")
    print(best_metrics.get("classification_report_text", ""))

    return meta_data


# Xử lý tham số dòng lệnh phục vụ thực thi kịch bản huấn luyện độc lập.
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Intent Classifier cho hệ thống Multi-Agent.")
    parser.add_argument("--engine", choices=["fast", "dense"], default="fast",
                        help="Kiến trúc mô hình: 'fast' (TF-IDF + Scikit-Learn) hoặc 'dense' (Dense SVD Lexical Embeddings)")
    parser.add_argument("--domain", default=None, help="Tên domain cần nạp thêm intents (nằm trong data/domains/<domain>/)")
    parser.add_argument("--cv", type=int, default=5, help="Số folds Stratified Cross-Validation (mặc định: 5)")
    parser.add_argument("--output-model", default=str(DEFAULT_MODEL_PATH), help="Đường dẫn file mô hình .joblib")
    parser.add_argument("--output-meta", default=str(DEFAULT_META_PATH), help="Đường dẫn file metadata .json")
    args = parser.parse_args()

    train_and_export(
        engine=args.engine,
        domain_name=args.domain,
        n_splits=args.cv,
        output_model=Path(args.output_model),
        output_meta=Path(args.output_meta),
    )


if __name__ == "__main__":
    main()
