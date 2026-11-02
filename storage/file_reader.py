"""
storage/file_reader.py — Bộ trích xuất văn bản đa định dạng (PDF, DOCX, XLSX, TXT, Ảnh).

Hoạt động 100% nội bộ, không gọi API bên ngoài.
Hỗ trợ: .txt, .md, .csv, .json, .py, .pdf, .docx, .xlsx, .png/.jpg (OCR cơ bản).
"""

from pathlib import Path
from typing import Optional

# Giới hạn trích xuất tối đa (tránh tràn RAM khi file quá lớn)
MAX_TEXT_LENGTH = 15_000


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    """Cắt ngắn văn bản nếu vượt quá giới hạn."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n\n... [Đã cắt ngắn, chỉ hiển thị {max_len:,} ký tự đầu tiên]"


def extract_text_from_txt(path: Path) -> Optional[str]:
    """Đọc file văn bản thuần (.txt, .md, .csv, .json, .py, .js, .html, .yaml)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return _truncate(text)
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc text file {path.name}: {exc}")
        return None


def extract_text_from_pdf(path: Path) -> Optional[str]:
    """Trích xuất văn bản từ PDF sử dụng PyMuPDF (fitz)."""
    try:
        import fitz  # pymupdf
    except ImportError:
        print("[FileReader] Thiếu thư viện pymupdf. Cài đặt: pip install pymupdf")
        return None
    try:
        doc = fitz.open(str(path))
        pages = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                pages.append(f"--- Trang {page_num} ---\n{text}")
        doc.close()
        return _truncate("\n".join(pages)) if pages else None
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc PDF {path.name}: {exc}")
        return None


def extract_text_from_docx(path: Path) -> Optional[str]:
    """Trích xuất văn bản từ file Word (.docx)."""
    try:
        from docx import Document
    except ImportError:
        print("[FileReader] Thiếu thư viện python-docx. Cài đặt: pip install python-docx")
        return None
    try:
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return _truncate("\n".join(paragraphs)) if paragraphs else None
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc DOCX {path.name}: {exc}")
        return None


def extract_text_from_xlsx(path: Path) -> Optional[str]:
    """Trích xuất dữ liệu từ file Excel (.xlsx)."""
    try:
        import openpyxl
    except ImportError:
        print("[FileReader] Thiếu thư viện openpyxl. Cài đặt: pip install openpyxl")
        return None
    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        rows = []
        for sheet in wb.worksheets:
            rows.append(f"=== Sheet: {sheet.title} ===")
            for row in sheet.iter_rows(values_only=True):
                values = [str(c) for c in row if c is not None]
                if values:
                    rows.append("\t".join(values))
        wb.close()
        return _truncate("\n".join(rows)) if rows else None
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc XLSX {path.name}: {exc}")
        return None


def extract_text_from_image(path: Path) -> Optional[str]:
    """
    Trích xuất chú thích cơ bản từ hình ảnh.
    Lưu ý: Không sử dụng Tesseract OCR (cần cài riêng).
    Chỉ trả về thông tin metadata cơ bản của ảnh.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[FileReader] Thiếu thư viện Pillow. Cài đặt: pip install Pillow")
        return None
    try:
        img = Image.open(str(path))
        info = (
            f"Hình ảnh: {path.name}\n"
            f"Kích thước: {img.width} x {img.height} pixels\n"
            f"Định dạng: {img.format}\n"
            f"Chế độ màu: {img.mode}\n"
        )
        # Đọc EXIF nếu có
        exif = img.getexif()
        if exif:
            info += "Metadata EXIF: có\n"
        img.close()
        return info
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc ảnh {path.name}: {exc}")
        return None


# Bảng ánh xạ phần mở rộng → hàm trích xuất
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".ts",
    ".java", ".yaml", ".yml", ".html", ".htm", ".xml", ".ini", ".cfg",
}

_EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".xlsx": extract_text_from_xlsx,
    ".xlsm": extract_text_from_xlsx,
    ".png": extract_text_from_image,
    ".jpg": extract_text_from_image,
    ".jpeg": extract_text_from_image,
    ".gif": extract_text_from_image,
    ".bmp": extract_text_from_image,
    ".webp": extract_text_from_image,
}


def extract_text(path: Path, content_type: Optional[str] = None) -> Optional[str]:
    """
    Hàm trích xuất tổng hợp — tự động chọn phương thức phù hợp theo phần mở rộng.

    Args:
        path: Đường dẫn tệp cần trích xuất.
        content_type: MIME type (tùy chọn, dùng khi phần mở rộng không rõ).

    Returns:
        Văn bản trích xuất hoặc None nếu không hỗ trợ.
    """
    if not path.exists():
        print(f"[FileReader] Tệp không tồn tại: {path}")
        return None

    suffix = path.suffix.lower()

    # Kiểm tra file văn bản thuần
    if suffix in _TEXT_EXTENSIONS:
        return extract_text_from_txt(path)

    # Kiểm tra theo phần mở rộng
    extractor = _EXTRACTORS.get(suffix)
    if extractor:
        return extractor(path)

    # Kiểm tra theo content_type
    if content_type:
        if content_type.startswith("text/"):
            return extract_text_from_txt(path)
        if "image" in content_type:
            return extract_text_from_image(path)

    print(f"[FileReader] Không hỗ trợ định dạng: {path.name} ({content_type})")
    return None


def scan_documents(docs_dir: str = "data/docs") -> list[dict]:
    """
    Quét thư mục tài liệu và trích xuất văn bản từ tất cả các file hỗ trợ.

    Returns:
        Danh sách dict: [{"filename": str, "path": str, "text": str, "char_count": int}]
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return []

    results = []
    supported = _TEXT_EXTENSIONS | set(_EXTRACTORS.keys())

    for file_path in sorted(docs_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            text = extract_text(file_path)
            if text and text.strip():
                results.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "text": text,
                    "char_count": len(text),
                })

    return results

