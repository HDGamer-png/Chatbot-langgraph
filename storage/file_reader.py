"""
storage/file_reader.py — Bộ trích xuất văn bản đa năng toàn diện v4.0.

Hỗ trợ mọi định dạng:
  1. Mã nguồn (Mọi ngôn ngữ): Python, JS, TS, Java, C, C++, C#, Go, Rust, PHP,
     Ruby, HTML, CSS, SQL, Shell, YAML, JSON, TOML, Dockerfile, Makefile...
     (Tự động phát hiện mã nguồn ngay cả với phần mở rộng hiếm hoặc không có đuôi file).
  2. Tài liệu văn phòng: PDF, Word (.docx), Excel (.xlsx, .csv, .tsv), PowerPoint (.pptx).
  3. Hình ảnh: PNG, JPG, JPEG, WEBP, BMP, TIFF, GIF, SVG (Metadata, EXIF, Kích thước, OCR quang học).
  4. Nén & Thư mục: Tệp .zip (tự giải nén và trích xuất cây thư mục), quét đệ quy mọi folder con.
"""

import os
import re
import json
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

# Giới hạn trích xuất tối đa mỗi file (tránh tràn RAM khi file quá lớn)
MAX_TEXT_LENGTH = 25_000
MAX_ZIP_FILES = 50

# Các thư mục rác cần bỏ qua khi duyệt đệ quy thư mục mã nguồn
IGNORED_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".idea", ".vscode", "dist", "build",
    ".next", ".nuxt", "coverage", ".pytest_cache"
}


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    """Cắt ngắn văn bản nếu vượt quá giới hạn an toàn."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n\n... [Đã cắt ngắn, hiển thị {max_len:,} ký tự đầu tiên để tối ưu bộ nhớ]"


def _is_binary_file(path: Path) -> bool:
    """Kiểm tra xem tệp có phải dạng binary (không phải text) hay không."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
            if b"\x00" in chunk:
                return True
        return False
    except Exception:
        return True


# ══════════════════════════════════════════════════════════════════
#  1. TRÍCH XUẤT MÃ NGUỒN & VĂN BẢN (MỌI NGÔN NGỮ LẬP TRÌNH)
# ══════════════════════════════════════════════════════════════════

CODE_AND_TEXT_EXTENSIONS = {
    # Python & Jupyter
    ".py", ".pyw", ".ipynb",
    # JavaScript, TypeScript & Web Frontend
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".svelte",
    ".html", ".htm", ".xhtml", ".css", ".scss", ".sass", ".less",
    # C, C++, C#
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".cs",
    # Java, Kotlin, JVM
    ".java", ".kt", ".kts", ".scala", ".groovy",
    # Go, Rust, Swift, Dart
    ".go", ".rs", ".swift", ".dart",
    # Backend, Scripting
    ".php", ".rb", ".pl", ".pm", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    # Database
    ".sql", ".prisma",
    # Cấu hình, dữ liệu, logs
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".toml",
    ".ini", ".cfg", ".conf", ".env", ".properties",
    # DevOps, Build tools
    ".dockerignore", ".gitignore", ".proto", ".graphql", ".tf", ".lock"
}

NAMED_TEXT_FILES = {
    "dockerfile", "makefile", "procfile", "vagrantfile", "jenkinsfile",
    "gemfile", "cmakelists.txt", "requirements.txt", "pipfile", "license",
    "readme", "changelog"
}


def extract_text_from_code(path: Path) -> Optional[str]:
    """
    Trích xuất nội dung mã nguồn hoặc văn bản thuần.
    Tự động phát hiện encoding (UTF-8, Latin-1, CP1252...) và cấu trúc tệp.
    """
    try:
        # Đọc nội dung với fallback encoding
        raw_text = None
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                raw_text = path.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, OSError):
                continue

        if raw_text is None:
            return None

        ext = path.suffix.lower()
        filename_lower = path.name.lower()

        # Nếu là Jupyter Notebook (.ipynb), trích xuất markdown và code cell
        if ext == ".ipynb":
            try:
                nb = json.loads(raw_text)
                cells = []
                for idx, cell in enumerate(nb.get("cells", []), 1):
                    cell_type = cell.get("cell_type", "code")
                    source = "".join(cell.get("source", []))
                    if source.strip():
                        cells.append(f"--- [Cell {idx} ({cell_type})] ---\n{source.strip()}")
                return _truncate(f"=== Jupyter Notebook: {path.name} ===\n\n" + "\n\n".join(cells))
            except Exception:
                pass

        lang_name = ext.replace(".", "").upper() if ext else filename_lower.upper()
        line_count = len(raw_text.splitlines())
        header = f"=== File mã nguồn: {path.name} ({lang_name} - {line_count:,} dòng) ==="
        return _truncate(f"{header}\n\n{raw_text}")
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc mã nguồn {path.name}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════
#  2. TRÍCH XUẤT TÀI LIỆU VĂN PHÒNG (PDF, WORD, EXCEL, PPTX)
# ══════════════════════════════════════════════════════════════════

def extract_text_from_pdf(path: Path) -> Optional[str]:
    """Trích xuất văn bản từ tệp PDF sử dụng PyMuPDF (fitz) hoặc pypdf."""
    try:
        try:
            import fitz
            doc = fitz.open(str(path))
            pages = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    pages.append(f"--- Trang {page_num} ---\n{text.strip()}")
            doc.close()
            if pages:
                return _truncate(f"=== Tài liệu PDF: {path.name} ({len(pages)} trang) ===\n\n" + "\n\n".join(pages))
        except ImportError:
            pass

        # Fallback qua pypdf nếu không có fitz
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = []
            for idx, p in enumerate(reader.pages, 1):
                t = p.extract_text()
                if t and t.strip():
                    pages.append(f"--- Trang {idx} ---\n{t.strip()}")
            if pages:
                return _truncate(f"=== Tài liệu PDF: {path.name} ({len(pages)} trang) ===\n\n" + "\n\n".join(pages))
        except ImportError:
            pass

        return f"[Tài liệu PDF '{path.name}' không có nội dung văn bản (hoặc là file scan ảnh thuần)]."
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc PDF {path.name}: {exc}")
        return None


def extract_text_from_docx(path: Path) -> Optional[str]:
    """Trích xuất văn bản và bảng biểu từ tệp Word (.docx)."""
    try:
        from docx import Document
        doc = Document(str(path))
        content = []

        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        if paragraphs:
            content.append("\n".join(paragraphs))

        for t_idx, table in enumerate(doc.tables, 1):
            table_lines = [f"--- Bảng {t_idx} ---"]
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                if any(cells):
                    table_lines.append(" | ".join(cells))
            content.append("\n".join(table_lines))

        if content:
            return _truncate(f"=== Tài liệu Word: {path.name} ===\n\n" + "\n\n".join(content))
        return f"[Tệp Word '{path.name}' rỗng]."
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc DOCX {path.name}: {exc}")
        return None


def extract_text_from_xlsx(path: Path) -> Optional[str]:
    """Trích xuất dữ liệu từ bảng tính Excel (.xlsx, .xlsm, .csv)."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        sheets_out = []
        for sheet in wb.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                vals = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if vals:
                    rows.append(" | ".join(vals))
            if rows:
                sheets_out.append(f"--- Sheet: {sheet.title} ({len(rows)} hàng) ---\n" + "\n".join(rows[:100]))
        wb.close()
        if sheets_out:
            return _truncate(f"=== Bảng tính Excel: {path.name} ===\n\n" + "\n\n".join(sheets_out))
        return f"[Bảng tính '{path.name}' không có dữ liệu]."
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc XLSX {path.name}: {exc}")
        return None


def extract_text_from_pptx(path: Path) -> Optional[str]:
    """Trích xuất nội dung slide trình chiếu từ PowerPoint (.pptx)."""
    try:
        from pptx import Presentation
        prs = Presentation(str(path))
        slides_out = []
        for s_idx, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        txt = para.text.strip()
                        if txt:
                            slide_texts.append(txt)
            if slide_texts:
                slides_out.append(f"--- Slide {s_idx} ---\n" + "\n".join(slide_texts))
        if slides_out:
            return _truncate(f"=== Bản trình chiếu PowerPoint: {path.name} ({len(prs.slides)} slides) ===\n\n" + "\n\n".join(slides_out))
        return f"[Tệp PowerPoint '{path.name}' không có nội dung văn bản]."
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc PPTX {path.name}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════
#  3. TRÍCH XUẤT THÔNG TIN HÌNH ẢNH (IMAGES & OCR)
# ══════════════════════════════════════════════════════════════════

def _find_tesseract_cmd() -> Optional[str]:
    """Tự động tìm kiếm vị trí thực thi của Tesseract trên Windows/Linux."""
    standard_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\DUC\AppData\Local\Tesseract-OCR\tesseract.exe",
    ]
    for p in standard_paths:
        if os.path.exists(p):
            return p
    return shutil.which("tesseract")


def extract_text_from_image(path: Path) -> Optional[str]:
    """
    Trích xuất metadata, kích thước, hệ màu, EXIF và nhận diện chữ quang học (OCR).
    Hỗ trợ: PNG, JPG, JPEG, WEBP, BMP, TIFF, GIF, SVG.
    """
    ext = path.suffix.lower()

    # Nếu là file SVG (Vector XML), đọc các thẻ văn bản bên trong
    if ext == ".svg":
        try:
            svg_text = path.read_text(encoding="utf-8", errors="replace")
            # Trích xuất các đoạn text trong thẻ <text>...</text>
            texts = re.findall(r"<text[^>]*>(.*?)</text>", svg_text, re.DOTALL | re.IGNORECASE)
            clean_texts = [re.sub(r"<[^>]+>", "", t).strip() for t in texts if t.strip()]
            header = f"=== File ảnh vector SVG: {path.name} ==="
            if clean_texts:
                return f"{header}\nCác nhãn chữ trong SVG:\n" + "\n".join(f"• {t}" for t in clean_texts)
            return f"{header}\n(Ảnh vector SVG chứa các hình khối đồ họa)."
        except Exception:
            pass

    try:
        from PIL import Image
        img = Image.open(str(path))

        file_size_kb = path.stat().st_size / 1024
        desc = [
            f"=== Tệp hình ảnh: {path.name} ===",
            f"• Định dạng: {img.format or ext.replace('.', '').upper()}",
            f"• Kích thước: {img.width} x {img.height} pixels",
            f"• Hệ màu: {img.mode}",
            f"• Dung lượng: {file_size_kb:.1f} KB",
        ]

        n_frames = getattr(img, "n_frames", 1)
        if n_frames > 1:
            desc.append(f"• Số khung hình động (Frames): {n_frames}")

        exif = img.getexif()
        if exif:
            exif_details = []
            for tag_id, val in exif.items():
                if isinstance(val, (str, int, float)) and len(str(val)) < 80:
                    exif_details.append(f"Tag {tag_id}: {val}")
            if exif_details:
                desc.append("• Thông tin EXIF: " + ", ".join(exif_details[:5]))

        # Thử nhận diện văn bản trong ảnh (OCR)
        ocr_result = None
        try:
            import pytesseract
            tess_cmd = _find_tesseract_cmd()
            if tess_cmd:
                pytesseract.pytesseract.tesseract_cmd = tess_cmd
            ocr_text = pytesseract.image_to_string(img, lang="vie+eng").strip()
            if ocr_text:
                ocr_result = ocr_text
        except Exception:
            pass

        img.close()

        if ocr_result:
            desc.append(f"\n--- [Nội dung văn bản nhận diện từ ảnh (OCR)] ---\n{ocr_result}")

        return "\n".join(desc)
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc ảnh {path.name}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════
#  4. TRÍCH XUẤT TẬP TIN NÉN ZIP & THƯ MỤC (FOLDERS & ARCHIVES)
# ══════════════════════════════════════════════════════════════════

def extract_text_from_zip(path: Path) -> Optional[str]:
    """
    Giải nén trong bộ nhớ và trích xuất nội dung toàn bộ các file bên trong file .zip.
    Cho phép đính kèm một folder được nén thành zip.
    """
    try:
        with zipfile.ZipFile(str(path), 'r') as z:
            namelist = [
                n for n in z.namelist()
                if not n.endswith('/') and not n.startswith('__MACOSX')
                and not any(ignored in n for ignored in IGNORED_DIRS)
            ]

            tree_view = [
                f"=== Lưu trữ ZIP (Thư mục nén): {path.name} ({len(namelist)} tệp) ===",
                "Cấu trúc thư mục bên trong ZIP:"
            ]
            for name in namelist[:30]:
                tree_view.append(f"  📄 {name}")
            if len(namelist) > 30:
                tree_view.append(f"  ... và còn {len(namelist) - 30} tệp khác.")

            extracted_files = ["\n".join(tree_view)]

            # Đọc nội dung tối đa 20 file văn bản/code quan trọng
            count = 0
            for name in namelist:
                if count >= 20:
                    break
                p_name = Path(name)
                ext = p_name.suffix.lower()

                if ext in CODE_AND_TEXT_EXTENSIONS or ext in (".pdf", ".docx"):
                    try:
                        with z.open(name) as f:
                            raw_data = f.read(30_000)
                            try:
                                text_content = raw_data.decode("utf-8")
                            except UnicodeDecodeError:
                                text_content = raw_data.decode("latin-1", errors="replace")

                            if text_content.strip():
                                extracted_files.append(f"\n--- [Tệp trong ZIP: {name}] ---\n{text_content.strip()[:2500]}")
                                count += 1
                    except Exception:
                        continue

            return _truncate("\n\n".join(extracted_files))
    except Exception as exc:
        print(f"[FileReader] Lỗi đọc ZIP {path.name}: {exc}")
        return None


def extract_text_from_folder(dir_path: Path) -> str:
    """
    Quét đệ quy toàn bộ thư mục và tổng hợp nội dung tất cả các file bên trong.
    Tự động lọc bỏ các thư mục rác (.git, node_modules, __pycache__, venv...).
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return f"[Thư mục không tồn tại: {dir_path}]"

    all_files = []
    for f in sorted(dir_path.rglob("*")):
        if f.is_file():
            # Bỏ qua nếu bất kỳ thư mục cha nào nằm trong IGNORED_DIRS
            parts = f.relative_to(dir_path).parts
            if any(p in IGNORED_DIRS or p.startswith('.') for p in parts[:-1]):
                continue
            if not f.name.startswith('.'):
                all_files.append(f)

    lines = [
        f"=== Thư mục dự án: {dir_path.name} (Tổng cộng: {len(all_files)} tệp) ===",
        "Cây cấu trúc thư mục:"
    ]
    for f in all_files[:40]:
        lines.append(f"  📄 {f.relative_to(dir_path).as_posix()}")
    if len(all_files) > 40:
        lines.append(f"  ... và còn {len(all_files) - 40} tệp khác.")

    docs_content = ["\n".join(lines)]

    # Đọc nội dung tối đa 25 file tiêu biểu trong thư mục
    count = 0
    for f in all_files:
        if count >= 25:
            break
        txt = extract_text(f)
        if txt and txt.strip():
            rel_name = f.relative_to(dir_path).as_posix()
            docs_content.append(f"\n--- [Nội dung tệp: {rel_name}] ---\n{txt}")
            count += 1

    return _truncate("\n\n".join(docs_content))


# ══════════════════════════════════════════════════════════════════
#  5. HÀM ĐIỀU PHỐI TỔNG HỢP (ROUTER EXTRACTOR)
# ══════════════════════════════════════════════════════════════════

def extract_text(path: Path, content_type: Optional[str] = None) -> Optional[str]:
    """
    Hàm trích xuất tổng hợp đa năng — Tự động nhận diện mọi định dạng:
    Mã nguồn (mọi ngôn ngữ), PDF, Word, Excel, PowerPoint, Ảnh, ZIP, Thư mục.
    """
    if not path.exists():
        print(f"[FileReader] Tệp không tồn tại: {path}")
        return None

    # Nếu là thư mục
    if path.is_dir():
        return extract_text_from_folder(path)

    suffix = path.suffix.lower()
    filename_lower = path.name.lower()

    # 1. Tập tin nén ZIP
    if suffix == ".zip":
        return extract_text_from_zip(path)

    # 2. Tài liệu văn phòng
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in (".docx", ".doc"):
        return extract_text_from_docx(path)
    if suffix in (".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"):
        return extract_text_from_xlsx(path)
    if suffix in (".pptx", ".ppt"):
        return extract_text_from_pptx(path)

    # 3. Hình ảnh
    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif", ".ico", ".svg"):
        return extract_text_from_image(path)

    # 4. Mã nguồn và văn bản thuần theo đuôi mở rộng hoặc tên đặc biệt
    if (suffix in CODE_AND_TEXT_EXTENSIONS or
            filename_lower in NAMED_TEXT_FILES or
            (content_type and any(content_type.startswith(p) for p in ("text/", "application/json", "application/javascript", "application/xml")))):
        return extract_text_from_code(path)

    # 5. Tệp không có đuôi mở rộng hoặc đuôi mở rộng tùy ý — Kiểm tra nếu không phải binary
    if not _is_binary_file(path):
        return extract_text_from_code(path)

    print(f"[FileReader] Định dạng nhị phân chưa được hỗ trợ: {path.name} ({suffix})")
    return None


def scan_documents(docs_dir: str = "data/docs") -> List[Dict[str, Any]]:
    """
    Quét đệ quy toàn bộ thư mục data/docs/ (bao gồm mọi thư mục con)
    và trích xuất văn bản từ TẤT CẢ các file được hỗ trợ.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return []

    results = []
    for file_path in sorted(docs_path.rglob("*")):
        if file_path.is_file():
            # Bỏ qua các file ẩn hoặc trong thư mục rác
            parts = file_path.relative_to(docs_path).parts
            if any(p in IGNORED_DIRS or p.startswith('.') for p in parts[:-1]):
                continue
            if file_path.name.startswith("."):
                continue

            text = extract_text(file_path)
            if text and text.strip():
                rel_path = file_path.relative_to(docs_path).as_posix()
                results.append({
                    "filename": file_path.name,
                    "rel_path": rel_path,
                    "path": str(file_path),
                    "text": text,
                    "char_count": len(text),
                    "extension": file_path.suffix.lower(),
                })

    return results
