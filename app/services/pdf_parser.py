from pathlib import Path
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: Path) -> dict:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    doc = fitz.open(pdf_path)
    page_count = len(doc)

    page_texts = []
    non_empty_pages = 0

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            non_empty_pages += 1
        page_texts.append(f"\n\n===== 第 {i} 页 =====\n{text}")

    full_text = "".join(page_texts).strip()
    char_count = len(full_text)

    return {
        "page_count": page_count,
        "non_empty_pages": non_empty_pages,
        "char_count": char_count,
        "text": full_text,
    }