"""Turn an uploaded invoice (PDF or image) into plain text.

Strategy (fast path first):
1. Digital PDF -> read the text layer directly (instant, no OCR errors).
2. Scanned PDF / image -> render, preprocess, run Tesseract.
"""
import os
import shutil
import time

import cv2
import pymupdf  # PyMuPDF
import numpy as np
import pytesseract

from .preprocess import preprocess

MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "5"))
MIN_TEXT_CHARS = 50  # below this, a PDF is treated as scanned
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class OCRUnavailable(Exception):
    pass


class UnreadableFile(Exception):
    pass


def _tesseract_cmd() -> str | None:
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and os.path.exists(cmd):
        return cmd
    return shutil.which("tesseract")


def tesseract_available() -> bool:
    return _tesseract_cmd() is not None


def _ocr_image(img_bgr: np.ndarray, use_preprocess: bool) -> str:
    cmd = _tesseract_cmd()
    if cmd is None:
        raise OCRUnavailable(
            "Tesseract is not installed or not found. Install it and set TESSERACT_CMD in .env."
        )
    pytesseract.pytesseract.tesseract_cmd = cmd
    img = preprocess(img_bgr) if use_preprocess else img_bgr
    # psm 6 = assume a uniform block of text; works well for invoices
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6")


def _page_to_bgr(page: "pymupdf.Page", dpi: int = 250) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def extract_text(data: bytes, filename: str, use_preprocess: bool = True) -> dict:
    start = time.perf_counter()
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".pdf":
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise UnreadableFile("Could not open this PDF.") from exc

        pages = min(len(doc), MAX_PDF_PAGES)
        layer = "\n".join(doc[i].get_text() for i in range(pages)).strip()
        if len(layer) >= MIN_TEXT_CHARS:
            text, source = layer, "pdf_text_layer"
        else:
            text = "\n".join(_ocr_image(_page_to_bgr(doc[i]), use_preprocess) for i in range(pages))
            source = "ocr_pdf"
        doc.close()
    else:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise UnreadableFile("Could not decode this image.")
        text, source, pages = _ocr_image(img, use_preprocess), "ocr_image", 1

    text = text.strip()
    return {
        "filename": filename,
        "source": source,
        "pages": pages,
        "preprocessed": use_preprocess and source != "pdf_text_layer",
        "chars": len(text),
        "ms": round((time.perf_counter() - start) * 1000),
        "text": text,
    }
