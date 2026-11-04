import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from decision.rules import decide, validate  # noqa: E402
from extraction.rule_based import extract_fields  # noqa: E402

from .ocr import (  # noqa: E402  (after load_dotenv so env vars are set)
    IMAGE_EXTS,
    OCRUnavailable,
    UnreadableFile,
    extract_text,
    tesseract_available,
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_EXTS = IMAGE_EXTS | {".pdf"}

app = FastAPI(title="DocuMind API", version="0.2.0")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _run_ocr(file: UploadFile, preprocess: bool) -> dict:
    ext = os.path.splitext((file.filename or "").lower())[1]
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use PDF or an image.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (limit {MAX_UPLOAD_MB} MB).")

    try:
        # OCR is CPU-heavy; run it off the event loop so the server stays responsive.
        return await run_in_threadpool(extract_text, data, file.filename, preprocess)
    except UnreadableFile as exc:
        raise HTTPException(422, str(exc))
    except OCRUnavailable as exc:
        raise HTTPException(503, str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "tesseract": tesseract_available()}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), preprocess: bool = True):
    """Step 1: upload an invoice, get raw text back."""
    return await _run_ocr(file, preprocess)


@app.post("/extract")
async def extract(file: UploadFile = File(...), preprocess: bool = True):
    """Step 2: upload an invoice -> OCR -> rule-based extraction -> validation -> decision."""
    ocr_result = await _run_ocr(file, preprocess)

    t0 = time.perf_counter()
    result = extract_fields(ocr_result["text"])
    issues = validate(result)
    outcome = decide(issues)
    extract_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "filename": ocr_result["filename"],
        "decision": outcome["decision"],
        "reasons": outcome["reasons"],
        "fields": result["fields"],
        "line_items": result["line_items"],
        "issues": issues,
        "timing_ms": {"ocr": ocr_result["ms"], "extraction": extract_ms},
        "ocr_source": ocr_result["source"],
        "text": ocr_result["text"],
    }
