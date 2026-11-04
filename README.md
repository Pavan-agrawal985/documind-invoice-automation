# DocuMind: Invoice Processing Automation

Final-year B.Tech project. DocuMind reads invoices, extracts key fields, checks them for errors, and sends only doubtful ones to a human. It also compares extraction methods (rule-based, classical ML, LLM) on accuracy, cost and speed.

> Status: work in progress. OCR, rule-based extraction and validation are done. LLM extraction, evaluation and the frontend are next.

## How it works

Upload (PDF or image) → OCR → field extraction → validation → **auto-approved** or **needs review**

## Progress

| Step | What | Status |
|---|---|---|
| 0 | Repo, environment, structure | Done |
| 1 | OCR endpoint (digital PDF text layer, else Tesseract) | Done |
| 2 | Rule-based extraction + validation + decision | Done |
| 3 | LLM extraction (Anthropic API) | Next |
| 4 | Classical ML extraction, SROIE evaluation | Planned |
| 5 | Next.js frontend on Vercel, hosted backend | Planned |

## Validation checks

| Check | Catches |
|---|---|
| Required fields present | Total or invoice number not found |
| Subtotal + tax = total | Misread tax or total |
| Qty × rate = amount (per line) | OCR digit errors |
| Line amounts add up to subtotal | A wrong line item |
| Date not in the future | Misread dates |
| Confidence below 0.70 | Guessed values |

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Server and Tesseract status |
| `POST /ocr` | Invoice file → raw text |
| `POST /extract` | Invoice file → fields, issues, decision |

## Run locally (Windows PowerShell)

Requires Python 3.11+ and [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
copy backend\.env.example backend\.env
cd backend
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoints. Sample invoices are in `backend/samples/`. Self-check: `python tests\run_checks.py`.

## Structure

```
backend/    FastAPI app, OCR, extraction methods, validation, evaluation
frontend/   Next.js app (planned, deployed on Vercel)
database/   SQL schema
data/       datasets (not committed)
thesis/     write-up chapters
```