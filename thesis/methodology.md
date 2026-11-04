## 3.x System Design and Implementation (Steps 1–2)

### Pipeline overview
The system processes an uploaded invoice in four stages: text acquisition, field extraction, validation, and decision. The backend is implemented in Python with FastAPI.

### Text acquisition
For PDF files, the embedded text layer is read first using PyMuPDF. If it contains fewer than 50 characters, the file is treated as scanned. Scanned PDFs (rendered at 250 DPI, up to 5 pages) and image files are passed to Tesseract OCR (engine mode 3, page segmentation mode 6). Images are preprocessed with OpenCV: grayscale conversion, upscaling to at least 1600 px on the longest side, median blur (kernel 3), and adaptive Gaussian thresholding. Preprocessing can be switched off, so its effect can be measured later.

### Method 1: rule-based extraction
OCR text is first normalized (for example, repairing split decimals such as "22500 .00"). Regular expressions and keyword rules then extract vendor name, invoice number, invoice date, subtotal, tax amount, total amount and line items. Numeric dates are read day-first. Split taxes (CGST + SGST) are summed. Each field is returned with a heuristic confidence score and the source line as evidence.

| Confidence | Condition |
|---|---|
| 0.90 | Field found next to its keyword (invoice no., total, date) |
| 0.85 | Tax summed from multiple lines |
| 0.75 / 0.60 | Vendor name taken from the first three lines / later lines |
| 0.65 | Date found without a "date" keyword |
| −0.05 | Date ambiguous between day-first and month-first |

### Validation and decision layer
Extraction output is checked using constraints the invoice must satisfy. Any failed check sends the invoice to human review; otherwise it is auto-approved.

| Check | Rule |
|---|---|
| Required fields | Vendor, invoice number, date and total must be present |
| Total consistency | Subtotal + tax = total (tolerance 0.01) |
| Line consistency | Quantity × rate = amount, per line (tolerance 0.01) |
| Subtotal consistency | Sum of line amounts = subtotal (tolerance 0.01) |
| Date sanity | Invoice date is not in the future |
| Confidence | Any extracted field below 0.70 is flagged |

The 0.70 threshold is a tunable parameter and will be studied as a research variable (accuracy versus reviewer workload).

### Deployment architecture
The frontend is planned in Next.js on Vercel. The Python backend runs separately, because OCR and ML libraries do not fit comfortably within Vercel's Python function limits (500 MB uncompressed bundle size at the time of writing; cite Vercel documentation).

*References to add: Tesseract overview paper (R. Smith, 2007; verify details before citing), PyMuPDF and FastAPI documentation.*