## 4.x Preliminary Observations (single-sample, not yet an evaluation)

These notes come from development runs on one synthetic invoice and will be replaced by a full evaluation on the SROIE test set.

| Observation | Detail |
|---|---|
| OCR digit errors occur | On a scanned PDF, Tesseract (Windows) read the rate "1500.00" as "1500.90" and the amount "15000.00" as "15000 .90" |
| OCR spacing artifacts | Subtotal and total were read as "22500 .00" and "26550 .00"; the normalization step repaired these |
| Header fields correct | Vendor, invoice number, date, subtotal, tax and total were all extracted correctly despite the line-item error |
| Validation caught the error | Two checks failed (line amount, subtotal sum); the invoice was routed to `needs_review` with exact reasons |
| Speed | OCR took about 883 ms, rule-based extraction about 2 ms (scanned PDF, one run) |
| Digital PDF | No OCR needed; [FILL IN: your measured ms] |

### Known limitations so far
- Numeric dates are read day-first, which is wrong for month-first invoices.
- Line-item parsing expects a "description, quantity, rate, amount" layout.
- Confidence scores are hand-set heuristics, not calibrated probabilities.
- The validator shows that something is inconsistent, but cannot tell which value is wrong.
- Rules were developed on a few samples and may not generalize.