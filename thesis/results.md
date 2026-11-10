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

| Observation | Detail |
|---|---|
| LLM extraction | All six fields and both line items were extracted correctly (Groq, gpt-oss-120b) and were grounded in the OCR text |
| Method agreement | Rule-based and LLM results agreed on every field |
| Shared OCR error | Both methods reproduced the OCR misread "1500.90"; only the arithmetic checks detected it, so cross-method agreement cannot catch OCR errors |
| Latency | LLM about 2,700 ms vs rule-based about 2 ms (one run) |
| Tokens | 387 prompt + 639 completion tokens for one invoice |

## Comparison of extraction methods (50 SROIE test receipts, seed 42)

OCR text was identical for both methods (Tesseract, OpenCV preprocessing). The LLM was openai/gpt-oss-120b via Groq. No tuning was done on these receipts.

| Field | Rules F1 | LLM F1 |
|---|---|---|
| Vendor (exact) | 0.265 | 0.460 |
| Vendor (near, ≥0.90) | 0.408 | 0.600 |
| Date | 0.747 | 0.723 |
| Total | 0.435 | 0.708 |

| Pipeline | Receipts with errors | Caught by review | Silent errors | Sent to review |
|---|---|---|---|---|
| Rules + validation | 45 | 38 | 7 (14%) | 76% |
| LLM + validation | 37 | 35 | 2 (4%) | 92% |
| Both + agreement check | 37 | 36 | 1 (2%) | 98% |

Mean latency (excluding OCR): rules about 1 ms, LLM about 4.8 s. LLM token use: about 594 prompt and 820 completion tokens per receipt. Mean OCR time was 934 ms.

**Findings.** (1) The LLM extracted totals and vendor names considerably better than hand-written rules. (2) Date accuracy was similar for both methods (F1 about 0.72-0.75), which suggests a limitation shared by both, such as OCR quality or date parsing, rather than the extraction method. (3) Stricter validation reduced silent errors (14% to 2%) but increased the share of receipts sent to human review (76% to 98%), so the system currently supports review triage more than automatic approval. (4) The LLM is about three to four orders of magnitude slower than the rules.

**Limitations.** Small sample (n=50); receipts rather than invoices; three comparable fields; one free-tier LLM.