"""Offline checks for the LLM method (no API calls, replies are simulated).
Run from the backend folder:  python tests\\run_llm_checks.py
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GROQ_API_KEY"] = "test"
os.environ["GEMINI_API_KEY"] = "test"
os.environ["LLM_CHAIN"] = "groq,gemini"

from decision.rules import compare_methods, decide, validate  # noqa: E402
from extraction import llm_prompting as llm  # noqa: E402
from extraction.rule_based import extract_fields  # noqa: E402

TODAY = date(2026, 11, 4)

# OCR text exactly as Tesseract returned it (with the 1500.90 digit error)
OCR = ("ABC TRADERS PVT LTD\n\n12 Market Road, New Delhi 110001\n\nTAX INVOICE\n\n"
       "Invoice No: INV-2026-0043\n\nDate: 12/07/2026\n\nItem Qty Rate Amount\n"
       "Steel Rods 10 1500.90 15000 .90\nCopper Wire 5 1500.00 7500.00\n"
       "Subtotal 22500 .00\nGST 18% 4050.00\nTOTAL 26550 .00")

GOOD = {"vendor_name": "ABC TRADERS PVT LTD", "invoice_number": "INV-2026-0043", "invoice_date": "2026-07-12",
        "subtotal": 22500.00, "tax_amount": 4050.00, "total_amount": 26550.00,
        "line_items": [{"description": "Steel Rods", "qty": 10, "rate": 1500.90, "amount": 15000.90},
                       {"description": "Copper Wire", "qty": 5, "rate": 1500.00, "amount": 7500.00}]}

failed = 0
total = 0


def check(name, condition, detail=""):
    global failed, total
    total += 1
    failed += not condition
    print(f"{'PASS' if condition else 'FAIL'}  {name}  {detail}")


def fake(replies):
    """replies: {provider: reply_text or Exception}"""
    def _call_raw(provider, text):
        r = replies[provider]
        if isinstance(r, Exception):
            raise r
        return r, {"prompt_tokens": 900, "completion_tokens": 200}, f"{provider}-model"
    llm._call_raw = _call_raw


# 1. faithful LLM answer: values are grounded, validator still catches the OCR digit error
fake({"groq": json.dumps(GOOD)})
r = llm.extract_with_llm(OCR)
f = r["fields"]
check("1. grounded fields get high confidence", all(f[k]["confidence"] >= 0.85 for k in f), str({k: f[k]["confidence"] for k in f}))
issues = validate(r, today=TODAY)
check("1b. OCR digit error still caught", any(i["code"] == "line_item_mismatch" for i in issues))
check("1c. decision = needs_review", decide(issues)["decision"] == "needs_review")

# 2. messy reply: code fences, numbers as strings with commas and currency
messy = "```json\n" + json.dumps({**GOOD, "total_amount": "Rs. 26,550.00", "subtotal": "22,500.00"}) + "\n```"
fake({"groq": messy})
r = llm.extract_with_llm(OCR)
check("2. fenced JSON + string numbers parsed", r["fields"]["total_amount"]["value"] == 26550.0 and r["fields"]["subtotal"]["value"] == 22500.0)

# 3. hallucination: model invents a total that is not in the invoice
fake({"groq": json.dumps({**GOOD, "total_amount": 28000.00})})
r = llm.extract_with_llm(OCR)
check("3. invented total gets low confidence", r["fields"]["total_amount"]["confidence"] <= 0.3, r["fields"]["total_amount"]["evidence"])
issues = validate(r, today=TODAY)
check("3b. flagged as low_confidence", any(i["code"] == "low_confidence" for i in issues))

# 4. LLM silently 'fixes' an OCR error -> not in text -> flagged (kept visible on purpose)
fixed = {**GOOD, "line_items": [{"description": "Steel Rods", "qty": 10, "rate": 1500.00, "amount": 15000.00},
                                {"description": "Copper Wire", "qty": 5, "rate": 1500.00, "amount": 7500.00}]}
fake({"groq": json.dumps(fixed)})
r = llm.extract_with_llm(OCR)
issues = validate(r, today=TODAY)
check("4. silently corrected line item flagged", any(i["code"] == "ungrounded_line_item" for i in issues))

# 5. provider fallback: Groq rate-limited -> Gemini answers
fake({"groq": RuntimeError("429 rate limit"), "gemini": json.dumps(GOOD)})
r = llm.extract_with_llm(OCR)
check("5. falls back to gemini", r["meta"]["provider"] == "gemini" and r["meta"]["failed_attempts"][0]["provider"] == "groq")

# 6. garbage reply from first provider -> fallback
fake({"groq": "Sorry, I cannot help with that.", "gemini": json.dumps(GOOD)})
r = llm.extract_with_llm(OCR)
check("6. bad reply -> next provider", r["meta"]["provider"] == "gemini")

# 7. everything fails -> clear error
fake({"groq": RuntimeError("down"), "gemini": RuntimeError("down")})
try:
    llm.extract_with_llm(OCR)
    check("7. all providers fail raises", False)
except llm.LLMUnavailable as exc:
    check("7. all providers fail raises LLMUnavailable", True)

# 8. no keys at all
saved = {k: os.environ.pop(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY")}
try:
    llm.extract_with_llm(OCR)
    check("8. no keys raises", False)
except llm.LLMUnavailable as exc:
    check("8. no keys -> helpful message", "GROQ_API_KEY" in str(exc))
os.environ.update(saved)

# 9. rules vs LLM disagreement is flagged
fake({"groq": json.dumps({**GOOD, "total_amount": 26500.00})})
r_llm = llm.extract_with_llm(OCR)
r_rules = extract_fields(OCR)
dis = compare_methods(r_rules, r_llm)
check("9. disagreement on total detected", any("total_amount" in i["message"] for i in dis), dis[0]["message"] if dis else "")

# 10. agreement produces no disagreement issue
fake({"groq": json.dumps(GOOD)})
check("10. agreeing methods -> no disagreement", compare_methods(extract_fields(OCR), llm.extract_with_llm(OCR)) == [])

print(f"\n{total - failed}/{total} checks passed")
sys.exit(1 if failed else 0)
