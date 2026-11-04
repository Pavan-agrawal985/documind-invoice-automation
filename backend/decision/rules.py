"""Validation + decision layer.

Extraction can be wrong. These checks catch impossible or suspicious values
BEFORE anything is approved, using math the invoice itself must satisfy.

Decision rule: no issues -> auto_approved, any issue -> needs_review.
"""
from datetime import date

TOLERANCE = 0.01          # money checks allow a 1-paisa rounding difference
LOW_CONFIDENCE = 0.70     # fields below this go to a human (tunable: a research variable)
REQUIRED = ["vendor_name", "invoice_number", "invoice_date", "total_amount"]


def _issue(code, severity, message):
    return {"code": code, "severity": severity, "message": message}


def _close(a, b):
    return abs(a - b) <= TOLERANCE


def validate(result: dict, today: date | None = None) -> list[dict]:
    today = today or date.today()
    fields = result["fields"]
    value = {k: v["value"] for k, v in fields.items()}
    issues = []

    # 1. required fields present
    for name in REQUIRED:
        if value.get(name) in (None, ""):
            issues.append(_issue("missing_field", "high", f"Missing required field: {name}"))

    # 2. low-confidence fields
    for name, f in fields.items():
        if f["value"] is not None and f["confidence"] < LOW_CONFIDENCE:
            issues.append(_issue("low_confidence", "medium",
                                 f"{name} has low confidence ({f['confidence']:.2f})"))

    # 3. date sanity
    if value.get("invoice_date"):
        if date.fromisoformat(value["invoice_date"]) > today:
            issues.append(_issue("future_date", "high", f"Invoice date {value['invoice_date']} is in the future"))

    # 4. subtotal + tax = total
    s, t, tot = value.get("subtotal"), value.get("tax_amount"), value.get("total_amount")
    if s is not None and t is not None and tot is not None and not _close(s + t, tot):
        issues.append(_issue("total_mismatch", "high",
                             f"Subtotal {s} + tax {t} = {round(s + t, 2)}, but total is {tot}"))

    # 5. each line: qty x rate = amount
    items = result.get("line_items", [])
    for it in items:
        expected = round(it["qty"] * it["rate"], 2)
        if not _close(expected, it["amount"]):
            issues.append(_issue("line_item_mismatch", "high",
                                 f"'{it['description']}': {it['qty']:g} x {it['rate']} = {expected}, "
                                 f"but amount reads {it['amount']}"))

    # 6. line amounts add up to subtotal
    if items and s is not None:
        line_sum = round(sum(it["amount"] for it in items), 2)
        if not _close(line_sum, s):
            issues.append(_issue("subtotal_mismatch", "high",
                                 f"Line items add up to {line_sum}, but subtotal is {s}"))
    return issues


def decide(issues: list[dict]) -> dict:
    return {
        "decision": "needs_review" if issues else "auto_approved",
        "reasons": [i["message"] for i in issues],
    }
