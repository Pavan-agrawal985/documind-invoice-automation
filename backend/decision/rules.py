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


def validate(result: dict, today: date | None = None, required: list[str] | None = None) -> list[dict]:
    today = today or date.today()
    required = REQUIRED if required is None else required
    fields = result["fields"]
    value = {k: v["value"] for k, v in fields.items()}
    issues = []

    # 1. required fields present
    for name in required:
        if value.get(name) in (None, ""):
            issues.append(_issue("missing_field", "high", f"Missing required field: {name}"))

    # 2. low-confidence fields
    for name, f in fields.items():
        if f["value"] is not None and f["confidence"] < LOW_CONFIDENCE:
            issues.append(_issue("low_confidence", "medium",
                                 f"{name} has low confidence ({f['confidence']:.2f})"))

    # 3. date sanity
    if value.get("invoice_date"):
        try:
            invoice_date = date.fromisoformat(str(value["invoice_date"]))
        except ValueError:
            issues.append(_issue("invalid_date", "high", f"Invoice date '{value['invoice_date']}' is not a valid date"))
        else:
            if invoice_date > today:
                issues.append(_issue("future_date", "high", f"Invoice date {value['invoice_date']} is in the future"))

    # 4. subtotal + tax = total
    s, t, tot = value.get("subtotal"), value.get("tax_amount"), value.get("total_amount")
    if s is not None and t is not None and tot is not None and not _close(s + t, tot):
        issues.append(_issue("total_mismatch", "high",
                             f"Subtotal {s} + tax {t} = {round(s + t, 2)}, but total is {tot}"))

    # 5. each line: qty x rate = amount
    items = result.get("line_items", [])
    for it in items:
        if it.get("grounded") is False:
            issues.append(_issue("ungrounded_line_item", "high",
                                 f"'{it['description']}': numbers not found in the invoice text (possible hallucination)"))
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


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _close(float(a), float(b))
    clean = lambda x: "".join(ch for ch in str(x).casefold() if ch.isalnum())
    ca, cb = clean(a), clean(b)
    return ca == cb or (len(ca) > 3 and len(cb) > 3 and (ca in cb or cb in ca))


def compare_methods(a: dict, b: dict, a_name: str = "rules", b_name: str = "llm") -> list[dict]:
    """Two independent methods disagreeing on a value is a strong sign one of them is wrong."""
    issues = []
    for name, fa in a["fields"].items():
        va, vb = fa["value"], b["fields"][name]["value"]
        if va is None or vb is None:
            continue
        if not _same(va, vb):
            issues.append({"code": "methods_disagree", "severity": "high", "method": "both",
                           "message": f"{name}: {a_name} read {va}, {b_name} read {vb}"})
    return issues
