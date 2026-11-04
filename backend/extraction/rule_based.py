"""Method 1: rule-based extraction (regex + keyword rules).

Fast and free, but brittle on layouts it has never seen. That is the point:
it is the baseline the other methods are compared against.

Every field comes back as {"value", "confidence", "evidence"} so later steps
(validation, human review, evaluation) know how much to trust it.
"""
import re
from datetime import date

from .normalize import normalize

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# A number such as 1,500.00 or 22500 (but not 18 in "18%")
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d{1,2})?(?!\d)(?!\s*%)")


def _field(value, confidence, evidence=None):
    return {"value": value, "confidence": confidence, "evidence": evidence}


def _empty():
    return _field(None, 0.0, None)


def _amounts(line: str) -> list[float]:
    out = []
    for m in NUM.finditer(line):
        try:
            out.append(round(float(m.group().replace(",", "")), 2))
        except ValueError:
            pass
    return out


# ---------- individual fields ----------

INV_NO = re.compile(
    r"\b(?:invoice|inv|bill|receipt|document)\s*(?:no\.?|number|num\.?|#)\s*[:\-#]?\s*([A-Z0-9][A-Z0-9\-/]{2,})", re.I)
INV_COLON = re.compile(r"\binvoice\s*:\s*([A-Z0-9][A-Z0-9\-/]{2,})", re.I)


def find_invoice_number(lines):
    for pattern in (INV_NO, INV_COLON):
        for line in lines:
            m = pattern.search(line)
            if m and any(c.isdigit() for c in m.group(1)):
                return _field(m.group(1).upper(), 0.9, line)
    return _empty()


VENDOR_SKIP = re.compile(r"invoice|bill|receipt|tax|date|gst|tel|phone|www|@|page|original|duplicate|copy", re.I)


def find_vendor(lines):
    for idx, line in enumerate(lines[:8]):
        s = line.strip()
        letters = sum(c.isalpha() for c in s)
        if letters >= 4 and letters / len(s) > 0.6 and not VENDOR_SKIP.search(s):
            return _field(s, 0.75 if idx < 3 else 0.6, s)
    return _empty()


D_NUM = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
D_ISO = re.compile(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b")
D_DMY = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")
D_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b")


def _safe_date(y, m, d):
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _parse_date(line):
    """Return (date, ambiguous) or None. Numeric dates are read day-first."""
    m = D_ISO.search(line)
    if m:
        d = _safe_date(int(m[1]), int(m[2]), int(m[3]))
        if d:
            return d, False
    m = D_NUM.search(line)
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        y += 2000 if y < 100 else 0
        d = _safe_date(y, b, a)  # day-first
        if d:
            return d, (a <= 12 and b <= 12 and a != b)
    m = D_DMY.search(line)
    if m and m[2][:3].lower() in MONTHS:
        d = _safe_date(int(m[3]), MONTHS[m[2][:3].lower()], int(m[1]))
        if d:
            return d, False
    m = D_MDY.search(line)
    if m and m[1][:3].lower() in MONTHS:
        d = _safe_date(int(m[3]), MONTHS[m[1][:3].lower()], int(m[2]))
        if d:
            return d, False
    return None


def find_date(lines):
    preferred = [l for l in lines if re.search(r"date", l, re.I) and not re.search(r"due", l, re.I)]
    for line in preferred + lines:
        parsed = _parse_date(line)
        if parsed:
            d, ambiguous = parsed
            keyword = line in preferred
            conf = (0.9 if keyword else 0.65) - (0.05 if ambiguous else 0)
            note = line + ("  [read as day-first]" if ambiguous else "")
            return _field(d.isoformat(), round(conf, 2), note)
    return _empty()


def _last_amount_for(lines, patterns, exclude=None):
    """Try patterns in priority order; use the last matching line's last number."""
    for pattern in patterns:
        hits = [l for l in lines
                if re.search(pattern, l, re.I) and not (exclude and re.search(exclude, l, re.I))]
        for line in reversed(hits):
            nums = _amounts(line)
            if nums:
                return _field(nums[-1], 0.9, line)
    return _empty()


def find_total(lines):
    return _last_amount_for(
        lines,
        [r"grand\s*total", r"total\s*(amount|due|payable)", r"amount\s*(due|payable)", r"balance\s*due", r"\btotal\b"],
        exclude=r"sub\s*-?\s*total|total\s*(tax|gst|vat|qty|quantity|items?)",
    )


def find_subtotal(lines):
    return _last_amount_for(lines, [r"sub\s*-?\s*total", r"taxable\s*(value|amount)", r"net\s*(amount|total)"])


TAX_LINE = re.compile(r"\b(?:c\.?gst|s\.?gst|i\.?gst|gst|vat|tax)\b", re.I)
TAX_EXCLUDE = re.compile(r"gstin|tax\s*invoice|taxable|before\s*tax", re.I)
TAX_TOTAL = re.compile(r"total\s*(?:tax|gst|vat)|(?:tax|gst|vat)\s*total", re.I)


def find_tax(lines):
    tax_lines = [l for l in lines if TAX_LINE.search(l) and not TAX_EXCLUDE.search(l)]
    totals = [l for l in tax_lines if TAX_TOTAL.search(l) and _amounts(l)]
    if totals:
        return _field(_amounts(totals[-1])[-1], 0.9, totals[-1])
    parts = [(l, _amounts(l)[-1]) for l in tax_lines if _amounts(l)]
    if parts:
        return _field(round(sum(v for _, v in parts), 2), 0.85, " | ".join(l for l, _ in parts))
    return _empty()


ITEM = re.compile(
    r"^(?P<desc>[A-Za-z][\w\s\-/&.()]*?)\s+(?P<qty>\d+(?:\.\d+)?)\s+(?P<rate>[\d,]+\.\d{2})\s+(?P<amount>[\d,]+\.\d{2})$")


def find_line_items(lines):
    items = []
    for line in lines:
        m = ITEM.match(line.strip())
        if m:
            items.append({
                "description": m["desc"].strip(),
                "qty": float(m["qty"]),
                "rate": float(m["rate"].replace(",", "")),
                "amount": float(m["amount"].replace(",", "")),
            })
    return items


# ---------- entry point ----------

def extract_fields(raw_text: str) -> dict:
    text = normalize(raw_text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return {
        "fields": {
            "vendor_name": find_vendor(lines),
            "invoice_number": find_invoice_number(lines),
            "invoice_date": find_date(lines),
            "subtotal": find_subtotal(lines),
            "tax_amount": find_tax(lines),
            "total_amount": find_total(lines),
        },
        "line_items": find_line_items(lines),
    }
