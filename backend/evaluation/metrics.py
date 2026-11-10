"""Scoring helpers for the SROIE evaluation.

SROIE labels four things per receipt: company, date, address, total.
We compare three of them with our fields (address is not extracted by DocuMind).
"""
import re
from difflib import SequenceMatcher

from extraction.rule_based import _parse_date

# our field -> (SROIE key, kind)
FIELDS = {
    "vendor_name": ("company", "text"),
    "invoice_date": ("date", "date"),
    "total_amount": ("total", "total"),
}
CORE_REQUIRED = ["vendor_name", "invoice_date", "total_amount"]  # receipts have no invoice number
NEAR_THRESHOLD = 0.90


def norm_text(s) -> str:
    return "".join(ch for ch in str(s).casefold() if ch.isalnum())


def norm_date(s):
    p = _parse_date(str(s))
    return p[0].isoformat() if p else norm_text(s)


def norm_total(s):
    if isinstance(s, (int, float)):
        return round(float(s), 2)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(s))
    return round(float(m.group().replace(",", "")), 2) if m else None


def is_correct(kind: str, pred, gt, near: bool = False) -> bool:
    if pred is None or gt is None:
        return False
    if kind == "total":
        a, b = norm_total(pred), norm_total(gt)
        return a is not None and b is not None and abs(a - b) <= 0.01
    if kind == "date":
        return norm_date(pred) == norm_date(gt)
    a, b = norm_text(pred), norm_text(gt)
    if near:
        return SequenceMatcher(None, a, b).ratio() >= NEAR_THRESHOLD
    return a == b


def field_scores(rows: list[dict], method: str) -> dict:
    """rows: [{'gt': {...}, 'pred': {field: value}}]. Returns precision/recall/F1 per field."""
    out = {}
    checks = [(f, k, False, f) for f, (_, k) in FIELDS.items()] + [("vendor_name", "text", True, "vendor_name (near)")]
    for field, kind, near, label in checks:
        key = FIELDS[field][0]
        n = predicted = correct = 0
        for r in rows:
            gt = r["gt"].get(key)
            if gt in (None, ""):
                continue
            n += 1
            pred = r["pred"].get(field)
            if pred is not None:
                predicted += 1
            if is_correct(kind, pred, gt, near):
                correct += 1
        p = correct / predicted if predicted else 0.0
        rc = correct / n if n else 0.0
        out[label] = {"n": n, "predicted": predicted, "correct": correct,
                      "precision": round(p, 3), "recall": round(rc, 3),
                      "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0}
    return out


def has_wrong_core_field(gt: dict, pred: dict) -> bool:
    """A 'wrong' invoice = any of vendor (near), date, total is not correct."""
    for field, (key, kind) in FIELDS.items():
        if gt.get(key) in (None, ""):
            continue
        if not is_correct(kind, pred.get(field), gt[key], near=(kind == "text")):
            return True
    return False
