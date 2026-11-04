"""Quick self-check: python tests/run_checks.py  (run from the backend folder)."""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.rules import decide, validate  # noqa: E402
from extraction.rule_based import extract_fields  # noqa: E402

TODAY = date(2026, 11, 4)

CLEAN = """ABC TRADERS PVT LTD
12 Market Road, New Delhi 110001
TAX INVOICE
Invoice No: INV-2026-0043
Date: 12/07/2026
Item Qty Rate Amount
Steel Rods 10 1500.00 15000.00
Copper Wire 5 1500.00 7500.00
Subtotal 22500.00
GST 18% 4050.00
TOTAL 26550.00"""

# Exact text returned by Tesseract on the user's Windows machine
REAL_OCR = ("ABC TRADERS PVT LTD\n\n12 Market Road, New Delhi 110001\n\nTAX INVOICE\n\n"
            "Invoice No: INV-2026-0043\n\nDate: 12/07/2026\n\nItem Qty Rate Amount\n"
            "Steel Rods 10 1500.90 15000 .90\nCopper Wire 5 1500.00 7500.00\n"
            "Subtotal 22500 .00\nGST 18% 4050.00\nTOTAL 26550 .00")

MISSING_TOTAL = CLEAN.replace("TOTAL 26550.00", "")

RECEIPT = """STAR GROCER SDN BHD
TEL: 03-1234 5678
INVOICE NO : TD01167104
DATE: 25/12/2018 8:13 PM
BREAD 2 3.50 7.00
GST @6% 0.42
TOTAL: RM 7.42
CASH RM 10.00
CHANGE RM 2.58"""

CSGST = """Sharma Electricals
Invoice Number: SE/2026/118
Invoice Date: 03 Aug 2026
Taxable Value 10,000.00
CGST 9% 900.00
SGST 9% 900.00
Total Amount 11,800.00"""

CASES = [
    ("clean invoice", CLEAN, "auto_approved"),
    ("REAL OCR output (digit errors)", REAL_OCR, "needs_review"),
    ("missing total", MISSING_TOTAL, "needs_review"),
    ("receipt style (RM, GST @6%)", RECEIPT, "auto_approved"),
    ("CGST + SGST split", CSGST, "auto_approved"),
]

failed = 0
for name, text, expected in CASES:
    result = extract_fields(text)
    issues = validate(result, today=TODAY)
    out = decide(issues)
    ok = out["decision"] == expected
    failed += not ok
    print(f"\n{'PASS' if ok else 'FAIL'}  {name}  ->  {out['decision']}")
    f = result["fields"]
    print("   fields:", {k: v["value"] for k, v in f.items()})
    for r in out["reasons"]:
        print("   issue :", r)

print(f"\n{len(CASES) - failed}/{len(CASES)} checks passed")
sys.exit(1 if failed else 0)
