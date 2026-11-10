"""Checks the evaluation script on a tiny fake SROIE-style dataset (no real data, no API calls).
Run from the backend folder:  python tests\\test_evaluate.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["GROQ_API_KEY"] = "test"
os.environ["LLM_CHAIN"] = "groq"

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from evaluation import evaluate  # noqa: E402
from extraction import llm_prompting as llm  # noqa: E402

RECEIPTS = [  # (company, date as written, total)
    ("STAR GROCER SDN BHD", "25/12/2018", "7.42"),
    ("BOOK TA K SDN BHD", "25-12-2018", "9.00"),
    ("MEGA MART TRADING", "03/01/2019", "112.50"),
    ("RESTORAN WAN SHENG", "23-03-2018", "6.70"),
    ("KEDAI RUNCIT AMIN", "15/09/2017", "27.00"),
    ("SEGI CASH CARRY SDN BHD", "02/02/2017", "38.55"),
]


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


TEXTS = {}  # clean receipt text per image, used instead of OCR so the test is machine-independent


def make_dataset(root: Path):
    (root / "test" / "img").mkdir(parents=True)
    (root / "test" / "entities").mkdir(parents=True)
    for i, (company, date, total) in enumerate(RECEIPTS):
        lines = [company, "NO 1 JALAN ABC TAMAN XYZ", "TEL 03-1234 5678", f"DATE: {date}", "",
                 "ITEM A      2   3.50   7.00", f"GST 6%                  {float(total) * 0.06:.2f}",
                 f"TOTAL: RM {total}", "CASH  RM 200.00"]
        img = Image.new("RGB", (900, 560), "white")
        d = ImageDraw.Draw(img)
        y = 20
        for line in lines:
            d.text((30, y), line, fill="black", font=font(28))
            y += 52
        stem = f"X5100{i:04d}"
        TEXTS[stem + ".jpg"] = "\n".join(lines)
        img.save(root / "test" / "img" / f"{stem}.jpg")
        (root / "test" / "entities" / f"{stem}.txt").write_text(
            json.dumps({"company": company, "date": date, "address": "NO 1 JALAN ABC", "total": total}), encoding="utf-8")


calls = {"n": 0}


def fake_call_raw(provider, text):
    """Simulated LLM: reads vendor/date/total from the text, but invents the total for one receipt."""
    calls["n"] += 1
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    vendor = lines[0]
    total = next(l.split("RM")[-1].strip() for l in lines if l.startswith("TOTAL"))
    date = next(l.split(":")[-1].strip() for l in lines if l.startswith("DATE"))
    d, m, y = date.replace("-", "/").split("/")
    if "MEGA MART" in vendor:
        total = "120.00"  # hallucinated: not present in the text
    reply = json.dumps({"vendor_name": vendor, "invoice_number": None, "invoice_date": f"{y}-{m}-{d}",
                        "subtotal": None, "tax_amount": None, "total_amount": total, "line_items": []})
    return reply, {"prompt_tokens": 400, "completion_tokens": 120}, "fake-model"


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    make_dataset(tmp / "SROIE2019")
    llm._call_raw = fake_call_raw
    # bypass real OCR: this test checks scoring and caching, not Tesseract
    evaluate.extract_text = lambda data, name, pre: {"text": TEXTS[name], "ms": 5, "source": "fake"}
    args = ["--data", str(tmp / "SROIE2019"), "--limit", "6", "--sleep", "0",
            "--cache", str(tmp / "cache"), "--out", str(tmp / "out")]

    print("######## RUN 1 (rules + simulated LLM)")
    evaluate.main(args)
    first_calls = calls["n"]

    print("\n######## RUN 2 (must reuse the cache: zero new LLM calls)")
    evaluate.main(args)
    ok_cache = calls["n"] == first_calls

    summary_file = sorted((tmp / "out").glob("summary_*.json"))[-1]
    s = json.loads(summary_file.read_text())
    ok_files = any((tmp / "out").glob("per_receipt_*.csv"))
    print("\nCHECK cache reused:", "PASS" if ok_cache else "FAIL", f"({first_calls} LLM calls total)")
    print("CHECK result files written:", "PASS" if ok_files else "FAIL")
    print("CHECK llm hallucinated total lowers llm total recall:", "PASS" if s["fields"]["llm"]["total_amount"]["correct"] == 5 else "FAIL",
          s["fields"]["llm"]["total_amount"])
    print("CHECK hallucination was routed to review (no silent llm error):", "PASS" if s["safety"]["llm"]["silent_errors"] == 0 else "FAIL",
          s["safety"]["llm"])
    sys.exit(0 if ok_cache and ok_files else 1)
