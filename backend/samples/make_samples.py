"""Creates 3 sample invoices to test the OCR endpoint: digital PDF, PNG image, scanned-style PDF."""
import pymupdf
from PIL import Image, ImageDraw, ImageFont

LINES = [
    "ABC TRADERS PVT LTD",
    "12 Market Road, New Delhi 110001",
    "TAX INVOICE",
    "Invoice No: INV-2026-0043",
    "Date: 12/07/2026",
    "",
    "Item            Qty    Rate      Amount",
    "Steel Rods       10   1500.00   15000.00",
    "Copper Wire       5   1500.00    7500.00",
    "",
    "Subtotal                        22500.00",
    "GST 18%                          4050.00",
    "TOTAL                           26550.00",
]


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# 1) Digital PDF (has a real text layer)
doc = pymupdf.open()
page = doc.new_page()
y = 72
for line in LINES:
    page.insert_text((60, y), line, fontname="cour", fontsize=11)
    y += 18
doc.save("invoice_digital.pdf")

# 2) PNG image
img = Image.new("RGB", (900, 620), "white")
d = ImageDraw.Draw(img)
f = font(24)
y = 30
for line in LINES:
    d.text((40, y), line, fill="black", font=f)
    y += 40
img.save("invoice_image.png")

# 3) Scanned-style PDF (image only, no text layer)
doc = pymupdf.open()
page = doc.new_page()
page.insert_image(page.rect, filename="invoice_image.png")
doc.save("invoice_scanned.pdf")
print("samples created")
