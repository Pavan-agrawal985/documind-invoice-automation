"""Clean raw OCR text so the extraction rules see consistent input."""
import re


def normalize(text: str) -> str:
    text = text.replace("\r", "")
    # OCR often splits decimals: "22500 .00" -> "22500.00"
    text = re.sub(r"(\d)\s+\.(\d{2})\b", r"\1.\2", text)
    # collapse runs of spaces/tabs, keep line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
