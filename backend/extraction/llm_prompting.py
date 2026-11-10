"""Method 3: LLM extraction through any OpenAI-compatible API (Groq, Gemini, ...).

Safeguards against LLM mistakes:
1. Strict output schema (Pydantic): malformed replies are rejected.
2. Grounding check: every value must literally appear in the OCR text.
   A value the model "invented" gets low confidence and goes to human review.
3. Provider fallback: if one provider fails or hits its free limit, try the next.
"""
import json
import os
import re
import time

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError, field_validator

from .normalize import normalize
from .rule_based import _amounts, _empty, _field, _parse_date

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        # Model names change often. If you get "model not found", set GEMINI_MODEL
        # in .env to a current Flash model name shown in Google AI Studio.
        "default_model": "gemini-2.5-flash",
    },
}

SYSTEM_PROMPT = """You extract data from invoice text produced by OCR. Reply with ONE JSON object and nothing else.

Rules:
- Use only what is written in the text. If a value is not present, use null. Never guess.
- The text may contain OCR errors. Copy numbers exactly as written; do not correct or recompute them.
- Numbers must be plain numbers (no currency symbols, no thousands separators).
- invoice_date must be YYYY-MM-DD. Dates like 12/07/2026 are day-first (12 July 2026).
- tax_amount is the total tax amount in money (add CGST and SGST if shown separately), not a percentage.
- total_amount is the final amount payable.

JSON keys: vendor_name, invoice_number, invoice_date, subtotal, tax_amount, total_amount,
line_items (a list of objects with description, qty, rate, amount)."""

MAX_TEXT_CHARS = 6000
MAX_OUTPUT_TOKENS = 2000  # hard cap so a bug cannot burn through a quota


class LLMUnavailable(Exception):
    """No provider produced a usable answer."""


# ---------- output schema ----------

def _to_number(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "")
    s = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", s)  # "1500 .90" -> "1500.90"
    m = re.search(r"-?\d+(?:\.\d+)?", s)       # first number; ignores "Rs." / "INR" / "$"
    return float(m.group()) if m else None


class LLMLineItem(BaseModel):
    description: str | None = None
    qty: float | None = None
    rate: float | None = None
    amount: float | None = None

    @field_validator("qty", "rate", "amount", mode="before")
    @classmethod
    def _num(cls, v):
        return _to_number(v)


class LLMInvoice(BaseModel):
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    subtotal: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    line_items: list[LLMLineItem] = []

    @field_validator("subtotal", "tax_amount", "total_amount", mode="before")
    @classmethod
    def _num(cls, v):
        return _to_number(v)

    @field_validator("vendor_name", "invoice_number", "invoice_date", mode="before")
    @classmethod
    def _str(cls, v):
        return None if v in (None, "") else str(v).strip()

    @field_validator("line_items", mode="before")
    @classmethod
    def _items(cls, v):
        return v or []


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the reply")
    return json.loads(raw[start:end + 1])


# ---------- provider call (the one place that talks to the network) ----------

def _call_raw(provider: str, text: str):
    """Returns (reply_text, usage_dict, model_name). Tests replace this function."""
    cfg = PROVIDERS[provider]
    model = os.getenv(cfg["model_env"], cfg["default_model"])
    client = OpenAI(base_url=cfg["base_url"], api_key=os.getenv(cfg["key_env"]), timeout=30, max_retries=1)
    kwargs = dict(
        model=model,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Invoice text:\n\n" + text[:MAX_TEXT_CHARS]},
        ],
    )
    try:
        resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except BadRequestError:
        # some models do not support JSON mode; retry without it
        resp = client.chat.completions.create(**kwargs)
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
    }
    return resp.choices[0].message.content or "", usage, model


# ---------- grounding: is each value really in the OCR text? ----------

def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).casefold().strip()


def _build_result(inv: LLMInvoice, ocr_text: str) -> dict:
    text = normalize(ocr_text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    squashed = _squash(text)
    numbers = _amounts(text)
    dates = {p[0].isoformat() for p in (_parse_date(l) for l in lines) if p}

    def text_field(value):
        if value is None:
            return _empty()
        if _squash(value) in squashed:
            return _field(value, 0.85, "found in OCR text")
        return _field(value, 0.30, "NOT found in OCR text (possible hallucination)")

    def number_field(value):
        if value is None:
            return _empty()
        if any(abs(n - value) <= 0.01 for n in numbers):
            return _field(value, 0.85, "found in OCR text")
        return _field(value, 0.30, "NOT found in OCR text (possible hallucination)")

    def date_field(value):
        if value is None:
            return _empty()
        if value in dates:
            return _field(value, 0.85, "found in OCR text")
        return _field(value, 0.30, "NOT found in OCR text (possible hallucination)")

    items = []
    for it in inv.line_items:
        if it.qty is None or it.rate is None or it.amount is None:
            continue
        grounded = all(any(abs(n - v) <= 0.01 for n in numbers) for v in (it.qty, it.rate, it.amount))
        items.append({"description": it.description or "", "qty": it.qty, "rate": it.rate,
                      "amount": it.amount, "grounded": grounded})

    return {
        "fields": {
            "vendor_name": text_field(inv.vendor_name),
            "invoice_number": text_field(inv.invoice_number),
            "invoice_date": date_field(inv.invoice_date),
            "subtotal": number_field(inv.subtotal),
            "tax_amount": number_field(inv.tax_amount),
            "total_amount": number_field(inv.total_amount),
        },
        "line_items": items,
    }


# ---------- entry point ----------

def available_providers() -> list[str]:
    chain = [p.strip() for p in os.getenv("LLM_CHAIN", "groq,gemini").split(",") if p.strip()]
    return [p for p in chain if p in PROVIDERS and os.getenv(PROVIDERS[p]["key_env"])]


def extract_with_llm(ocr_text: str) -> dict:
    providers = available_providers()
    if not providers:
        raise LLMUnavailable("No LLM API key found. Add GROQ_API_KEY (and optionally GEMINI_API_KEY) to backend/.env.")

    attempts = []
    for provider in providers:
        t0 = time.perf_counter()
        try:
            reply, usage, model = _call_raw(provider, ocr_text)
            inv = LLMInvoice(**_parse_json(reply))
        except (ValidationError, ValueError, TypeError) as exc:
            attempts.append({"provider": provider, "error": f"bad reply: {str(exc)[:120]}"})
            continue
        except Exception as exc:  # network, rate limit (429), auth, server errors
            attempts.append({"provider": provider, "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue

        result = _build_result(inv, ocr_text)
        result["meta"] = {
            "provider": provider,
            "model": model,
            "ms": round((time.perf_counter() - t0) * 1000),
            **usage,
            "failed_attempts": attempts,
        }
        return result

    raise LLMUnavailable("All LLM providers failed: " + "; ".join(f"{a['provider']} ({a['error']})" for a in attempts))
