"""Evaluate DocuMind's extraction methods on SROIE receipts.

Example (from the backend folder):
  python -m evaluation.evaluate --data ..\\data\\raw\\SROIE2019 --limit 50 --no-llm
  python -m evaluation.evaluate --data ..\\data\\raw\\SROIE2019 --limit 50

OCR and LLM answers are cached on disk, so re-running never repeats work
(important for free-tier LLM limits).
"""
import argparse
import ast
import csv
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.ocr import extract_text  # noqa: E402
from decision.rules import compare_methods, decide, validate  # noqa: E402
from evaluation.metrics import CORE_REQUIRED, FIELDS, field_scores, has_wrong_core_field  # noqa: E402
from extraction import llm_prompting  # noqa: E402
from extraction.rule_based import extract_fields  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def read_entities(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def load_samples(data_dir: str, split: str, limit: int, seed: int):
    root = Path(data_dir) / split
    img_dir, ent_dir = root / "img", root / "entities"
    if not img_dir.exists() or not ent_dir.exists():
        sys.exit(f"Could not find {img_dir} and {ent_dir}. Expected SROIE2019/<split>/img and /entities.")
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT and (ent_dir / f"{p.stem}.txt").exists()}
    stems = sorted(imgs)
    random.Random(seed).shuffle(stems)  # fixed seed -> same sample every run
    return [(s, imgs[s], read_entities(ent_dir / f"{s}.txt")) for s in stems[:limit]]


def cached(path: Path, producer):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def llm_cache_tag() -> str:
    first = (llm_prompting.available_providers() or ["none"])[0]
    cfg = llm_prompting.PROVIDERS.get(first)
    model = os.getenv(cfg["model_env"], cfg["default_model"]) if cfg else "none"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{first}_{model}")


def values(result: dict) -> dict:
    return {k: v["value"] for k, v in result["fields"].items()}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 1) if xs else None


def pipeline_summary(name, rows, issue_fn):
    wrong = caught = silent = auto = review = 0
    for r in rows:
        issues = issue_fn(r)
        decision = decide(issues)["decision"]
        bad = has_wrong_core_field(r["gt"], r["pred"][name])
        auto += decision == "auto_approved"
        review += decision == "needs_review"
        wrong += bad
        caught += bad and decision == "needs_review"
        silent += bad and decision == "auto_approved"
        r.setdefault("decision", {})[name] = decision
        r.setdefault("wrong", {})[name] = bad
    n = len(rows)
    return {"invoices": n, "auto_approved": auto, "needs_review": review,
            "invoices_with_errors": wrong, "errors_caught_by_review": caught,
            "silent_errors": silent, "silent_error_rate": round(silent / n, 3) if n else 0.0,
            "review_rate": round(review / n, 3) if n else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path("..") / "data" / "raw" / "SROIE2019"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true", help="evaluate the rule-based method only")
    ap.add_argument("--no-preprocess", action="store_true", help="skip OpenCV cleanup before OCR")
    ap.add_argument("--sleep", type=float, default=2.2, help="seconds between LLM calls (free-tier rate limits)")
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parent / "cache"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args(argv)

    use_pre = not a.no_preprocess
    cache, samples = Path(a.cache), load_samples(a.data, a.split, a.limit, a.seed)
    use_llm = not a.no_llm
    if use_llm and not llm_prompting.available_providers():
        sys.exit("No LLM API key found in backend/.env. Add GROQ_API_KEY, or run with --no-llm.")
    tag = llm_cache_tag() if use_llm else None
    print(f"{len(samples)} receipts | split={a.split} | preprocess={use_pre} | llm={tag if use_llm else 'off'}")

    rows, ocr_fail, llm_fail = [], 0, 0
    for i, (stem, img_path, gt) in enumerate(samples, 1):
        try:
            ocr = cached(cache / f"ocr_{'pre' if use_pre else 'raw'}" / f"{stem}.json",
                         lambda: extract_text(img_path.read_bytes(), img_path.name, use_pre))
        except Exception as exc:  # unreadable image etc.
            ocr_fail += 1
            print(f"  [{i}/{len(samples)}] {stem}: OCR failed ({exc})")
            continue

        t0 = time.perf_counter()
        rules = extract_fields(ocr["text"])
        rules_ms = (time.perf_counter() - t0) * 1000

        llm, llm_meta = None, None
        if use_llm:
            path = cache / f"llm_{tag}" / f"{stem}.json"
            fresh = not path.exists()
            try:
                llm = cached(path, lambda: llm_prompting.extract_with_llm(ocr["text"]))
                llm_meta = llm.get("meta")
                if fresh:
                    time.sleep(a.sleep)
            except llm_prompting.LLMUnavailable as exc:
                llm_fail += 1
                print(f"  [{i}/{len(samples)}] {stem}: LLM unavailable ({str(exc)[:90]})")

        rows.append({"stem": stem, "gt": gt, "ocr": ocr, "rules": rules, "llm": llm,
                     "rules_ms": rules_ms, "llm_meta": llm_meta})
        if i % 10 == 0:
            print(f"  processed {i}/{len(samples)}")

    if use_llm:  # fair comparison: only receipts where every method produced an answer
        dropped = len([r for r in rows if r["llm"] is None])
        rows = [r for r in rows if r["llm"] is not None]
    else:
        dropped = 0
    if not rows:
        sys.exit("No receipts could be evaluated.")

    for r in rows:
        r["pred"] = {"rules": values(r["rules"])}
        if use_llm:
            r["pred"]["llm"] = values(r["llm"])
            r["pred"]["both"] = r["pred"]["llm"]

    req = CORE_REQUIRED
    summary = {"settings": {"split": a.split, "requested": a.limit, "evaluated": len(rows), "seed": a.seed,
                            "preprocess": use_pre, "llm": tag, "ocr_failed": ocr_fail,
                            "llm_failed_or_dropped": dropped}}
    summary["fields"] = {"rules": field_scores([{"gt": r["gt"], "pred": r["pred"]["rules"]} for r in rows], "rules")}
    summary["safety"] = {"rules": pipeline_summary("rules", rows, lambda r: validate(r["rules"], required=req))}
    if use_llm:
        summary["fields"]["llm"] = field_scores([{"gt": r["gt"], "pred": r["pred"]["llm"]} for r in rows], "llm")
        summary["safety"]["llm"] = pipeline_summary("llm", rows, lambda r: validate(r["llm"], required=req))
        summary["safety"]["both"] = pipeline_summary(
            "both", rows,
            lambda r: validate(r["rules"], required=req) + validate(r["llm"], required=req)
            + compare_methods(r["rules"], r["llm"]))
    summary["speed_and_cost"] = {
        "ocr_ms_mean": mean(r["ocr"]["ms"] for r in rows),
        "rules_ms_mean": mean(r["rules_ms"] for r in rows),
        "llm_ms_mean": mean(r["llm_meta"]["ms"] for r in rows if r["llm_meta"]) if use_llm else None,
        "llm_prompt_tokens_mean": mean(r["llm_meta"].get("prompt_tokens") for r in rows if r["llm_meta"]) if use_llm else None,
        "llm_completion_tokens_mean": mean(r["llm_meta"].get("completion_tokens") for r in rows if r["llm_meta"]) if use_llm else None,
    }

    print_report(summary)
    save(summary, rows, a.out, a.split, use_llm)


def print_report(s):
    st = s["settings"]
    print(f"\n=== Evaluated {st['evaluated']} receipts (OCR failed: {st['ocr_failed']}, LLM dropped: {st['llm_failed_or_dropped']}) ===")
    print("\nField accuracy (precision / recall / F1)")
    print(f"{'method':<8}{'field':<22}{'n':>4}{'prec':>8}{'recall':>8}{'F1':>8}")
    for method, fields in s["fields"].items():
        for field, m in fields.items():
            print(f"{method:<8}{field:<22}{m['n']:>4}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")
    print("\nSafety: what happens to wrong extractions (wrong = vendor/date/total incorrect)")
    print(f"{'pipeline':<10}{'wrong':>7}{'caught':>8}{'SILENT':>8}{'silent%':>9}{'review%':>9}")
    for name, m in s["safety"].items():
        print(f"{name:<10}{m['invoices_with_errors']:>7}{m['errors_caught_by_review']:>8}{m['silent_errors']:>8}"
              f"{m['silent_error_rate'] * 100:>8.1f}%{m['review_rate'] * 100:>8.1f}%")
    print("\nSpeed / tokens (means):", s["speed_and_cost"])


def save(summary, rows, out_dir, split, use_llm):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"summary_{split}_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / f"per_receipt_{split}_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["receipt", "field", "ground_truth", "rules", "llm", "rules_ok", "llm_ok", "decision_rules", "decision_both"])
        from evaluation.metrics import is_correct
        for r in rows:
            for field, (key, kind) in FIELDS.items():
                gt = r["gt"].get(key)
                rp = r["pred"]["rules"].get(field)
                lp = r["pred"]["llm"].get(field) if use_llm else ""
                w.writerow([r["stem"], field, gt, rp, lp,
                            is_correct(kind, rp, gt, near=(kind == "text")),
                            is_correct(kind, lp, gt, near=(kind == "text")) if use_llm else "",
                            r["decision"].get("rules"), r["decision"].get("both", "")])
    print(f"\nSaved: {out / f'summary_{split}_{stamp}.json'}\n       {out / f'per_receipt_{split}_{stamp}.csv'}")


if __name__ == "__main__":
    main()
