#!/usr/bin/env python3
"""
Minimal, functioning PDF extractor for medical documents.
- Primary: PyMuPDF (fast, layout-aware text)
- Fallback: OCR via Tesseract (pytesseract + pdf2image)

Outputs TWO JSON files per input PDF in --outdir:
  - <sha256>.archive.json  (detailed, for manual debugging/audit)
  - <sha256>.llm.json      (compact payload for LLM classification)

Usage locally:
  python extractor.py /path/to/input.pdf --outdir ./out

Dependencies:
  pip install pymupdf pytesseract pdf2image Pillow
System deps:
  - Tesseract OCR (tesseract)
  - Poppler (for pdf2image rasterization)

Note: Focused on medical docs (reports, prescriptions, orders, etc.).
      Signal extractors include MRN/DOB/ICD-10/Rx cues.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageFilter
from pdf2image import convert_from_path
import pytesseract

#============================================================================
#                                   Utilities 
#============================================================================

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r", "\n")

    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def slugify_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', '_', name).strip()

def unique_path(outdir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(outdir, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(outdir, f"{base}-{i}{ext}")
        i += 1
    return candidate

def is_probably_scan(doc: fitz.Document, text: str) -> bool:
    """
    Trigger OCR if:
    - very few alpha characters per page, OR
    - high image-page ratio, OR
    - low word density (PyMuPDF extracted but it's mostly noise)
    """
    pages = len(doc) or 1
    t = text or ""
    alpha = sum(ch.isalpha() for ch in t)
    words = len(re.findall(r"\b\w+\b", t))
    alpha_per_page = alpha / pages
    words_per_page = words / pages
    img_pages = 0
    for pno in range(len(doc)):
        if doc[pno].get_images(full=True):
            img_pages += 1
    img_ratio = img_pages / pages

    if alpha_per_page < 120:            
        return True
    if words_per_page < 25:             
        return True
    if img_ratio >= 0.5:                
        return True
    return False

#============================================================================
#                            PyMuPDF extraction
#============================================================================

def extract_text_pymupdf(doc: fitz.Document) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    pages_layout: List[Dict[str, Any]] = []
    page_headings: List[str] = []
    all_text_parts: List[str] = []

    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("dict")["blocks"]
        # first pass: find largest font for heading heuristic
        largest_font = 0.0
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    size = float(s.get("size", 0.0))
                    if size > largest_font:
                        largest_font = size
        heading_thr = max(12.0, largest_font * 0.9)

        elements = []
        page_text_parts = []
        local_headings = []
        for b in blocks:
            text = "".join(
                s.get("text", "") + " "
                for l in b.get("lines", [])
                for s in l.get("spans", [])
            ).strip()
            if not text:
                continue
            max_font = max(
                [float(s.get("size", 0.0)) for l in b.get("lines", []) for s in l.get("spans", [])]
                or [0.0]
            )
            el_type = "heading" if (max_font >= heading_thr or text.isupper()) else "paragraph"
            bbox = [float(v) for v in b.get("bbox", [0, 0, 0, 0])]
            elements.append({"type": el_type, "text": text, "bbox": bbox})
            page_text_parts.append(text)
            if el_type == "heading":
                local_headings.append(text)
        pages_layout.append({
            "index": pno,
            "width": float(page.rect.width),
            "height": float(page.rect.height),
            "rotation": int(page.rotation or 0),
            "elements": elements,
        })
        if local_headings:
            page_headings.extend(local_headings[:3])
        all_text_parts.append("\n".join(page_text_parts))

    full_text = "\n\n".join(all_text_parts)
    return full_text, pages_layout, page_headings

#============================================================================
#                            OCR path 
#============================================================================

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=8))
    return g

def ocr_pdf(pdf_path: str, dpi: int = 650, lang: str = "eng", psm: int = 6) -> Tuple[str, List[str]]:
    """
    OCR a PDF. For noisy scans/faxes, higher DPI helps.
    We also try a small set of PSM modes and pick the better result per page.
    """
    from pdf2image.exceptions import PDFInfoNotInstalledError

    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except PDFInfoNotInstalledError as e:
        raise RuntimeError("Poppler (pdftoppm) is not installed or not in PATH") from e

    def score(txt: str) -> int:
        return sum(c.isalpha() for c in txt) + 2 * len(re.findall(r"\b\w+\b", txt))

    psm_candidates = [psm, 4, 11]  
    out_texts, headings = [], []

    for im in images:
        pim = preprocess_for_ocr(im)
        best_txt, best_score = "", -1
        for p in psm_candidates:
            raw = pytesseract.image_to_string(pim, lang=lang, config=f"--psm {p} --oem 3")
            txt_n = normalize_text(raw)
            sc = score(txt_n)
            if sc > best_score:
                best_txt, best_score = txt_n, sc
        out_texts.append(best_txt)
        first_lines = [ln.strip() for ln in best_txt.splitlines() if ln.strip()][:2]
        headings.extend(first_lines)

    return "\n\n".join(out_texts), headings

# ------------------------------ Signals ------------------------------------

MED_TYPES = [
    "complaince_report",
    "prescription",
    "physician_notes",
    "orders",
    "delivery_receipt",
    "study_report",
    "others",
]

MED_KV_PATTERNS = [
    ("patient_name", r"\b(Patient Name|Pt\.\s*Name|Name)\s*[:\-]\s*([A-Z][A-Za-z'\-]+\s+[A-Z][A-Za-z'\-]+)"),
    ("mrn", r"\b(MRN|Med(?:ical)? Record Number)\s*[:#\-]?\s*([A-Z0-9\-]+)\b"),
    ("dob", r"\b(DOB|Date of Birth)\s*[:\-]\s*([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b"),
    ("dos", r"\b(DOS|Date of Service)\s*[:\-]\s*([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b"),
    ("account_number", r"\b(Account\s*#|Account Number)\s*[:#\-]?\s*([A-Z0-9\-]+)\b"),
    ("provider", r"\b(Provider|Physician|Doctor)\s*[:\-]\s*([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)+)\b"),
]

ICD10_RE = r"\b([A-TV-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?)\b"  # simple ICD-10 pattern
CPT_RE = r"\b(\d{5})\b"
RX_CUES = ["rx", "prescription", "sig:", "take", "dispense", "refill", "mg", "tablet", "capsule"]
ORDER_CUES = ["order", "ordered", "request", "stat", "standing order"]
REPORT_CUES = ["impression:", "findings:", "result:", "interpretation", "technique:"]
SOAP_KEYS = ["soap", "subjective", "objective", "assessment", "plan"]
PHARMACY_CUES = [
    "pharmacy", "dea", "dea#", "dea #", "npi", "dispense", "refill", "refills",
    "qty", "quantity", "substitution", "do not substitute", "daw", "cpap", "dme" 
]

MED_LINE_RE = re.compile(
    r"^[^\n]{0,120}?\b([A-Za-z][A-Za-z\-() ]{2,})\b.*\b(\d{1,4}\s?(mg|mcg|g|unit|units|ml|iu))\b",
    re.IGNORECASE
)


def detect_soap(text: str) -> dict:
    low = text.lower()
    found = {k: (k in low) for k in SOAP_KEYS}
    found_count = sum(found.values())
    section_hits = {}
    for k in ["Subjective:", "Objective:", "Assessment:", "Plan:"]:
        section_hits[k[:-1].lower()] = (k.lower() in low)
    return {
        "present": found_count >= 2 or any(section_hits.values()),
        "keys": [k for k, v in found.items() if v],
        "sections": section_hits
    }

def medication_lines_stats(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines()]
    med_lines = [ln for ln in lines if MED_LINE_RE.search(ln)]
    narrative_lines = [ln for ln in lines if len(ln) > 40 and not MED_LINE_RE.search(ln)]
    return {
        "med_line_count": len(med_lines),
        "narrative_line_count": len(narrative_lines),
        "total_lines": len(lines),
        "narrative_ratio": (len(narrative_lines) / max(1, len(lines))),
        "sample_meds": med_lines[:6]
    }

def pharmacy_cues(text: str) -> dict:
    low = text.lower()
    cues = {c: (c in low) for c in PHARMACY_CUES}
    return {"present": any(cues.values()), "hits": [k for k, v in cues.items() if v]}
        
def cheap_signals(text: str, headings: List[str] = None) -> Dict[str, Any]:
    s = text
    signals: Dict[str, Any] = {
        "page_headings": [],
        "kv_candidates": [],
        "icd10": [],
        "cpt": [],
        "rx_cues": False,
        "order_cues": False,
        "report_cues": False,
        "money": {"currency_candidates": [], "amounts": []},
    }

    if headings:
        signals["page_headings"] = headings[:10]
    else:
        first_lines = [ln.strip() for ln in s.splitlines() if ln.strip()][:5]
        signals["page_headings"] = first_lines

    for label, pat in MED_KV_PATTERNS:
        for m in re.finditer(pat, s, flags=re.IGNORECASE):
            signals["kv_candidates"].append({"key": label, "value": m.group(2).strip()})

    signals["icd10"] = list({m.group(1) for m in re.finditer(ICD10_RE, s)})[:10]
    signals["cpt"] = list({m.group(1) for m in re.finditer(CPT_RE, s)})[:10]

    low = s.lower()
    signals["rx_cues"] = any(c in low for c in RX_CUES)
    signals["order_cues"] = any(c in low for c in ORDER_CUES)
    signals["report_cues"] = any(c in low for c in REPORT_CUES)

    curr = set()
    amts = []
    for m in re.finditer(r"(\$|USD|EUR|€|GBP|£)\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))", s, re.I):
        cur = m.group(1)
        curr.add({"$":"USD","€":"EUR","£":"GBP"}.get(cur, cur.upper()))
        try:
            amts.append({"label":"amount", "value": float(m.group(2).replace(",",""))})
        except Exception:
            pass
    signals["money"] = {"currency_candidates": sorted(list(curr)), "amounts": amts[:5]}
    signals["soap"] = detect_soap(s)
    signals["med_lines"] = medication_lines_stats(s)
    signals["pharmacy_cues"] = pharmacy_cues(s)

    return signals

#=============================================================================
#                            LLM payload 
#=============================================================================

def build_llm_payload(archive: Dict[str, Any], max_chars: int = 4000) -> Dict[str, Any]:
    text = archive.get("text", "")
    head = archive.get("signals", {}).get("page_headings", [])
    kv = archive.get("signals", {}).get("kv_candidates", [])
    sig = archive.get("signals", {})

    lines = [ln for ln in text.splitlines() if ln.strip()]
    snippet_candidates = []
    if lines:
        snippet_candidates.append({"page": 1, "role": "header", "text": lines[0][:300]})
    if len(lines) > 1:
        snippet_candidates.append({"page": 1, "role": "header2", "text": lines[1][:300]})
   
    for key in ["IMPRESSION", "Impression:", "Findings:", "RESULTS:", "Assessment:"]:
        i = text.find(key)
        if i != -1:
            snippet_candidates.append({"page": 1, "role": key.lower().strip(":"), "text": text[i:i+500]})
            break

    llm = {
        "schema_version": "1.0",
        "task": "classify_medical_document_type",
        "allowed_types": MED_TYPES,
        "doc_id": archive.get("doc_id"),
        "hints": {
            "ocr_used": archive.get("extraction", {}).get("ocr_used", False),
            "pages": archive.get("extraction", {}).get("pages", 0),
            "language": "eng"
        },
        "signals": {
            "page_headings": head[:6],
            "kv": {kv_i.get("key"): kv_i.get("value") for kv_i in kv[-10:]},
            "icd10": sig.get("icd10", [])[:10],
            "cpt": sig.get("cpt", [])[:10],
            "rx_cues": sig.get("rx_cues", False),
            "order_cues": sig.get("order_cues", False),
            "report_cues": sig.get("report_cues", False),
            "money": sig.get("money", {}),
            "soap": sig.get("soap", {}),
            "med_lines": sig.get("med_lines", {}),
            "pharmacy_cues": sig.get("pharmacy_cues", {}),
        },
        "snippets": snippet_candidates[:3],
        "fulltext_budgeted": {
            "chars": max_chars,
            "text": text[:max_chars]
        },
        "routing_policy": {
            "return_format": {
                "type": "object",
                "properties": {
                    "predicted_type": {"type": "string"},
                    "confidence_reasoning": {"type": "string"},
                    "key_facts": {"type": "array", "items": {"type": "string"}},
                    "needs_human_review": {"type": "boolean", "default": False}
                },
                "required": ["predicted_type"]
            }
        }
    }
    return llm

#============================================================================
#                           Main extractor 
#============================================================================

@dataclass
class ExtractionInfo:
    engine: str
    ocr_used: bool
    char_count: int
    duration_ms: int
    languages: List[str]
    encrypted: bool
    pages: int


def process_pdf(input_path: str, outdir: str, ocr_lang: str = "eng",
                ocr_if_needed: bool = True, force_ocr: bool = False,
                ocr_dpi: int = 650) -> Tuple[str, str]:
    """Process one PDF. Returns paths to (archive_json_path, llm_json_path)."""
    ensure_dir(outdir)
    file_hash = sha256_file(input_path)
    doc_id = f"sha256:{file_hash}"
    t0 = time.time()

    try:
        with fitz.open(input_path) as doc:
            if doc.is_encrypted:
                raise RuntimeError("Encrypted PDFs are not supported in this script.")
            pymu_text, layout_pages, headings = extract_text_pymupdf(doc)
            pages_count = len(doc)

            suspected_scan = is_probably_scan(doc, pymu_text)
    except Exception as e:
        raise RuntimeError(f"Failed to open/extract PDF: {e}")

    engine = "pymupdf"
    ocr_used = False
    text_full = normalize_text(pymu_text)

    # Heuristic: if too little/low-quality text, fall back to OCR
    if force_ocr or (ocr_if_needed and suspected_scan):
        print("[info] using OCR fallback (scan heuristics or --force-ocr)")
        engine = "tesseract"
        ocr_used = True
        
        if pages_count > 10:
            actual_ocr_dpi = 350
            print(f"[info] Using reduced DPI {actual_ocr_dpi} for {pages_count} pages")
        else:
            actual_ocr_dpi = ocr_dpi
            
        text_full, ocr_headings = ocr_pdf(input_path, dpi=actual_ocr_dpi, lang=ocr_lang, psm=6)
        text_full = normalize_text(text_full)
        layout_pages = []  
        headings = ocr_headings

    duration_ms = int((time.time() - t0) * 1000)

    signals = cheap_signals(text_full, headings)

    archive = {
        "schema_version": "1.0",
        "doc_id": doc_id,
        "source": {
            "filename": os.path.basename(input_path),
            "received_at": now_iso(),
            "size_bytes": os.path.getsize(input_path),
        },
        "extraction": asdict(ExtractionInfo(
            engine=engine,
            ocr_used=ocr_used,
            char_count=len(text_full),
            duration_ms=duration_ms,
            languages=[ocr_lang],
            encrypted=False,
            pages=pages_count,
        )),
        "layout": layout_pages,
        "text": text_full,
        "metadata": {},
        "signals": signals,
        "privacy": {"redactions": [], "contains_pii": True}
    }

    llm_payload = build_llm_payload(archive)


    in_stem = os.path.splitext(os.path.basename(input_path))[0]
    slug = slugify_filename(in_stem)

    archive_path = unique_path(outdir, f"{slug}.archive.json")
    llm_path = unique_path(outdir, f"{slug}.llm.json")

    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_payload, f, ensure_ascii=False, indent=2)

    return archive_path, llm_path

#============================================================================
#                               CLI
#=============================================================================

def main():
    ap = argparse.ArgumentParser(description="PDF extractor with OCR fallback (medical docs)")
    ap.add_argument("input", help="Path to a PDF file")
    ap.add_argument("--outdir", default="out", help="Output directory for JSON files")
    ap.add_argument("--ocr-lang", default="eng", help="Tesseract language code (e.g., eng, spa)")
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    ap.add_argument("--force-ocr", action="store_true", help="Force OCR even if text is found")
    ap.add_argument("--dpi", type=int, default=650, help="DPI for rasterizing PDFs during OCR")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    try:
        archive_path, llm_path = process_pdf(
            args.input,
            outdir=args.outdir,
            ocr_lang=args.ocr_lang,
            ocr_if_needed=not args.no_ocr,
            force_ocr=args.force_ocr,
            ocr_dpi=args.dpi,
        )
    except Exception as e:
        raise SystemExit(f"Error: {e}")

    print("Archive JSON:", archive_path)
    print("LLM JSON:", llm_path)

if __name__ == "__main__":
    main()
