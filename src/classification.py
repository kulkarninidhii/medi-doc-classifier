#!/usr/bin/env python3
"""
Batch LLM classification for medical documents with a clear 4-step validation loop.

Flow per document:
  Step 1) CLASSIFY  – initial decision from analysis_pack
  Step 2) REVIEW    – second pass inspects the decision for issues
  Step 3) REVISE    – third pass corrects the decision as needed
  Step 4) VALIDATE  – final pass enforces hard rules; its output is authoritative

Outputs:
  - classifications.jsonl
  - classifications.csv

Usage:
  export OPENAI_API_KEY=sk-...
  python classification.py ./samples --outdir ./out_cls

Requires:
  pip install openai pandas packaging
"""

from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import openai._base_client

import pandas as pd


# =============================================================================
# OpenAI client bootstrap
# =============================================================================
try:
    from openai import OpenAI  # official SDK (2024+)
except Exception as e:
    print("Please `pip install openai` (>=1.0).", file=sys.stderr)
    raise

# =============================================================================
# Data model for the thing we save
# =============================================================================
@dataclass
class LLMResult:
    doc_id: str
    source_file: str
    predicted_type: str
    dated: str
    confidence_reasoning: str
    key_facts: List[str]
    needs_human_review: bool
    model: str
    raw_response: Dict[str, Any]

# =============================================================================
# Utilities: payload loading, SDK compatibility helpers, JSON parsing
# =============================================================================
def iter_llm_payloads(root: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (path, payload_dict) for each *.llm.json in folder recursion."""
    pattern = os.path.join(root, "**", "*.llm.json")
    for path in glob.iglob(pattern, recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            yield path, data
        except Exception as e:
            print(f"Skipping {path}: {e}")

def _supports_structured_outputs() -> bool:
    """Return True if the installed openai SDK likely supports response_format json_schema."""
    try:
        import openai
        from packaging import version  # pip install packaging
        v = getattr(openai, "__version__", "0.0.0")
        return version.parse(v) >= version.parse("1.14.0")
    except Exception:
        return False

def _extract_text_from_resp(resp: Any) -> Optional[str]:
    """Try multiple SDK response shapes to recover the textual JSON."""
    try:
        if getattr(resp, "output_text", None):
            return resp.output_text
        if getattr(resp, "output", None):
            chunks = []
            for item in resp.output:
                for part in getattr(item, "content", []):
                    if getattr(part, "type", None) in ("output_text", "text") and getattr(part, "text", None):
                        chunks.append(part.text)
            if chunks:
                return "".join(chunks)
        if getattr(resp, "choices", None):
            ch = resp.choices[0]
            if hasattr(ch, "message"):
                return ch.message.get("content") or getattr(ch.message, "content", None)
            return ch.get("message", {}).get("content")
    except Exception:
        pass
    return None

def _parse_json_safely(txt: str) -> Dict[str, Any]:
    """Parse JSON; if the model wrapped it in prose, extract the first {...} block."""
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        return json.loads(m.group(0))
    raise RuntimeError("Could not parse model output as JSON")

def call_llm_with_schema(client: OpenAI, model: str, system_prompt: str,
                         schema: Dict[str, Any], user_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generic SO/JSON call that reuses compatibility fallbacks."""
    print(f"Using model: {model}")
    use_structured = _supports_structured_outputs()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]
    if use_structured:
        try:
            resp = client.responses.create(
                model=model,
                input=messages,
                response_format={"type": "json_schema", "json_schema": schema, "strict": True},
                temperature=0,
            )
            txt = _extract_text_from_resp(resp)
            if txt:
                return _parse_json_safely(txt)
            try:
                first = resp.output[0].content[0]
                if getattr(first, "type", "") == "output_json" and getattr(first, "json", None):
                    return first.json
            except Exception:
                pass
        except Exception:
            pass
    # fallback: plain prompt enforcing JSON
    fallback = "Return ONLY valid JSON that matches this schema: " + json.dumps(schema["schema"])
    msgs = [
        {"role": "system", "content": system_prompt + " " + fallback},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    try:
        resp = client.responses.create(model=model, input=msgs, temperature=0)
    except Exception:
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=0)
    txt = _extract_text_from_resp(resp)
    if not txt:
        raise RuntimeError("Empty response from model")
    return _parse_json_safely(txt)

# =============================================================================
# Analysis pack (compact evidence we pass to every step)
# =============================================================================
CATEGORIES = [
    "Prescription", "Physician Notes", "Orders", "Compliance Report",
    "Delivery Receipt", "Study Report", "Other"
]

def _mk_keywords(payload: Dict[str, Any]) -> List[str]:
    """Extract high-signal keywords/codes/names from llm.json."""
    sig = payload.get("signals", {}) or {}
    kws = []
    kws += sig.get("icd10", []) or []
    kws += sig.get("cpt", []) or []
    kv = sig.get("kv", {}) or {}
    kws += [f"{k}:{v}" for k, v in kv.items() if isinstance(v, str) and len(v) <= 64]
    kws += (sig.get("page_headings") or [])[:5]

    text = (payload.get("fulltext_budgeted", {}) or {}).get("text", "")
    for ln in (text.splitlines() if text else []):
        if re.search(r"\b\d{1,4}\s?(mg|mcg|g|ml|units?)\b", ln, re.I):
            kws.append(ln.strip()[:120])
    
    seen = set(); out = []
    for k in kws:
        k2 = k.strip()
        if k2 and k2 not in seen:
            seen.add(k2); out.append(k2)
    return out[:32]

def build_analysis_pack(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compact pack we pass to the LLM in every step."""
    return {
        "doc_id": payload.get("doc_id"),
        "allowed_types": payload.get("allowed_types") or CATEGORIES,
        "signals": payload.get("signals", {}),
        "snippets": payload.get("snippets", [])[:4],
        "text_excerpt": (payload.get("fulltext_budgeted", {}) or {}).get("text", "")[:2500],
        "keywords": _mk_keywords(payload),
        "hints": payload.get("hints", {})
    }

# =============================================================================
# STEP 1 — CLASSIFY: prompt + schema + function
# =============================================================================
SYSTEM_PROMPT = """
<instructions>
You are an expert medical professional who can classify prescriptions, physician notes, study reports, orders, delivery receipts and compliance reports used in patient care and regulatory compliance.  

</instructions>

<approach>

1. First Identify for any obvious headings or explicit headers explaining the nature of the document
2. Then look for any medical code indicating any diagnosis, procedures, equipments or drugs
3. Identify key terminalogies and formatting patterns
4. Extract all the relevant dates from the file
5. Assess the content flow and the document structure

</approach>

<categories>

1. "Prescription" - Medication orders with Rx symbols, drug names, dosages, DEA numbers, CPAP DME
2. "Physician Notes" - Clinical notes with SOAP format, patient history, symptoms, assessments, complaints
3. "Orders" - Laboratory, imaging, or equipment requests with CPT/HCPCS codes
4. "Compliance Report" - Regulatory compliance documentation with usage statistics and compliance periods
5. "Delivery Receipt" - Bills, invoices, payment records with financial amounts
6. "Study Report" - Clinical studies, research reports with methodology and participant data
7. "Other" - Documents that don't fit above categories

</categories>

<indicator>

- Prescriptions: "Rx" symbols, medication names, "Sig:", dosage instructions, pharmacy info
- Physician Notes: "SOAP", "Assessment", "Plan", vital signs, patient complaints, patient history, Q&A
- Orders: "Order for", test names, procedure codes, "STAT", equipment requests
- Compliance Reports:"Compliance period", usage percentages, regulatory citations
- Delivery Receipts: dollar amounts, "Amount due", billing codes, insurance claims
- Study Reports: "Protocol", "participants", "methodology", "inclusion criteria" and research

</indicator>

<examples>

Prescription Examples:
- "Rx: Lisinopril 10mg, Take 1 tablet daily, #30, Refills: 2"
- "DEA# AB1234567, NDC: 12345-678-90, Sig: Apply CPAP nightly"
- "Type of Interface and Accessories: (A7038) Disposable filter (2/1mo) (MASK FIT TO PATIENTS COMFORT)"
- "CPAP DME (Order ID: 701783674)"

Physician Notes Examples:
- Document with "SOAP Note" header
- "Subjective: Patient complains of chest pain. Objective: BP 140/90, HR 82..."
- Patient questionnaire: "Over the last 2 weeks, how often have you felt down? 1-Not at all 2-Several days..."
- Clinical assessment with progress notes, patient history and treatment plans and 

Orders Examples:
- Header sections containing "Standard Order Written"
- "Proc. Code Item Name/Narrative"
- "Please fax a confirmation that you received this order"
- Equipment request forms for medical devices 
- "EQUIPMENT/SERVICES"

Study Report Examples:
- Procedure of conducting study in detail - "PROCEDURE IN DETAIL: The patient underwent overnight home...'
- Document with diagnostic conclusions and methodology sections
- Research overview with statistical analysis and findings
- Sections containing sleep stage summary

Delivery Receipt Examples:
- "Amount due: $125.00, Insurance claim #ABC123"
- Invoice with billing codes and payment terms
- Receipt showing medical equipment delivery charges

Compliance Report Examples:
- Sections indicating Initial compliance period
- Monitoring reports with usage statistics and compliance metrics

</examples>

<important>
- Base classification on concrete evidence, not assumptions
- If SOAP structure is present (e.g., “Subjective/Objective/Assessment/Plan” or “SOAP”), classify as “Physician Notes”, even if the document lists many medications.
- If headers/titles are present, they override content-based classification
- Extract dates in standardized YYYY-MM-DD format when possible
- Set needs_human_review=true if confidence is below 80% or evidence conflicts
- Use exact category names as listed above

</important>
"""

CLASSIFY_PROMPT = SYSTEM_PROMPT + """
<task>
You will be given an analysis pack containing:
- allowed_types: list of valid labels
- signals: structured cues (codes, kv, soap, pharmacy cues, rx/order/report cues, ocr_quality)
- snippets: 1-4 short excerpts
- text_excerpt: up to ~2500 chars of normalized text
- keywords: high-signal tokens extracted for you
- hints: meta (e.g., ocr_used, pages)
</task>

<output_format>
Return a JSON object with fields:
- predicted_type: string (one of allowed_types)
- dated: string (YYYY-MM-DD if found else empty)
- confidence_reasoning: string (cite headings, KVs, codes, cues)
- key_facts: string[] (MRN, DOS, codes, notable cues)
- needs_human_review: boolean
</output_format>

<analysis_guidelines>
When you analyze the analysis pack, consider both the cues and the content:

Content Pattern Recognition:
- Questions like "Little interest or pleasure in doing things?" → Physician Notes
- Conclusions drawn from tests or diagnostic results → Study Report
- CPAP equipment or Rx medications mentioned or DEA # → Prescription
- Equipment names and/or medical services requested → Orders

Decision Process:
- Weigh both structured cues and content patterns
- Prioritize explicit headers and titles when present
- Use medical codes and terminology as supporting evidence
- Come to a balanced conclusion based on all available evidence
</analysis_guidelines>

<classification_priority>
1. Document headers and explicit titles (highest priority)
2. SOAP structure indicators
3. Medical codes and structured cues
4. Content patterns and terminology
5. Keywords and contextual hints
</classification_priority>
"""

SCHEMA_CLASSIFY = {
    "name": "med_doc_classify_v1",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_type": {"type": "string", "enum": CATEGORIES},
            "dated": {"type": "string"},
            "confidence_reasoning": {"type": "string"},
            "key_facts": {"type": "array", "items": {"type": "string"}},
            "needs_human_review": {"type": "boolean"}
        },
        "required": ["predicted_type"],
        "additionalProperties": False
    },
    "strict": True
}

def step1_classify(client: OpenAI, model: str, analysis_pack: Dict[str, Any]) -> Dict[str, Any]:
    return call_llm_with_schema(
        client, model, CLASSIFY_PROMPT, SCHEMA_CLASSIFY,
        {"analysis_pack": analysis_pack}
    )

# =============================================================================
# STEP 2 — REVIEW: prompt + schema + function
# =============================================================================
REVIEW_PROMPT = """
You are the reviewer conducting a critical second look at the classification. Given:
- the original analysis pack
- the model's initial classification result

CRITICAL REVIEW CHECKLIST:
1. Document Title takes precedence over content. If the explicit header says "Compliance Report" then it is most likely a compliance report.
2. Presence of Progress Note or Pro Note most likely indicates it being a Physician Notes
3. If there are questions asked like "after lunch?" or "usual sleep?" it is Physician Notes
3. If a file is handwritten, contains dea # or cpap dme it is likely a Prescription 
4. Presence of "Proc. Code: Item Name / Narrative" or "EQUIPMENT / SERVICES" indicates an Order

COMMON MISCLASSIFICATION PATTERNS TO CATCH:
- Compliance Report incorrectly labeled as "Study Report"
- Prescriptions incorrectly labeled as "Orders"
- Physician Notes incorrectly labeled as  "Study Report"

Return JSON assessing whether you agree with the initial classification. If the document contains sleep study diagnostic data or conclusions, it should be "Study Report" regardless of other content.

Return JSON with: agree (bool), issues[] (strings), suggested_type (enum), confidence (High/Medium/Low),
needs_human_review (bool), missing_evidence[] (strings).
"""

SCHEMA_REVIEW = {
    "name": "med_doc_review_v1",
    "schema": {
        "type": "object",
        "properties": {
            "agree": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "suggested_type": {"type": "string", "enum": CATEGORIES},
            "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "needs_human_review": {"type": "boolean"},
            "missing_evidence": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["agree", "suggested_type", "needs_human_review"],
        "additionalProperties": False
    },
    "strict": True
}

def step2_review(client: OpenAI, model: str, analysis_pack: Dict[str, Any],
                 initial_result: Dict[str, Any]) -> Dict[str, Any]:
    return call_llm_with_schema(
        client, model, REVIEW_PROMPT, SCHEMA_REVIEW,
        {"analysis_pack": analysis_pack, "initial_result": initial_result}
    )

# =============================================================================
# STEP 3 — REVISE: prompt + schema + function
# =============================================================================
REVISE_PROMPT = """
You are the reviser. Using the analysis pack and the review findings,
produce a corrected classification JSON with fields:
- predicted_type
- dated
- confidence_reasoning
- key_facts[]
- needs_human_review

Make sure the categories are proper and aligned - for example it should be "Physician Notes" and not "Physician Order". 
Revise all incorrect values and produce the best output.
If the reviewer disagreed or flagged issues, address them explicitly in confidence_reasoning.
"""

SCHEMA_REVISE = {
    "name": "med_doc_revise_v1",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_type": {"type": "string", "enum": CATEGORIES},
            "dated": {"type": "string"},
            "confidence_reasoning": {"type": "string"},
            "key_facts": {"type": "array", "items": {"type": "string"}},
            "needs_human_review": {"type": "boolean"}
        },
        "required": ["predicted_type", "needs_human_review"],
        "additionalProperties": False
    },
    "strict": True
}

def step3_revise(client: OpenAI, model: str, analysis_pack: Dict[str, Any],
                 initial_result: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    return call_llm_with_schema(
        client, model, REVISE_PROMPT, SCHEMA_REVISE,
        {"analysis_pack": analysis_pack, "initial_result": initial_result, "review": review}
    )

# =============================================================================
# STEP 4 — VALIDATE: prompt + schema + function
# =============================================================================
VALIDATE_PROMPT = """
You are the validator. Enforce these rules:
- Headers/titles override content heuristics when explicit
- Prescriptions require pharmacy/DEA/refill/dispense cues or Rx form cues
- Set needs_human_review=true if evidence conflicts or confidence is Low

Validate that the output is complete and adheres to the schema. Validate that there are no errors in the output.

Given the analysis pack and revised decision, return a FINAL JSON with the same fields plus rule_violations[] (if any).
"""

SCHEMA_VALIDATE = {
    "name": "med_doc_validate_v1",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_type": {"type": "string", "enum": CATEGORIES},
            "dated": {"type": "string"},
            "confidence_reasoning": {"type": "string"},
            "key_facts": {"type": "array", "items": {"type": "string"}},
            "needs_human_review": {"type": "boolean"},
            "rule_violations": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["predicted_type", "needs_human_review"],
        "additionalProperties": False
    },
    "strict": True
}

def step4_validate(client: OpenAI, model: str, analysis_pack: Dict[str, Any],
                   initial_result: Dict[str, Any], review: Dict[str, Any],
                   revised_result: Dict[str, Any]) -> Dict[str, Any]:
    return call_llm_with_schema(
        client, model, VALIDATE_PROMPT, SCHEMA_VALIDATE,
        {
            "analysis_pack": analysis_pack,
            "initial_result": initial_result,
            "review": review,
            "revised_result": revised_result
        }
    )

# =============================================================================
# Orchestrator: run 4 steps and produce final result
# =============================================================================
def classify_with_validation(client: OpenAI, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    pack = build_analysis_pack(payload)

    # 1) classify
    initial = step1_classify(client, model, pack)

    # 2) review
    review = step2_review(client, model, pack, initial)

    # 3) revise
    revised = step3_revise(client, model, pack, initial, review)

    # 4) validate 
    final_res = step4_validate(client, model, pack, initial, review, revised)

    out = {
        "predicted_type": final_res.get("predicted_type", revised.get("predicted_type", initial.get("predicted_type", "Other"))),
        "dated": final_res.get("dated", revised.get("dated", "")),
        "confidence_reasoning": final_res.get("confidence_reasoning", revised.get("confidence_reasoning", initial.get("confidence_reasoning", ""))),
        "key_facts": final_res.get("key_facts", revised.get("key_facts", initial.get("key_facts", []))),
        "needs_human_review": bool(final_res.get("needs_human_review", revised.get("needs_human_review", False))),
    }

    if final_res.get("rule_violations"):
        out["needs_human_review"] = True
        rv = "; ".join(final_res["rule_violations"])
        out["confidence_reasoning"] = (out["confidence_reasoning"] + f" | Rule checks: {rv}").strip()
    return out

# =============================================================================
# Adapter to your existing result dataclass and file writers
# =============================================================================
def classify_payload(path: str, payload: Dict[str, Any], client: OpenAI, model: str) -> LLMResult:
    result = classify_with_validation(client, model, payload)
    doc_id = payload.get("doc_id") or os.path.basename(path).split(".")[0]
    return LLMResult(
        doc_id=doc_id,
        source_file=path,
        predicted_type=result.get("predicted_type", "unknown"),
        dated=result.get("dated", ""),
        confidence_reasoning=result.get("confidence_reasoning", ""),
        key_facts=list(result.get("key_facts", [])),
        needs_human_review=bool(result.get("needs_human_review", False)),
        model=model,
        raw_response=result,
    )

def write_outputs(outdir: str, results: List[LLMResult]) -> Tuple[str, str]:
    jsonl_path = os.path.join(outdir, "classifications.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in results:
            jf.write(json.dumps({
                "doc_id": r.doc_id,
                "source_file": r.source_file,
                "predicted_type": r.predicted_type,
                "dated": r.dated,
                "confidence_reasoning": r.confidence_reasoning,
                "key_facts": r.key_facts,
                "needs_human_review": r.needs_human_review,
                "model": r.model
            }, ensure_ascii=False) + "\n")

    try:
        df = pd.DataFrame([{
            "doc_id": r.doc_id,
            "predicted_type": r.predicted_type,
            "dated": r.dated,
            "needs_human_review": r.needs_human_review,
            "source_file": os.path.basename(r.source_file)
        } for r in results])
        print(df.head().to_string(index=False))
    except Exception:
        pass

    return jsonl_path


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Classify medical documents using a 4-step LLM validation loop")
    ap.add_argument("input", help="Folder containing *.llm.json (or PDFs with --ingest-pdfs)")
    ap.add_argument("--outdir", default="out_cls", help="Where to write classifications.jsonl/csv")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model (e.g., gpt-4.1, gpt-4.1-mini, gpt-5)")
    ap.add_argument("--rate-limit", type=float, default=5.0, help="Seconds to sleep between calls")
    ap.add_argument("--max", type=int, default=0, help="Max number of files to process (0 = all)")
    ap.add_argument("--batch-size", type=int, default=5, help="Process files in batches")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    original_init = openai._base_client.SyncHttpxClientWrapper.__init__
    
    def patched_init(self, **kwargs):
        kwargs.pop('proxies', None)
        return original_init(self, **kwargs)
    
    openai._base_client.SyncHttpxClientWrapper.__init__ = patched_init    

    client = OpenAI()
    llm_root = args.input

    all_files = list(iter_llm_payloads(llm_root))
    results: List[LLMResult] = []
    count = 0

    for i in range(0, len(all_files), args.batch_size):
        batch = all_files[i:i + args.batch_size]
        batch_num = i//args.batch_size + 1
        print(f"Processing batch {batch_num}: {len(batch)} files")

        for path, payload in batch:
            try:
                res = classify_payload(path, payload, client, args.model)
                results.append(res)
                count += 1
                print(f"[final] {os.path.basename(path)} -> {res.predicted_type} (review={res.needs_human_review})")
            except Exception as e:
                print(f"[error] {path}: {e}")
            
            if args.rate_limit > 0:
                time.sleep(args.rate_limit)
            
            if args.max and count >= args.max:
                break

        if i + args.batch_size < len(all_files):
            print(f"Completed batch {batch_num}, pausing 10 seconds...")
            time.sleep(10)
        
        if args.max and count >= args.max:
            break

    jsonl_path = write_outputs(args.outdir, results)
    print("\nWrote:")
    print("  ", jsonl_path)

if __name__ == "__main__":
    main()