"""
test.py — Synthetic Document Generation + Redaction Pipeline Evaluation
========================================================================

What this does:
  1. Generates synthetic PDFs with known PII (ground truth is auto-known)
  2. Runs the SanitizationPipeline on each synthetic PDF
  3. Compares detected entities against ground truth
  4. Computes Precision, Recall, F1, FNR per entity type and overall
  5. Scans redacted output PDFs for any leaked PII patterns
  6. Prints a full evaluation report

Usage:
    pip install faker reportlab pymupdf
    python test.py
"""

from __future__ import annotations

import os
import re
import json
import random
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    from faker import Faker
except ImportError:
    raise SystemExit("Install faker first:  pip install faker")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ImportError:
    raise SystemExit("Install reportlab first:  pip install reportlab")

try:
    import fitz  # PyMuPDF — for leak scanning the redacted PDF
except ImportError:
    raise SystemExit("Install pymupdf first:  pip install pymupdf")

# ── Your pipeline ─────────────────────────────────────────────────────────────
from pipeline import SanitizationPipeline


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GroundTruthEntity:
    """One known-sensitive item planted in the synthetic document."""
    text: str
    tag:  str


@dataclass
class SyntheticDocument:
    """A generated PDF path together with its full ground-truth entity list."""
    pdf_path:     str
    ground_truth: list[GroundTruthEntity]
    doc_type:     str   # e.g. "medical", "financial", "hr"


@dataclass
class PageMetrics:
    """Evaluation results for a single document."""
    doc_type:  str
    pdf_path:  str
    TP:        int = 0
    FP:        int = 0
    FN:        int = 0
    leaks:     dict = field(default_factory=dict)
    per_tag:   dict = field(default_factory=dict)   # tag → {TP,FP,FN}

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def precision(self) -> float:
        return self.TP / (self.TP + self.FP) if (self.TP + self.FP) else 0.0

    @property
    def recall(self) -> float:
        return self.TP / (self.TP + self.FN) if (self.TP + self.FN) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r) / (p + r) if (p + r) else 0.0

    @property
    def fnr(self) -> float:
        """False Negative Rate — the most critical metric for redaction."""
        return self.FN / (self.TP + self.FN) if (self.TP + self.FN) else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FAKE-DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

fake_in = Faker("en_IN")   # Indian locale  → Aadhaar / PAN style data
fake_us = Faker("en_US")   # US locale      → SSN / credit-card style data
Faker.seed(42)
random.seed(42)


def _fake_pan() -> str:
    """Generate a syntactically valid fake Indian PAN."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(random.choices(letters, k=5))
        + "".join(random.choices("0123456789", k=4))
        + random.choice(letters)
    )


def _fake_aadhaar() -> str:
    """Generate a fake 12-digit Aadhaar (starts with 2-9)."""
    first = str(random.randint(2, 9))
    rest  = "".join(random.choices("0123456789", k=11))
    raw   = first + rest
    return f"{raw[:4]} {raw[4:8]} {raw[8:]}"


def _fake_account() -> str:
    return "".join(random.choices("0123456789", k=random.randint(10, 14)))


def _fake_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DOCUMENT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

def _make_medical_doc() -> tuple[str, list[GroundTruthEntity]]:
    """Return (text_body, ground_truth_entities)."""
    name    = fake_in.name()
    dob     = fake_in.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d/%m/%Y")
    phone   = fake_in.phone_number()
    email   = fake_us.email()
    address = fake_in.address().replace("\n", ", ")
    aadhaar = _fake_aadhaar()
    diag    = random.choice([
        "Type 2 Diabetes Mellitus",
        "Hypertensive Heart Disease",
        "Chronic Kidney Disease Stage 3",
        "Major Depressive Disorder",
    ])
    med     = random.choice(["Metformin 500mg", "Amlodipine 5mg", "Atorvastatin 10mg"])

    text = f"""
PATIENT MEDICAL RECORD
======================

Patient Name       : {name}
Date of Birth      : {dob}
Contact Number     : {phone}
Email Address      : {email}
Residential Address: {address}
Aadhaar Number     : {aadhaar}

CLINICAL NOTES
--------------
Primary Diagnosis  : {diag}
Prescribed Drug    : {med}
Attending Physician: Dr. Rajesh Kumar (keep — not patient PII)

This record is confidential and intended solely for the treating physician.
"""

    gt = [
        GroundTruthEntity(name,    "PERSON"),
        GroundTruthEntity(dob,     "DOB"),
        GroundTruthEntity(phone,   "PHONE"),
        GroundTruthEntity(email,   "EMAIL"),
        GroundTruthEntity(address, "ADDRESS"),
        GroundTruthEntity(aadhaar, "AADHAAR"),
        GroundTruthEntity(diag,    "MEDICAL_HISTORY"),
        GroundTruthEntity(med,     "MEDICAL_HISTORY"),
    ]
    return text, gt


def _make_financial_doc() -> tuple[str, list[GroundTruthEntity]]:
    name    = fake_in.name()
    pan     = _fake_pan()
    account = _fake_account()
    phone   = fake_in.phone_number()
    email   = fake_us.email()
    salary  = f"Rs. {random.randint(30000, 200000):,}"
    address = fake_in.address().replace("\n", ", ")

    text = f"""
SALARY DISBURSEMENT RECORD
==========================

Employee Name      : {name}
PAN Number         : {pan}
Bank Account No.   : {account}
Registered Mobile  : {phone}
Email              : {email}
Monthly Salary     : {salary}
Office Address     : {address}

This document contains confidential payroll information.
Unauthorised disclosure is strictly prohibited.
"""

    gt = [
        GroundTruthEntity(name,    "PERSON"),
        GroundTruthEntity(pan,     "PAN"),
        GroundTruthEntity(account, "ACCOUNT_NUMBER"),
        GroundTruthEntity(phone,   "PHONE"),
        GroundTruthEntity(email,   "EMAIL"),
        GroundTruthEntity(salary,  "MONEY"),
        GroundTruthEntity(address, "ADDRESS"),
    ]
    return text, gt


def _make_hr_doc() -> tuple[str, list[GroundTruthEntity]]:
    name    = fake_in.name()
    dob     = fake_in.date_of_birth(minimum_age=22, maximum_age=55).strftime("%d/%m/%Y")
    phone   = fake_in.phone_number()
    email   = fake_us.email()
    address = fake_in.address().replace("\n", ", ")
    pan     = _fake_pan()
    aadhaar = _fake_aadhaar()

    text = f"""
EMPLOYEE ONBOARDING FORM
========================

Full Name          : {name}
Date of Birth      : {dob}
Personal Email     : {email}
Mobile Number      : {phone}
Current Address    : {address}
PAN Card           : {pan}
Aadhaar            : {aadhaar}

Emergency Contact  : Jane Doe  +91 9000000000  (anonymised for this test)

I hereby confirm that all details above are accurate.

Signature: ___________         Date: __________
"""

    gt = [
        GroundTruthEntity(name,    "PERSON"),
        GroundTruthEntity(dob,     "DOB"),
        GroundTruthEntity(email,   "EMAIL"),
        GroundTruthEntity(phone,   "PHONE"),
        GroundTruthEntity(address, "ADDRESS"),
        GroundTruthEntity(pan,     "PAN"),
        GroundTruthEntity(aadhaar, "AADHAAR"),
    ]
    return text, gt


def _make_legal_doc() -> tuple[str, list[GroundTruthEntity]]:
    name1   = fake_in.name()
    name2   = fake_in.name()
    address = fake_in.address().replace("\n", ", ")
    ip      = _fake_ip()
    email   = fake_us.email()
    phone   = fake_in.phone_number()
    account = _fake_account()

    text = f"""
NON-DISCLOSURE AGREEMENT
========================

This Agreement is entered into between {name1} ("Disclosing Party")
and {name2} ("Receiving Party").

Registered Address : {address}
Point of Contact   : {email}  |  {phone}
Payment Account    : {account}
System IP Address  : {ip}

Both parties agree to keep all shared information strictly confidential.
Breach of this agreement will attract legal proceedings.
"""

    gt = [
        GroundTruthEntity(name1,   "PERSON"),
        GroundTruthEntity(name2,   "PERSON"),
        GroundTruthEntity(address, "ADDRESS"),
        GroundTruthEntity(email,   "EMAIL"),
        GroundTruthEntity(phone,   "PHONE"),
        GroundTruthEntity(account, "ACCOUNT_NUMBER"),
        GroundTruthEntity(ip,      "IP_ADDRESS"),
    ]
    return text, gt


# Map doc type → generator function
DOC_GENERATORS = {
    "medical":   _make_medical_doc,
    "financial": _make_financial_doc,
    "hr":        _make_hr_doc,
    "legal":     _make_legal_doc,
}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PDF WRITER  (text → real PDF via reportlab)
# ══════════════════════════════════════════════════════════════════════════════

def _write_pdf(text: str, output_path: str) -> None:
    """Write `text` as a simple single-page PDF at `output_path`."""
    c    = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4
    c.setFont("Courier", 10)
    margin   = 50
    y        = h - margin
    line_h   = 14

    for line in text.split("\n"):
        # Wrap long lines
        while len(line) > 95:
            c.drawString(margin, y, line[:95])
            line = "    " + line[95:]
            y   -= line_h
            if y < margin:
                c.showPage()
                c.setFont("Courier", 10)
                y = h - margin
        c.drawString(margin, y, line)
        y -= line_h
        if y < margin:
            c.showPage()
            c.setFont("Courier", 10)
            y = h - margin

    c.save()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SYNTHETIC DOCUMENT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_documents(
    n_per_type: int = 2,
    output_dir: Optional[str] = None,
) -> list[SyntheticDocument]:
    """
    Generate `n_per_type` PDFs for each document type.
    Returns a list of SyntheticDocument objects.
    """
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="synth_docs_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    docs: list[SyntheticDocument] = []

    for doc_type, generator in DOC_GENERATORS.items():
        for i in range(n_per_type):
            text, gt = generator()
            pdf_name = out_dir / f"{doc_type}_{i+1}.pdf"
            _write_pdf(text, str(pdf_name))

            docs.append(SyntheticDocument(
                pdf_path=str(pdf_name),
                ground_truth=gt,
                doc_type=doc_type,
            ))
            print(f"  [GEN]  {pdf_name.name}  ({len(gt)} ground-truth entities)")

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# 6.  LEAK SCANNER  (post-redaction regex scan)
# ══════════════════════════════════════════════════════════════════════════════

LEAK_PATTERNS: dict[str, str] = {
    "EMAIL":   r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    "PHONE":   r'\b(\+91[\-\s]?)?[6-9]\d{9}\b',
    "AADHAAR": r'\b[2-9]\d{3}\s\d{4}\s\d{4}\b',
    "PAN":     r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
    "IP":      r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "ACCOUNT": r'\b\d{10,14}\b',
}


def scan_for_leaks(pdf_path: str) -> dict[str, list[str]]:
    """
    Extract text from a (redacted) PDF and run regex patterns over it.
    Returns a dict of tag → [leaked_values].  Empty dict = clean.
    """
    try:
        doc  = fitz.open(pdf_path)
        text = " ".join(page.get_text() for page in doc)
    except Exception as e:
        return {"ERROR": [str(e)]}

    leaks: dict[str, list[str]] = {}
    for label, pattern in LEAK_PATTERNS.items():
        found = re.findall(pattern, text)
        if found:
            leaks[label] = found
    return leaks


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace — for fuzzy matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def evaluate_document(
    synth_doc:    SyntheticDocument,
    pipeline:     SanitizationPipeline,
    redacted_dir: Path,
) -> PageMetrics:
    """
    Run the pipeline on one synthetic PDF, compare against ground truth,
    scan the redacted output for leaks, and return PageMetrics.
    """
    metrics = PageMetrics(doc_type=synth_doc.doc_type, pdf_path=synth_doc.pdf_path)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print(f"\n  [EVAL]  Running pipeline on {Path(synth_doc.pdf_path).name} …")
    try:
        result = pipeline.ingest(synth_doc.pdf_path)
    except Exception as e:
        print(f"  [ERROR] Pipeline failed: {e}")
        metrics.FN = len(synth_doc.ground_truth)
        return metrics

    # ── Build sets for comparison ─────────────────────────────────────────────
    detected_texts = {_normalise(e.text) for e in result.entities}
    gt_texts       = {_normalise(g.text) for g in synth_doc.ground_truth}

    TP_texts = detected_texts & gt_texts
    FP_texts = detected_texts - gt_texts
    FN_texts = gt_texts - detected_texts

    metrics.TP = len(TP_texts)
    metrics.FP = len(FP_texts)
    metrics.FN = len(FN_texts)

    # ── Per-tag breakdown ─────────────────────────────────────────────────────
    tag_map = {_normalise(g.text): g.tag for g in synth_doc.ground_truth}
    det_tag_map = {_normalise(e.text): e.tag for e in result.entities}

    all_tags = {g.tag for g in synth_doc.ground_truth}
    for tag in all_tags:
        gt_tag  = {_normalise(g.text) for g in synth_doc.ground_truth if g.tag == tag}
        det_tag = {t for t in detected_texts if det_tag_map.get(t) == tag}

        tp = len(gt_tag & detected_texts)   # detected regardless of tag label
        fn = len(gt_tag - detected_texts)
        fp = len(det_tag - gt_tag)
        metrics.per_tag[tag] = {"TP": tp, "FP": fp, "FN": fn}

    # ── Export redacted PDF and scan for leaks ────────────────────────────────
    redacted_path = redacted_dir / (Path(synth_doc.pdf_path).stem + "_redacted.pdf")
    try:
        result.export(str(redacted_path))
        metrics.leaks = scan_for_leaks(str(redacted_path))
    except Exception as e:
        print(f"  [WARN]  Could not export/scan redacted PDF: {e}")

    # ── Print per-doc summary ─────────────────────────────────────────────────
    print(f"         GT={len(gt_texts)}  Detected={len(detected_texts)}  "
          f"TP={metrics.TP}  FP={metrics.FP}  FN={metrics.FN}")
    print(f"         Precision={metrics.precision:.2%}  "
          f"Recall={metrics.recall:.2%}  F1={metrics.f1:.2%}  "
          f"FNR={metrics.fnr:.2%}")
    if metrics.leaks:
        print(f"  ⚠️  LEAKS DETECTED: {metrics.leaks}")
    else:
        print(f"         ✅  No regex-detectable leaks in redacted output")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 8.  REPORT PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_report(all_metrics: list[PageMetrics]) -> None:
    SEP  = "═" * 72
    SEP2 = "─" * 72

    print(f"\n{SEP}")
    print("  REDACTION EVALUATION REPORT")
    print(SEP)

    # ── Per-document table ────────────────────────────────────────────────────
    print(f"\n{'Document':<35} {'P':>7} {'R':>7} {'F1':>7} {'FNR':>7} {'Leaks':>6}")
    print(SEP2)
    for m in all_metrics:
        name   = Path(m.pdf_path).name[:34]
        leaks  = "YES ⚠️" if m.leaks else "none"
        print(f"{name:<35} {m.precision:>7.1%} {m.recall:>7.1%} "
              f"{m.f1:>7.1%} {m.fnr:>7.1%} {leaks:>6}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total_TP = sum(m.TP for m in all_metrics)
    total_FP = sum(m.FP for m in all_metrics)
    total_FN = sum(m.FN for m in all_metrics)

    agg_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
    agg_recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
    agg_f1        = ((2 * agg_precision * agg_recall) /
                     (agg_precision + agg_recall) if (agg_precision + agg_recall) else 0)
    agg_fnr       = total_FN / (total_TP + total_FN) if (total_TP + total_FN) else 0

    print(SEP2)
    print(f"{'AGGREGATE':<35} {agg_precision:>7.1%} {agg_recall:>7.1%} "
          f"{agg_f1:>7.1%} {agg_fnr:>7.1%}")

    # ── Per entity-type breakdown ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PER ENTITY-TYPE BREAKDOWN (across all documents)")
    print(SEP)
    tag_agg: dict[str, dict] = {}
    for m in all_metrics:
        for tag, counts in m.per_tag.items():
            agg = tag_agg.setdefault(tag, {"TP": 0, "FP": 0, "FN": 0})
            for k in ("TP", "FP", "FN"):
                agg[k] += counts[k]

    print(f"\n{'Entity Type':<22} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'Precision':>10} {'Recall':>8} {'Status':>10}")
    print(SEP2)
    for tag in sorted(tag_agg):
        tp = tag_agg[tag]["TP"]
        fp = tag_agg[tag]["FP"]
        fn = tag_agg[tag]["FN"]
        p  = tp / (tp + fp) if (tp + fp) else 0
        r  = tp / (tp + fn) if (tp + fn) else 0
        status = "✅ Good" if r >= 0.80 else ("⚠️  OK" if r >= 0.50 else "❌ Poor")
        print(f"{tag:<22} {tp:>4} {fp:>4} {fn:>4} {p:>10.1%} {r:>8.1%} {status:>10}")

    # ── Leak summary ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  LEAK SCAN SUMMARY")
    print(SEP)
    total_docs_with_leaks = sum(1 for m in all_metrics if m.leaks)
    if total_docs_with_leaks == 0:
        print("\n  ✅  All redacted documents passed the regex leak scan.")
    else:
        print(f"\n  ⚠️  {total_docs_with_leaks}/{len(all_metrics)} documents have detectable leaks:\n")
        for m in all_metrics:
            if m.leaks:
                print(f"    {Path(m.pdf_path).name}")
                for leak_type, values in m.leaks.items():
                    print(f"      {leak_type}: {values}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  INTERPRETATION")
    print(SEP)
    print(f"""
  Overall Recall  : {agg_recall:.1%}  {'✅ Acceptable (>= 80%)' if agg_recall >= 0.8 else '❌ Too low — PII is leaking through'}
  Overall FNR     : {agg_fnr:.1%}  {'✅ Good (< 20% missed)' if agg_fnr < 0.20 else '⚠️  Too many missed entities'}
  Overall Precision: {agg_precision:.1%}  {'✅ Good' if agg_precision >= 0.7 else '⚠️  High over-redaction'}

  Key:
    Recall / FNR   — Most important. Low recall = missed PII = privacy risk.
    Precision      — Over-redaction. Low precision = too much blacked out.
    F1             — Harmonic mean. Balanced view of both.
    Leaks          — Regex scan of exported redacted PDF for surviving PII.
""")

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "aggregate": {
            "precision": round(agg_precision, 4),
            "recall":    round(agg_recall,    4),
            "f1":        round(agg_f1,        4),
            "fnr":       round(agg_fnr,       4),
            "TP": total_TP, "FP": total_FP, "FN": total_FN,
        },
        "per_entity_type": {
            tag: {
                **counts,
                "precision": round(counts["TP"] / (counts["TP"] + counts["FP"]), 4)
                             if (counts["TP"] + counts["FP"]) else 0,
                "recall":    round(counts["TP"] / (counts["TP"] + counts["FN"]), 4)
                             if (counts["TP"] + counts["FN"]) else 0,
            }
            for tag, counts in tag_agg.items()
        },
        "per_document": [
            {
                "name":      Path(m.pdf_path).name,
                "doc_type":  m.doc_type,
                "precision": round(m.precision, 4),
                "recall":    round(m.recall,    4),
                "f1":        round(m.f1,        4),
                "fnr":       round(m.fnr,       4),
                "TP": m.TP, "FP": m.FP, "FN": m.FN,
                "leaks": m.leaks,
            }
            for m in all_metrics
        ],
    }

    report_path = Path("eval_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  📄  Full report saved to: {report_path.resolve()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 72)
    print("  SYNTHETIC DOCUMENT GENERATION + REDACTION EVALUATION")
    print("═" * 72)

    # ── Directories ───────────────────────────────────────────────────────────
    synth_dir    = Path("test_synthetic_docs")
    redacted_dir = Path("test_redacted_docs")
    synth_dir.mkdir(exist_ok=True)
    redacted_dir.mkdir(exist_ok=True)

    # ── Step 1: Generate synthetic PDFs ──────────────────────────────────────
    print("\n[STEP 1]  Generating synthetic PDFs …\n")
    docs = generate_synthetic_documents(
        n_per_type=2,          # 2 docs per type × 4 types = 8 PDFs total
        output_dir=str(synth_dir),
    )
    print(f"\n  → {len(docs)} synthetic PDFs written to: {synth_dir.resolve()}")

    # ── Step 2: Initialise pipeline ───────────────────────────────────────────
    print("\n[STEP 2]  Initialising SanitizationPipeline …")
    pipeline = SanitizationPipeline(
        registry_path="test_registry.json",
        ocr_backend="tesseract",
        dpi=300,
        conf_threshold=0.30,   # lower threshold → catch more, useful for eval
    )

    # ── Step 3: Evaluate each document ────────────────────────────────────────
    print("\n[STEP 3]  Running evaluation …")
    all_metrics: list[PageMetrics] = []
    for doc in docs:
        m = evaluate_document(doc, pipeline, redacted_dir)
        all_metrics.append(m)

    # ── Step 4: Print report ──────────────────────────────────────────────────
    print("\n[STEP 4]  Building report …")
    print_report(all_metrics)


if __name__ == "__main__":
    main()